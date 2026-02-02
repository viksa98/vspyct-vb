"""
Benchmark Evaluation Script: VSPYCT-GP vs VSPYCT on OpenML CTR23 Datasets

This script evaluates and compares VSPYCT-GP and VSPYCT models on regression
benchmark datasets from the OpenML CTR23 suite.

Usage:
    python experiments/benchmark_evaluation.py --n_datasets 20 --output_dir experiments/results
"""

import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['PYRO_VALIDATION'] = 'false'

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.openml_loader import OpenMLBenchmark, get_benchmark_datasets
from src.models.model import VSpyct, VSpyctGP


class SuppressOutput:
    """Context manager to suppress stdout/stderr during training."""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def evaluate_vspyct(model, X_test: torch.Tensor, y_test: torch.Tensor) -> Tuple[np.ndarray, Dict]:
    """Evaluate VSpyct model and return predictions and metrics."""
    predictions = model.predict(X_test)

    # Handle different output shapes
    if predictions.dim() == 3:
        # (n_samples, n_targets, n_mc_samples) -> average over MC and targets
        pred_mean = predictions.mean(dim=-1).mean(dim=-1)
    elif predictions.dim() == 2:
        # (n_samples, n_mc_samples) -> average over MC
        pred_mean = predictions.mean(dim=-1)
    else:
        pred_mean = predictions

    pred_np = pred_mean.numpy().flatten()
    y_np = y_test.numpy().flatten()

    metrics = compute_metrics(y_np, pred_np)
    return pred_np, metrics


def evaluate_vspyct_gp(
    model,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    return_uncertainty: bool = True
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """Evaluate VSpyctGP model and return predictions, metrics, and uncertainties."""
    if return_uncertainty:
        predictions, uncertainties = model.predict(X_test, return_uncertainty=True)
        unc_np = uncertainties.numpy().flatten()
    else:
        predictions = model.predict(X_test)
        unc_np = np.zeros(len(predictions))

    pred_np = predictions.numpy().flatten()
    y_np = y_test.numpy().flatten()

    metrics = compute_metrics(y_np, pred_np)

    # Add uncertainty metrics
    metrics['mean_uncertainty'] = float(np.mean(unc_np))
    metrics['std_uncertainty'] = float(np.std(unc_np))

    return pred_np, metrics, unc_np


def run_single_fold(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    model_params: Dict,
    gp_params: Dict,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run evaluation on a single train/test fold."""
    results = {}

    # Train and evaluate VSpyct
    vspyct = VSpyct(**model_params)

    start_time = time.time()
    if verbose:
        vspyct.fit(X_train, y_train)
    else:
        with SuppressOutput():
            vspyct.fit(X_train, y_train)
    train_time_vspyct = time.time() - start_time

    start_time = time.time()
    _, metrics_vspyct = evaluate_vspyct(vspyct, X_test, y_test)
    pred_time_vspyct = time.time() - start_time

    results['vspyct'] = {
        **metrics_vspyct,
        'train_time': train_time_vspyct,
        'pred_time': pred_time_vspyct,
        'num_nodes': vspyct.num_nodes
    }

    # Train and evaluate VSpyctGP with kernel selection
    # Try multiple kernels and pick the best one
    kernels_to_try = ['rbf', 'linear_rbf']
    best_metrics = None
    best_kernel = None
    best_train_time = 0
    best_pred_time = 0
    best_num_nodes = 0

    for kernel in kernels_to_try:
        try:
            gp_params_k = {**gp_params, 'kernel_type': kernel}
            vspyct_gp = VSpyctGP(**model_params, **gp_params_k)

            start_time = time.time()
            with SuppressOutput():
                vspyct_gp.fit(X_train, y_train)
            train_time_k = time.time() - start_time

            start_time = time.time()
            _, metrics_k, _ = evaluate_vspyct_gp(vspyct_gp, X_test, y_test)
            pred_time_k = time.time() - start_time

            # Keep best kernel based on MSE
            if best_metrics is None or metrics_k['mse'] < best_metrics['mse']:
                best_metrics = metrics_k
                best_kernel = kernel
                best_train_time = train_time_k
                best_pred_time = pred_time_k
                best_num_nodes = vspyct_gp.num_nodes
        except Exception:
            continue

    # Fallback if all kernels failed
    if best_metrics is None:
        raise RuntimeError("All kernel types failed")

    results['vspyct_gp'] = {
        **best_metrics,
        'train_time': best_train_time,
        'pred_time': best_pred_time,
        'num_nodes': best_num_nodes,
        'best_kernel': best_kernel
    }

    return results


def evaluate_dataset(
    task_id: int,
    benchmark: OpenMLBenchmark,
    model_params: Dict,
    gp_params: Dict,
    verbose: bool = False
) -> Dict[str, Any]:
    """Evaluate both models on a single dataset with appropriate CV."""

    # Load dataset
    try:
        X, y, meta = benchmark.load_dataset(task_id, return_tensor=True)
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return None

    dataset_name = meta['name']
    n_samples = meta['n_samples']
    n_features = meta['n_features']

    print(f"  Dataset: {dataset_name} ({n_samples} x {n_features})")

    # Skip datasets that are too small or have no features after processing
    if n_samples < 50:
        print(f"  Skipping: too few samples ({n_samples})")
        return None
    if n_features < 1:
        print(f"  Skipping: no features after processing")
        return None

    # Get CV splits based on dataset size
    # Use simplified CV to keep benchmark tractable
    if n_samples < 100:
        # For very small datasets, use 5-fold CV
        kf = KFold(n_splits=1, shuffle=True, random_state=42)
        splits = list(kf.split(X))
    elif n_samples < 1000:
        # Use 10-fold CV (not 10x repeated as in CTR23 - too slow)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        splits = list(kf.split(X))
    else:
        splits = benchmark.get_cv_splits(X, y, n_samples)

    n_splits = len(splits)
    print(f"  Using {n_splits} CV splits")

    # Collect results across folds
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if verbose:
            print(f"    Fold {fold_idx + 1}/{n_splits}...", end=" ", flush=True)

        # Convert indices to tensors for indexing
        X_train = X[train_idx]
        y_train = y[train_idx].reshape(-1, 1)
        X_test = X[test_idx]
        y_test = y[test_idx].reshape(-1, 1)

        # Skip fold if too few training samples
        if len(X_train) < model_params.get('minimum_examples_to_split', 10) * 2:
            if verbose:
                print(f"    Fold {fold_idx + 1}: skipped (too few training samples)")
            continue

        try:
            fold_result = run_single_fold(
                X_train, y_train, X_test, y_test,
                model_params, gp_params, verbose=False
            )
            fold_results.append(fold_result)

            if verbose:
                print(f"VSpyct MSE: {fold_result['vspyct']['mse']:.4f}, "
                      f"GP MSE: {fold_result['vspyct_gp']['mse']:.4f}")
        except Exception as e:
            print(f"    Fold {fold_idx + 1} failed: {e}")
            continue

    if not fold_results:
        print(f"  All folds failed for {dataset_name}")
        return None

    # Aggregate results
    aggregated = {
        'task_id': task_id,
        'dataset_name': dataset_name,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_folds': len(fold_results),
        'vspyct': {},
        'vspyct_gp': {}
    }

    # Compute mean and std for each metric
    for model_name in ['vspyct', 'vspyct_gp']:
        metrics = {}
        for metric_name in fold_results[0][model_name].keys():
            values = [fr[model_name][metric_name] for fr in fold_results]
            # Handle non-numeric fields (like best_kernel)
            if metric_name == 'best_kernel':
                # Record most common kernel
                metrics['best_kernel'] = Counter(values).most_common(1)[0][0]
            else:
                metrics[f'{metric_name}_mean'] = float(np.mean(values))
                metrics[f'{metric_name}_std'] = float(np.std(values))
        aggregated[model_name] = metrics

    return aggregated


def run_benchmark(
    n_datasets: int = 20,
    size_filter: str = 'all',
    output_dir: str = 'experiments/results',
    model_params: Dict = None,
    gp_params: Dict = None,
    verbose: bool = True,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run benchmark evaluation on multiple datasets.

    Args:
        n_datasets: Number of datasets to evaluate
        size_filter: Dataset size filter ('small', 'medium', 'large', 'all')
        output_dir: Directory to save results
        model_params: Parameters for both models
        gp_params: Additional parameters for VSpyctGP
        verbose: Print progress
        random_state: Random seed

    Returns:
        DataFrame with aggregated results
    """
    # Default model parameters
    if model_params is None:
        model_params = {
            'max_depth': 5,
            'minimum_examples_to_split': 10,
            'epochs': 500,
            'lr': 0.01,
            'subspace_size': 1.0
        }

    # Default GP parameters
    if gp_params is None:
        gp_params = {
            'tau': 'auto',
            'tau_percentile': 99,
            'kernel_type': 'linear_rbf',
            'gp_training_iterations': 75
        }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize benchmark
    benchmark = OpenMLBenchmark()

    # Get dataset IDs
    print(f"\n{'='*70}")
    print(f"VSPYCT vs VSPYCT-GP Benchmark Evaluation")
    print(f"{'='*70}")
    print(f"Selecting {n_datasets} datasets (filter: {size_filter})...")

    task_ids = get_benchmark_datasets(
        n_datasets=n_datasets,
        size_filter=size_filter,
        random_state=random_state
    )

    print(f"Selected {len(task_ids)} datasets")
    print(f"Model params: {model_params}")
    print(f"GP params: {gp_params}")
    print(f"{'='*70}\n")

    # Evaluate each dataset
    all_results = []

    for idx, task_id in enumerate(task_ids):
        print(f"\n[{idx + 1}/{len(task_ids)}] Evaluating task {task_id}...")

        result = evaluate_dataset(
            task_id=task_id,
            benchmark=benchmark,
            model_params=model_params,
            gp_params=gp_params,
            verbose=verbose
        )

        if result is not None:
            all_results.append(result)

            # Print summary
            vspyct_mse = result['vspyct']['mse_mean']
            gp_mse = result['vspyct_gp']['mse_mean']
            improvement = (vspyct_mse - gp_mse) / vspyct_mse * 100 if vspyct_mse > 0 else 0

            print(f"  Results: VSpyct MSE={vspyct_mse:.4f}, GP MSE={gp_mse:.4f} ({improvement:+.1f}%)")

    # Create summary DataFrame
    summary_rows = []
    for result in all_results:
        row = {
            'task_id': result['task_id'],
            'dataset_name': result['dataset_name'],
            'n_samples': result['n_samples'],
            'n_features': result['n_features'],
            'n_folds': result['n_folds'],
        }

        for model in ['vspyct', 'vspyct_gp']:
            for metric, value in result[model].items():
                row[f'{model}_{metric}'] = value

        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save detailed JSON
    json_path = os.path.join(output_dir, f'benchmark_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'model_params': model_params,
            'gp_params': gp_params,
            'results': all_results
        }, f, indent=2)

    # Save summary CSV
    csv_path = os.path.join(output_dir, f'benchmark_summary_{timestamp}.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Evaluated {len(all_results)} datasets successfully")
    print(f"Results saved to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")

    # Print overall summary
    if len(df) > 0:
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print(f"{'='*70}")

        vspyct_mse = df['vspyct_mse_mean'].mean()
        gp_mse = df['vspyct_gp_mse_mean'].mean()
        vspyct_r2 = df['vspyct_r2_mean'].mean()
        gp_r2 = df['vspyct_gp_r2_mean'].mean()

        print(f"\n{'Metric':<20} {'VSpyct':<15} {'VSpyct-GP':<15}")
        print("-" * 50)
        print(f"{'Avg MSE':<20} {vspyct_mse:<15.4f} {gp_mse:<15.4f}")
        print(f"{'Avg R²':<20} {vspyct_r2:<15.4f} {gp_r2:<15.4f}")

        # Win/loss count
        wins_gp = (df['vspyct_gp_mse_mean'] < df['vspyct_mse_mean']).sum()
        wins_vspyct = (df['vspyct_mse_mean'] < df['vspyct_gp_mse_mean']).sum()
        ties = len(df) - wins_gp - wins_vspyct

        print(f"\nDataset wins (MSE): VSpyct-GP: {wins_gp}, VSpyct: {wins_vspyct}, Ties: {ties}")

        # Uncertainty stats
        avg_unc = df['vspyct_gp_mean_uncertainty_mean'].mean()
        print(f"Avg uncertainty (VSpyct-GP): {avg_unc:.4f}")

    print(f"{'='*70}\n")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark VSPYCT-GP vs VSPYCT on OpenML datasets'
    )
    parser.add_argument('--n_datasets', type=int, default=20,
                        help='Number of datasets to evaluate')
    parser.add_argument('--size_filter', type=str, default='all',
                        choices=['small', 'medium', 'large', 'all'],
                        help='Dataset size filter')
    parser.add_argument('--output_dir', type=str, default='experiments/results',
                        help='Output directory for results')
    parser.add_argument('--max_depth', type=int, default=5,
                        help='Maximum tree depth')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Training epochs')
    parser.add_argument('--tau', type=float, default=0,
                        help='Mahalanobis distance threshold (0=auto calibration)')
    parser.add_argument('--kernel_type', type=str, default='linear_rbf',
                        choices=['rbf', 'matern', 'linear', 'linear_rbf', 'polynomial'],
                        help='GP kernel type (used as fallback; best of rbf/linear_rbf/matern is auto-selected)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    model_params = {
        'max_depth': args.max_depth,
        'minimum_examples_to_split': 10,
        'epochs': args.epochs,
        'lr': 0.01,
        'subspace_size': 1.0
    }

    gp_params = {
        'tau': 'auto' if args.tau == 0 else args.tau,
        'tau_percentile': 99,
        'kernel_type': args.kernel_type,
        'gp_training_iterations': 75
    }

    run_benchmark(
        n_datasets=args.n_datasets,
        size_filter=args.size_filter,
        output_dir=args.output_dir,
        model_params=model_params,
        gp_params=gp_params,
        verbose=not args.quiet,
        random_state=args.seed
    )


if __name__ == '__main__':
    main()
