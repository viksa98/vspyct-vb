"""
Fast benchmark evaluation for paper results.
Uses single 80/20 train/test splits for efficiency.
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
os.environ['PYRO_VALIDATION'] = 'false'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.openml_loader import OpenMLBenchmark
from src.models.model import VSpyct, VSpyctGP


class SuppressOutput:
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


def evaluate_single_dataset(task_id, benchmark, model_params, gp_params, test_size=0.2, seed=42):
    """Evaluate both models on a single dataset with train/test split."""
    try:
        X, y, meta = benchmark.load_dataset(task_id, return_tensor=True)
    except Exception as e:
        print(f"  Error loading: {e}")
        return None

    name = meta['name']
    n_samples = meta['n_samples']
    n_features = meta['n_features']

    # Train/test split
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)

    X_train, y_train = X[train_idx], y[train_idx].reshape(-1, 1)
    X_test, y_test = X[test_idx], y[test_idx].reshape(-1, 1)

    results = {
        'task_id': task_id,
        'name': name,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_train': len(train_idx),
        'n_test': len(test_idx)
    }

    # Train VSpyct
    try:
        vspyct = VSpyct(**model_params)
        t0 = time.time()
        with SuppressOutput():
            vspyct.fit(X_train, y_train)
        train_time_v = time.time() - t0

        pred_v = vspyct.predict(X_test)
        if pred_v.dim() == 3:
            pred_v = pred_v.mean(dim=1).squeeze()
        elif pred_v.dim() == 2:
            pred_v = pred_v.mean(dim=-1)

        y_np = y_test.numpy().flatten()
        pred_v_np = pred_v.numpy().flatten()

        results['vspyct_mse'] = mean_squared_error(y_np, pred_v_np)
        results['vspyct_rmse'] = np.sqrt(results['vspyct_mse'])
        results['vspyct_r2'] = r2_score(y_np, pred_v_np)
        results['vspyct_train_time'] = train_time_v
        results['vspyct_nodes'] = vspyct.num_nodes
    except Exception as e:
        print(f"  VSpyct error: {e}")
        results['vspyct_mse'] = np.nan
        results['vspyct_rmse'] = np.nan
        results['vspyct_r2'] = np.nan

    # Train VSpyctGP
    try:
        vspyct_gp = VSpyctGP(**model_params, **gp_params)
        t0 = time.time()
        with SuppressOutput():
            vspyct_gp.fit(X_train, y_train)
        train_time_gp = time.time() - t0

        pred_gp, unc_gp = vspyct_gp.predict(X_test, return_uncertainty=True)
        pred_gp_np = pred_gp.numpy().flatten()

        results['vspyct_gp_mse'] = mean_squared_error(y_np, pred_gp_np)
        results['vspyct_gp_rmse'] = np.sqrt(results['vspyct_gp_mse'])
        results['vspyct_gp_r2'] = r2_score(y_np, pred_gp_np)
        results['vspyct_gp_train_time'] = train_time_gp
        results['vspyct_gp_nodes'] = vspyct_gp.num_nodes
        results['vspyct_gp_mean_unc'] = unc_gp.mean().item()
    except Exception as e:
        print(f"  VSpyctGP error: {e}")
        results['vspyct_gp_mse'] = np.nan
        results['vspyct_gp_rmse'] = np.nan
        results['vspyct_gp_r2'] = np.nan

    return results


def main():
    print("=" * 70)
    print("VSPYCT vs VSPYCT-GP Benchmark Evaluation")
    print("=" * 70)

    # Model parameters
    model_params = {
        'max_depth': 5,
        'minimum_examples_to_split': 15,
        'epochs': 300,
        'lr': 0.01,
        'subspace_size': 1.0
    }

    gp_params = {
        'tau': 4.5,
        'kernel_type': 'linear_rbf',
        'gp_training_iterations': 75,
        'use_gp_mean_for_extrapolation': False  # Use prototype mean with GP variance for uncertainty
    }

    # Initialize benchmark
    benchmark = OpenMLBenchmark()

    # Get all datasets
    print("\nLoading CTR23 suite...")
    df_datasets = benchmark.list_datasets(verbose=False)
    print(f"Found {len(df_datasets)} datasets")

    # Sort by sample size for progress tracking
    df_datasets = df_datasets.sort_values('n_samples')

    all_results = []

    for idx, row in df_datasets.iterrows():
        task_id = row['task_id']
        name = row['name']
        n_samples = row['n_samples']

        print(f"\n[{len(all_results)+1}/{len(df_datasets)}] {name} ({int(n_samples)} samples)...", end=" ", flush=True)

        result = evaluate_single_dataset(task_id, benchmark, model_params, gp_params)

        if result is not None:
            all_results.append(result)
            v_mse = result.get('vspyct_mse', np.nan)
            gp_mse = result.get('vspyct_gp_mse', np.nan)
            if not np.isnan(v_mse) and not np.isnan(gp_mse):
                imp = (v_mse - gp_mse) / v_mse * 100 if v_mse > 0 else 0
                print(f"VSPYCT: {v_mse:.4f}, GP: {gp_mse:.4f} ({imp:+.1f}%)")
            else:
                print("Error")

    # Save results
    df_results = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    output_dir = 'experiments/results'
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f'benchmark_full_{timestamp}.csv')
    df_results.to_csv(csv_path, index=False)

    json_path = os.path.join(output_dir, f'benchmark_full_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'model_params': model_params,
            'gp_params': gp_params,
            'results': all_results
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid = df_results.dropna(subset=['vspyct_mse', 'vspyct_gp_mse'])

    print(f"\nSuccessfully evaluated: {len(valid)}/{len(df_datasets)} datasets")

    avg_v_rmse = valid['vspyct_rmse'].mean()
    avg_gp_rmse = valid['vspyct_gp_rmse'].mean()
    avg_v_r2 = valid['vspyct_r2'].mean()
    avg_gp_r2 = valid['vspyct_gp_r2'].mean()

    print(f"\n{'Metric':<20} {'VSPYCT':<15} {'VSPYCT-GP':<15}")
    print("-" * 50)
    print(f"{'Avg RMSE':<20} {avg_v_rmse:<15.4f} {avg_gp_rmse:<15.4f}")
    print(f"{'Avg R²':<20} {avg_v_r2:<15.4f} {avg_gp_r2:<15.4f}")

    wins_gp = (valid['vspyct_gp_rmse'] < valid['vspyct_rmse']).sum()
    wins_v = (valid['vspyct_rmse'] < valid['vspyct_gp_rmse']).sum()
    ties = len(valid) - wins_gp - wins_v

    print(f"\nWins (RMSE): VSPYCT-GP: {wins_gp}, VSPYCT: {wins_v}, Ties: {ties}")

    print(f"\nResults saved to: {csv_path}")
    print("=" * 70)

    # Also save a LaTeX-ready table
    latex_path = os.path.join(output_dir, f'benchmark_table_{timestamp}.tex')
    generate_latex_table(valid, latex_path)
    print(f"LaTeX table saved to: {latex_path}")

    return df_results


def generate_latex_table(df, output_path):
    """Generate a LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Benchmark results on CTR23 regression datasets. Bold indicates best performance.}")
    lines.append(r"\label{tab:benchmark-results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & $n$ & $d$ & \multicolumn{2}{c}{RMSE} & \multicolumn{2}{c}{$R^2$} & Unc. \\")
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r" & & & VSPYCT & VSPYCT-GP & VSPYCT & VSPYCT-GP & (GP) \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        name = row['name'][:20]  # Truncate long names
        n = int(row['n_samples'])
        d = int(row['n_features'])

        v_rmse = row['vspyct_rmse']
        gp_rmse = row['vspyct_gp_rmse']
        v_r2 = row['vspyct_r2']
        gp_r2 = row['vspyct_gp_r2']
        unc = row.get('vspyct_gp_mean_unc', 0)

        # Bold the better one
        if gp_rmse < v_rmse:
            rmse_v = f"{v_rmse:.3f}"
            rmse_gp = f"\\textbf{{{gp_rmse:.3f}}}"
        else:
            rmse_v = f"\\textbf{{{v_rmse:.3f}}}"
            rmse_gp = f"{gp_rmse:.3f}"

        if gp_r2 > v_r2:
            r2_v = f"{v_r2:.3f}"
            r2_gp = f"\\textbf{{{gp_r2:.3f}}}"
        else:
            r2_v = f"\\textbf{{{v_r2:.3f}}}"
            r2_gp = f"{gp_r2:.3f}"

        lines.append(f"{name} & {n} & {d} & {rmse_v} & {rmse_gp} & {r2_v} & {r2_gp} & {unc:.3f} \\\\")

    lines.append(r"\midrule")

    # Add average row
    avg_v_rmse = df['vspyct_rmse'].mean()
    avg_gp_rmse = df['vspyct_gp_rmse'].mean()
    avg_v_r2 = df['vspyct_r2'].mean()
    avg_gp_r2 = df['vspyct_gp_r2'].mean()
    avg_unc = df['vspyct_gp_mean_unc'].mean()

    if avg_gp_rmse < avg_v_rmse:
        avg_rmse_v = f"{avg_v_rmse:.3f}"
        avg_rmse_gp = f"\\textbf{{{avg_gp_rmse:.3f}}}"
    else:
        avg_rmse_v = f"\\textbf{{{avg_v_rmse:.3f}}}"
        avg_rmse_gp = f"{avg_gp_rmse:.3f}"

    if avg_gp_r2 > avg_v_r2:
        avg_r2_v = f"{avg_v_r2:.3f}"
        avg_r2_gp = f"\\textbf{{{avg_gp_r2:.3f}}}"
    else:
        avg_r2_v = f"\\textbf{{{avg_v_r2:.3f}}}"
        avg_r2_gp = f"{avg_gp_r2:.3f}"

    lines.append(f"\\textit{{Average}} & -- & -- & {avg_rmse_v} & {avg_rmse_gp} & {avg_r2_v} & {avg_r2_gp} & {avg_unc:.3f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
