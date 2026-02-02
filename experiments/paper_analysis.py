"""
Comprehensive Analysis for VSPYCT-GP Paper.

Generates publication-quality figures for:
1. Hyperparameter sensitivity analysis (tau)
2. Uncertainty calibration analysis
3. Extrapolation vs interpolation experiment
4. Kernel comparison ablation

Usage:
    python experiments/paper_analysis.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl

# Publication-quality plot settings
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

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


# =============================================================================
# 1. HYPERPARAMETER SENSITIVITY ANALYSIS
# =============================================================================

def run_tau_sensitivity_analysis(datasets=['solar_flare', 'abalone', 'concrete_compressive_strength'],
                                  tau_values=[0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0],
                                  n_seeds=1):
    """
    Analyze how the support threshold τ affects model performance.
    """
    print("=" * 70)
    print("TAU SENSITIVITY ANALYSIS")
    print("=" * 70)

    benchmark = OpenMLBenchmark()
    df_datasets = benchmark.list_datasets(verbose=False)

    results = []

    model_params = {
        'max_depth': 5,
        'minimum_examples_to_split': 15,
        'epochs': 300,
        'lr': 0.01,
        'subspace_size': 1.0
    }

    for dataset_name in datasets:
        row = df_datasets[df_datasets['name'] == dataset_name]
        if len(row) == 0:
            print(f"Dataset '{dataset_name}' not found, skipping...")
            continue

        task_id = int(row.iloc[0]['task_id'])
        X, y, meta = benchmark.load_dataset(task_id, return_tensor=True)

        print(f"\nDataset: {dataset_name} ({meta['n_samples']} samples)")

        for seed in range(n_seeds):
            indices = np.arange(len(X))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42 + seed)
            X_train, y_train = X[train_idx], y[train_idx].reshape(-1, 1)
            X_test, y_test = X[test_idx], y[test_idx].reshape(-1, 1)

            # Train base VSPYCT for reference
            vspyct = VSpyct(**model_params)
            with SuppressOutput():
                vspyct.fit(X_train, y_train)

            pred_v = vspyct.predict(X_test)
            if pred_v.dim() == 3:
                pred_v = pred_v.mean(dim=1).squeeze()
            elif pred_v.dim() == 2:
                pred_v = pred_v.mean(dim=-1)
            rmse_vspyct = np.sqrt(mean_squared_error(y_test.numpy().flatten(), pred_v.numpy().flatten()))

            # Train VSPYCT-GP once, vary tau at prediction time
            vspyct_gp = VSpyctGP(**model_params, tau=4.5, kernel_type='linear', tau_percentile=99,
                                  gp_training_iterations=75)
            with SuppressOutput():
                vspyct_gp.fit(X_train, y_train)

            for tau in tau_values:
                vspyct_gp.tau = tau
                pred_gp, unc_gp = vspyct_gp.predict(X_test, return_uncertainty=True)
                rmse_gp = np.sqrt(mean_squared_error(y_test.numpy().flatten(), pred_gp.numpy().flatten()))

                results.append({
                    'dataset': dataset_name,
                    'seed': seed,
                    'tau': tau,
                    'rmse_vspyct': rmse_vspyct,
                    'rmse_vspyct_gp': rmse_gp,
                    'mean_uncertainty': unc_gp.mean().item(),
                    'relative_improvement': (rmse_vspyct - rmse_gp) / rmse_vspyct * 100
                })
                print(f"  tau={tau:.1f}, seed={seed}: RMSE={rmse_gp:.4f}")

    return pd.DataFrame(results)


def plot_tau_sensitivity(df, output_dir):
    """Create tau sensitivity plot."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Aggregate across seeds (will just be mean if n_seeds=1)
    agg = df.groupby(['dataset', 'tau']).agg({
        'rmse_vspyct_gp': 'mean',
        'rmse_vspyct': 'mean',
        'mean_uncertainty': 'mean'
    }).reset_index()
    agg.columns = ['dataset', 'tau', 'rmse_gp', 'rmse_baseline', 'unc_mean']

    # Left plot: RMSE vs tau
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['dataset'].unique())))

    for i, dataset in enumerate(df['dataset'].unique()):
        data = agg[agg['dataset'] == dataset]
        ax.plot(data['tau'], data['rmse_gp'], 'o-', color=colors[i], label=dataset, linewidth=1.5, markersize=5)
        # Add baseline
        ax.axhline(data['rmse_baseline'].iloc[0], color=colors[i], linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel(r'Support threshold $\tau$')
    ax.set_ylabel('RMSE')
    ax.set_title(r'(a) Effect of $\tau$ on prediction accuracy')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xscale('log')

    # Right plot: Uncertainty vs tau
    ax = axes[1]
    for i, dataset in enumerate(df['dataset'].unique()):
        data = agg[agg['dataset'] == dataset]
        ax.plot(data['tau'], data['unc_mean'], 's-', color=colors[i], label=dataset, linewidth=1.5, markersize=5)

    ax.set_xlabel(r'Support threshold $\tau$')
    ax.set_ylabel('Mean uncertainty')
    ax.set_title(r'(b) Effect of $\tau$ on uncertainty')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tau_sensitivity.pdf'))
    plt.savefig(os.path.join(output_dir, 'tau_sensitivity.png'))
    plt.close()
    print(f"Saved: tau_sensitivity.pdf")


# =============================================================================
# 2. UNCERTAINTY CALIBRATION ANALYSIS
# =============================================================================

def run_calibration_analysis(datasets=['solar_flare', 'abalone', 'concrete_compressive_strength'], n_seeds=1):
    """
    Analyze uncertainty calibration: do confidence intervals contain
    true values at the expected rate?
    """
    print("\n" + "=" * 70)
    print("UNCERTAINTY CALIBRATION ANALYSIS")
    print("=" * 70)

    benchmark = OpenMLBenchmark()
    df_datasets = benchmark.list_datasets(verbose=False)

    all_errors = []
    all_uncertainties = []
    all_datasets = []

    model_params = {
        'max_depth': 5,
        'minimum_examples_to_split': 15,
        'epochs': 300,
        'lr': 0.01,
        'subspace_size': 1.0
    }

    gp_params = {
        'tau': 'auto',
        'tau_percentile': 99,
        'kernel_type': 'linear',
        'gp_training_iterations': 75
    }

    for dataset_name in datasets:
        row = df_datasets[df_datasets['name'] == dataset_name]
        if len(row) == 0:
            continue

        task_id = int(row.iloc[0]['task_id'])
        X, y, meta = benchmark.load_dataset(task_id, return_tensor=True)

        print(f"\nDataset: {dataset_name}")

        for seed in range(n_seeds):
            indices = np.arange(len(X))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42 + seed)
            X_train, y_train = X[train_idx], y[train_idx].reshape(-1, 1)
            X_test, y_test = X[test_idx], y[test_idx].reshape(-1, 1)

            vspyct_gp = VSpyctGP(**model_params, **gp_params)
            with SuppressOutput():
                vspyct_gp.fit(X_train, y_train)

            pred_gp, unc_gp = vspyct_gp.predict(X_test, return_uncertainty=True)

            errors = np.abs(y_test.numpy().flatten() - pred_gp.numpy().flatten())
            uncertainties = np.sqrt(unc_gp.numpy().flatten())  # Convert variance to std

            all_errors.extend(errors.tolist())
            all_uncertainties.extend(uncertainties.tolist())
            all_datasets.extend([dataset_name] * len(errors))

    return pd.DataFrame({
        'dataset': all_datasets,
        'error': all_errors,
        'uncertainty': all_uncertainties
    })


def plot_calibration(df, output_dir):
    """Create calibration plots."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    errors = df['error'].values
    uncertainties = df['uncertainty'].values

    # Left plot: Error vs Uncertainty scatter with trend
    ax = axes[0]

    # Bin by uncertainty and compute mean error in each bin
    n_bins = 20
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    bin_mean_errors = []
    bin_std_errors = []

    for i in range(n_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
        if mask.sum() > 5:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_mean_errors.append(errors[mask].mean())
            bin_std_errors.append(errors[mask].std() / np.sqrt(mask.sum()))

    # Scatter (subsampled for visibility)
    subsample = np.random.choice(len(errors), min(1000, len(errors)), replace=False)
    ax.scatter(uncertainties[subsample], errors[subsample], alpha=0.3, s=10, c='gray', label='Predictions')

    # Trend line
    ax.plot(bin_centers, bin_mean_errors, 'o-', color='blue', linewidth=2, markersize=6, label='Binned mean')

    # Perfect calibration line (error = uncertainty for 68% CI)
    max_val = max(max(uncertainties), max(errors))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect calibration')

    ax.set_xlabel('Predicted uncertainty (std)')
    ax.set_ylabel('Absolute error')
    ax.set_title('(a) Uncertainty vs. actual error')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(0, np.percentile(uncertainties, 99))
    ax.set_ylim(0, np.percentile(errors, 99))

    # Right plot: Coverage plot
    ax = axes[1]

    confidence_levels = np.linspace(0.1, 0.99, 20)
    observed_coverage = []

    for conf in confidence_levels:
        z = 1.96 * (conf / 0.95)  # Scale z-score for different confidence levels
        # For a Gaussian, what fraction of errors fall within z * uncertainty?
        within_interval = errors <= z * uncertainties
        observed_coverage.append(within_interval.mean())

    ax.plot(confidence_levels, observed_coverage, 'o-', color='blue', linewidth=2, markersize=5, label='Observed')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect calibration')

    ax.set_xlabel('Expected coverage')
    ax.set_ylabel('Observed coverage')
    ax.set_title('(b) Coverage calibration')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration.pdf'))
    plt.savefig(os.path.join(output_dir, 'calibration.png'))
    plt.close()
    print(f"Saved: calibration.pdf")


# =============================================================================
# 3. EXTRAPOLATION EXPERIMENT
# =============================================================================

def run_extrapolation_experiment(n_train=300, n_test=100, noise_std=0.2, n_seeds=1):
    """
    Controlled synthetic experiment comparing interpolation vs extrapolation.
    """
    print("\n" + "=" * 70)
    print("EXTRAPOLATION EXPERIMENT")
    print("=" * 70)

    results = []

    model_params = {
        'max_depth': 4,
        'minimum_examples_to_split': 10,
        'epochs': 400,
        'lr': 0.01,
        'subspace_size': 1.0
    }

    def target_function(x):
        """Smooth nonlinear function for testing."""
        return 2.0 * np.sin(2 * np.pi * x[:, 0]) + 1.5 * x[:, 1]**2 + x[:, 2]

    for seed in range(n_seeds):
        np.random.seed(42 + seed)
        torch.manual_seed(42 + seed)

        n_features = 5

        # Training data: x in [0, 1]
        X_train = np.random.uniform(0, 1, size=(n_train, n_features))
        y_train = target_function(X_train) + np.random.normal(0, noise_std, n_train)

        # Test data: interpolation (same range)
        X_test_interp = np.random.uniform(0, 1, size=(n_test, n_features))
        y_test_interp = target_function(X_test_interp) + np.random.normal(0, noise_std, n_test)

        # Test data: mild extrapolation
        X_test_mild = np.random.uniform(0.8, 1.3, size=(n_test, n_features))
        y_test_mild = target_function(X_test_mild) + np.random.normal(0, noise_std, n_test)

        # Test data: strong extrapolation
        X_test_strong = np.random.uniform(1.2, 1.8, size=(n_test, n_features))
        y_test_strong = target_function(X_test_strong) + np.random.normal(0, noise_std, n_test)

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

        test_sets = [
            ('Interpolation', X_test_interp, y_test_interp),
            ('Mild extrapolation', X_test_mild, y_test_mild),
            ('Strong extrapolation', X_test_strong, y_test_strong)
        ]

        # Train VSPYCT
        vspyct = VSpyct(**model_params)
        with SuppressOutput():
            vspyct.fit(X_train_t, y_train_t)

        # Train VSPYCT-GP with linear kernel for extrapolation
        vspyct_gp = VSpyctGP(**model_params, tau='auto', tau_percentile=95,
                              kernel_type='linear', gp_training_iterations=100)
        with SuppressOutput():
            vspyct_gp.fit(X_train_t, y_train_t)

        for regime, X_test, y_test in test_sets:
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

            # VSPYCT prediction
            pred_v = vspyct.predict(X_test_t)
            if pred_v.dim() == 3:
                pred_v = pred_v.mean(dim=1).squeeze()
            elif pred_v.dim() == 2:
                pred_v = pred_v.mean(dim=-1)
            rmse_v = np.sqrt(mean_squared_error(y_test, pred_v.numpy().flatten()))

            # VSPYCT-GP prediction
            pred_gp, unc_gp = vspyct_gp.predict(X_test_t, return_uncertainty=True)
            rmse_gp = np.sqrt(mean_squared_error(y_test, pred_gp.numpy().flatten()))

            results.append({
                'seed': seed,
                'regime': regime,
                'rmse_vspyct': rmse_v,
                'rmse_vspyct_gp': rmse_gp,
                'mean_uncertainty': unc_gp.mean().item(),
                'improvement': (rmse_v - rmse_gp) / rmse_v * 100
            })

        print(f"Seed {seed}: completed")

    return pd.DataFrame(results)


def plot_extrapolation_experiment(df, output_dir):
    """Create extrapolation comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Aggregate (will just be mean if n_seeds=1)
    agg = df.groupby('regime').agg({
        'rmse_vspyct': 'mean',
        'rmse_vspyct_gp': 'mean',
        'mean_uncertainty': 'mean'
    }).reset_index()
    agg.columns = ['regime', 'rmse_v', 'rmse_gp', 'unc_mean']

    # Order regimes
    regime_order = ['Interpolation', 'Mild extrapolation', 'Strong extrapolation']
    agg['order'] = agg['regime'].map({r: i for i, r in enumerate(regime_order)})
    agg = agg.sort_values('order')

    x = np.arange(len(regime_order))
    width = 0.35

    # Left plot: RMSE comparison
    ax = axes[0]
    ax.bar(x - width/2, agg['rmse_v'], width, label='VSPYCT', color='#2ecc71')
    ax.bar(x + width/2, agg['rmse_gp'], width, label='VSPYCT-GP', color='#e74c3c')

    ax.set_xlabel('Test regime')
    ax.set_ylabel('RMSE')
    ax.set_title('(a) Prediction accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(['Interp.', 'Mild\nextrap.', 'Strong\nextrap.'], fontsize=9)
    ax.legend(loc='upper left', framealpha=0.9)

    # Right plot: Uncertainty by regime
    ax = axes[1]
    ax.bar(x, agg['unc_mean'], width * 1.5, color='#3498db')

    ax.set_xlabel('Test regime')
    ax.set_ylabel('Mean uncertainty')
    ax.set_title('(b) VSPYCT-GP uncertainty')
    ax.set_xticks(x)
    ax.set_xticklabels(['Interp.', 'Mild\nextrap.', 'Strong\nextrap.'], fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extrapolation_experiment.pdf'))
    plt.savefig(os.path.join(output_dir, 'extrapolation_experiment.png'))
    plt.close()
    print(f"Saved: extrapolation_experiment.pdf")


# =============================================================================
# 4. KERNEL COMPARISON ABLATION
# =============================================================================

def run_kernel_ablation(datasets=['solar_flare', 'abalone', 'concrete_compressive_strength'],
                         kernels=['rbf', 'matern', 'linear', 'linear_rbf'],
                         n_seeds=1):
    """
    Compare different GP kernel choices.
    """
    print("\n" + "=" * 70)
    print("KERNEL ABLATION STUDY")
    print("=" * 70)

    benchmark = OpenMLBenchmark()
    df_datasets = benchmark.list_datasets(verbose=False)

    results = []

    model_params = {
        'max_depth': 5,
        'minimum_examples_to_split': 15,
        'epochs': 300,
        'lr': 0.01,
        'subspace_size': 1.0
    }

    for dataset_name in datasets:
        row = df_datasets[df_datasets['name'] == dataset_name]
        if len(row) == 0:
            continue

        task_id = int(row.iloc[0]['task_id'])
        X, y, meta = benchmark.load_dataset(task_id, return_tensor=True)

        print(f"\nDataset: {dataset_name}")

        for seed in range(n_seeds):
            indices = np.arange(len(X))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42 + seed)
            X_train, y_train = X[train_idx], y[train_idx].reshape(-1, 1)
            X_test, y_test = X[test_idx], y[test_idx].reshape(-1, 1)

            for kernel in kernels:
                try:
                    vspyct_gp = VSpyctGP(**model_params, tau=4.5, tau_percentile=99,
                                          kernel_type=kernel, gp_training_iterations=75)
                    with SuppressOutput():
                        vspyct_gp.fit(X_train, y_train)

                    pred_gp, unc_gp = vspyct_gp.predict(X_test, return_uncertainty=True)
                    rmse = np.sqrt(mean_squared_error(y_test.numpy().flatten(), pred_gp.numpy().flatten()))
                    r2 = r2_score(y_test.numpy().flatten(), pred_gp.numpy().flatten())

                    results.append({
                        'dataset': dataset_name,
                        'seed': seed,
                        'kernel': kernel,
                        'rmse': rmse,
                        'r2': r2,
                        'mean_uncertainty': unc_gp.mean().item()
                    })
                    print(f"  {kernel}, seed={seed}: RMSE={rmse:.4f}")
                except Exception as e:
                    print(f"  {kernel}, seed={seed}: FAILED - {e}")

    return pd.DataFrame(results)


def plot_kernel_ablation(df, output_dir):
    """Create kernel comparison plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Aggregate across datasets (will just be mean if n_seeds=1)
    agg = df.groupby('kernel').agg({
        'rmse': 'mean'
    }).reset_index()
    agg.columns = ['kernel', 'rmse']

    # Sort by RMSE
    agg = agg.sort_values('rmse')

    # Nice kernel names
    kernel_names = {'rbf': 'RBF', 'matern': 'Matérn', 'linear': 'Linear', 'linear_rbf': 'Linear+RBF'}
    agg['kernel_name'] = agg['kernel'].map(kernel_names)

    x = np.arange(len(agg))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(agg)))

    bars = ax.bar(x, agg['rmse'], color=colors)

    ax.set_xlabel('GP kernel')
    ax.set_ylabel('RMSE')
    ax.set_title('Effect of GP kernel choice on prediction accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(agg['kernel_name'])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kernel_ablation.pdf'))
    plt.savefig(os.path.join(output_dir, 'kernel_ablation.png'))
    plt.close()
    print(f"Saved: kernel_ablation.pdf")


# =============================================================================
# 5. COMBINED SUMMARY FIGURE
# =============================================================================

def create_summary_figure(tau_df, calib_df, extrap_df, kernel_df, output_dir):
    """Create a 2x2 summary figure combining key results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Tau sensitivity - one dataset example
    ax = axes[0, 0]
    if len(tau_df) > 0:
        dataset = tau_df['dataset'].iloc[0]
        data = tau_df[tau_df['dataset'] == dataset]
        agg = data.groupby('tau').agg({'rmse_vspyct_gp': 'mean', 'rmse_vspyct': 'mean'}).reset_index()

        ax.plot(agg['tau'], agg['rmse_vspyct_gp'], 'o-', color='#e74c3c', label='VSPYCT-GP', linewidth=2, markersize=6)
        ax.axhline(agg['rmse_vspyct'].iloc[0], color='#2ecc71', linestyle='--', linewidth=2, label='VSPYCT')

        ax.set_xlabel(r'Support threshold $\tau$')
        ax.set_ylabel('RMSE')
        ax.set_title(r'(a) Sensitivity to $\tau$')
        ax.legend(loc='best', framealpha=0.9)
        ax.set_xscale('log')

    # (b) Calibration
    ax = axes[0, 1]
    if len(calib_df) > 0:
        errors = calib_df['error'].values
        uncertainties = calib_df['uncertainty'].values

        confidence_levels = np.linspace(0.1, 0.99, 15)
        observed_coverage = []
        for conf in confidence_levels:
            z = 1.96 * (conf / 0.95)
            within_interval = errors <= z * uncertainties
            observed_coverage.append(within_interval.mean())

        ax.plot(confidence_levels, observed_coverage, 'o-', color='#3498db', linewidth=2, markersize=5)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
        ax.fill_between(confidence_levels, observed_coverage, confidence_levels,
                        alpha=0.2, color='#3498db')

        ax.set_xlabel('Expected coverage')
        ax.set_ylabel('Observed coverage')
        ax.set_title('(b) Uncertainty calibration')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    # (c) Extrapolation
    ax = axes[1, 0]
    if len(extrap_df) > 0:
        agg = extrap_df.groupby('regime').agg({
            'rmse_vspyct': 'mean',
            'rmse_vspyct_gp': 'mean',
            'mean_uncertainty': 'mean'
        }).reset_index()

        regime_order = ['Interpolation', 'Mild extrapolation', 'Strong extrapolation']
        agg['order'] = agg['regime'].map({r: i for i, r in enumerate(regime_order)})
        agg = agg.sort_values('order')

        x = np.arange(len(regime_order))
        width = 0.35

        ax.bar(x - width/2, agg['rmse_vspyct'], width, label='VSPYCT', color='#2ecc71')
        ax.bar(x + width/2, agg['rmse_vspyct_gp'], width, label='VSPYCT-GP', color='#e74c3c')

        ax.set_xlabel('Test regime')
        ax.set_ylabel('RMSE')
        ax.set_title('(c) Interpolation vs. extrapolation')
        ax.set_xticks(x)
        ax.set_xticklabels(['Interp.', 'Mild ext.', 'Strong ext.'], fontsize=9)
        ax.legend(loc='upper left', framealpha=0.9)

    # (d) Kernel ablation
    ax = axes[1, 1]
    if len(kernel_df) > 0:
        agg = kernel_df.groupby('kernel')['rmse'].mean().reset_index()
        agg.columns = ['kernel', 'rmse']
        agg = agg.sort_values('rmse')

        kernel_names = {'rbf': 'RBF', 'matern': 'Matérn', 'linear': 'Linear', 'linear_rbf': 'Lin+RBF'}
        agg['kernel_name'] = agg['kernel'].map(kernel_names)

        colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c'][:len(agg)]
        bars = ax.bar(range(len(agg)), agg['rmse'], color=colors)

        ax.set_xlabel('GP kernel')
        ax.set_ylabel('RMSE')
        ax.set_title('(d) Kernel comparison')
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg['kernel_name'])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'analysis_summary.pdf'))
    plt.savefig(os.path.join(output_dir, 'analysis_summary.png'))
    plt.close()
    print(f"Saved: analysis_summary.pdf")


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = 'experiments/figures'
    os.makedirs(output_dir, exist_ok=True)

    # Use datasets from CTR23 suite
    datasets = ['solar_flare', 'abalone', 'concrete_compressive_strength']

    print("Starting comprehensive analysis for paper...")
    print(f"Output directory: {output_dir}")

    # 1. Tau sensitivity
    print("\n" + "=" * 70)
    tau_df = run_tau_sensitivity_analysis(datasets=datasets)
    tau_df.to_csv(os.path.join(output_dir, 'tau_sensitivity.csv'), index=False)
    plot_tau_sensitivity(tau_df, output_dir)

    # 2. Calibration analysis
    calib_df = run_calibration_analysis(datasets=datasets)
    calib_df.to_csv(os.path.join(output_dir, 'calibration.csv'), index=False)
    plot_calibration(calib_df, output_dir)

    # 3. Extrapolation experiment
    extrap_df = run_extrapolation_experiment()
    extrap_df.to_csv(os.path.join(output_dir, 'extrapolation.csv'), index=False)
    plot_extrapolation_experiment(extrap_df, output_dir)

    # 4. Kernel ablation
    kernel_df = run_kernel_ablation(datasets=datasets)
    kernel_df.to_csv(os.path.join(output_dir, 'kernel_ablation.csv'), index=False)
    plot_kernel_ablation(kernel_df, output_dir)

    # 5. Combined summary figure
    create_summary_figure(tau_df, calib_df, extrap_df, kernel_df, output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nGenerated figures in {output_dir}/:")
    print("  - tau_sensitivity.pdf")
    print("  - calibration.pdf")
    print("  - extrapolation_experiment.pdf")
    print("  - kernel_ablation.pdf")
    print("  - analysis_summary.pdf (combined)")


if __name__ == '__main__':
    main()
