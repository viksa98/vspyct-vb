"""
Test script comparing VSpyct and VSpyctGP models on synthetic data.
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings
import os
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.environ['PYRO_VALIDATION'] = 'false'

from src.models.model import VSpyct, VSpyctGP


def generate_synthetic_data(n_samples=500, n_features=10, noise_std=0.3, seed=42):
    """
    Generate synthetic regression data with a complex nonlinear function.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate features in [0, 1] range for training
    X_train = np.random.uniform(0, 1, size=(n_samples, n_features))

    # Complex nonlinear target function
    def target_function(X):
        y = (2.0 * np.sin(2 * np.pi * X[:, 0]) +  # Sinusoidal
             1.5 * X[:, 1] ** 2 +  # Quadratic
             1.0 * X[:, 2] * X[:, 3] +  # Interaction
             0.8 * np.cos(np.pi * X[:, 4]) +  # Cosine
             0.5 * X[:, 5] +  # Linear
             0.3 * X[:, 6] * X[:, 7] * X[:, 8] +  # Three-way interaction
             0.2 * np.exp(-X[:, 9]))  # Exponential decay
        return y

    y_train = target_function(X_train) + np.random.normal(0, noise_std, n_samples)

    # In-distribution test set
    X_test_in = np.random.uniform(0, 1, size=(150, n_features))
    y_test_in = target_function(X_test_in) + np.random.normal(0, noise_std, 150)

    # Extrapolation test set (moderate extrapolation)
    X_test_extrap = np.random.uniform(1.0, 1.4, size=(150, n_features))
    y_test_extrap = target_function(X_test_extrap) + np.random.normal(0, noise_std, 150)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test_in = torch.tensor(X_test_in, dtype=torch.float32)
    y_test_in = torch.tensor(y_test_in, dtype=torch.float32).reshape(-1, 1)
    X_test_extrap = torch.tensor(X_test_extrap, dtype=torch.float32)
    y_test_extrap = torch.tensor(y_test_extrap, dtype=torch.float32).reshape(-1, 1)

    return (X_train, y_train, X_test_in, y_test_in, X_test_extrap, y_test_extrap)


def evaluate_vspyct(model, X_test, y_test):
    """Evaluate VSpyct model and return MSE."""
    predictions = model.predict(X_test)

    # VSpyct returns shape (n_samples, n_targets, n_mc_samples) or (n_samples, n_mc_samples)
    # We need to average over MC samples and targets
    if predictions.dim() == 3:
        # (n_samples, n_targets, n_mc_samples) -> average over MC and targets
        pred_mean = predictions.mean(dim=-1).mean(dim=-1)
    elif predictions.dim() == 2:
        # (n_samples, n_mc_samples) -> average over MC
        pred_mean = predictions.mean(dim=-1)
    else:
        pred_mean = predictions

    mse = mean_squared_error(y_test.numpy().flatten(), pred_mean.numpy().flatten())
    return mse, pred_mean


def evaluate_vspyct_gp(model, X_test, y_test, return_uncertainty=False):
    """Evaluate VSpyctGP model and return MSE."""
    if return_uncertainty:
        predictions, uncertainties = model.predict(X_test, return_uncertainty=True)
        mse = mean_squared_error(y_test.numpy().flatten(), predictions.numpy().flatten())
        return mse, predictions, uncertainties
    else:
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test.numpy().flatten(), predictions.numpy().flatten())
        return mse, predictions


class SuppressOutput:
    """Context manager to suppress stdout/stderr."""
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


import sys


def main():
    print("=" * 60)
    print("VSPYCT vs VSPYCT-GP Comparison")
    print("=" * 60)

    # Generate complex data
    print("\nGenerating data...")
    X_train, y_train, X_test_in, y_test_in, X_test_extrap, y_test_extrap = generate_synthetic_data(
        n_samples=800, n_features=10, noise_std=0.2
    )
    print(f"  Train: {X_train.shape[0]}, Test-ID: {X_test_in.shape[0]}, Test-Extrap: {X_test_extrap.shape[0]}")
    print(f"  Target range - Train: [{y_train.min():.2f}, {y_train.max():.2f}], Extrap: [{y_test_extrap.min():.2f}, {y_test_extrap.max():.2f}]")

    # Model parameters
    model_params = {
        'max_depth': 5,
        'minimum_examples_to_split': 10,
        'epochs': 500,
        'lr': 0.01,
        'subspace_size': 1.0  # Must be 1.0 to avoid dimension mismatch
    }

    # Train VSpyct (suppress output)
    print("\nTraining VSpyct...", end=" ", flush=True)
    vspyct = VSpyct(**model_params)
    with SuppressOutput():
        vspyct.fit(X_train, y_train)
    print(f"Done. Nodes: {vspyct.num_nodes}")

    # Train VSpyctGP (suppress output)
    print("Training VSpyctGP...", end=" ", flush=True)
    vspyct_gp = VSpyctGP(
        **model_params,
        tau='auto',  # Auto-calibrate support threshold
        kernel_type='linear',  # Linear kernel for extrapolation
        gp_training_iterations=100
    )
    with SuppressOutput():
        vspyct_gp.fit(X_train, y_train)
    print(f"Done. Nodes: {vspyct_gp.num_nodes}")


    # Evaluate
    print("\nEvaluating...")
    mse_vspyct_in, pred_vspyct_in = evaluate_vspyct(vspyct, X_test_in, y_test_in)
    mse_vspyct_gp_in, pred_vspyct_gp_in, unc_gp_in = evaluate_vspyct_gp(vspyct_gp, X_test_in, y_test_in, return_uncertainty=True)
    mse_vspyct_ext, pred_vspyct_ext = evaluate_vspyct(vspyct, X_test_extrap, y_test_extrap)
    mse_vspyct_gp_ext, pred_vspyct_gp_ext, unc_gp_ext = evaluate_vspyct_gp(vspyct_gp, X_test_extrap, y_test_extrap, return_uncertainty=True)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'VSpyct':<12} {'VSpyctGP':<12} {'Diff':<10}")
    print("-" * 60)

    diff_in = ((mse_vspyct_in - mse_vspyct_gp_in) / mse_vspyct_in * 100) if mse_vspyct_in > 0 else 0
    diff_ext = ((mse_vspyct_ext - mse_vspyct_gp_ext) / mse_vspyct_ext * 100) if mse_vspyct_ext > 0 else 0

    print(f"{'MSE (In-Dist)':<25} {mse_vspyct_in:<12.4f} {mse_vspyct_gp_in:<12.4f} {diff_in:+.1f}%")
    print(f"{'MSE (Extrapolation)':<25} {mse_vspyct_ext:<12.4f} {mse_vspyct_gp_ext:<12.4f} {diff_ext:+.1f}%")
    print(f"{'Uncertainty (In-Dist)':<25} {'N/A':<12} {unc_gp_in.mean():<12.4f}")
    print(f"{'Uncertainty (Extrap)':<25} {'N/A':<12} {unc_gp_ext.mean():<12.4f}")

    unc_ratio = unc_gp_ext.mean() / unc_gp_in.mean() if unc_gp_in.mean() > 0 else 0
    print(f"\nUncertainty ratio (Extrap/In-Dist): {unc_ratio:.2f}x")

    if unc_gp_ext.mean() > unc_gp_in.mean():
        print("-> GP correctly shows higher uncertainty for extrapolation")
    else:
        print("-> WARNING: Extrapolation uncertainty should be higher!")

    print("=" * 60)

    # Plot predicted vs actual values
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    y_in_np = y_test_in.numpy().flatten()
    y_ext_np = y_test_extrap.numpy().flatten()

    # VSpyct - In-distribution
    ax = axes[0, 0]
    ax.scatter(y_in_np, pred_vspyct_in.numpy().flatten(), alpha=0.6, s=20)
    ax.plot([y_in_np.min(), y_in_np.max()], [y_in_np.min(), y_in_np.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'VSpyct - In-Distribution (MSE={mse_vspyct_in:.3f})')
    ax.grid(True, alpha=0.3)

    # VSpyctGP - In-distribution
    ax = axes[0, 1]
    ax.scatter(y_in_np, pred_vspyct_gp_in.numpy().flatten(), alpha=0.6, s=20, c='green')
    ax.plot([y_in_np.min(), y_in_np.max()], [y_in_np.min(), y_in_np.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'VSpyctGP - In-Distribution (MSE={mse_vspyct_gp_in:.3f})')
    ax.grid(True, alpha=0.3)

    # VSpyct - Extrapolation
    ax = axes[1, 0]
    ax.scatter(y_ext_np, pred_vspyct_ext.numpy().flatten(), alpha=0.6, s=20, c='orange')
    ax.plot([y_ext_np.min(), y_ext_np.max()], [y_ext_np.min(), y_ext_np.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'VSpyct - Extrapolation (MSE={mse_vspyct_ext:.3f})')
    ax.grid(True, alpha=0.3)

    # VSpyctGP - Extrapolation with uncertainty
    ax = axes[1, 1]
    pred_ext = pred_vspyct_gp_ext.numpy().flatten()
    unc_ext = np.sqrt(unc_gp_ext.numpy().flatten())  # Convert variance to std
    ax.errorbar(y_ext_np, pred_ext, yerr=unc_ext, fmt='o', alpha=0.5, markersize=4,
                color='purple', ecolor='lightgray', capsize=2)
    ax.plot([y_ext_np.min(), y_ext_np.max()], [y_ext_np.min(), y_ext_np.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'VSpyctGP - Extrapolation (MSE={mse_vspyct_gp_ext:.3f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    print(f"\nPlot saved to: model_comparison.png")
    plt.show()

    return {
        'vspyct_mse_in': mse_vspyct_in,
        'vspyct_mse_ext': mse_vspyct_ext,
        'vspyct_gp_mse_in': mse_vspyct_gp_in,
        'vspyct_gp_mse_ext': mse_vspyct_gp_ext,
        'vspyct_gp_unc_in': unc_gp_in.mean().item(),
        'vspyct_gp_unc_ext': unc_gp_ext.mean().item()
    }


if __name__ == "__main__":
    results = main()
