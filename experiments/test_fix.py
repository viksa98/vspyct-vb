"""
Quick test to verify VSPYCT-GP fixes.
Tests that:
1. GP mean is now used for extrapolation (not just prototype)
2. Tau calibration works
3. VSPYCT-GP outperforms VSPYCT in extrapolation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.models.model import VSpyct, VSpyctGP
from src.data.openml_loader import OpenMLBenchmark


class SuppressOutput:
    """Context manager to suppress print output."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def test_on_real_data():
    """Test on concrete_compressive_strength dataset."""
    print("=" * 60)
    print("TEST ON REAL DATA: concrete_compressive_strength")
    print("=" * 60)

    # Load dataset using OpenMLBenchmark for robust handling
    from src.data.openml_loader import OpenMLBenchmark
    benchmark = OpenMLBenchmark()
    df_datasets = benchmark.list_datasets(verbose=False)
    row = df_datasets[df_datasets['name'] == 'concrete_compressive_strength']
    if len(row) == 0:
        print("Dataset not found, skipping real data test")
        return None, None
    task_id = int(row.iloc[0]['task_id'])
    X, y, _ = benchmark.load_dataset(task_id)
    X = X.numpy() if hasattr(X, 'numpy') else X
    y = y.numpy() if hasattr(y, 'numpy') else y

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Model parameters
    model_params = {
        'max_depth': 4,
        'minimum_examples_to_split': 10,
        'epochs': 200,
        'lr': 0.005
    }

    # Train VSPYCT
    print("\nTraining VSPYCT...")
    vspyct = VSpyct(**model_params)
    with SuppressOutput():
        vspyct.fit(X_train, y_train)

    pred_v = vspyct.predict(X_test)
    if pred_v.dim() == 3:
        pred_v = pred_v.mean(dim=1).squeeze()
    if pred_v.dim() == 2:
        pred_v = pred_v.mean(dim=-1)
    rmse_vspyct = np.sqrt(mean_squared_error(y_test.numpy().flatten(), pred_v.numpy().flatten()))
    print(f"VSPYCT RMSE: {rmse_vspyct:.4f}")

    # Train VSPYCT-GP with auto tau
    print("\nTraining VSPYCT-GP (tau='auto')...")
    vspyct_gp = VSpyctGP(**model_params, tau='auto', tau_percentile=99,
                         kernel_type='linear', gp_training_iterations=50)
    vspyct_gp.fit(X_train, y_train)

    pred_gp, unc_gp = vspyct_gp.predict(X_test, return_uncertainty=True)
    rmse_gp = np.sqrt(mean_squared_error(y_test.numpy().flatten(), pred_gp.numpy().flatten()))
    print(f"VSPYCT-GP RMSE: {rmse_gp:.4f}")
    print(f"Mean uncertainty: {unc_gp.mean():.4f}")

    print(f"\nDifference: {rmse_vspyct - rmse_gp:.4f} (positive = GP better)")

    return rmse_vspyct, rmse_gp


def test_extrapolation_synthetic():
    """Test extrapolation on synthetic data where GP should clearly help."""
    print("\n" + "=" * 60)
    print("TEST ON SYNTHETIC DATA: EXTRAPOLATION")
    print("=" * 60)

    np.random.seed(42)

    # Generate training data in [0, 1] - use a simpler linear function for clearer extrapolation
    n_train = 300
    X_train = np.random.uniform(0, 1, (n_train, 3))
    # Linear function: y = 2*x1 + 3*x2 + x3, clearly extrapolates linearly
    y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] + X_train[:, 2] + np.random.normal(0, 0.1, n_train)

    # Test sets
    n_test = 100

    # Interpolation: [0, 1]
    X_interp = np.random.uniform(0, 1, (n_test, 3))
    y_interp = 2 * X_interp[:, 0] + 3 * X_interp[:, 1] + X_interp[:, 2]

    # Mild extrapolation: [0.8, 1.5]
    X_mild = np.random.uniform(0.8, 1.5, (n_test, 3))
    y_mild = 2 * X_mild[:, 0] + 3 * X_mild[:, 1] + X_mild[:, 2]

    # Strong extrapolation: [1.3, 2.0]
    X_strong = np.random.uniform(1.3, 2.0, (n_test, 3))
    y_strong = 2 * X_strong[:, 0] + 3 * X_strong[:, 1] + X_strong[:, 2]

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    # Model parameters - more epochs for better tree building
    model_params = {
        'max_depth': 4,
        'minimum_examples_to_split': 10,
        'epochs': 300,
        'lr': 0.005
    }

    # Train VSPYCT
    print("\nTraining VSPYCT...")
    vspyct = VSpyct(**model_params)
    with SuppressOutput():
        vspyct.fit(X_train_t, y_train_t)
    print(f"VSPYCT nodes: {vspyct.num_nodes}")

    # Train VSPYCT-GP with linear kernel for extrapolation
    print("Training VSPYCT-GP...")
    vspyct_gp = VSpyctGP(**model_params, tau='auto', tau_percentile=95,
                         kernel_type='linear', gp_training_iterations=100)
    vspyct_gp.fit(X_train_t, y_train_t)
    print(f"VSPYCT-GP nodes: {vspyct_gp.num_nodes}")

    print(f"\nEffective tau: {vspyct_gp._get_effective_tau():.3f}")

    # Evaluate on all test sets
    print("\n" + "-" * 40)
    print(f"{'Regime':<20} {'VSPYCT':<12} {'VSPYCT-GP':<12} {'Diff':<10} {'Unc':<10}")
    print("-" * 40)

    for name, X_t, y_t in [('Interpolation', X_interp, y_interp),
                           ('Mild extrap.', X_mild, y_mild),
                           ('Strong extrap.', X_strong, y_strong)]:
        X_t_tensor = torch.tensor(X_t, dtype=torch.float32)

        # VSPYCT prediction
        pred_v = vspyct.predict(X_t_tensor)
        if pred_v.dim() == 3:
            pred_v = pred_v.mean(dim=1).squeeze()
        if pred_v.dim() == 2:
            pred_v = pred_v.mean(dim=-1)
        rmse_v = np.sqrt(mean_squared_error(y_t, pred_v.numpy().flatten()))

        # VSPYCT-GP prediction
        pred_gp, unc_gp = vspyct_gp.predict(X_t_tensor, return_uncertainty=True)
        rmse_gp = np.sqrt(mean_squared_error(y_t, pred_gp.numpy().flatten()))

        diff = rmse_v - rmse_gp
        print(f"{name:<20} {rmse_v:<12.4f} {rmse_gp:<12.4f} {diff:<+10.4f} {unc_gp.mean():<10.4f}")

    print("-" * 40)


def test_tau_effect():
    """Test that tau actually affects predictions."""
    print("\n" + "=" * 60)
    print("TEST: TAU EFFECT ON PREDICTIONS")
    print("=" * 60)

    np.random.seed(42)

    # Simple synthetic data
    n_train = 100
    X_train = np.random.uniform(0, 1, (n_train, 2))
    y_train = X_train[:, 0] + X_train[:, 1] + np.random.normal(0, 0.1, n_train)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    # Out-of-distribution test point
    X_test = torch.tensor([[1.5, 1.5]], dtype=torch.float32)  # Outside [0,1] range
    y_test = np.array([3.0])  # 1.5 + 1.5 = 3.0

    model_params = {
        'max_depth': 2,
        'minimum_examples_to_split': 5,
        'epochs': 100,
        'lr': 0.01
    }

    print("\nPredictions for out-of-distribution point [1.5, 1.5]:")
    print(f"True value: {y_test[0]:.4f}")
    print("-" * 50)

    tau_values = [0.5, 1.0, 2.0, 5.0, 10.0, 100.0]

    for tau in tau_values:
        vspyct_gp = VSpyctGP(**model_params, tau=tau, kernel_type='linear',
                             gp_training_iterations=50)
        with SuppressOutput():
            vspyct_gp.fit(X_train_t, y_train_t)

        pred, unc = vspyct_gp.predict(X_test, return_uncertainty=True)
        print(f"tau={tau:<6.1f} -> pred={pred.item():<8.4f} unc={unc.item():<8.4f}")

    print("-" * 50)
    print("Note: Lower tau -> more GP usage -> predictions should approach true value")
    print("      Higher tau -> more prototype usage -> predictions bounded by training range")


if __name__ == '__main__':
    test_tau_effect()
    test_extrapolation_synthetic()
    test_on_real_data()
