"""
Debug script to understand why VSPYCT-GP might perform worse than VSPYCT.
"""

import os
import sys
import warnings
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
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


def get_dataset_by_name(name):
    """Find dataset task_id by name."""
    benchmark = OpenMLBenchmark()
    df = benchmark.list_datasets(verbose=False)
    row = df[df['name'] == name]
    if len(row) == 0:
        print(f"Dataset '{name}' not found. Available datasets:")
        print(df['name'].tolist())
        return None
    return int(row.iloc[0]['task_id'])


def analyze_support_detection(model, X_test, y_test):
    """Analyze how support detection works for test points."""
    in_support_count = 0
    out_support_count = 0
    distances = []

    # Get all leaves
    leaves = []
    def collect_leaves(node):
        if node.is_leaf():
            leaves.append(node)
        else:
            if node.left:
                collect_leaves(node.left)
            if node.right:
                collect_leaves(node.right)
    collect_leaves(model.root_node)

    print(f"\n=== Support Detection Analysis ===")
    print(f"Number of leaves: {len(leaves)}")
    print(f"Tau (threshold): {model.tau}")

    # Check each leaf
    for i, leaf in enumerate(leaves):
        if leaf.leaf_X is not None:
            n_samples = len(leaf.leaf_X)
            print(f"  Leaf {i}: {n_samples} training samples")
            if leaf.cov_matrix_inv is not None:
                print(f"    Has covariance matrix inverse")
            else:
                print(f"    NO covariance matrix (using Euclidean)")

    # For each test point, check support status
    for i in range(min(len(X_test), 50)):  # Check first 50 points
        x = X_test[i]

        # Route to leaf (using first MC sample)
        leaf = model._route_to_leaf(model.root_node, x)

        # Check distance
        dist = leaf.compute_mahalanobis_distance(x)
        distances.append(dist)

        if leaf.is_in_support(x, model.tau):
            in_support_count += 1
        else:
            out_support_count += 1

    print(f"\n=== Test Point Support Status (first 50) ===")
    print(f"In-support: {in_support_count}")
    print(f"Out-of-support: {out_support_count}")
    print(f"Distance stats: min={min(distances):.2f}, max={max(distances):.2f}, mean={np.mean(distances):.2f}")

    return distances, in_support_count, out_support_count


def compare_predictions(vspyct, vspyct_gp, X_test, y_test):
    """Compare predictions from both models point-by-point."""
    print(f"\n=== Prediction Comparison ===")

    # Get VSPYCT predictions
    pred_v = vspyct.predict(X_test)
    if pred_v.dim() == 3:
        pred_v = pred_v.mean(dim=1).squeeze()
    elif pred_v.dim() == 2:
        pred_v = pred_v.mean(dim=-1)

    # Get VSPYCT-GP predictions
    pred_gp, unc_gp = vspyct_gp.predict(X_test, return_uncertainty=True)

    y_np = y_test.numpy().flatten()
    pred_v_np = pred_v.numpy().flatten()
    pred_gp_np = pred_gp.numpy().flatten()

    rmse_v = np.sqrt(mean_squared_error(y_np, pred_v_np))
    rmse_gp = np.sqrt(mean_squared_error(y_np, pred_gp_np))

    print(f"VSPYCT RMSE: {rmse_v:.4f}")
    print(f"VSPYCT-GP RMSE: {rmse_gp:.4f}")
    print(f"Difference: {(rmse_gp - rmse_v) / rmse_v * 100:+.1f}%")

    # Analyze per-point differences
    errors_v = np.abs(y_np - pred_v_np)
    errors_gp = np.abs(y_np - pred_gp_np)

    gp_better = np.sum(errors_gp < errors_v)
    v_better = np.sum(errors_v < errors_gp)

    print(f"\nPer-point comparison:")
    print(f"  GP better: {gp_better} points")
    print(f"  VSPYCT better: {v_better} points")
    print(f"  Equal: {len(y_np) - gp_better - v_better} points")

    # Look at worst GP predictions
    gp_worse_idx = np.argsort(errors_gp - errors_v)[-5:]
    print(f"\nWorst GP predictions (compared to VSPYCT):")
    for idx in gp_worse_idx:
        print(f"  Point {idx}: y={y_np[idx]:.2f}, VSPYCT={pred_v_np[idx]:.2f}, GP={pred_gp_np[idx]:.2f}, unc={unc_gp[idx]:.4f}")

    return rmse_v, rmse_gp


def main(dataset_name='solar_flare'):
    print("=" * 70)
    print(f"Debugging VSPYCT-GP Performance on {dataset_name}")
    print("=" * 70)

    # Get dataset
    task_id = get_dataset_by_name(dataset_name)
    if task_id is None:
        return

    benchmark = OpenMLBenchmark()
    X, y, meta = benchmark.load_dataset(task_id, return_tensor=True)

    print(f"\nDataset: {meta['name']}")
    print(f"Samples: {meta['n_samples']}, Features: {meta['n_features']}")

    # Train/test split
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, y_train = X[train_idx], y[train_idx].reshape(-1, 1)
    X_test, y_test = X[test_idx], y[test_idx].reshape(-1, 1)

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Target range - Train: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"Target range - Test: [{y_test.min():.2f}, {y_test.max():.2f}]")

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
        'use_gp_mean_for_extrapolation': True
    }

    print(f"\nModel params: {model_params}")
    print(f"GP params: {gp_params}")

    # Train VSPYCT
    print("\n--- Training VSPYCT ---")
    vspyct = VSpyct(**model_params)
    with SuppressOutput():
        vspyct.fit(X_train, y_train)
    print(f"Nodes: {vspyct.num_nodes}")

    # Train VSPYCT-GP
    print("\n--- Training VSPYCT-GP ---")
    vspyct_gp = VSpyctGP(**model_params, **gp_params)
    vspyct_gp.fit(X_train, y_train)
    print(f"Nodes: {vspyct_gp.num_nodes}")

    # Analyze support detection
    analyze_support_detection(vspyct_gp, X_test, y_test)

    # Compare predictions
    rmse_v, rmse_gp = compare_predictions(vspyct, vspyct_gp, X_test, y_test)

    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING")
    print("=" * 70)

    # Test different tau values
    print("\n--- Effect of tau (support threshold) ---")
    for tau in [1.0, 2.0, 3.0, 4.5, 6.0, 10.0, 100.0]:
        vspyct_gp.tau = tau
        pred_gp, _ = vspyct_gp.predict(X_test, return_uncertainty=True)
        rmse = np.sqrt(mean_squared_error(y_test.numpy().flatten(), pred_gp.numpy().flatten()))
        print(f"  tau={tau:5.1f}: RMSE={rmse:.4f}")

    # Reset tau
    vspyct_gp.tau = gp_params['tau']

    # Test with use_gp_mean_for_extrapolation = False
    print("\n--- Effect of use_gp_mean_for_extrapolation ---")
    vspyct_gp.use_gp_mean_for_extrapolation = False
    pred_gp_proto, _ = vspyct_gp.predict(X_test, return_uncertainty=True)
    rmse_proto = np.sqrt(mean_squared_error(y_test.numpy().flatten(), pred_gp_proto.numpy().flatten()))
    print(f"  use_gp_mean=False: RMSE={rmse_proto:.4f}")

    vspyct_gp.use_gp_mean_for_extrapolation = True
    pred_gp_mean, _ = vspyct_gp.predict(X_test, return_uncertainty=True)
    rmse_mean = np.sqrt(mean_squared_error(y_test.numpy().flatten(), pred_gp_mean.numpy().flatten()))
    print(f"  use_gp_mean=True:  RMSE={rmse_mean:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"VSPYCT RMSE:    {rmse_v:.4f}")
    print(f"VSPYCT-GP RMSE: {rmse_gp:.4f}")
    if rmse_gp > rmse_v:
        print("\n*** VSPYCT-GP performs WORSE ***")
        print("Possible causes:")
        print("1. Test data is IID - no extrapolation benefit")
        print("2. GP predictions for out-of-support points are worse than prototypes")
        print("3. Support detection might be too aggressive (tau too small)")
        print("4. GP hyperparameters might not be well-tuned")


if __name__ == '__main__':
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'solar_flare'
    main(dataset)
