"""
OpenML Benchmark Dataset Loader for VSPYCT-GP Evaluation

This module provides utilities for loading regression benchmark datasets
from OpenML, specifically the CTR23 (Curated Tabular Regression 2023) suite.

Usage:
    from src.data.openml_loader import OpenMLBenchmark

    benchmark = OpenMLBenchmark()
    datasets = benchmark.list_datasets()
    X, y, meta = benchmark.load_dataset(dataset_id)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold

try:
    import openml
except ImportError:
    raise ImportError(
        "OpenML is required for benchmark evaluation. "
        "Install it with: pip install openml"
    )


# OpenML CTR23 Benchmark Suite ID
CTR23_SUITE_ID = 353

# Cache directory for downloaded datasets
CACHE_DIR = os.path.join(os.path.dirname(__file__), '../../data/openml_cache')


class OpenMLBenchmark:
    """
    Handler for OpenML benchmark datasets for regression evaluation.

    Attributes:
        suite_id: OpenML study/suite ID (default: CTR23 = 353)
        cache_dir: Directory to cache downloaded datasets
        datasets_info: Metadata about available datasets
    """

    def __init__(self, suite_id: int = CTR23_SUITE_ID, cache_dir: str = None):
        """
        Initialize the benchmark loader.

        Args:
            suite_id: OpenML suite ID (default: 353 for CTR23)
            cache_dir: Directory to cache datasets (default: data/openml_cache)
        """
        self.suite_id = suite_id
        self.cache_dir = cache_dir or CACHE_DIR
        self.datasets_info = {}
        self._suite = None
        self._task_to_dataset = {}

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_suite(self):
        """Load the OpenML suite if not already loaded."""
        if self._suite is None:
            print(f"Loading OpenML suite {self.suite_id}...")
            self._suite = openml.study.get_suite(self.suite_id)
            print(f"Suite contains {len(self._suite.tasks)} regression tasks")
        return self._suite

    def list_datasets(self, verbose: bool = True) -> pd.DataFrame:
        """
        List all datasets in the benchmark suite with metadata.

        Args:
            verbose: Whether to print dataset information

        Returns:
            DataFrame with dataset metadata (id, name, n_samples, n_features, etc.)
        """
        suite = self._load_suite()

        datasets_list = []

        for task_id in suite.tasks:
            try:
                task = openml.tasks.get_task(task_id)
                dataset = task.get_dataset()

                # Get basic info
                info = {
                    'task_id': task_id,
                    'dataset_id': dataset.dataset_id,
                    'name': dataset.name,
                    'n_samples': dataset.qualities.get('NumberOfInstances', 'N/A'),
                    'n_features': dataset.qualities.get('NumberOfFeatures', 'N/A'),
                    'n_numeric': dataset.qualities.get('NumberOfNumericFeatures', 'N/A'),
                    'n_categorical': dataset.qualities.get('NumberOfSymbolicFeatures', 'N/A'),
                    'missing_values': dataset.qualities.get('NumberOfMissingValues', 0),
                    'target': dataset.default_target_attribute
                }

                datasets_list.append(info)
                self._task_to_dataset[task_id] = dataset.dataset_id

            except Exception as e:
                print(f"Warning: Could not load task {task_id}: {e}")
                continue

        df = pd.DataFrame(datasets_list)
        self.datasets_info = {row['task_id']: row for _, row in df.iterrows()}

        if verbose:
            print(f"\n{'='*80}")
            print(f"CTR23 Benchmark Suite: {len(df)} regression datasets")
            print(f"{'='*80}")
            print(df.to_string(index=False))
            print(f"{'='*80}\n")

        return df

    def load_dataset(
        self,
        task_id: int,
        return_tensor: bool = True,
        standardize: bool = True,
        handle_missing: str = 'mean',
        handle_categorical: str = 'onehot'
    ) -> Tuple[Any, Any, Dict]:
        """
        Load a single dataset from the benchmark.

        Args:
            task_id: OpenML task ID
            return_tensor: If True, return PyTorch tensors; else numpy arrays
            standardize: Whether to standardize features
            handle_missing: How to handle missing values ('mean', 'median', 'drop')
            handle_categorical: How to handle categorical features ('onehot', 'label')

        Returns:
            X: Feature matrix (tensor or ndarray)
            y: Target vector (tensor or ndarray)
            meta: Dictionary with dataset metadata
        """
        # Load task and dataset
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        # Get data
        X, y, categorical_indicator, feature_names = dataset.get_data(
            target=dataset.default_target_attribute
        )

        # Convert to DataFrame for easier preprocessing
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name='target')

        # Store original info
        meta = {
            'task_id': task_id,
            'dataset_id': dataset.dataset_id,
            'name': dataset.name,
            'n_samples_original': len(X),
            'n_features_original': X.shape[1],
            'target_name': dataset.default_target_attribute,
            'feature_names': list(feature_names),
            'categorical_features': [f for f, is_cat in zip(feature_names, categorical_indicator) if is_cat]
        }

        # Handle missing values
        X, y = self._handle_missing(X, y, method=handle_missing)

        # Handle categorical features
        X = self._handle_categorical(X, categorical_indicator, feature_names, method=handle_categorical)

        # Remove any remaining non-numeric columns that couldn't be processed
        X = self._remove_problematic_columns(X)

        # Ensure all columns are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Drop rows with NaN values created by coercion
        valid_mask = ~X.isna().any(axis=1)
        if valid_mask.sum() < len(X):
            n_dropped = len(X) - valid_mask.sum()
            print(f"    Dropped {n_dropped} rows with non-numeric values")
            X = X[valid_mask].reset_index(drop=True)
            y = y[valid_mask].reset_index(drop=True)

        # Standardize features
        if standardize:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            meta['scaler'] = scaler

        # Update metadata
        meta['n_samples'] = len(X)
        meta['n_features'] = X.shape[1]

        # Convert to numpy
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.float32)

        # Convert to tensors if requested
        if return_tensor:
            X_out = torch.tensor(X_np, dtype=torch.float32)
            y_out = torch.tensor(y_np, dtype=torch.float32)
        else:
            X_out = X_np
            y_out = y_np

        return X_out, y_out, meta

    def _handle_missing(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'mean'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle missing values in features and target."""

        # Remove rows with missing target
        valid_idx = ~y.isna()
        X = X[valid_idx].reset_index(drop=True)
        y = y[valid_idx].reset_index(drop=True)

        if method == 'drop':
            # Drop rows with any missing values
            valid_idx = ~X.isna().any(axis=1)
            X = X[valid_idx].reset_index(drop=True)
            y = y[valid_idx].reset_index(drop=True)
        else:
            # Impute missing values
            for col in X.columns:
                if X[col].isna().any():
                    if X[col].dtype in ['object', 'category']:
                        # For categorical: use mode
                        X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'missing')
                    else:
                        # For numeric: use mean or median
                        if method == 'median':
                            X[col] = X[col].fillna(X[col].median())
                        else:  # mean
                            X[col] = X[col].fillna(X[col].mean())

        return X, y

    def _handle_categorical(
        self,
        X: pd.DataFrame,
        categorical_indicator: List[bool],
        feature_names: List[str],
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """Handle categorical features."""

        cat_cols = [f for f, is_cat in zip(feature_names, categorical_indicator) if is_cat and f in X.columns]

        # Also detect string columns that weren't marked as categorical
        for col in X.columns:
            if col not in cat_cols and X[col].dtype == 'object':
                cat_cols.append(col)

        if not cat_cols:
            return X

        if method == 'onehot':
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        else:  # label encoding
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        return X

    def _remove_problematic_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that cannot be converted to numeric."""
        cols_to_drop = []
        for col in X.columns:
            try:
                # Try converting to numeric
                pd.to_numeric(X[col], errors='raise')
            except (ValueError, TypeError):
                # If conversion fails, mark for removal
                cols_to_drop.append(col)

        if cols_to_drop:
            print(f"    Removing {len(cols_to_drop)} non-numeric columns: {cols_to_drop}")
            X = X.drop(columns=cols_to_drop)

        return X

    def load_all_datasets(
        self,
        max_samples: int = None,
        max_features: int = None,
        **kwargs
    ) -> Dict[int, Tuple[Any, Any, Dict]]:
        """
        Load all datasets from the benchmark suite.

        Args:
            max_samples: Skip datasets with more samples than this
            max_features: Skip datasets with more features than this
            **kwargs: Additional arguments passed to load_dataset

        Returns:
            Dictionary mapping task_id to (X, y, meta) tuples
        """
        suite = self._load_suite()
        datasets = {}

        for task_id in suite.tasks:
            try:
                X, y, meta = self.load_dataset(task_id, **kwargs)

                # Apply filters
                if max_samples and meta['n_samples'] > max_samples:
                    print(f"Skipping {meta['name']}: too many samples ({meta['n_samples']})")
                    continue
                if max_features and meta['n_features'] > max_features:
                    print(f"Skipping {meta['name']}: too many features ({meta['n_features']})")
                    continue

                datasets[task_id] = (X, y, meta)
                print(f"Loaded: {meta['name']} ({meta['n_samples']} x {meta['n_features']})")

            except Exception as e:
                print(f"Error loading task {task_id}: {e}")
                continue

        return datasets

    def get_cv_splits(
        self,
        X: Any,
        y: Any,
        n_samples: int,
        random_state: int = 42
    ) -> List[Tuple]:
        """
        Get cross-validation splits according to CTR23 methodology.

        - <1,000 samples: 10x repeated 10-fold CV
        - 1,000-10,000 samples: 10-fold CV
        - >10,000 samples: 33% holdout split

        Args:
            X: Feature matrix
            y: Target vector
            n_samples: Number of samples
            random_state: Random seed

        Returns:
            List of (train_idx, test_idx) tuples
        """
        if n_samples < 1000:
            # 10x repeated 10-fold CV
            splits = []
            for repeat in range(10):
                kf = KFold(n_splits=10, shuffle=True, random_state=random_state + repeat)
                for train_idx, test_idx in kf.split(X):
                    splits.append((train_idx, test_idx))
            return splits

        elif n_samples <= 10000:
            # 10-fold CV
            kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
            return list(kf.split(X))

        else:
            # 33% holdout
            n = len(X) if hasattr(X, '__len__') else X.shape[0]
            indices = np.arange(n)
            train_idx, test_idx = train_test_split(
                indices, test_size=0.33, random_state=random_state
            )
            return [(train_idx, test_idx)]

    def save_dataset_list(self, filepath: str = None):
        """Save the list of datasets to a JSON file for reproducibility."""
        if not self.datasets_info:
            self.list_datasets(verbose=False)

        filepath = filepath or os.path.join(self.cache_dir, 'ctr23_datasets.json')

        # Convert to serializable format
        info_serializable = {}
        for task_id, info in self.datasets_info.items():
            info_serializable[str(task_id)] = {
                k: (int(v) if isinstance(v, (np.integer, np.int64)) else
                    float(v) if isinstance(v, (np.floating, np.float64)) else v)
                for k, v in info.items()
            }

        with open(filepath, 'w') as f:
            json.dump(info_serializable, f, indent=2)

        print(f"Dataset list saved to: {filepath}")


def get_benchmark_datasets(
    n_datasets: int = 20,
    size_filter: str = 'medium',
    random_state: int = 42
) -> List[int]:
    """
    Get a list of task IDs for benchmark evaluation.

    Args:
        n_datasets: Number of datasets to select
        size_filter: Size category ('small', 'medium', 'large', 'all')
        random_state: Random seed for selection

    Returns:
        List of OpenML task IDs
    """
    benchmark = OpenMLBenchmark()
    df = benchmark.list_datasets(verbose=False)

    # Convert to numeric
    df['n_samples'] = pd.to_numeric(df['n_samples'], errors='coerce')

    # Filter by size
    if size_filter == 'small':
        df = df[df['n_samples'] < 1000]
    elif size_filter == 'medium':
        df = df[(df['n_samples'] >= 1000) & (df['n_samples'] <= 10000)]
    elif size_filter == 'large':
        df = df[df['n_samples'] > 10000]

    # Select n_datasets
    if len(df) <= n_datasets:
        return df['task_id'].tolist()
    else:
        np.random.seed(random_state)
        selected = df.sample(n=n_datasets, random_state=random_state)
        return selected['task_id'].tolist()


if __name__ == '__main__':
    # Quick test
    print("Testing OpenML Benchmark Loader...")

    benchmark = OpenMLBenchmark()

    # List all datasets
    df = benchmark.list_datasets()

    # Load first dataset as test
    if len(df) > 0:
        task_id = df.iloc[0]['task_id']
        print(f"\nLoading test dataset (task_id={task_id})...")
        X, y, meta = benchmark.load_dataset(task_id)
        print(f"  Name: {meta['name']}")
        print(f"  Shape: X={X.shape}, y={y.shape}")
        print(f"  Features: {meta['n_features']}")

        # Test CV splits
        splits = benchmark.get_cv_splits(X, y, meta['n_samples'])
        print(f"  CV splits: {len(splits)}")

    print("\nTest complete!")
