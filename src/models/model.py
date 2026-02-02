import torch
from src.models.node import Node, VNode, VNodeGP, load_and_prediction, batch_load_and_prediction
from src.models.split import learn_split, learn_split_vb
import numpy as np
import pyro
from pyro.infer import Predictive
from sklearn.preprocessing import MinMaxScaler

def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output

def multi_target_impurity(tensor, target_dim=1):
    """
    Compute impurity for multiple target variables.
    
    tensor: Input tensor where each row represents a sample and each column represents a different target variable.
    target_dim: The dimension along which the targets are stored (e.g., 1 if columns are different targets).
    """
    if tensor.isnan().all():
        return float('inf')
    
    variances = nanvar(tensor, dim=0 if target_dim == 1 else 1)
    
    total_variance = torch.sum(variances)

    return total_variance

class Spyct:
    def __init__(self, max_depth=np.inf, subspace_size=1, minimum_examples_to_split=2,
                 device='cpu', epochs=100, bs=None, lr=0.1):
        self.minimum_examples_to_split = minimum_examples_to_split
        self.root_node = None
        self.num_nodes = 0
        self.device = device
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.max_depth = max_depth
        self.subspace_size = subspace_size

    def fit(self, descriptive_data, target_data, clustering_data=None, rows=None, enable_mc_dropout=True):
        if clustering_data is None: clustering_data = target_data
        if rows is None: rows = torch.arange(descriptive_data.shape[0])

        self.num_training_instances = descriptive_data.shape[0]
        total_variance = multi_target_impurity(clustering_data)
        self.root_node = Node(depth=0, enable_mc_dropout=enable_mc_dropout)
        splitting_queue = [(self.root_node, rows, total_variance)]
        order = 0
        while splitting_queue:
            node, rows, total_variance = splitting_queue.pop()
            node.order = order
            node.num_instances = rows.shape[0]
            order += 1
            if total_variance > 0 and node.depth < self.max_depth and rows.size(0) >= self.minimum_examples_to_split:
                split_model = learn_split(
                    rows, descriptive_data[rows], clustering_data[rows],
                    device=self.device, epochs=self.epochs, bs=self.bs, lr=self.lr, subspace_size=self.subspace_size)
                split = split_model(descriptive_data[rows]).squeeze()
                rows_right = rows[split > torch.tensor(0., device=self.device)]
                var_right = multi_target_impurity(clustering_data[rows_right])
                rows_left = rows[split <= torch.tensor(0., device=self.device)]
                var_left = multi_target_impurity(clustering_data[rows_left])
                if var_left < total_variance or var_right < total_variance:
                    node.split_model = split_model
                    node.left = Node(depth=node.depth+1)
                    node.right = Node(depth=node.depth+1)
                    splitting_queue.append((node.left, rows_left, var_left, ))
                    splitting_queue.append((node.right, rows_right, var_right, ))
                else: node.prototype = torch.nanmean(target_data[rows], dim=0)

            else: node.prototype = torch.nanmean(target_data[rows], dim=0)

        self.num_nodes = order

    def predict(self, descriptive_data):
        raw_predictions = [self.root_node.predict(descriptive_data[i]) for i in range(descriptive_data.size(0))]
        return torch.stack(raw_predictions)
    
    def feature_importances(self, num_samples=200, k=None):
        non_leaves = []
        def _traverse(node):
            if node is not None:
                if node.left is not None or node.right is not None: non_leaves.append(node)
                _traverse(node.left)
                _traverse(node.right)
        
        _traverse(self.root_node)
        weights = []
        for node in non_leaves:
            weights.append(node.split_model.linear.weight.detach().numpy())
        weights = np.abs(np.array(weights))
        weights = weights.reshape(weights.shape[0], -1)
        weights_normalized = (weights - weights.min()) / (weights.max() - weights.min() + 0.0001)
        importances = torch.zeros((weights_normalized.shape[1]))
        for i, node in enumerate(non_leaves):importances += node.num_instances/self.num_training_instances*(weights_normalized[i, :]/np.linalg.norm(weights_normalized[i, :]))
        if k is not None: return dict(zip(torch.topk(importances, k=k).indices.tolist(), torch.topk(importances, k=k).values.tolist()))
        else: return importances


class VSpyct:
    def __init__(self, max_depth=np.inf, subspace_size=1, minimum_examples_to_split=2,
                 device='cpu', epochs=500, bs=None, lr=0.001):
        self.root_node = None
        self.num_nodes = 0
        self.minimum_examples_to_split = minimum_examples_to_split
        self.device = device
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.max_depth = max_depth
        self.subspace_size = subspace_size
        self.scaler = MinMaxScaler(feature_range=(0, 100))

    def fit(self, descriptive_data, target_data, clustering_data=None, rows=None):
        target_data = torch.Tensor(self.scaler.fit_transform(target_data))
        if clustering_data is None: clustering_data = target_data
        if rows is None: rows = torch.arange(descriptive_data.shape[0])

        self.num_training_instances = descriptive_data.shape[0]
        self.descriptive_data_shape = descriptive_data.shape
        total_variance = multi_target_impurity(clustering_data)
        print(f'Total variance: {total_variance}')
        self.root_node = VNode(descriptive_values=descriptive_data, depth=0, root = True)
        splitting_queue = [(self.root_node, rows, total_variance)]
        order = 0

        while splitting_queue:
            node, rows, total_variance = splitting_queue.pop()
            node.order = order
            node.num_instances = rows.shape[0]
            order += 1
            if total_variance > 0 and node.depth < self.max_depth and rows.size(0) >= self.minimum_examples_to_split:
                split_model, guide, losses = learn_split_vb(
                    rows, descriptive_data[rows], clustering_data[rows],
                    device=self.device, epochs=self.epochs, bs=self.bs, lr=self.lr, subspace_size=self.subspace_size)
                
                predictive = Predictive(model = split_model.linear.to(self.device),
                            guide=guide,
                            num_samples=100,
                            return_sites=("linear.weight", "linear.bias"))
                sdata = predictive(descriptive_data[rows].clone().detach())
                sdata_lin = torch.mean(sdata['linear.weight'], dim=0).reshape(-1, 1).T.T
                sdata_b = torch.mean(sdata['linear.bias'], dim=0).reshape(-1, 1).T
                ssplit = (descriptive_data[rows] @ sdata_lin + sdata_b).sigmoid()
                split = ssplit.reshape(-1)

                # split_model = split_model.linear
                # split = split_model(descriptive_data[rows]).squeeze()
                
                rows_right = rows[split > torch.tensor(0.5, device=self.device)]
                var_right = multi_target_impurity(clustering_data[rows_right])
                rows_left = rows[split <= torch.tensor(0.5, device=self.device)]
                var_left = multi_target_impurity(clustering_data[rows_left])

                print('Rows left: ', rows_left.shape, 'Var left', var_left)
                print('Rows right: ', rows_right.shape, 'Var right', var_right)

                if (var_left < total_variance or var_right < total_variance) and (self.minimum_examples_to_split < rows_right.shape[0] and self.minimum_examples_to_split < rows_left.shape[0]):
                    node.split_model = split_model.linear
                    node.guide = guide
                    node.losses = losses
                    node.left = VNode(descriptive_values=descriptive_data[rows_left], depth=node.depth+1)
                    node.right = VNode(descriptive_values=descriptive_data[rows_right], depth=node.depth+1)
                    splitting_queue.append((node.left, rows_left, var_left, ))
                    splitting_queue.append((node.right, rows_right, var_right, ))
                # potentially to be fixed
                else:
                    if len(target_data.shape)==1: node.prototype = torch.nanmean(target_data[rows], dim=0)
                    else:
                        # node.prototype = torch.nanmean(target_data[rows], dim=0)
                        target_data_np = target_data.numpy()
                        prototype = []
                        for col in range(target_data_np.shape[1]):
                            non_nan_values = target_data_np[rows, col][~torch.isnan(target_data[rows, col])]
                            mean_value = np.nanmean(non_nan_values)
                            prototype.append(mean_value)
                        node.prototype = torch.tensor(np.array(prototype))
            else:
                # node.prototype = torch.nanmean(target_data[rows], dim=0)
                if len(target_data.shape)==1: node.prototype = torch.nanmean(target_data[rows], dim=0)
                else:
                    target_data_np = target_data.numpy()
                    prototype = []
                    for col in range(target_data_np.shape[1]):
                        non_nan_values = target_data_np[rows, col][~torch.isnan(target_data[rows, col])]
                        mean_value = np.nanmean(non_nan_values)
                        prototype.append(mean_value)
                    node.prototype = torch.tensor(np.array(prototype))
        self.num_nodes = order

    def predict(self, descriptive_data, num_mc_samples=10):
        """
        Make predictions with Monte Carlo sampling (optimized batched version).

        Args:
            descriptive_data: Input features (n_samples, n_features)
            num_mc_samples: Number of Monte Carlo samples

        Returns:
            predictions: Predicted values with shape (n_samples, num_mc_samples)
        """
        n_samples = descriptive_data.size(0)

        # Collect MC samples: shape (num_mc_samples, n_samples)
        all_predictions = []

        # Clear param store once at start
        pyro.clear_param_store()

        for _ in range(num_mc_samples):
            # Route all samples through tree and get leaf assignments
            leaf_assignments = self._batch_route_to_leaves(descriptive_data)

            # Collect predictions for this MC sample
            mc_preds = torch.zeros(n_samples)

            for leaf, sample_indices in leaf_assignments.items():
                if len(sample_indices) == 0:
                    continue

                # Get prototype for this leaf
                proto = leaf.prototype
                if isinstance(proto, torch.Tensor):
                    if proto.numel() > 1:
                        proto = proto.mean()
                    else:
                        proto = proto.flatten()[0] if proto.numel() == 1 else proto

                # Assign prototype to all samples in this leaf
                for idx in sample_indices:
                    mc_preds[idx] = proto

            all_predictions.append(mc_preds)

        # Stack: shape (n_samples, num_mc_samples)
        predictions = torch.stack(all_predictions, dim=1)

        # Apply inverse transform
        predictions_reshaped = predictions.cpu().numpy()
        predictions_reshaped = self.scaler.inverse_transform(predictions_reshaped)
        predictions = torch.Tensor(predictions_reshaped)

        return predictions

    def _batch_route_to_leaves(self, descriptive_data):
        """
        Route all samples through tree and return leaf assignments.

        Args:
            descriptive_data: Input features (n_samples, n_features)

        Returns:
            Dictionary mapping leaf nodes to list of sample indices
        """
        n_samples = descriptive_data.size(0)
        leaf_assignments = {}

        # Start with all samples at root
        queue = [(self.root_node, list(range(n_samples)))]

        while queue:
            node, indices = queue.pop()

            if not indices:
                continue

            if node.is_leaf():
                leaf_assignments[node] = indices
            else:
                # Get samples for this node
                X_node = descriptive_data[indices]

                # Sample split parameters once for this node
                data = batch_load_and_prediction(
                    node.split_model, node.guide, X_node,
                    num_samples=1, clear_store=False
                )

                # Compute splits for all samples
                weight = data['linear.weight'].squeeze(0)  # (1, n_features)
                bias = data['linear.bias'].squeeze(0)  # (1,)

                splits = (X_node @ weight.T + bias).sigmoid().squeeze()

                # Handle single sample case
                if splits.dim() == 0:
                    splits = splits.unsqueeze(0)

                # Split indices based on split values
                left_mask = splits <= 0.5
                right_mask = ~left_mask

                left_indices = [indices[i] for i in range(len(indices)) if left_mask[i]]
                right_indices = [indices[i] for i in range(len(indices)) if right_mask[i]]

                if left_indices:
                    queue.append((node.left, left_indices))
                if right_indices:
                    queue.append((node.right, right_indices))

        return leaf_assignments
    
    @classmethod
    def create_edge_list(cls, root):
        edges = []
        cls.traverse(root, edges)
        return edges

    @classmethod
    def traverse(cls, node, edges):
        if node is None:
            return

        if node.left is not None:
            edges.append((node, node.left))
            cls.traverse(node.left, edges)

        if node.right is not None:
            edges.append((node, node.right))
            cls.traverse(node.right, edges)

    def get_nodes(self, node, list):
        list.append(node)
        if node.left is not None:
            self.get_nodes(node.left, list)
        if node.right is not None:
            self.get_nodes(node.right, list)

    def get_leaves(self, node, non_leaves):
        if node is not None:
            if node.left is None or node.right is None: non_leaves.append(node)
            self.get_leaves(node.left, non_leaves)
            self.get_leaves(node.right, non_leaves)

    def feature_importances(self, num_samples=100, k=None):
        assert self.num_training_instances!=0, 'The model is not fitted! Unable to calculate feature importances.'
        non_leaves = []
        def _traverse(node):
            if node is not None:
                if node.left is not None or node.right is not None: non_leaves.append(node)
                _traverse(node.left)
                _traverse(node.right)

        _traverse(self.root_node)
        weights = []
        vars = []
        for node in non_leaves:
            predictive = Predictive(model=node.split_model.to(self.device),
                                    guide=node.guide,
                                    num_samples=num_samples,
                                    return_sites=(["linear.weight"]))
            data = predictive(node.descriptive_values)['linear.weight']
            weights.append(data.mean(axis=0).numpy())
            vars.append(data.var(axis=0).numpy())
        weights = np.abs(np.array(weights))
        weights = weights.reshape(weights.shape[0], -1)
        vars = np.abs(np.array(vars))
        vars = vars.reshape(vars.shape[0], -1)
        weights_normalized = weights # (weights - weights.min()) / (weights.max() - weights.min() + 0.0001)
        importances = torch.zeros((weights_normalized.shape[1]))
        for i, node in enumerate(non_leaves):
            importances += (node.num_instances/self.num_training_instances)*(weights_normalized[i, :]/(vars[i]+ 0.0001))
        if k is not None: return dict(zip(torch.topk(importances, k=k).indices.tolist(), torch.topk(importances, k=k).values.tolist()))
        else: return importances


class VSpyctGP:
    """
    Variational Oblique Predictive Clustering Tree with Gaussian Process Leaves (VSPYCT-GP).

    This model extends VSPYCT by replacing constant leaf prototypes with Gaussian Process
    predictors, enabling uncertainty-aware extrapolation beyond the training target range.

    Key features:
    - Bayesian oblique splits with variational inference (same as VSPYCT)
    - GP leaf models that capture local functional behavior
    - Extrapolation-aware gating: uses prototype for in-support, GP for out-of-support
    - Principled uncertainty quantification combining routing and functional uncertainty

    Args:
        max_depth: Maximum tree depth
        subspace_size: Fraction of features to consider for each split
        minimum_examples_to_split: Minimum samples required to split a node
        device: PyTorch device ('cpu' or 'cuda')
        epochs: Number of epochs for split learning
        bs: Batch size for split learning
        lr: Learning rate for split learning
        tau: Threshold for support detection ('auto' or float Mahalanobis distance)
        epsilon_sq: Noise floor for in-support predictions
        kernel_type: GP kernel type ('linear', 'rbf', 'matern', 'linear_rbf', 'polynomial')
        gp_training_iterations: Number of iterations for GP hyperparameter optimization
        tau_percentile: Percentile for auto tau calibration (default 99 = conservative)
        temperature: Sharpness of soft gating transition (lower = sharper, default 0.5)
    """

    def __init__(self, max_depth=np.inf, subspace_size=1, minimum_examples_to_split=2,
                 device='cpu', epochs=500, bs=None, lr=0.001, tau='auto', epsilon_sq=1e-4,
                 kernel_type='linear', gp_training_iterations=50, tau_percentile=99,
                 temperature=0.5):
        self.root_node = None
        self.num_nodes = 0
        self.minimum_examples_to_split = minimum_examples_to_split
        self.device = device
        self.epochs = epochs
        self.bs = bs
        self.lr = lr
        self.max_depth = max_depth
        self.subspace_size = subspace_size
        self.scaler = MinMaxScaler(feature_range=(0, 100))

        # VSPYCT-GP specific parameters
        self.tau = tau  # Can be 'auto' or a float value
        self.tau_percentile = tau_percentile  # Percentile for auto tau calibration (99 = conservative)
        self.epsilon_sq = epsilon_sq
        self.kernel_type = kernel_type
        self.gp_training_iterations = gp_training_iterations
        self.temperature = temperature  # Controls sharpness of soft gating transition
        self._calibrated_tau = None  # Set during fit if tau='auto'

        # Store data for leaf assignment
        self.descriptive_data = None
        self.target_data = None
        self.num_training_instances = 0
        self.descriptive_data_shape = None

    def fit(self, descriptive_data, target_data, clustering_data=None, rows=None):
        """
        Fit the VSPYCT-GP model.

        Args:
            descriptive_data: Input features (n_samples, n_features)
            target_data: Target values (n_samples,) or (n_samples, n_targets)
            clustering_data: Data used for clustering (defaults to target_data)
            rows: Indices of rows to use (defaults to all rows)
        """
        # Scale target data
        target_data_scaled = torch.Tensor(self.scaler.fit_transform(target_data.numpy() if isinstance(target_data, torch.Tensor) else target_data))

        if clustering_data is None:
            clustering_data = target_data_scaled
        if rows is None:
            rows = torch.arange(descriptive_data.shape[0])

        # Store data for later use
        self.descriptive_data = descriptive_data
        self.target_data = target_data_scaled
        self.num_training_instances = descriptive_data.shape[0]
        self.descriptive_data_shape = descriptive_data.shape

        total_variance = multi_target_impurity(clustering_data)
        print(f'Total variance: {total_variance}')

        # Initialize root node with GP capability
        self.root_node = VNodeGP(descriptive_values=descriptive_data, depth=0, root=True, device=self.device)

        # Track which rows belong to each leaf for GP fitting
        leaf_assignments = {}

        splitting_queue = [(self.root_node, rows, total_variance)]
        order = 0

        while splitting_queue:
            node, rows, total_variance = splitting_queue.pop()
            node.order = order
            node.num_instances = rows.shape[0]
            order += 1

            if total_variance > 0 and node.depth < self.max_depth and rows.size(0) >= self.minimum_examples_to_split:
                # Learn variational split
                split_model, guide, losses = learn_split_vb(
                    rows, descriptive_data[rows], clustering_data[rows],
                    device=self.device, epochs=self.epochs, bs=self.bs,
                    lr=self.lr, subspace_size=self.subspace_size)

                # Compute split using posterior mean
                predictive = Predictive(model=split_model.linear.to(self.device),
                            guide=guide,
                            num_samples=100,
                            return_sites=("linear.weight", "linear.bias"))
                sdata = predictive(descriptive_data[rows].clone().detach())
                sdata_lin = torch.mean(sdata['linear.weight'], dim=0).reshape(-1, 1).T.T
                sdata_b = torch.mean(sdata['linear.bias'], dim=0).reshape(-1, 1).T
                ssplit = (descriptive_data[rows] @ sdata_lin + sdata_b).sigmoid()
                split = ssplit.reshape(-1)

                rows_right = rows[split > torch.tensor(0.5, device=self.device)]
                var_right = multi_target_impurity(clustering_data[rows_right])
                rows_left = rows[split <= torch.tensor(0.5, device=self.device)]
                var_left = multi_target_impurity(clustering_data[rows_left])

                print('Rows left: ', rows_left.shape, 'Var left', var_left)
                print('Rows right: ', rows_right.shape, 'Var right', var_right)

                if (var_left < total_variance or var_right < total_variance) and \
                   (self.minimum_examples_to_split < rows_right.shape[0] and
                    self.minimum_examples_to_split < rows_left.shape[0]):
                    # Store split model and guide
                    node.split_model = split_model.linear
                    node.guide = guide
                    node.losses = losses

                    # Create child nodes
                    node.left = VNodeGP(descriptive_values=descriptive_data[rows_left],
                                       depth=node.depth+1, device=self.device)
                    node.right = VNodeGP(descriptive_values=descriptive_data[rows_right],
                                        depth=node.depth+1, device=self.device)

                    splitting_queue.append((node.left, rows_left, var_left))
                    splitting_queue.append((node.right, rows_right, var_right))
                else:
                    # This is a leaf node - compute prototype and fit GP
                    self._setup_leaf_node(node, rows, descriptive_data, target_data_scaled)
            else:
                # This is a leaf node - compute prototype and fit GP
                self._setup_leaf_node(node, rows, descriptive_data, target_data_scaled)

        self.num_nodes = order
        print(f'Tree built with {self.num_nodes} nodes')

        # Calibrate tau if set to 'auto'
        if self.tau == 'auto':
            self._calibrate_tau(descriptive_data)
            print(f'Calibrated tau = {self._calibrated_tau:.3f} (percentile={self.tau_percentile})')

    def _calibrate_tau(self, descriptive_data):
        """
        Calibrate tau based on training data distribution in each leaf.

        Computes Mahalanobis distances for all training points to their
        leaf centroid and sets tau as the specified percentile.
        """
        all_distances = []

        leaves = self.get_leaves()
        for leaf in leaves:
            if leaf.leaf_X is not None and len(leaf.leaf_X) > 1:
                # Compute distances for all training points in this leaf
                for i in range(len(leaf.leaf_X)):
                    x = leaf.leaf_X[i]
                    dist = leaf.compute_mahalanobis_distance(x)
                    if dist != float('inf'):
                        all_distances.append(dist)

        if len(all_distances) > 0:
            # Set tau as the specified percentile of training distances
            self._calibrated_tau = float(np.percentile(all_distances, self.tau_percentile))
        else:
            # Fallback to default
            self._calibrated_tau = 2.0

    def _setup_leaf_node(self, node, rows, descriptive_data, target_data):
        """Set up a leaf node with prototype and GP model."""
        # Compute prototype (mean of targets)
        if len(target_data.shape) == 1:
            node.prototype = torch.nanmean(target_data[rows], dim=0)
        else:
            target_data_np = target_data.numpy()
            prototype = []
            for col in range(target_data_np.shape[1]):
                non_nan_values = target_data_np[rows, col][~torch.isnan(target_data[rows, col])]
                mean_value = np.nanmean(non_nan_values)
                prototype.append(mean_value)
            node.prototype = torch.tensor(np.array(prototype))

        # Fit GP for extrapolation
        X_leaf = descriptive_data[rows]

        # For GP, we need single-target output
        if len(target_data.shape) == 1:
            y_leaf = target_data[rows]
        else:
            # For multi-target, fit GP on first target or average
            y_leaf = target_data[rows, 0] if target_data.shape[1] > 0 else target_data[rows].mean(dim=1)

        # Filter out samples with NaN in y_leaf (for semi-supervised learning)
        valid_mask = ~torch.isnan(y_leaf)
        X_leaf_valid = X_leaf[valid_mask]
        y_leaf_valid = y_leaf[valid_mask]

        # Only fit GP if we have enough valid (non-NaN) data points
        if len(y_leaf_valid) >= 3:
            try:
                node.fit_gp(X_leaf_valid, y_leaf_valid,
                           kernel_type=self.kernel_type,
                           training_iterations=self.gp_training_iterations)
                print(f'  Fitted GP at leaf (depth={node.depth}) with {len(y_leaf_valid)} valid samples (of {len(rows)} total)')
            except Exception as e:
                print(f'  Warning: Could not fit GP at leaf (depth={node.depth}): {e}')
                node.gp_model = None
        else:
            print(f'  Leaf (depth={node.depth}) has only {len(y_leaf_valid)} valid samples, skipping GP')
            node.gp_model = None

    def predict(self, descriptive_data, num_mc_samples=10, return_uncertainty=False,
                uncertainty_type='functional'):
        """
        Make predictions with extrapolation-aware gating (optimized batched version).

        Implements Algorithm 1 from the paper: Monte Carlo sampling over split
        parameters combined with gated GP/prototype predictions.

        Args:
            descriptive_data: Input features (n_samples, n_features)
            num_mc_samples: Number of Monte Carlo samples
            return_uncertainty: If True, return (predictions, uncertainties)
            uncertainty_type: 'total' for full variance (functional + routing),
                            'functional' for just GP/prototype variance (default)

        Returns:
            predictions: Predicted values
            uncertainties: (optional) Predictive variances
        """
        n_samples = descriptive_data.size(0)
        effective_tau = self._get_effective_tau()

        # Collect MC samples: shape (num_mc_samples, n_samples)
        all_means = torch.zeros(num_mc_samples, n_samples)
        all_variances = torch.zeros(num_mc_samples, n_samples)

        # Clear param store once at start
        pyro.clear_param_store()

        for mc_idx in range(num_mc_samples):
            # Route all samples through tree and get leaf assignments
            leaf_assignments = self._batch_route_to_leaves(descriptive_data)

            # Process each leaf's samples in batch
            for leaf, sample_indices in leaf_assignments.items():
                if len(sample_indices) == 0:
                    continue

                # Get samples for this leaf
                X_leaf = descriptive_data[sample_indices]

                # Batch prediction with gating
                means, vars = leaf.predict_with_gating_batch(
                    X_leaf, effective_tau, self.epsilon_sq, self.temperature
                )

                # Store results
                for i, idx in enumerate(sample_indices):
                    all_means[mc_idx, idx] = means[i].float()
                    all_variances[mc_idx, idx] = vars[i].float()

        # Aggregate across MC samples
        # E[Y] = E[E[Y|L]] = mean of means across MC samples
        pred_means = all_means.mean(dim=0)

        # Uncertainty computation
        if uncertainty_type == 'total':
            # Law of total variance: Var[Y] = E[Var[Y|L]] + Var[E[Y|L]]
            pred_vars = all_variances.mean(dim=0) + all_means.var(dim=0)
        else:
            # Functional variance only: E[Var[Y|L]]
            pred_vars = all_variances.mean(dim=0)

        # Inverse transform predictions
        predictions_np = pred_means.numpy().reshape(-1, 1)
        predictions_transformed = self.scaler.inverse_transform(predictions_np)
        predictions = torch.tensor(predictions_transformed).squeeze()

        if return_uncertainty:
            # Scale uncertainty by scaler scale
            scale_factor = (self.scaler.data_max_ - self.scaler.data_min_) / 100.0
            uncertainties = pred_vars * (scale_factor ** 2)
            return predictions, uncertainties.squeeze()

        return predictions

    def _batch_route_to_leaves(self, descriptive_data):
        """
        Route all samples through tree and return leaf assignments.

        Args:
            descriptive_data: Input features (n_samples, n_features)

        Returns:
            Dictionary mapping leaf nodes to list of sample indices
        """
        n_samples = descriptive_data.size(0)
        leaf_assignments = {}

        # Start with all samples at root
        # Queue contains (node, sample_indices)
        queue = [(self.root_node, list(range(n_samples)))]

        while queue:
            node, indices = queue.pop()

            if not indices:
                continue

            if node.is_leaf():
                leaf_assignments[node] = indices
            else:
                # Get samples for this node
                X_node = descriptive_data[indices]

                # Sample split parameters once for this node
                data = batch_load_and_prediction(
                    node.split_model, node.guide, X_node,
                    num_samples=1, clear_store=False
                )

                # Compute splits for all samples
                # data['linear.weight'] shape: (1, 1, n_features)
                # data['linear.bias'] shape: (1, 1)
                weight = data['linear.weight'].squeeze(0)  # (1, n_features)
                bias = data['linear.bias'].squeeze(0)  # (1,)

                splits = (X_node @ weight.T + bias).sigmoid().squeeze()

                # Handle single sample case
                if splits.dim() == 0:
                    splits = splits.unsqueeze(0)

                # Split indices based on split values
                left_mask = splits <= 0.5
                right_mask = ~left_mask

                left_indices = [indices[i] for i in range(len(indices)) if left_mask[i]]
                right_indices = [indices[i] for i in range(len(indices)) if right_mask[i]]

                if left_indices:
                    queue.append((node.left, left_indices))
                if right_indices:
                    queue.append((node.right, right_indices))

        return leaf_assignments

    def _mc_predict_single(self, x, num_mc_samples):
        """
        Monte Carlo prediction for a single instance.

        Args:
            x: Single input (1D tensor)
            num_mc_samples: Number of MC samples

        Returns:
            means: Tensor of predicted means
            variances: Tensor of predicted variances
        """
        means = []
        variances = []

        # Use calibrated tau if available, otherwise use provided tau
        effective_tau = self._get_effective_tau()

        for _ in range(num_mc_samples):
            # Route to leaf with sampled split parameters
            leaf = self._route_to_leaf(self.root_node, x)

            # Get prediction using soft gating mechanism
            mean, var = leaf.predict_with_gating(x, effective_tau, self.epsilon_sq, self.temperature)

            # Handle tensor vs scalar
            if isinstance(mean, torch.Tensor):
                mean = mean.float()
            else:
                mean = torch.tensor(mean).float()
            if isinstance(var, torch.Tensor):
                var = var.float()
            else:
                var = torch.tensor(var).float()

            means.append(mean)
            variances.append(var)

        return torch.stack(means), torch.stack(variances)

    def _get_effective_tau(self):
        """Get the effective tau value (user-specified takes precedence if numeric)."""
        # User-specified numeric tau takes precedence (allows runtime override)
        if isinstance(self.tau, (int, float)):
            return float(self.tau)
        # Use calibrated tau if available
        elif self._calibrated_tau is not None:
            return self._calibrated_tau
        else:
            # Default fallback
            return 2.0

    def _route_to_leaf(self, node, x):
        """
        Route x to a leaf node by sampling split parameters.

        Args:
            node: Current node
            x: Input to route

        Returns:
            Leaf node
        """
        if node.is_leaf():
            return node

        # Sample split parameters and compute split
        data = load_and_prediction(node.split_model, node.guide, x, num_samples=1)
        split = (data['linear.weight'] @ x + data['linear.bias']).sigmoid().item()

        if split <= 0.5:
            return self._route_to_leaf(node.left, x)
        else:
            return self._route_to_leaf(node.right, x)

    def predict_standard(self, descriptive_data):
        """
        Standard prediction (like VSPYCT, without gating).

        Uses Monte Carlo sampling over splits but always uses prototype at leaves.

        Args:
            descriptive_data: Input features

        Returns:
            predictions: Predicted values
        """
        raw_predictions = [self.root_node.mc_predict(descriptive_data[i])
                          for i in range(descriptive_data.size(0))]
        predictions = torch.stack(raw_predictions)

        # Get the original shape
        original_shape = predictions.shape

        # Reshape and inverse transform
        predictions_reshaped = predictions.view(-1, original_shape[-1]).cpu().numpy()
        predictions_reshaped = self.scaler.inverse_transform(predictions_reshaped)
        predictions = torch.Tensor(predictions_reshaped).view(original_shape)

        return predictions

    def get_leaves(self, node=None, leaves=None):
        """Get all leaf nodes in the tree."""
        if leaves is None:
            leaves = []
        if node is None:
            node = self.root_node

        if node.is_leaf():
            leaves.append(node)
        else:
            if node.left is not None:
                self.get_leaves(node.left, leaves)
            if node.right is not None:
                self.get_leaves(node.right, leaves)
        return leaves

    def get_nodes(self, node=None, nodes=None):
        """Get all nodes in the tree."""
        if nodes is None:
            nodes = []
        if node is None:
            node = self.root_node

        nodes.append(node)
        if node.left is not None:
            self.get_nodes(node.left, nodes)
        if node.right is not None:
            self.get_nodes(node.right, nodes)
        return nodes

    @classmethod
    def create_edge_list(cls, root):
        """Create edge list for tree visualization."""
        edges = []
        cls.traverse(root, edges)
        return edges

    @classmethod
    def traverse(cls, node, edges):
        if node is None:
            return
        if node.left is not None:
            edges.append((node, node.left))
            cls.traverse(node.left, edges)
        if node.right is not None:
            edges.append((node, node.right))
            cls.traverse(node.right, edges)

    def feature_importances(self, num_samples=100, k=None):
        """
        Compute feature importances based on split weights.

        Args:
            num_samples: Number of samples for posterior estimation
            k: If provided, return top-k features

        Returns:
            Feature importance scores
        """
        assert self.num_training_instances != 0, 'The model is not fitted!'

        non_leaves = []
        def _traverse(node):
            if node is not None:
                if node.left is not None or node.right is not None:
                    non_leaves.append(node)
                _traverse(node.left)
                _traverse(node.right)

        _traverse(self.root_node)

        weights = []
        vars = []
        for node in non_leaves:
            predictive = Predictive(model=node.split_model.to(self.device),
                                    guide=node.guide,
                                    num_samples=num_samples,
                                    return_sites=(["linear.weight"]))
            data = predictive(node.descriptive_values)['linear.weight']
            weights.append(data.mean(axis=0).numpy())
            vars.append(data.var(axis=0).numpy())

        weights = np.abs(np.array(weights))
        weights = weights.reshape(weights.shape[0], -1)
        vars = np.abs(np.array(vars))
        vars = vars.reshape(vars.shape[0], -1)

        importances = torch.zeros((weights.shape[1]))
        for i, node in enumerate(non_leaves):
            importances += (node.num_instances / self.num_training_instances) * \
                          (weights[i, :] / (vars[i] + 0.0001))

        if k is not None:
            return dict(zip(torch.topk(importances, k=k).indices.tolist(),
                           torch.topk(importances, k=k).values.tolist()))
        return importances

    def get_uncertainty_decomposition(self, x, num_mc_samples=10):
        """
        Decompose predictive uncertainty into routing and functional components.

        Implements the variance decomposition from Section 4.2 of the paper:
        Var[Y|x] = E[Var[Y|x,L]] + Var[E[Y|x,L]]

        Args:
            x: Single input
            num_mc_samples: Number of MC samples

        Returns:
            dict with 'total_variance', 'functional_variance', 'routing_variance'
        """
        means, variances = self._mc_predict_single(x, num_mc_samples)

        # Functional uncertainty: E[Var[Y|x,L]]
        functional_var = variances.mean()

        # Routing uncertainty: Var[E[Y|x,L]]
        routing_var = means.var()

        # Total variance
        total_var = functional_var + routing_var

        return {
            'total_variance': total_var.item(),
            'functional_variance': functional_var.item(),
            'routing_variance': routing_var.item()
        }