import torch
import pyro
import numpy as np
from pyro.infer import Predictive
import random
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, LinearKernel, PolynomialKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood


def load_and_prediction(model, guide, x_test, num_samples = 1, device = 'cpu'):
    pyro.clear_param_store()
    predictive = Predictive(model = model.to(device),
                            guide=guide,
                            num_samples=num_samples,
                            return_sites=("linear.weight", "linear.bias"))
    data = predictive(x_test.clone().detach())#, None,x.shape[0])
    return data


def batch_load_and_prediction(model, guide, x_test, num_samples=1, device='cpu', clear_store=True):
    """
    Batch-optimized version of load_and_prediction.

    Args:
        model: Pyro model
        guide: Pyro guide
        x_test: Input tensor (can be 1D for single sample or 2D for batch)
        num_samples: Number of MC samples
        device: Device to use
        clear_store: Whether to clear param store (set False when called in loop)

    Returns:
        Dictionary with 'linear.weight' and 'linear.bias' samples
    """
    if clear_store:
        pyro.clear_param_store()
    predictive = Predictive(model=model.to(device),
                            guide=guide,
                            num_samples=num_samples,
                            return_sites=("linear.weight", "linear.bias"))
    data = predictive(x_test.clone().detach())
    return data


class LeafGP(ExactGP):
    """Gaussian Process model for leaf-level predictions."""
    def __init__(self, train_x, train_y, likelihood, kernel_type='rbf'):
        super(LeafGP, self).__init__(train_x, train_y, likelihood)

        input_dim = train_x.shape[1] if train_x.dim() > 1 else 1

        if kernel_type == 'rbf':
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(RBFKernel())
        elif kernel_type == 'matern':
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(MaternKernel(nu=2.5))
        elif kernel_type == 'linear':
            # Linear kernel for extrapolation
            self.mean_module = LinearMean(input_dim)
            self.covar_module = ScaleKernel(LinearKernel())
        elif kernel_type == 'linear_rbf':
            # Linear + RBF: captures trends AND local variations
            self.mean_module = LinearMean(input_dim)
            self.covar_module = ScaleKernel(LinearKernel()) + ScaleKernel(RBFKernel())
        elif kernel_type == 'polynomial':
            # Polynomial kernel for nonlinear extrapolation
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(PolynomialKernel(power=2))
        else:
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class Node:
    def __init__(self, depth=0, enable_mc_dropout=False, num_instances=0):
        self.left = None
        self.right = None
        self.prototype = None
        self.split_model = None
        self.order = None
        self.depth = depth
        self.enable_mc_dropout = enable_mc_dropout
        self.num_instances = num_instances

    def predict(self, x):
        if self.is_leaf():
            return self.prototype
        else:
            splits = self.split_model(x, self.enable_mc_dropout)
            print(splits.shape)
            # print(f'splits: {splits}')
            if splits.shape[0]>1:
                return torch.stack([self.left.predict(x) if split<=0 else self.right.predict(x) for split in splits])
            else:
                if splits <= 0: return self.left.predict(x)
                else: return self.right.predict(x)

    def is_leaf(self): return self.left is None and self.right is None


class VNode:
    def __init__(self, descriptive_values, depth=0, num_instances=0, root=False):
        self.left = None
        self.right = None
        self.prototype = None   
        self.split_model = None
        self.guide = None
        self.order = None
        self.depth = depth
        self.is_root = root
        self.param_store = None
        self.losses = None
        self.num_instances = num_instances
        self.descriptive_values = descriptive_values

    def is_leaf(self): return self.left is None and self.right is None

    def predict(self, x):
        if self.is_leaf():
            return self.prototype
        else:
            data = load_and_prediction(self.split_model, self.guide, x)
            splits = (data['linear.weight'] @ x + data['linear.bias']).sigmoid()
            splits = splits.reshape(-1)
            if splits.shape[0]>1:
                return torch.stack([self.left.predict(x) if split<=0.5 else self.right.predict(x) for split in splits])
            else:
                if splits <= 0.5: return self.left.predict(x)
                else: return self.right.predict(x)

    def mc_predict(self, x, num_samples=10):
        pred_lst = []
        for i in range(num_samples):
            pred_lst.append(self.predict(x))
        return torch.stack(pred_lst)


class VNodeGP:
    """
    Variational Node with Gaussian Process leaf models for VSPYCT-GP.

    This node extends VNode by replacing constant prototypes with GP predictors
    that enable uncertainty-aware extrapolation.
    """
    def __init__(self, descriptive_values, depth=0, num_instances=0, root=False, device='cpu'):
        self.left = None
        self.right = None
        self.prototype = None  # Constant prediction (used for in-support)
        self.split_model = None
        self.guide = None
        self.order = None
        self.depth = depth
        self.is_root = root
        self.param_store = None
        self.losses = None
        self.num_instances = num_instances
        self.descriptive_values = descriptive_values
        self.device = device

        # GP-specific attributes
        self.gp_model = None
        self.gp_likelihood = None
        self.leaf_X = None  # Training inputs for this leaf
        self.leaf_y = None  # Training targets for this leaf
        self.cov_matrix_inv = None  # Inverse covariance matrix for Mahalanobis distance
        self.leaf_mean = None  # Mean of leaf training data

    def is_leaf(self):
        return self.left is None and self.right is None

    def compute_mahalanobis_distance(self, x):
        """
        Compute the Mahalanobis distance from x to the centroid of training data in this leaf.

        Args:
            x: Test input (1D tensor)

        Returns:
            Mahalanobis distance to leaf centroid
        """
        if self.leaf_X is None or len(self.leaf_X) == 0:
            return float('inf')

        x = x.reshape(1, -1)

        # Compute distance from x to the leaf centroid (not to nearest point)
        diff = x - self.leaf_mean.reshape(1, -1)

        if self.cov_matrix_inv is not None:
            # Mahalanobis distance: sqrt((x-mu)^T * Sigma^-1 * (x-mu))
            mahal_sq = torch.sum(diff @ self.cov_matrix_inv * diff, dim=1)
            distance = torch.sqrt(torch.clamp(mahal_sq, min=0))
        else:
            # Fallback to scaled Euclidean distance
            # Use feature-wise std for scaling
            std = self.leaf_X.std(dim=0) + 1e-6
            scaled_diff = diff / std
            distance = torch.norm(scaled_diff, dim=1)

        return distance.item()

    def is_in_support(self, x, tau=2.0):
        """
        Check if x is within the training support of this leaf.

        Args:
            x: Test input
            tau: Threshold for support detection (default 2.0 standard deviations)

        Returns:
            True if x is in-support, False otherwise
        """
        min_dist = self.compute_mahalanobis_distance(x)
        return min_dist <= tau

    def fit_gp(self, X, y, kernel_type='rbf', training_iterations=50):
        """
        Fit a Gaussian Process model on the leaf data.

        Args:
            X: Training inputs for this leaf
            y: Training targets for this leaf
            kernel_type: Type of GP kernel ('rbf' or 'matern')
            training_iterations: Number of iterations for GP optimization
        """
        self.leaf_X = X.clone().detach()
        self.leaf_y = y.clone().detach()

        # Compute covariance matrix for Mahalanobis distance
        if len(X) > 1:
            try:
                cov_matrix = torch.cov(X.T)
                if cov_matrix.dim() == 0:
                    cov_matrix = cov_matrix.reshape(1, 1)
                # Add small regularization for numerical stability
                cov_matrix = cov_matrix + 1e-6 * torch.eye(cov_matrix.shape[0], device=X.device)
                self.cov_matrix_inv = torch.inverse(cov_matrix)
            except:
                self.cov_matrix_inv = None
        else:
            self.cov_matrix_inv = None

        self.leaf_mean = X.mean(dim=0)

        # For single-target regression, ensure y is 1D
        if y.dim() > 1 and y.shape[1] == 1:
            y_train = y.squeeze(-1)
        else:
            y_train = y

        # Initialize likelihood and model
        self.gp_likelihood = GaussianLikelihood()
        self.gp_model = LeafGP(X, y_train, self.gp_likelihood, kernel_type=kernel_type)

        # Set to training mode
        self.gp_model.train()
        self.gp_likelihood.train()

        # Optimize GP hyperparameters
        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model)

        for _ in range(training_iterations):
            optimizer.zero_grad()
            output = self.gp_model(X)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

        # Set to evaluation mode
        self.gp_model.eval()
        self.gp_likelihood.eval()

    def gp_predict(self, x):
        """
        Get GP posterior prediction for x.

        Args:
            x: Test input

        Returns:
            (mean, variance) tuple from GP posterior
        """
        if self.gp_model is None:
            return self.prototype, torch.tensor(0.0)

        x = x.reshape(1, -1)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.gp_likelihood(self.gp_model(x))
            mean = pred.mean
            variance = pred.variance

        return mean.squeeze(), variance.squeeze()

    def predict_with_gating(self, x, tau=2.0, epsilon_sq=1e-4, temperature=0.5):
        """
        Predict using soft extrapolation-aware gating mechanism.

        Uses sigmoid weighting for smooth transition between prototype and GP:
        - Points well within support (d << tau): ~100% prototype
        - Points well outside support (d >> tau): ~100% GP
        - Boundary region: smooth interpolation

        Args:
            x: Test input
            tau: Support threshold (Mahalanobis distance)
            epsilon_sq: Noise floor for in-support predictions
            temperature: Controls sharpness of transition (lower = sharper)

        Returns:
            (mean, variance) tuple - both as scalar tensors
        """
        if self.is_leaf():
            # Get prototype as scalar
            proto = self.prototype
            if isinstance(proto, torch.Tensor):
                if proto.numel() > 1:
                    proto = proto.mean()
                else:
                    proto = proto.flatten()[0] if proto.numel() == 1 else proto
            else:
                proto = torch.tensor(proto, dtype=torch.float32)

            # Compute Mahalanobis distance
            d = self.compute_mahalanobis_distance(x)

            # Soft gating: sigmoid weight based on distance relative to tau
            # weight = 0 when d << tau (use prototype)
            # weight = 1 when d >> tau (use GP)
            weight = torch.sigmoid(torch.tensor((d - tau) / temperature))

            # Get GP prediction
            gp_mean, gp_var = self.gp_predict(x)
            if isinstance(gp_mean, torch.Tensor) and gp_mean.numel() > 1:
                gp_mean = gp_mean.mean()
            if isinstance(gp_var, torch.Tensor) and gp_var.numel() > 1:
                gp_var = gp_var.mean()

            # Soft interpolation between prototype and GP
            mean = (1 - weight) * proto + weight * gp_mean
            var = (1 - weight) * epsilon_sq + weight * gp_var

            return mean, var
        else:
            raise ValueError("predict_with_gating should only be called on leaf nodes")

    def predict(self, x):
        """Standard prediction (route through tree)."""
        if self.is_leaf():
            return self.prototype
        else:
            data = load_and_prediction(self.split_model, self.guide, x)
            splits = (data['linear.weight'] @ x + data['linear.bias']).sigmoid()
            splits = splits.reshape(-1)
            if splits.shape[0] > 1:
                return torch.stack([self.left.predict(x) if split <= 0.5 else self.right.predict(x) for split in splits])
            else:
                if splits <= 0.5:
                    return self.left.predict(x)
                else:
                    return self.right.predict(x)

    def mc_predict(self, x, num_samples=10):
        """Monte Carlo prediction (standard VSPYCT style)."""
        pred_lst = []
        for _ in range(num_samples):
            pred_lst.append(self.predict(x))
        return torch.stack(pred_lst)

    def mc_predict_with_gating(self, x, num_samples=10, tau=2.0, epsilon_sq=1e-4):
        """
        Monte Carlo prediction with extrapolation-aware gating.

        This implements Algorithm 1 from the paper: samples split parameters,
        routes to leaf, then applies gating mechanism.

        Args:
            x: Test input
            num_samples: Number of MC samples
            tau: Support threshold
            epsilon_sq: Noise floor

        Returns:
            Tensor of shape (num_samples, 2) containing [mean, variance] for each sample
        """
        means = []
        variances = []

        for _ in range(num_samples):
            # Route to leaf with sampled split parameters
            leaf = self._route_to_leaf(x)

            # Get prediction using gating
            mean, var = leaf.predict_with_gating(x, tau, epsilon_sq)
            means.append(mean)
            variances.append(var)

        return torch.stack(means), torch.stack(variances)

    def _route_to_leaf(self, x):
        """Route x to a leaf node by sampling split parameters."""
        if self.is_leaf():
            return self

        data = load_and_prediction(self.split_model, self.guide, x, num_samples=1)
        split = (data['linear.weight'] @ x + data['linear.bias']).sigmoid().item()

        if split <= 0.5:
            return self.left._route_to_leaf(x)
        else:
            return self.right._route_to_leaf(x)

    def gp_predict_batch(self, X):
        """
        Batch GP posterior prediction for multiple inputs.

        Args:
            X: Test inputs (n_samples, n_features)

        Returns:
            (means, variances) tuple from GP posterior, both shape (n_samples,)
        """
        if self.gp_model is None:
            n = X.shape[0] if X.dim() > 1 else 1
            proto = self.prototype
            if isinstance(proto, torch.Tensor):
                if proto.numel() > 1:
                    proto = proto.mean()
                else:
                    proto = proto.flatten()[0] if proto.numel() == 1 else proto
            else:
                proto = torch.tensor(proto, dtype=torch.float32)
            return proto.expand(n), torch.zeros(n)

        if X.dim() == 1:
            X = X.reshape(1, -1)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.gp_likelihood(self.gp_model(X))
            means = pred.mean
            variances = pred.variance

        return means, variances

    def compute_mahalanobis_distance_batch(self, X):
        """
        Compute Mahalanobis distances for batch of inputs.

        Args:
            X: Test inputs (n_samples, n_features)

        Returns:
            Tensor of distances, shape (n_samples,)
        """
        if self.leaf_X is None or len(self.leaf_X) == 0:
            return torch.full((X.shape[0],), float('inf'))

        if X.dim() == 1:
            X = X.reshape(1, -1)

        # Compute distance from each x to the leaf centroid
        diff = X - self.leaf_mean.reshape(1, -1)

        if self.cov_matrix_inv is not None:
            # Mahalanobis distance: sqrt((x-mu)^T * Sigma^-1 * (x-mu))
            mahal_sq = torch.sum(diff @ self.cov_matrix_inv * diff, dim=1)
            distances = torch.sqrt(torch.clamp(mahal_sq, min=0))
        else:
            # Fallback to scaled Euclidean distance
            std = self.leaf_X.std(dim=0) + 1e-6
            scaled_diff = diff / std
            distances = torch.norm(scaled_diff, dim=1)

        return distances

    def predict_with_gating_batch(self, X, tau=2.0, epsilon_sq=1e-4, temperature=0.5):
        """
        Batch prediction using soft extrapolation-aware gating.

        Args:
            X: Test inputs (n_samples, n_features)
            tau: Support threshold
            epsilon_sq: Noise floor for in-support predictions
            temperature: Controls sharpness of transition

        Returns:
            (means, variances) tuple, both shape (n_samples,)
        """
        if not self.is_leaf():
            raise ValueError("predict_with_gating_batch should only be called on leaf nodes")

        if X.dim() == 1:
            X = X.reshape(1, -1)

        # Get prototype as scalar
        proto = self.prototype
        if isinstance(proto, torch.Tensor):
            if proto.numel() > 1:
                proto = proto.mean()
            else:
                proto = proto.flatten()[0] if proto.numel() == 1 else proto
        else:
            proto = torch.tensor(proto, dtype=torch.float32)

        # Compute Mahalanobis distances for all samples
        distances = self.compute_mahalanobis_distance_batch(X)

        # Soft gating weights for all samples
        weights = torch.sigmoid((distances - tau) / temperature)

        # Get batch GP predictions
        gp_means, gp_vars = self.gp_predict_batch(X)

        # Soft interpolation between prototype and GP
        means = (1 - weights) * proto + weights * gp_means
        vars = (1 - weights) * epsilon_sq + weights * gp_vars

        return means, vars