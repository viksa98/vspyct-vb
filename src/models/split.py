import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os

from tqdm import trange
import pyro
import pyro.distributions as dist
import pyro.optim
import pyro.infer
import pyro.contrib.autoguide as autoguide
from pyro.contrib.autoguide import AutoDiagonalNormal
import pyro.poutine as poutine
from torch.distributions import constraints
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from pyro.infer.autoguide.guides import AutoDiagonalNormal
from pyro.infer.autoguide.initialization import init_to_mean
from pyro import infer, optim
from pyro.nn.module import to_pyro_module_
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
from pyro.infer import Predictive
import pyro.distributions as dist

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

# def weighted_variance(values, weights, weight_sum):
#     mean = torch.matmul(weights, values) / weight_sum
#     return -torch.sum(mean*mean)

# def weighted_variance(values, weights, weight_sum):
#     values = torch.nan_to_num(values, nan=0.0)
#     mean = torch.matmul(weights, values) / weight_sum
#     return -torch.sum(mean*mean)

# def weighted_variance(values, weights, weight_sum):
#     values = torch.nan_to_num(values, nan=0.0)
#     weights = weights.view(weights.shape[0], 1)
#     weighted_squares = weights*values
#     return torch.sum((weighted_squares-weighted_squares.mean(axis=0))**2)/ weight_sum

def weighted_variance(values, weights, weight_sum):
    """
    Compute weighted variance, handling NaN values for semi-supervised learning.

    For multi-target data with NaN values, computes variance only over non-NaN entries.
    Behavior is unchanged when there are no NaN values.
    """
    # Check if there are any NaN values
    has_nan = torch.isnan(values).any()

    if not has_nan:
        # Original behavior - no NaN values
        mean = torch.matmul(weights, values) / weight_sum
        squared_diff = (values - mean) ** 2
        weighted_var = torch.sum(torch.matmul(weights, squared_diff)) / weight_sum
        return weighted_var
    else:
        # NaN-aware computation for semi-supervised learning
        nan_mask = torch.isnan(values)
        values_filled = torch.where(nan_mask, torch.zeros_like(values), values)

        if values.dim() > 1:
            # Multi-target case: compute per-column weighted variance
            weights_col = weights.unsqueeze(1).expand_as(values)
            # Zero out weights for NaN entries
            weights_adjusted = torch.where(nan_mask, torch.zeros_like(weights_col), weights_col)
            weight_sums_col = weights_adjusted.sum(dim=0) + 1e-8

            # Weighted mean per column
            weighted_sum = (weights_adjusted * values_filled).sum(dim=0)
            mean = weighted_sum / weight_sums_col

            # Weighted variance per column
            squared_diff = torch.where(nan_mask, torch.zeros_like(values), (values_filled - mean) ** 2)
            var_per_col = (weights_adjusted * squared_diff).sum(dim=0) / weight_sums_col

            return var_per_col.sum()
        else:
            # Single-target case with NaN
            valid_mask = ~nan_mask
            valid_weights = weights * valid_mask.float()
            valid_weight_sum = valid_weights.sum() + 1e-8
            mean = (valid_weights * values_filled).sum() / valid_weight_sum
            squared_diff = valid_mask.float() * (values_filled - mean) ** 2
            weighted_var = (valid_weights * squared_diff).sum() / valid_weight_sum
            return weighted_var


class MC_Dropout_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.2):
        super(MC_Dropout_Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x, enable_mc_dropout=False):
        if not enable_mc_dropout: return self.linear(x)
        else:
            num_samples = 100  # choose the number of samples
            outputs = torch.zeros(num_samples, x.shape[0] if x.dim()>1 else 1, self.output_dim, device=x.device)
            for i in range(num_samples):
                mask = F.dropout(torch.ones_like(x), p=self.dropout_prob, training=enable_mc_dropout)
                outputs[i] = self.linear(x * mask)

            return outputs

class Impurity(PyroModule):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super(Impurity, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = PyroModule[nn.Linear](self.input_dim, self.output_dim)

        self.linear.weight = PyroSample(dist.Normal(0, 1).expand([self.output_dim, self.input_dim]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0, 1).expand([self.output_dim]).to_event(1))

    def forward(self, descriptive_data, clustering_data):
        # sigma = pyro.sample("sigma", dist.Normal(0., 1.))
        right_selection = self.linear(descriptive_data).reshape(-1).sigmoid()   
        left_selection = torch.tensor(1., device=self.device) - right_selection

        right_weight_sum = torch.sum(right_selection)
        left_weight_sum = torch.sum(left_selection)

        var_left = weighted_variance(clustering_data, left_selection, left_weight_sum)
        var_right = weighted_variance(clustering_data, right_selection, right_weight_sum)
        
        impurity = (left_weight_sum * var_left + right_weight_sum * var_right)#+ torch.norm(self.linear.weight, p=0.5)

        
        try:
            with pyro.plate("data", descriptive_data.shape[0]):
                obs = pyro.sample("obs", dist.Normal(impurity, 1.0), obs=impurity/3)#torch.tensor(0.))
        except: print('NAN!')   
        
        
        return impurity
    

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, relative_delta=0.001, min_epochs=50):
        self.patience = patience
        self.min_delta = min_delta
        self.relative_delta = relative_delta  # Use relative improvement threshold
        self.min_epochs = min_epochs  # Don't stop before this many epochs
        self.counter = 0
        self.best_loss = float('inf')
        self.initial_loss = None
        self.current_epoch = 0
        self.early_stop = False

    def is_converged(self, loss):
        self.current_epoch += 1

        # Set initial loss on first call
        if self.initial_loss is None:
            self.initial_loss = loss

        # Don't allow early stopping before min_epochs
        if self.current_epoch < self.min_epochs:
            if loss < self.best_loss:
                self.best_loss = loss
            return False

        # Use relative delta: improvement must be at least relative_delta * initial_loss
        effective_delta = max(self.min_delta, self.relative_delta * abs(self.initial_loss))

        if loss < self.best_loss - effective_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
        
def learn_split(rows, descriptive_data, clustering_data, device, epochs, bs, lr, subspace_size):
    selected_attributes = np.random.choice(a=[False, True],
                                           size=descriptive_data.size(1),
                                           p=[1-subspace_size, subspace_size])
    descriptive_subset = descriptive_data[:, selected_attributes]
    # model = torch.nn.Linear(in_features=descriptive_subset.size(1), out_features=1).to(device)
    model = MC_Dropout_Linear(input_dim=descriptive_subset.size(1), output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if bs is None: bs = rows.size(0)
    num_batches = math.ceil(rows.size(0) / bs)

    for _ in range(epochs):
        for b in range(num_batches):
            descr = descriptive_subset[b*bs:(b+1)*bs]
            clustr = clustering_data[b*bs:(b+1)*bs]
            right_selection = model(descr).reshape(-1).sigmoid()
            left_selection = torch.tensor(1., device=device) - right_selection
            right_weight_sum = torch.sum(right_selection)
            left_weight_sum = torch.sum(left_selection)
            var_left = weighted_variance(clustr, left_selection, left_weight_sum)
            var_right = weighted_variance(clustr, right_selection, right_weight_sum)
            impurity = (left_weight_sum * var_left + right_weight_sum * var_right) + torch.norm(model.linear.weight, p=0.5)
            impurity.backward()
            optimizer.step()
            model.zero_grad()

    return model.to(device).eval()

def learn_split_vb(rows, descriptive_data, clustering_data, device, epochs, bs, lr, subspace_size, patience=20, enable_validation=False):
    pyro.enable_validation(enable_validation)
    selected_attributes = np.random.choice(a=[False, True],
                                            size=descriptive_data.size(1),
                                            p=[1-subspace_size, subspace_size])
    descriptive_subset = descriptive_data[:, selected_attributes]

    pyro.clear_param_store()
    model = Impurity(input_dim=descriptive_subset.size(1), output_dim=1).to(device)
    guide = AutoDiagonalNormal(model)
    # adam_params = {"lr": lr }#, "betas": (0.90, 0.999)}
    optimizer = torch.optim.Adam
    scheduler = pyro.optim.LinearLR({'optimizer': optimizer, 'optim_args': {'lr': lr}})

    # Use relative delta (0.1% improvement) with increased patience for multi-target problems
    # Require at least 100 epochs before early stopping can kick in
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0, relative_delta=0.001, min_epochs=100)

    # Setup SVI
    svi = infer.SVI(model,
                    guide,
                    scheduler,
                    loss=infer.TraceMeanField_ELBO(num_particles=5))
    losses = []
    # _steps = 0
    if bs is None: bs = rows.size(0)
    num_batches = math.ceil(rows.size(0) / bs)
    for epoch in trange(epochs, desc="Epochs"):
        train_loss = 0.0
        for b in range(num_batches):
            descr = descriptive_subset[b*bs:(b+1)*bs]
            clustr = clustering_data[b*bs:(b+1)*bs]
            train_loss += svi.step(descr, clustr)
            # if _steps<1:
            #     param_store = pyro.get_param_store()['AutoDiagonalNormal.loc']
            # _steps+=1
        if early_stopping.is_converged(train_loss):
              print(f"Early stopping at epoch {epoch}.")
              break
        losses.append(train_loss)
        scheduler.step()
        # print("[iteration %04d] loss: %.4f" % (epoch+1, train_loss))
    print(model)
    return (model, guide, losses)