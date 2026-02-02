import sys
sys.path.append('.')
from src.utils import fix_predictions
from src.data.dataset import SurvivalDataset
import numpy as np
import torch

class IPCWBrier:
  def __init__(self, times, event_observed):
    from lifelines import KaplanMeierFitter
    from scipy.interpolate import interp1d
    self.times = times
    self.event_observed = event_observed
    self.num_events = sum(event_observed)
    self.kmf = KaplanMeierFitter()
    self.kmf.fit(self.times, self.event_observed)
    
    # Calculate cumulative censoring probabilities
    cumulative_censoring_probs = 1 - self.kmf.survival_function_.values.flatten()
    
    # Get the time points corresponding to the survival function estimates
    kmf_times = self.kmf.survival_function_.index.values
    
    # Interpolate to match the original number of time points
    # Create an interpolation function based on the kmf times and the cumulative censoring probabilities
    interp_func = interp1d(kmf_times, cumulative_censoring_probs, fill_value="extrapolate")
    
    # Use the interpolation function to estimate ipcw_coeffs for each original time point
    self.ipcw_coeffs = interp_func(self.times)
    
    # Add a small epsilon value to avoid division by zero
    epsilon = 1e-10
    self.ipcw_coeffs = np.clip(self.ipcw_coeffs, epsilon, 1)
    
    # Inverse to get IPCW coefficients
    self.ipcw_coeffs = 1 / self.ipcw_coeffs

  def evaluate(self, times_arr, predicted_probs):
    if isinstance(times_arr, torch.Tensor): times_arr = times_arr.numpy()
    if isinstance(predicted_probs, torch.Tensor): predicted_probs = predicted_probs.numpy()
    diff = torch.tensor(np.power(times_arr - predicted_probs, 2))
    ipcw_coeffs = torch.tensor(self.ipcw_coeffs, dtype=torch.float32)  # Ensure it's a PyTorch tensor
    ipcw_coeffs = ipcw_coeffs.unsqueeze(-1)
    brier_score = torch.nanmean(ipcw_coeffs * diff, axis=0) / self.num_events
    return brier_score


if __name__ == '__main__':
  data = SurvivalDataset(fname='pbc.rda', path='data/raw/')
  _, _, y_train, y_test = data.get_tensors()
  X_train, T_train, E_train, X_test, T_test, E_test = data.pysurvival_split()
  from pysurvival.models.multi_task import LinearMultiTaskModel
  mtlr = LinearMultiTaskModel()
  mtlr.fit(X_train, T_train, E_train, lr=0.0001, l2_reg=1e-2, init_method='zeros')
  predicted_mtlr = mtlr.predict_survival(X_test)
  transformed_predictions_mtlr = fix_predictions(predicted_mtlr, mtlr.times, T_train.max())
  ipcw_brier = IPCWBrier(T_test, E_test)
  bs_mtlr = ipcw_brier.evaluate(y_test, transformed_predictions_mtlr)
  import matplotlib.pyplot as plt
  plt.plot(bs_mtlr)
  plt.show()