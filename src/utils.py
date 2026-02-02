import properscoring as ps
import numpy as np
import torch
import matplotlib.pyplot as plt

calculate_bs = lambda x: np.nanmean(ps.brier_score(x[1], x[0]), axis=0)

def fix_predictions(predictions, times, end_time_point):
  transformed_predictions = np.zeros((predictions.shape[0], end_time_point))
  transformed_predictions[:, :int(times[0])] = 1
  for row_idx in range(predictions.shape[0]):
    for interval_idx in range(predictions.shape[1]-1):
      start_time = int(times[interval_idx])
      end_time = int(times[interval_idx+1])
      interval_value = predictions[row_idx, interval_idx]
      transformed_predictions[row_idx, start_time:end_time] = interval_value
  transformed_predictions[:, end_time:end_time_point] = 0
  return transformed_predictions

def fix_brier_scores(brier_scores, times, end_time_point):
  transformed_brier_scores = np.zeros((end_time_point, ))
  for interval_idx in range(brier_scores.shape[0]-1):
    start_time = int(times[interval_idx])
    end_time = int(times[interval_idx+1])
    interval_value = brier_scores[interval_idx]
    transformed_brier_scores[start_time:end_time] = interval_value
  transformed_brier_scores[end_time:end_time_point] = interval_value
  return transformed_brier_scores


def plot_brier(preds1, preds2, y_test):
  bs1 = calculate_bs((preds1.mean(axis=1), y_test))
  bs2 = calculate_bs((preds2, y_test))
  plt.figure(figsize=(15,5))
  plt.plot(bs1, label = 'Variational SPYCT')
  plt.plot(bs2, label='SPYCT')
  # preds_bayes_10 = torch.quantile(preds1[:, :, :], 0.1, axis=1)
  # preds_bayes_90 = torch.quantile(preds1[:, :, :], 0.9, axis=1)
  # bs_bayes_10 = calculate_bs((preds_bayes_10.numpy(), y_test))
  # bs_bayes_90 = calculate_bs((preds_bayes_90.numpy(), y_test))
  # plt.fill_between(range(preds1.shape[-1]), bs_bayes_10, bs_bayes_90, color='blue', alpha=0.4)
  plt.legend()