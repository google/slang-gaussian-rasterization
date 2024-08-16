import torch

def sort_by_keys_torch(keys, values):
  """Sorts a values tensor by a corresponding keys tensor."""
  sorted_keys, idxs = torch.sort(keys)
  sorted_val = values[idxs]
  return sorted_keys, sorted_val
