# @Author: Yaming Duan, BNU
# @Purpose: 
#   1.Calculate the confusion matrix through two images.

import torch
import numpy as np

def NumpyGetConfusionMatrix(prediction, reference, upper_bound=None):
    if upper_bound == None:
        upper_bound = max([np.max(reference), np.max(prediction)])

    n_cls = upper_bound + 1
    available_idx = (reference >= 0) & (reference <= upper_bound) #  > 0 means without background
    
    return np.bincount(
        n_cls * reference[available_idx].astype(int) +prediction[available_idx], 
        minlength=n_cls**2
    ).reshape(n_cls, n_cls)

def TorchGetConfusionMatrix(prediction, reference, upper_bound=None):
    if upper_bound is None:
        upper_bound =int(max([max(prediction), max(reference)]))

    n_cls = upper_bound + 1
    available_idx = (reference >= 0) & (reference <= upper_bound)
    
    return torch.bincount(
        n_cls * reference[available_idx].int() + prediction[available_idx], 
        minlength= n_cls**2
    ).cpu().numpy().reshape(n_cls, n_cls)
