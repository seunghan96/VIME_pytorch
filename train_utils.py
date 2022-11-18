import torch
import numpy as np
from torch.utils.data import Dataset

class Tabular_DS(Dataset): 
  def __init__(self, X, device):
    self.X = X
    self.p = X.shape[1]
    self.device = device

  def __len__(self): 
    return len(self.X)

  def __getitem__(self, idx): 
    X = torch.FloatTensor(self.X[idx]).to(self.device)
    return X

def mask_generator(mask_prob, x):
    
    """
    ==== Generate mask vector ====
    Args:
        - mask_prob: corruption probability
        - x: feature matrix
        
    Returns:
        - mask: binary mask matrix 
    """
    mask = np.random.binomial(1, mask_prob, x.shape)
    mask = torch.FloatTensor(mask)
    return mask


def pretext_generator(mask, x, device):  
    """
    ==== Generate corrupted samples ====
    Args:
        - mask: mask matrix
        - x: feature matrix
        
    Returns:
        - mask_new: final mask matrix after corruption
        - x_tilde: corrupted feature matrix
    """

    N, p = x.shape  
    x_bar = torch.zeros([N, p])

    # step 1) shuffle dataset
    ## -- original data : x
    ## -- shuffled data : x_bar
    for i in range(p):
        idx = np.random.permutation(N)
        x_bar[:, i] = x[idx, i]
        
    # step 2) Corrupt samples
    ## -- new data : Mix(x, x_bar)
    mask = mask.to(device)
    x_bar = x_bar.to(device)
    x_tilde = x * (1-mask) + x_bar * mask  
    
    # step 3) (new) mask matrix
    mask_new = 1 * (x != x_tilde)
    mask_new = mask_new.type(torch.FloatTensor).to(device)

    return mask_new, x_tilde