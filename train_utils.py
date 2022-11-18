import torch
import numpy as np
import pandas as pd
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


def one_hot_encode(df, cat_cols):
  assert isinstance(df, pd.DataFrame)
  for col in cat_cols:
    df = pd.concat([df,pd.get_dummies(df[col])], axis=1)
    df.drop(col, axis=1, inplace=True)
  
  return df.astype('float64').values


class EarlyStopping:
    def __init__(self, patience=50, verbose=False, delta=0, checkpoint_pth='chechpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_pth = checkpoint_pth

    def __call__(self, val_loss, model):
        loss = val_loss
        if self.best_loss is None:
            self.best_loss = loss
        elif loss >= self.best_loss + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.save_checkpoint(val_loss, model, self.checkpoint_pth)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, checkpoint_pth):
        if model is not None:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f}).  Saving model ...')
            torch.save(model.state_dict(), checkpoint_pth)
            self.val_loss_min = val_loss
            