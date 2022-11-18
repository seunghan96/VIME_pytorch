import torch.nn as nn

class VIME(nn.Module):
  def __init__(self, dim):
    super(VIME, self).__init__()
    self.dim = dim
    self.encoder = nn.Linear(dim, dim)
    self.mask_model = nn.Linear(dim, dim)
    self.feature_model = nn.Linear(dim, dim)

  def forward(self, x):
    assert x.shape[1] == self.dim
    h = self.encoder(x)
    
    X_pred = self.feature_model(h)
    X_pred = nn.Sigmoid()(X_pred)

    M_pred = self.mask_model(h)
    M_pred = nn.Sigmoid()(M_pred)
    return X_pred, M_pred


