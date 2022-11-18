import torch.nn as nn


class VIME(nn.Module):
  def __init__(self, backbone, input_dim , z_dim):
    super(VIME, self).__init__()
    self.backbone = backbone
    self.input_dim = input_dim
    self.z_dim = z_dim
    self.encoder = backbone
    self.mask_model = nn.Linear(self.z_dim, self.input_dim)
    self.feature_model = nn.Linear(self.z_dim, self.input_dim)

  def forward(self, x):
    assert x.shape[1] == self.input_dim
    h = self.backbone(x)
    
    X_pred = self.feature_model(h)
    X_pred = nn.Sigmoid()(X_pred)

    M_pred = self.mask_model(h)
    M_pred = nn.Sigmoid()(M_pred)
    return X_pred, M_pred


