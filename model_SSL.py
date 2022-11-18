import torch.nn as nn

def layers_FC(num_layers, in_, mid_, out_, dropout = 0.2):
    layers_ = []
    if num_layers==1:
        mid_=out_
    layers_.append(nn.Linear(in_, mid_))
    for i in range(num_layers-2):
        layers_.append(nn.ReLU(inplace = True))
        layers_.append(nn.Dropout(dropout))
        layers_.append(nn.Linear(mid_, mid_))
    if num_layers>1:    
        layers_.append(nn.ReLU(inplace = True))
        layers_.append(nn.Dropout(dropout))
        layers_.append(nn.Linear(mid_, out_))
    return nn.Sequential(*layers_)

class VIME(nn.Module):
  def __init__(self, input_dim, hidden_dim, z_dim, enc_depth):
    super(VIME, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.z_dim = z_dim
    self.encoder = layers_FC(num_layers = enc_depth, in_= self.input_dim, 
      mid_ = self.hidden_dim, out_ = self.z_dim)
    self.mask_model = nn.Linear(self.z_dim, self.input_dim)
    self.feature_model = nn.Linear(self.z_dim, self.input_dim)

  def forward(self, x):
    assert x.shape[1] == self.input_dim
    h = self.encoder(x)
    
    X_pred = self.feature_model(h)
    X_pred = nn.Sigmoid()(X_pred)

    M_pred = self.mask_model(h)
    M_pred = nn.Sigmoid()(M_pred)
    return X_pred, M_pred


