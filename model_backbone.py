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


class Backbone_FC(nn.Module):
    def __init__(self, num_layers, in_, mid_, out_, dropout = 0.2):
        super(Backbone_FC, self).__init__()
        self.num_layers = num_layers
        self.in_ = in_
        self.mid_ = mid_
        self.out_ = out_
        self.dropout = dropout
        self.encoder = layers_FC(num_layers, in_, mid_, out_, dropout = 0.2)
        
    def forward(self, x):
        out = self.encoder(x)
        return out