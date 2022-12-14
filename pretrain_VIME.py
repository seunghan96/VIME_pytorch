import time
import random
import os
import argparse

import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from train_utils import one_hot_encode, Tabular_DS, mask_generator, pretext_generator, EarlyStopping
#from train_utils import *
from model_SSL import VIME
from model_backbone import Backbone_FC

#python train.py --gcn_bool --adjtype doubletransition --addaptadj  --adddynadj --randomadj

parser = argparse.ArgumentParser()
#==========================================================#
parser.add_argument('--bs', default=64, type=int)
parser.add_argument('--data_file', type=str)
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--prob_mask', default=0.1, type=float)
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--alpha', default=0.3, type=float)
parser.add_argument('--hidden_dim', default=256, type=int)
parser.add_argument('--z_dim', default=256, type=int)
parser.add_argument('--enc_depth', default=3, type=int)
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--print_epochs', default=5, type=int)
parser.add_argument('--device',type=str,default='cuda:0')
parser.add_argument('--seed',type=int,default=19960729)

args = parser.parse_args()



def main():
    # (1) Set Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    # (2) Project Path
    PROJECT_PATH = '/content/drive/Shareddrives/contrastive_learning/SSL_TS'
    os.chdir(PROJECT_PATH)

    WEIGHT_PATH = os.path.join(PROJECT_PATH,'ckpt')
    os.makedirs(WEIGHT_PATH, exist_ok=True) 

    settings = f"_z_dim{args.z_dim}_hidden_dim{args.hidden_dim}_depth{args.enc_depth}_lr{args.lr}_prob_mask{args.prob_mask}_alpha{args.alpha}_do{args.dropout}_"
    WEIGHT_FILE_NAME = "BACKBONE_BEST" + settings + ".tar"

    WEIGHT_PATH = os.path.join(WEIGHT_PATH, WEIGHT_FILE_NAME)

    
    # (3) Read dataset
    categorical_cols = []
    
    with open(args.data_file,'rb') as f: 
      data = pickle.load(f)
    '''
    X = np.array([[3,'a',0.5,'f'],[5,'b',0.2,'d'],[5,'b',0.2,'d']])
    data = pd.DataFrame(X, columns=['A','B','C','D'])
    numeric_cols = ['C']
    categorical_cols = ['A','B','D']
    '''
    data = one_hot_encode(data, categorical_cols)

    print('Shape of data :', data.shape)
    print('='*100)
    ds = Tabular_DS(data, args.device)
    train_loader = DataLoader(ds, batch_size = args.bs, shuffle=True, drop_last = True)
    
    # (4) Model
    backbone = Backbone_FC(args.enc_depth, ds.p, args.hidden_dim, args.z_dim, args.dropout).to(args.device)
    model = VIME(backbone = backbone, input_dim = ds.p, z_dim = args.z_dim)
    model = model.to(args.device)
    
    early_stopping = EarlyStopping(args.patience, verbose=True, checkpoint_pth = WEIGHT_PATH)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    BCE_loss = nn.BCELoss()
    MSE_loss = nn.MSELoss()

    loss_best = np.infty
    epoch_best = 0
    

    current_patience = 0

    for epoch in range(1,args.num_epochs+1):
      loss_epoch = 0
      for X in train_loader:
          mask = mask_generator(args.prob_mask, X)    
          mask_new, X_new = pretext_generator(mask, X, args.device)

          X_pred, mask_pred = model(X_new)
          loss_X = MSE_loss(X_pred, X)
          loss_M = BCE_loss(mask_pred, mask_new)
          loss = loss_X*args.alpha + loss_M
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          loss_epoch += loss.item()

      loss_epoch /= len(train_loader)

      if loss_epoch < loss_best:
        loss_best = loss_epoch
        epoch_best = epoch

      if epoch%args.print_epochs ==0:
        print('-'*100)
        print(f'[Epoch = {epoch}] loss = {np.round(loss.item(), 3)} || [Best] epoch = {epoch_best}, loss = {np.round(loss_best, 3)}')  
        print(f'\t MSE loss = {np.round(loss_X.item(), 3)}')
        print(f'\t BCE loss = {np.round(loss_M.item(), 3)}')

      early_stopping(loss_epoch, backbone)

      if early_stopping.early_stop:
          print("Early stopping")
          break


if __name__ == "__main__":
  print('='*100)
  print(f"Settings : z_dim={args.z_dim}, depth={args.enc_depth}, lr={args.lr}, prob_mask={args.prob_mask}, alpha={args.alpha}")
  print('='*100)
  t1 = time.time()
  main()
  t2 = time.time()
  print("Total time spent: {:.1f}".format(t2-t1))

