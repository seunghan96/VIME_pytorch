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
    EMBEDDING_PATH = os.path.join(PROJECT_PATH,'embeddings')
    os.makedirs(WEIGHT_PATH, exist_ok=True) 
    os.makedirs(EMBEDDING_PATH, exist_ok=True) 

    settings = f"_z_dim{args.z_dim}_hidden_dim{args.hidden_dim}_depth{args.enc_depth}_lr{args.lr}_prob_mask{args.prob_mask}_alpha{args.alpha}_do{args.dropout}_"
    WEIGHT_FILE_NAME = "BACKBONE_BEST" + settings + ".tar"
    EMBEDDING_FILE_NAME = "EMBEDDING" + settings + ".npy"

    BEST_WEIGHT_PATH = os.path.join(WEIGHT_PATH, WEIGHT_FILE_NAME)
    BEST_EMBEDDING_PATH = os.path.join(EMBEDDING_PATH, EMBEDDING_FILE_NAME)
    print(BEST_WEIGHT_PATH)

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
    val_loader = DataLoader(ds, batch_size = args.bs, shuffle=False, drop_last = False)
    
    # (4) Model
    backbone = Backbone_FC(args.enc_depth, ds.p, args.hidden_dim, args.z_dim, args.dropout).to(args.device)
    best_weight = torch.load(BEST_WEIGHT_PATH)
    backbone.load_state_dict(best_weight)

    backbone.eval()
    Z_out = []
    for X in val_loader:
      Z = backbone(X)
      Z_out.append(Z.detach().cpu().numpy())
    Z_out = np.vstack(Z_out)

    print(111,BEST_EMBEDDING_PATH)
    with open(BEST_EMBEDDING_PATH, 'wb') as f:
      np.save(f, Z_out)

    

if __name__ == "__main__":
  print('='*100)
  main()
  print('Saved Embeddings!')


