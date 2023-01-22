
import time
import os
import argparse
import yaml

import pickle
import itertools
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cluster import KMeans
from preprocess_utils import *


#python train.py --gcn_bool --adjtype doubletransition --addaptadj  --adddynadj --randomadj

parser = argparse.ArgumentParser()
#==========================================================#

parser.add_argument('--config_filename', default = 'xxxx', type=str)
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--device',type=str,default='cuda:0')

args = parser.parse_args()



def main():
    with open(args.config_filename) as f:
        configs = yaml.load(f, Loader = yaml.Loader)
    
    # Configurations
    DATA_PATH = configs['data_path'] #'/root/jupyter/data/pkl'
    TABULAR_FNAME = configs['tabular_fname'] # 'MST_OPER_PART_220701.pkl'
    parts_id = configs['parts_id'] # FCST_PART_NO
    
    IDX_DIR = configs['target_prod_idx_dir'] # /root/ ... /FIT_DATA_TR_2206.pkl
    
    ECO_DATA_PATH = configs['eco_data_path'] # '/root/jupyter/data/target_data_0809/evaluation/Yearly'
    ECO_LONG_FNAME = configs['eco_long_fname'] # 'YDMD_MST_ECO_PART_202206.pkl'
    
    CLUSTER_PATH = configs['nlp_cluster_path']
    CLUSTER_FNAME = configs['nlp_cluster_fname']
    CLUSTER_SAVE_PATH = os.path.join(CLUSTER_PATH, CLUSTER_FNAME)
    
    NUM_WORDS_RATIO = configs['num_words_ratio'] # 0.7
    PCA_EVR_RATIO = configs['pca_evr_ratio'] # 0.7
    NUM_CLUSTERS = configs['num_clusters']
    
    TABULAR_DIR = os.path.join(DATA_PATH, TABULAR_FNAME)
    ECO_LONG_DIR = os.path.join(ECO_DATA_PATH, ECO_LONG_FNAME)
    
    
    # Tabular (All)
    tabular_all = pd.read_pickle(TABULAR_DIR)
    tabular_all.set_index(parts_id, inplace = True)
    
    # Index of TARGET & ECO
    target_idx = pd.read_pickle(IDX_DIR).index.unique()
    eco_idx = pd.read_pickle(ECO_LONG_DIR).index.unique()
    
    tabular_target = tabular_all[tabular_all.index.isin(target_idx)]
    tabular_eco = tabular_all[tabular_all.index.isin(eco_idx)]
    tabular = pd.concat([tabular_target, tabular_eco], axis=0)
    n_data = len(tabular)
    print(f'Shape of ALL tabular data : {tabular_all.shape}')
    print(f'Shape of TARGET tabular data : {tabular_all.shape}')
    print(f'Shape of ECO tabular data : {tabular_eco.shape}')
    print(f'Shape of TARGET + ECO tabular data : {tabular_raw.shape}')
    
    oper_part = tabular.OPER_PART_NM.values
    oper_part = [str(x) for x in oper_part]
    oper_part_unq = list(set(oper_part))
    print(f'Number of unique OPER_PART_NM : {len(oper_part_unq)}')
    
    oper_part_token = [x.split(' ') for x in oper_part]
    oper_part_token_flat = list(itertools.chain.from_iterable(oper_part_token))
    text_cnt = Counter(oper_part_token_flat)
    text_cnt_freq = text_cnt.most_common()
    text_cnt_freq_text = np.array(i[0] for i in text_cnt_freq)
    text_cnt_freq_val = np.array(i[1] for i in text_cnt_freq)
    text_cnt_freq_val_cumsum = text_cnt_freq_val.cumsum()
    text_cnt_freq_val_cumsum_ratio = text_cnt_freq_val_cumsum / text_cnt_freq_val.sum() 
    num_cols_K = sum(text_cnt_freq_val_cumsum_ratio < NUM_WORDS_RATIO)
    oper_part_word = get_oper_part_word(oper_part_token, text_cnt_freq_text, top_K = num_cols_K)
    
    
    word_df = get_dummy_df(oper_part_word, top_K = num_cols_K)
    
    pca = PCA()
    pca.fit(word_df.values)
    used_n_pc = sum(pca.explained_variance_ratio_.cumsum() < PCA_EVR_RATIO)
    word_pca_df = pca.fit_transform(word_df.values)[:, :used_n_pc]
    word_pca_df = pd.DataFrame(word_pca_df, 
                               columns = [f'PC{i}' for i in range(1, used_n_pc + 1)])
    
    kmeans = KMeans(n_clusters = NUM_CLUSTERS)
    kmeans.fit(word_pca_df.values)
    label_assignment = kmeans.labels_
    print('Cluster Assignment Results')
    print(pd.Series(label_assignment).value_counts())
    cluster_df = pd.DataFrame({'FCST_PART_NO' : tabular[parts_id],
                               'cluster' : label_assignment})
    cluster_df.to_csv(CLUSTER_SAVE_PATH,index = False)
    