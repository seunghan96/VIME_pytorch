
import time
import os
import argparse
import yaml

import pickle
import numpy as np
import pandas as pd

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
    SAVE_PATH = configs['save_path']
    
    DATA_PATH = configs['data_path'] #'/root/jupyter/data/pkl'
    TABULAR_FNAME = configs['tabular_fname'] # 'MST_OPER_PART_220701.pkl'
    parts_id = configs['parts_id'] # FCST_PART_NO
    
    IDX_DIR = configs['target_prod_idx_dir'] # /root/ ... /FIT_DATA_TR_2206.pkl
    
    ECO_DATA_PATH = configs['eco_data_path'] # '/root/jupyter/data/target_data_0809/evaluation/Yearly'
    ECO_LONG_FNAME = configs['eco_long_fname'] # 'YDMD_MST_ECO_PART_202206.pkl'
    
    CLUSTER_PATH = configs['nlp_cluster_path']
    CLUSTER_FNAME = configs['nlp_cluster_fname']
    
    clip_min_quantile = configs['CLIP_MIN'] # 0.00
    clip_max_quantile = configs['CLIP_MAX'] # 0.95
    
    TABULAR_DIR = os.path.join(DATA_PATH, TABULAR_FNAME)
    ECO_LONG_DIR = os.path.join(ECO_DATA_PATH, ECO_LONG_FNAME)
    CLUSTER_DIR = os.path.join(CLUSTER_PATH, CLUSTER_FNAME)
    
    
    # Tabular (All)
    tabular_all = pd.read_pickle(TABULAR_DIR)
    tabular_all.set_index(parts_id, inplace = True)
    
    # Index of TARGET & ECO
    target_idx = pd.read_pickle(IDX_DIR).index.unique()
    eco_idx = pd.read_pickle(ECO_LONG_DIR).index.unique()
    
    tabular_target = tabular_all[tabular_all.index.isin(target_idx)]
    tabular_eco = tabular_all[tabular_all.index.isin(eco_idx)]
    tabular_raw = pd.concat([tabular_target, tabular_eco], axis=0)
    n_data = len(tabular_raw)
    print(f'Shape of ALL tabular data : {tabular_all.shape}')
    print(f'Shape of TARGET tabular data : {tabular_all.shape}')
    print(f'Shape of ECO tabular data : {tabular_eco.shape}')
    print(f'Shape of TARGET + ECO tabular data : {tabular_raw.shape}')

    # Drop Unused columns
    na_thres = configs['na_drop_thres'] # 0.3
    drop_na_mask = tabular_raw.isna().sum(axis=0)/n_data < na_thres
    tabular = tabular_raw.loc[:, drop_na_mask]
    cols = tabular.columns
    print(f'Before dropping NA columns : {tabular_raw.shape}')
    print(f'After dropping NA columns : {tabular.shape}')
    
    tabular_type = [str(i.type) for i in tabular.dtypes.values]
    column_category = cols[[True if 'ategor' in x else False for x in tabular_type]]
    column_float = cols[[True if 'float' in x else False for x in tabular_type]]
    column_int32 = cols[[True if 'int32' in x else False for x in tabular_type]]
    column_int16 = cols[[True if 'int16' in x else False for x in tabular_type]]
    column_int8 = cols[[True if 'int8' in x else False for x in tabular_type]]
    column_date = cols[[True if 'date' in x else False for x in tabular_type]]    
    print(f'# of CATEGORICAL columns : {len(column_category)}')
    print(f'# of FLOAT columns : {len(column_float)}')
    print(f'# of INTEGER32 columns : {len(column_int32)}')
    print(f'# of INTEGER16 columns : {len(column_int16)}')
    print(f'# of INTEGER8 columns : {len(column_int8)}')
    print(f'# of DATE columns : {len(column_date)}')
    
    # a) OPER_PART_NO
    tabular['OPER_PART_NO_part1'] = tabular['OPER_PART_NO'].apply(lambda x : x[:5])
    tabular['OPER_PART_NO_part2'] = tabular['OPER_PART_NO'].apply(lambda x : get_fifth_str(x))
    tabular['OPER_PART_NO_part3'] = tabular['OPER_PART_NO'].apply(lambda x : x[6:])
    part1_etc = get_count_percentage(tabular, 'OPER_PART_NO_part1', n_data, perc = 0.001)
    part2_etc = get_count_percentage(tabular, 'OPER_PART_NO_part2', n_data, perc = 0.0001)
    part3_etc = get_count_percentage(tabular, 'OPER_PART_NO_part3', n_data, perc = 0.001)
    tabular['OPER_PART_NO_part1'][tabular['OPER_PART_NO_part1'].isin(part1_etc)] = 'etc'
    tabular['OPER_PART_NO_part2'][tabular['OPER_PART_NO_part2'].isin(part2_etc)] = 'etc'
    tabular['OPER_PART_NO_part3'][tabular['OPER_PART_NO_part3'].isin(part3_etc)] = 'etc'
    
    # b) PART_NM_CD
    tabular['PART_NM_CD_part1'] = tabular['PART_NM_CD'].apply(lambda x : x[:3])
    tabular['PART_NM_CD_part2'] = tabular['PART_NM_CD'].apply(lambda x : x[3:5])
    part1_etc = get_count_percentage(tabular, 'PART_NM_CD_part1', n_data, perc = 0.001)
    part2_etc = get_count_percentage(tabular, 'PART_NM_CD_part2', n_data, perc = 0.001)
    tabular['PART_NM_CD_part1'][tabular['PART_NM_CD_part1'].isin(part2_etc)] = 'etc'
    tabular['PART_NM_CD_part2'][tabular['PART_NM_CD_part2'].isin(part3_etc)] = 'etc'
    
    # c) clustering with NLP clusters
    cluster_idx = pd.read_csv(CLUSTER_DIR)['cluster']
    tabular['cluster'] = cluster_idx
    
    # d) VENDOR_CD
    tabular['VENDOR_CD_part1'] = tabular['VENDOR_CD'].apply(lambda x : x[0:2])
    tabular['VENDOR_CD_part2'] = tabular['VENDOR_CD'].apply(lambda x : x[2:])
    tabular['VENDOR_CD_part1'].fillna('etc', inplace = True)
    tabular['VENDOR_CD_part2'].fillna('etc', inplace = True)
    
    # e) Drop columns
    cat_unused_cols = ['LT_CUT_DT'] + [x for x in cols if 'DATA' in x]
    cat_substituted_cols = ['PART_NM_CD', 'OPER_PART_NO', 'OPER_PART_NM', 'VENDOR_CD']
    cat_added_cols = ['OPER_PART_NO_part1', 'OPER_PART_NO_part2', 'OPER_PART_NO_part3', 
                  'PART_NM_CD_part1', 'PART_NM_CD_part2',
                  'VENDOR_CD_part1', 'VENDOR_CD_part2']
    cat_drop_cols = cat_unused_cols + cat_substituted_cols
    column_category  = list(set(column_category) - set(cat_drop_cols) | set(cat_added_cols))
    
    column_int = list(column_int8) + list(column_int16) + list(column_int32)
    for clip_cols in column_int :
        tabular[clip_cols] = clip_with_quantile(tabular, clip_cols, clip_min_quantile, clip_max_quantile)
    
    for int_col in column_int:
        tabular[int_col] = (tabular[int_col] - np.mean(tabular[int_col])) / np.std(tabular[int_col])
    
    tabular.drop(column_date, axis = 1, inplace = True)
    
    final_cols = list(column_float) + list(column_int) + list(column_category)
    na_ratio = tabular[final_cols].isna().mean()
    na_drop_cols = list(na_ratio[na_ratio > 0.05].index)
    
    drop_cols = na_drop_cols + ['OPER_PART_EOP_MONTH', 'PAT_TYPE', 'MODELS', 'ADI',
                                'FIRST_ACT_YYYYMM', 'LT_CUT_YN', 'LT_CUT_DT']
    drop_cols = drop_cols + [x for x in tabular.columns if 'DATE' in x]
    drop_cols = list(set(drop_cols))
    drop_cols = [col for col in drop_cols if col in tabular.columns]
    final_cols = list(set(final_cols) - set(drop_cols))
    tabular.drop(drop_cols, axis=1, inplace=True)
    
    etc_replace_cols = ['MAIN_FUNC','PART_NM_CD_part1', 'SUB_FUNC',
                        'USAGE_TYPE', 'CAR_USAGE_TYPE','CAR_TYPE',
                        'WH_MAIN','PART_NM_CD_part2', 'SOURCE_TYPE']
    mode_replace_cols = ['GIM_CLASS', 'PROD_TYPE']
    tabular[etc_replace_cols] = tabular[etc_replace_cols].astype('object')
    for col in etc_replace_cols:
        tabular[col].fillna('etc', inplace=True)
    for col in mode_replace_cols:
        tabular[col].fillna(tabular[col].moide()[0], inplace=True)
    
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(tabular[final_cols], f, pickle.HIGHEST_PROTOCOL)
    
    
    
    
    
if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.1f}".format(t2-t1))

