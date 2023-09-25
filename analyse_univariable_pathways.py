# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:47:47 2023

@author: d07321ow
"""


import pandas as pd

import numpy as np
import sys
import os
import seaborn as sns
#import scanpy as sc
#import torch
import scipy

import seaborn as sns
import matplotlib.pyplot as plt

path_to_data = '/home/owysocki/Documents/KI_dataset'
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset'

df_clinical_features = pd.read_csv( os.path.join( r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_model', 'CCE_clinical_features.csv') )

# %% load data

#path_to_data = '/home/owysocki/Documents/KI_dataset/data_to_model'
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'
#path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_model'
#path_to_data = 'KI_dataset/data_to_model'

#df_fus = feather.read_feather(os.path.join(path_to_data, 'CCE_fusions_to_model') )
#df_mut = feather.read_feather( os.path.join(path_to_data, 'CCE_mutations_to_model') ,)
#df_amp = feather.read_feather( os.path.join(path_to_data, 'CCE_amplifications_to_model') )
#df_del = feather.read_feather( os.path.join(path_to_data, 'CCE_deletions_to_model') , )
df_exp = feather.read_feather( os.path.join(path_to_data, 'CCE_expressions_to_model') , )



df_fus = pd.read_csv(os.path.join(path_to_data, 'CCE_fusions_to_model.csv') ,index_col = 0)
df_mut = pd.read_csv( os.path.join(path_to_data, 'CCE_mutations_to_model.csv') ,index_col = 0)
df_amp = pd.read_csv( os.path.join(path_to_data, 'CCE_amplifications_to_model.csv'),index_col = 0 )
df_del = pd.read_csv( os.path.join(path_to_data, 'CCE_deletions_to_model.csv') ,index_col = 0 )
#df_exp = pd.read_csv( os.path.join(path_to_data, 'CCE_expressions_to_model.csv') ,index_col = 0 )

#df_clinical_features = pd.read_csv( os.path.join(path_to_data, 'CCE_clinical_features.csv') )

# %% preprocess data
df_exp = df_exp.apply(lambda x: np.log(x + 1))
df_exp_stand = (df_exp-df_exp.mean(axis=0))/df_exp.std(axis=0)

df_mut_scale = (df_mut-df_mut.min(axis=0))/df_mut.max(axis=0)
df_fus_scale = (df_fus-df_fus.min(axis=0))/df_fus.max(axis=0)

df_amp[df_amp==2] =1


# %% data to model

data = pd.concat((df_exp_stand, df_mut_scale, df_amp, df_del, df_fus_scale), axis = 1)
# %% genes from pathways

genes_pathways = pd.read_csv(os.path.join(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\PATHWAYS', 'genes_pathways_pancanatlas_matched_cce.csv'))
genes_pathways_set = list(set(genes_pathways['cce_match'].dropna()))


# %% exp vs exp

genes_pathways_list = [gene + '_exp' for gene in  genes_pathways_set]
genes_pathways_list.sort()

cols_to_select = set.intersection(set(genes_pathways_list), set(df_exp.columns))
df_exp_path = df_exp[list(cols_to_select)]

corrs = df_exp_path.corr(method = 'spearman')

sns.heatmap(corrs, mask = corrs.abs() < 0.5)



# %%






























