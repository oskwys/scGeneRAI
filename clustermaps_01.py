# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:03:09 2023

@author: d07321ow
"""


import pickle
import pandas as pd

import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.cm as cm


# %% get input data
# %% load data

#path_to_data = '/home/owysocki/Documents/KI_dataset/data_to_model'
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_model'
#path_to_data = 'KI_dataset/data_to_model'

#df_fus = feather.read_feather(os.path.join(path_to_data, 'CCE_fusions_to_model') )
#df_mut = feather.read_feather( os.path.join(path_to_data, 'CCE_mutations_to_model') ,)
#df_amp = feather.read_feather( os.path.join(path_to_data, 'CCE_amplifications_to_model') )
#df_del = feather.read_feather( os.path.join(path_to_data, 'CCE_deletions_to_model') , )
#df_exp = feather.read_feather( os.path.join(path_to_data, 'CCE_expressions_to_model') , )



df_fus = pd.read_csv(os.path.join(path_to_data, 'CCE_fusions_to_model.csv') ,index_col = 0)
df_mut = pd.read_csv( os.path.join(path_to_data, 'CCE_mutations_to_model.csv') ,index_col = 0)
df_amp = pd.read_csv( os.path.join(path_to_data, 'CCE_amplifications_to_model.csv'),index_col = 0 )
df_del = pd.read_csv( os.path.join(path_to_data, 'CCE_deletions_to_model.csv') ,index_col = 0 )
df_exp = pd.read_csv( os.path.join(path_to_data, 'CCE_expressions_to_model.csv') ,index_col = 0 )



df_clinical_features = pd.read_csv( os.path.join(path_to_data, 'CCE_clinical_features.csv') )


df_exp = df_exp.apply(lambda x: np.log(x + 1))

df_exp_stand = (df_exp-df_exp.mean(axis=0))/df_exp.std(axis=0)#.apply(lambda x: np.log(x +1))
df_mut_scale = (df_mut-df_mut.min(axis=0))/df_mut.max(axis=0)
df_fus_scale = (df_fus-df_fus.min(axis=0))/df_fus.max(axis=0)


df_exp_stand.columns = [col + '_exp' for col in df_exp_stand.columns]
df_mut_scale.columns = [col + '_mut' for col in df_mut.columns]
df_amp.columns = [col + '_amp' for col in df_amp.columns]
df_del.columns = [col + '_del' for col in df_del.columns]
df_fus_scale.columns = [col + '_fus' for col in df_fus.columns]


data = pd.concat((df_exp_stand, df_mut_scale, df_amp, df_del, df_fus_scale), axis = 1)

# %% analyse mutations
data = data[(df_clinical_features['acronym']=='BRCA').values]

df_clinical_features = df_clinical_features[(df_clinical_features['acronym']=='BRCA').values]

# %%

col_mut = [x  for x in  data.columns if '_mut' in x]

sns.clustermap(data.loc[:, col_mut], mask = (data.loc[:, col_mut]==0), method = 'ward')


.sum()






































