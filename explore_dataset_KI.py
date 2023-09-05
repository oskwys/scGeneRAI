#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:37:13 2023

@author: owysocki
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


# %% FUSION
file = 'CCE_fusion_genes.csv'
df= pd.read_csv(os.path.join(path_to_data, file) , index_col = 0)



sns.heatmap(df, cmap = 'Reds')
plt.show()

df.sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (20,5))

(df != 0).sum()


# Prepare fusions to model
# condition
min_n_with_condition = 2

df_fusions = df.iloc[:, ((df != 0).sum() > min_n_with_condition).values]

# %% MUTATIONS
file = 'CCE_final_analysis_mutations.csv'
df= pd.read_csv(os.path.join(path_to_data, file) , index_col = 0)



sns.heatmap(df, cmap = 'Reds')
plt.show()

df.sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (20,5))
plt.show()

# Prepare mutations to model
# condition
min_n_with_condition = 10

df_mutations = df.iloc[:, ((df != 0).sum() > min_n_with_condition).values]

sns.clustermap(df_mutations, cmap = 'Reds', method = 'ward')



# %% CNA
file = 'CCE_data_CNA_paper.csv'
df= pd.read_csv(os.path.join(path_to_data, file) , index_col = 0)



sns.heatmap(df, cmap = 'Reds')
sns.clustermap(df, cmap = 'RdBu', method = 'ward')
plt.show()

(df==2).sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (20,5), title = 'CNA = 2')
(df==1).sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (20,5), title = 'CNA = 1')
(df==-1).sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (20,5), title = 'CNA = -1')
(df==-2).sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (20,5), title = 'CNA = -2')
plt.show()


# Prepare CNA amplification to model
# condition
min_n_with_condition = 10
df_amp = df[df>0].fillna(0)
# any amplificaiton (1 or 2) is a binary == 1
df_amp[df_amp >0] = 1
df_amp = df_amp.iloc[:, ((df_amp > 0).sum() > min_n_with_condition).values]

sns.clustermap(df_amp, cmap = 'Reds', method = 'ward')

# Prepare CNA deletion to model
# condition
min_n_with_condition = 10
df_del = df[df<0].fillna(0)
# any amplificaiton (1 or 2) is a binary == 1
df_del[df_del <0] = 1
df_del = df_del.iloc[:, ((df_del > 0).sum() > min_n_with_condition).values]

sns.clustermap(df_del, cmap = 'Reds', method = 'ward')


# %% Expressions
#file = 'CCE_gene_expression.csv'
#df= pd.read_csv(os.path.join(path_to_data, file) , index_col = 0)

import pyarrow.feather as feather
#feather.write_feather(df, os.path.join(path_to_data, 'CCE_gene_expression') )
df = feather.read_feather(os.path.join(path_to_data, 'CCE_gene_expression'))
df = df.fillna(0)

# adjust per each patient
df = df.div(df.sum(axis=1), axis=0) * 10000


std_ = df.std()
std_.sort_values(ascending=False)[:1000].plot(kind='bar')

mean_ = np.log(df+1).mean()
mean_.sort_values(ascending=False).plot()

np.log(df+1).mean().sort_values(ascending=False).plot()
np.log(df+1).max().sort_values(ascending=False).plot()
np.log(df+1).min().sort_values(ascending=False).plot()

# condition
# highest average expression values
#std_min = 5000

mean_min = df.mean().sort_values(ascending=False)[1000]

df_expr = df.loc[:, list(df.mean()[df.mean() > mean_min].index)]


sns.clustermap(np.log(df_expr.values+1), cmap = 'Reds', method = 'ward')

# %% SAVE DATA 
path_to_save = '/home/owysocki/Documents/KI_dataset/data_to_model'
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_model'


feather.write_feather(df_fusions, os.path.join(path_to_save, 'CCE_fusions_to_model') )
feather.write_feather(df_mutations, os.path.join(path_to_save, 'CCE_mutations_to_model') )
feather.write_feather(df_amp, os.path.join(path_to_save, 'CCE_amplifications_to_model') )
feather.write_feather(df_del, os.path.join(path_to_save, 'CCE_deletions_to_model') )
feather.write_feather(df_expr, os.path.join(path_to_save, 'CCE_expressions_to_model') )

df_fusions.to_csv(os.path.join(path_to_save, 'CCE_fusions_to_model.csv') )
df_mutations.to_csv(os.path.join(path_to_save, 'CCE_mutations_to_model.csv') )
df_amp.to_csv( os.path.join(path_to_save, 'CCE_amplifications_to_model.csv') )
df_del.to_csv(os.path.join(path_to_save, 'CCE_deletions_to_model.csv') )
df_expr.to_csv(os.path.join(path_to_save, 'CCE_expressions_to_model.csv') )




