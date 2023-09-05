# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 14:00:19 2023

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

# %%  get only BRCA patients
subtype = 'BRCA'
index_ = (df_clinical_features['acronym'] == subtype).values
index_ = (df_clinical_features['acronym'] != 0).values


# %% FUSION
file = 'CCE_fusion_genes.csv'
df= pd.read_csv(os.path.join(path_to_data, file) , index_col = 0)
df = df[index_]

# Prepare fusions to model
# condition
min_n_with_condition = 2

df_fus = df.iloc[:, ((df != 0).sum() > min_n_with_condition).values]
#(df_fus>0).sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (10,5),  title = 'No. samples with Fusions')


# %% MUTATIONS
file = 'CCE_final_analysis_mutations.csv'
df= pd.read_csv(os.path.join(path_to_data, file) , index_col = 0)
df = df[index_]

# Prepare mutations to model
# condition
min_n_with_condition = 2

df_mut = df.iloc[:, ((df != 0).sum() > min_n_with_condition).values]

#(df_mut>0).sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (20,5), title = 'No. samples with Mutations')
#plt.show()

#(df_mut>0).sum().plot(kind = 'hist', bins = 20)

# %% CNA
file = 'CCE_data_CNA_paper.csv'
df= pd.read_csv(os.path.join(path_to_data, file) , index_col = 0)
df = df[index_]


#sns.heatmap(df, cmap = 'Reds')
#sns.clustermap(df, cmap = 'RdBu', method = 'ward')
#plt.show()


# Prepare CNA amplification to model
# condition
min_n_with_condition = 2
df_amp = df[df>0].fillna(0)
# any amplificaiton (1 or 2) is a binary == 1
df_amp[df_amp >0] = 2
df_amp = df_amp.iloc[:, ((df_amp > 0).sum() > min_n_with_condition).values]

# Prepare CNA deletion to model
# condition
min_n_with_condition = 2
df_del = df[df<0].fillna(0)
# any amplificaiton (1 or 2) is a binary == 1
df_del[df_del <0] = 1
df_del = df_del.iloc[:, ((df_del > 0).sum() > min_n_with_condition).values]


#(df_amp>0).sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (20,5), title = 'No. samples with Amplifications')
#plt.show()
#(df_del>0).sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (20,5), title = 'No. samples with Deletions')
#plt.show()
# %% Expressions
#file = 'CCE_gene_expression.csv'
#df= pd.read_csv(os.path.join(path_to_data, file) , index_col = 0)

import pyarrow.feather as feather
#feather.write_feather(df, os.path.join(path_to_data, 'CCE_gene_expression') )
df = feather.read_feather(os.path.join(path_to_data, 'CCE_gene_expression'))
df = df[index_]
df_exp = df.fillna(0)

# adjust per each patient
df_exp = df_exp.div(df.sum(axis=1), axis=0) * 10000


# select gene expression
genes_to_select = pd.read_csv(os.path.join(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection', 'brca_genes_selected.csv'),index_col = 0)
genes_to_select = set(genes_to_select['genes'].to_list())
genes_exp = set(df_exp.columns)

missing = set.difference(genes_to_select, genes_exp)
intersection = set.intersection(genes_to_select, genes_exp)
df_exp = df_exp[ intersection]


# %% add names to columns
df_exp.columns = [col + '_exp' for col in df_exp.columns]
df_mut.columns = [col + '_mut' for col in df_mut.columns]
df_amp.columns = [col + '_amp' for col in df_amp.columns]
df_del.columns = [col + '_del' for col in df_del.columns]
df_fus.columns = [col + '_fus' for col in df_fus.columns]

print(np.sum(index_))


# comput z score for each column EXPR
df_exp_z_score = (df_exp - df_exp.mean()) / df_exp.std()

# remove nans
cols_to_keep = df_exp_z_score.columns[df_exp_z_score.isna().sum() == 0]
df_exp_z_score = df_exp_z_score[cols_to_keep]
df_exp = df_exp[cols_to_keep]



# %% SAVE DATA 
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_model'


feather.write_feather(df_fus, os.path.join(path_to_save, 'CCE_fusions_to_model') )
feather.write_feather(df_mut, os.path.join(path_to_save, 'CCE_mutations_to_model') )
feather.write_feather(df_amp, os.path.join(path_to_save, 'CCE_amplifications_to_model') )
feather.write_feather(df_del, os.path.join(path_to_save, 'CCE_deletions_to_model') )
feather.write_feather(df_exp, os.path.join(path_to_save, 'CCE_expressions_to_model') )

df_fus.to_csv(os.path.join(path_to_save, 'CCE_fusions_to_model.csv') )
df_mut.to_csv(os.path.join(path_to_save, 'CCE_mutations_to_model.csv') )
df_amp.to_csv( os.path.join(path_to_save, 'CCE_amplifications_to_model.csv') )
df_del.to_csv(os.path.join(path_to_save, 'CCE_deletions_to_model.csv') )
#df_exp.to_csv(os.path.join(path_to_save, 'CCE_expressions_to_model.csv') )





