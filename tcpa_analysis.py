# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:14:29 2024

@author: d07321ow
"""



%matplotlib inline 
import pickle
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.cm as cm

import matplotlib
import pyarrow.feather as feather
import itertools

#from scGeneRAI import scGeneRAI
import functions as f
from datetime import datetime

import importlib, sys
importlib.reload(f)

 #%%
 
path_to_data = r'G:\My Drive\SAFE_AI\CCE_DART\TCPA_data\tmp'
tcpa_data = pd.read_csv(os.path.join(path_to_data, 'TCGA-BRCA-L4.csv'))
tcpa_data['bcr_patient_barcode'] = tcpa_data['Sample_ID'].str[:12]
# cce data

samples = pd.read_csv(r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()
data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(r'G:\My Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model')
df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)
df_clinical_features = f.add_cluster2(df_clinical_features)

samples_groups = f.get_samples_by_group(df_clinical_features)
#df_clinical_features=df_clinical_features.set_index('bcr_patient_barcode')

samples_intersection = set(df_clinical_features['bcr_patient_barcode']).intersection(set(tcpa_data['bcr_patient_barcode']))

# save samples_intersection to file
df_samples_intersection = pd.DataFrame(samples_intersection, columns = ['samples'])
df_samples_intersection.to_csv(os.path.join(r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_TCPA', 'samples_intersection.txt'))


tcpa_to_model = tcpa_data[tcpa_data['bcr_patient_barcode'].isin(list(df_samples_intersection['samples']))]
# check if bcr_patient_barcode is unique
print('check if bcr_patient_barcode has only unique IDs: ', tcpa_to_model['bcr_patient_barcode'].nunique() == tcpa_to_model.shape[0])
# find duplicated bcr_patient_barcode
tcpa_to_model['duplicated'] = tcpa_to_model.duplicated(subset = 'bcr_patient_barcode', keep = False)
tcpa_to_model[tcpa_to_model['duplicated']].sort_values('bcr_patient_barcode')

# get list of duplicated bcr_patient_barcode
duplicated_samples = tcpa_to_model[tcpa_to_model['duplicated']]['bcr_patient_barcode'].to_list()

# reomve duplicated samples from df_smamples_intersection
df_samples_intersection = df_samples_intersection[~df_samples_intersection['samples'].isin(duplicated_samples)]

# repeat the process of filtering the tcpa data
tcpa_to_model = tcpa_data[tcpa_data['bcr_patient_barcode'].isin(list(df_samples_intersection['samples']))]
print('check if bcr_patient_barcode has only unique IDs: ', tcpa_to_model['bcr_patient_barcode'].nunique() == tcpa_to_model.shape[0])

# sort tcpa_to_model by bcr_patient_barcode
tcpa_to_model = tcpa_to_model.sort_values('bcr_patient_barcode').reset_index(drop = True)
# set index to bcr_patient_barcode
tcpa_to_model = tcpa_to_model.set_index('bcr_patient_barcode')

# %% EDA on the dataset

df_clinical_features_ = df_clinical_features.loc[df_clinical_features['bcr_patient_barcode'].isin(list(df_samples_intersection['samples'])), :]
# set index of df_clinical_features_ to bcr_patient_barcode
df_clinical_features_ = df_clinical_features_.set_index('bcr_patient_barcode')

column_colors = pd.DataFrame(f.map_subtypes_to_col_color(df_clinical_features_)).T
column_colors = column_colors.rename(
    columns={"Estrogen_receptor": "ER", "Progesterone_receptor": "PR"}
)
# describe the dataset
tcpa_describe = tcpa_data.describe().T


vmax = 5
g = sns.clustermap(
    tcpa_to_model.iloc[:, 4:].fillna(0).T, 
    #vmax=vmax,
    method="ward",
    cmap="jet",
    #col_linkage=Z_col,
    #row_linkage=Z_row,
    yticklabels=False,
    xticklabels=False,
    col_colors=column_colors,
    #row_colors=row_colors,
    #cbar_kws={"orientation": "horizontal", "fraction": 0.1, "pad": 0.1},
)

# %% prepare data to model

path_to_data = r'C:\Users\owysocky\Documents\GitHub\scGeneRAI\data\data_BRCA'
df_exp = feather.read_feather( os.path.join(path_to_data, 'CCE_expressions_to_model') , )
df_fus = pd.read_csv(os.path.join(path_to_data, 'CCE_fusions_to_model.csv') ,index_col = 0)
df_mut = pd.read_csv( os.path.join(path_to_data, 'CCE_mutations_to_model.csv') ,index_col = 0)
df_amp = pd.read_csv( os.path.join(path_to_data, 'CCE_amplifications_to_model.csv'),index_col = 0 )
df_del = pd.read_csv( os.path.join(path_to_data, 'CCE_deletions_to_model.csv') ,index_col = 0 )


# filter the data to the samples_intersection
df_exp = df_exp.loc[df_exp.index.isin(list(df_samples_intersection['samples'])), :]
df_fus = df_fus.loc[df_fus.index.isin(list(df_samples_intersection['samples'])), :]
df_mut = df_mut.loc[df_mut.index.isin(list(df_samples_intersection['samples'])), :]
df_amp = df_amp.loc[df_amp.index.isin(list(df_samples_intersection['samples'])), :]
df_del = df_del.loc[df_del.index.isin(list(df_samples_intersection['samples'])), :]

df_prot = tcpa_to_model.iloc[:, 4:].fillna(0)
# add suffix to each column name '_prot'
df_prot.columns = [x + '_prot' for x in df_prot.columns]


df_exp = df_exp.apply(lambda x: np.log(x + 1))
df_exp_stand = (df_exp-df_exp.mean(axis=0))/df_exp.std(axis=0)

df_mut_scale = (df_mut-df_mut.min(axis=0))/df_mut.max(axis=0)
df_fus_scale = (df_fus-df_fus.min(axis=0))/df_fus.max(axis=0)

df_amp[df_amp==2] =1

data_to_model = pd.concat((df_exp_stand, df_mut_scale, df_amp, df_del, df_fus_scale, df_prot), axis = 1)

data_to_model.to_csv(os.path.join(r'C:\Users\owysocky\Documents\GitHub\scGeneRAI\data\data_BRCA', 'data_to_model_proteomics.csv'))


