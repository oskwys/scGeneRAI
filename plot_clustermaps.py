# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:01:14 2023

@author: d07321ow
"""

#%matplotlib inline 
import pickle
import pandas as pd

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

# %% load data

path_to_data = '/home/d07321ow/scratch/scGeneRAI/data/data_BRCA'
#path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'

path_to_save = '/home/d07321ow/scratch/results_LRP_BRCA/networks'
#path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\networks'


#path_to_pathways = r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI\PATHWAYS'
path_to_pathways ='/home/d07321ow/scratch/scGeneRAI/PATHWAYS'

#path_to_lrp_results = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\results'
path_to_lrp_results = '/home/d07321ow/scratch/results_LRP_BRCA/results'

data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(path_to_data)

# %% get samples

samples = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()

data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model')

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

df_clinical_features = f.add_cluster0(df_clinical_features)


samples_groups = f.get_samples_by_group(df_clinical_features)


# %%
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'

topn = 100

LRP_pd = pd.read_csv(os.path.join(path_to_save, 'LRP_individual_top{}.csv'.format(topn)), index_col = 0)


# %% CLUSTERMAP ALL
df_clinical_features_ = f.get_column_colors_from_clinical_df(df_clinical_features, LRP_pd)
column_colors = f.map_subtypes_to_col_color(df_clinical_features_)


g = sns.clustermap(LRP_pd, method = 'ward', vmin = 0.00, vmax = 0.05, cmap = 'jet', yticklabels = False, xticklabels = False,
               #mask = LRP_pd<0.01, 
               col_colors=column_colors)
g.ax_heatmap.set_xlabel('Samples')
g.ax_heatmap.set_ylabel('Interactions')

plt.savefig(os.path.join(path_to_save, 'clustermap_top{}.png'.format(topn)))

fig,ax = plt.subplots()
ax.hist(LRP_pd.values.ravel(), bins = 30)
LRP_pd.melt().describe()


# %% CLUSTERMAP PAM50




