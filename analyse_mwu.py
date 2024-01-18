# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 00:00:53 2024

@author: owysocky
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
from kneefinder import KneeFinder
from kneed import KneeLocator
from matplotlib.gridspec import GridSpec
import math
import textwrap
# %% samples
path_to_data = r'G:\My Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'
data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(path_to_data)

samples = f.get_samples_with_lrp(path_to_lrp_results)

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)
df_clinical_features = f.add_cluster2(df_clinical_features)

samples_groups = f.get_samples_by_group(df_clinical_features)



# %% analyse MWU LRP
path_to_data = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\mwu'
path_to_lrp_mean = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'
path_to_save = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\lrp_top_interactions\plots'
path_to_save_csv = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\lrp_top_interactions\data_to_plots'
import os

df_lrp_mean = pd.read_csv(os.path.join(path_to_lrp_mean, 'lrp_mean_list.csv'), index_col=0)
sns.histplot(df_lrp_mean['LRP_mean'])
sns.distplot(df_lrp_mean['LRP_mean'])
#df_lrp_mean = df_lrp_mean.sort_values()

for group in list(samples_groups.keys())[:2]:
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)
    
    df = pd.read_csv(os.path.join(path_to_data, 'mwu_edges_LRP_{}.csv'.format(group)), index_col=0)
    df['temp'] = temp['edge'].values
    df['LRP_mean'] = df_lrp_mean['LRP_mean']
    
    
    
# Create a figure and axis object
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='CLES', y='p-val', ax=ax)
ax.set_yscale('log')
ax.invert_yaxis()

fig, ax = plt.subplots()
sns.scatterplot(data=df, x='LRP_mean', y='p-val', ax=ax, alpha = 0.2, size = 3)
ax.set_yscale('log')
ax.invert_yaxis()

sns.displot(data = df, x ='CLES')

fig, ax = plt.subplots()
sns.histplot(data = df, x ='p-val',ax=ax)
ax.set_xscale('log')





