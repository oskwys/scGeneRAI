# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:32:47 2023

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
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\umaps'

path_to_save = '/home/d07321ow/scratch/results_LRP_BRCA/umaps'
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\umaps'

# %%
samples = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()
data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model')
df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)
samples_groups = f.get_samples_by_group(df_clinical_features)

# %% load PCA
from collections import Counter

pca_files = files = [f for f in os.listdir(path_to_data) if f.startswith('PCA_t')]
#cluster_results = pd.DataFrame()

explained_variances = pd.read_csv(os.path.join(path_to_data, 'PCA_explained_variance_thres0001.csv'), index_col = 0)

for file in pca_files:
    print(file)
    threshold = file.split('_')[1].split('.')[0]
        
    X_pca = pd.read_csv(os.path.join(path_to_data, file), index_col = 0)
    
    explained_variance = explained_variances.loc[explained_variances['thresholds'].astype('str').str.replace('.','') == threshold.replace('thres',''), :].iloc[:,:-1].round(2).values[0]
    fig, ax = plt.subplots(figsize = (8,8))
    ax.scatter(X_pca['PCA_1'], X_pca['PCA_2'], cmap='viridis', s=10)
    
    ax.set_xlabel('Principal Component 1 explained variance: {}'.format(explained_variance[0]))
    ax.set_ylabel('Principal Component 2 explained variance: {}'.format(explained_variance[1]))

    title = 'Threshold ' + threshold
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, 'PCAplot_{}.png'.format(threshold)), dpi = 400)        
    