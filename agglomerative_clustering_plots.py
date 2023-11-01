# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:56:58 2023

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
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\umaps_median'

path_to_save = '/home/d07321ow/scratch/results_LRP_BRCA/umaps'
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\umaps_median'

# %%
samples = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()
data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model')
df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

df_clinical_features = f.add_cluster0(df_clinical_features)

samples_groups = f.get_samples_by_group(df_clinical_features)

# %% load PCA
from collections import Counter
from scipy.cluster.hierarchy import dendrogram

z_files = files = [f for f in os.listdir(path_to_data) if f.startswith('Z_')]
#cluster_results = pd.DataFrame()

for file in z_files:
    print(file)
    threshold = file.split('_')[2].split('.')[0]
    method = file.split('_')[1]
    
    Z = pd.read_csv(os.path.join(path_to_data, file), index_col = 0).values


    fig, ax = plt.subplots(figsize = (15,7))

    # Create a dendrogram based on the linkage matrix
    dendrogram(Z, ax=ax)
    
    # Enhance the plot

    ax.set_xlabel('sample index or (cluster size)')
    ax.set_ylabel('distance')

    title = 'Threshold ' + threshold + ' ' + method
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, 'dendrogram_{}_{}.png'.format(method, threshold)), dpi = 300)        
    