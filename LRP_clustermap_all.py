# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:50:50 2024

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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
 
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'
df = pd.read_csv(os.path.join(path_to_data, 'lrp_mean_matrix.csv'))

df = df.set_index('source_gene')

data_to_dendrogram = df.T.values
Z = linkage(data_to_dendrogram, method='ward')

# Define the cutoff threshold to determine the clusters
cutoff = .3

# Calculate the dendrogram
fig, ax = plt.subplots(figsize=(12, 5))
dn = dendrogram(
    Z, 
    color_threshold=cutoff, 
    above_threshold_color='gray', 
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=30,  # show only the last 12 merged clusters
    show_leaf_counts=True,  # show the number of samples in each cluster
    ax=ax
)

# Draw a line to signify the cutoff on the dendrogram
plt.axhline(y=cutoff, color='r', linestyle='--')

# Label the axes
ax.set_xlabel('Cluster Size/ID')
ax.set_ylabel('Distance')
 
cluster_labels = fcluster(Z, t=cutoff, criterion='distance')
unique_labels, counts = np.unique(cluster_labels, return_counts=True)


sns.clustermap(df, method = 'ward', mask = (df == 0) , cmap = 'jet', 
               row_linkage = Z, col_linkage = Z,
               yticklabels = False, xticklabels = False,)


genes = list(df.index)
df_genes = pd.DataFrame()
df_genes['genes'] = genes
df_genes['cluster'] = cluster_labels


# %% plot only exp-exp

df_expexp = pd.read_csv(os.path.join(path_to_data, 'lrp_mean_matrix.csv'))
df_expexp = df_expexp.set_index('source_gene')
cols = df_expexp.columns
cols = [x for x in cols if '_exp' in x ]
df_expexp = df_expexp[cols]
df_expexp = df_expexp.loc[df_expexp.index.isin(cols)]


data_to_dendrogram = df_expexp.T.values
Z = linkage(data_to_dendrogram, method='ward')

# Define the cutoff threshold to determine the clusters
cutoff = .2

# Calculate the dendrogram
fig, ax = plt.subplots(figsize=(12, 3))
dn = dendrogram(
    Z, 
    color_threshold=cutoff, 
    above_threshold_color='gray', 
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=30,  # show only the last 12 merged clusters
    show_leaf_counts=False,  # show the number of samples in each cluster
    ax=ax
)

# Draw a line to signify the cutoff on the dendrogram
plt.axhline(y=cutoff, color='r', linestyle='--')

 
cluster_labels = fcluster(Z, t=cutoff, criterion='distance')
unique_labels, counts = np.unique(cluster_labels, return_counts=True)


sns.clustermap(df_expexp, method = 'ward', mask = (df_expexp == 0) , cmap = 'jet', 
               row_linkage = Z, col_linkage = Z,
               yticklabels = False, xticklabels = False,)


genes = list(df_expexp.index)
df_genes = pd.DataFrame()
df_genes['genes'] = genes
df_genes['cluster'] = cluster_labels
df_genes.pivot(columns = 'cluster').to_excel(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\expexp_lrp_mean_clusters.xlsx')





