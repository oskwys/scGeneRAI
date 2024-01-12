# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:27:12 2024

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
from kneefinder import KneeFinder
from kneed import KneeLocator


# %% get samples

#samples = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()
samples = pd.read_csv(r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()

#data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model')
data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(r'G:\My Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model')

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

df_clinical_features = f.add_cluster2(df_clinical_features)


samples_groups = f.get_samples_by_group(df_clinical_features)


# %%
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300'

topn = 1000

LRP_pd = pd.read_csv(r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\LRP_individual_top1000_noexpexp.csv', index_col = 0)

# %% histograms

sns.histplot(LRP_pd.values.ravel())



# %% clustermap
data_to_clustermap = LRP_pd

Z_x = linkage(data_to_clustermap, method = 'ward')
Z_y = linkage(data_to_clustermap.T, method = 'ward')
#sns.clustermap(a, cmap = 'Reds', vmin = 0, col_colors=col_colors, yticklabels = True, figsize = (20,20))
fg = sns.clustermap(data_to_clustermap, method = 'ward', cmap = 'jet', vmax = 0.01, mask = data_to_clustermap==0,
                    yticklabels = False, figsize = (10,10), row_linkage = Z_x, col_linkage = Z_y,
                    xticklabels = False,
                    #col_cluster =False
                    )
fg.ax_heatmap.set_xlabel('Sample')
fg.ax_heatmap.set_ylabel('Interaction')

plt.title('Overlap: {}'.format(threshold))
plt.savefig(os.path.join(path_to_save, 'clustermap_lrp_1000_filtered.png'))
plt.savefig(os.path.join(path_to_save, 'clustermap_lrp_1000_filtered.pdf'), format= 'pdf')



dendrogram_cutoff = 7
cutoff = dendrogram_cutoff
fig, ax = plt.subplots(figsize=(10, 3))
dn = dendrogram(
    Z_y, 
    color_threshold=cutoff, 
    above_threshold_color='gray', 
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=30,  # show only the last 12 merged clusters
    #show_leaf_counts=True,  # show the number of samples in each cluster
    ax=ax
)

plt.axhline(y=cutoff, color='r', linestyle='--')
ax.set_ylabel('Distance')
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'dendrogram_samples_lrp_filtered1000.png'))
plt.show()

cluster_labels = fcluster(Z_y, t=cutoff, criterion='distance')
samples_cluster2 = pd.Series(samples)[cluster_labels==2]
samples_i_cluster2 = pd.Series(range(988))[cluster_labels==2].values




dendrogram_cutoff = 7
cutoff = dendrogram_cutoff
fig, ax = plt.subplots(figsize=(10, 3))
dn = dendrogram(
    Z_x, 
    color_threshold=cutoff, 
    above_threshold_color='gray', 
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=30,  # show only the last 12 merged clusters
    #show_leaf_counts=True,  # show the number of samples in each cluster
    ax=ax
)

plt.axhline(y=cutoff, color='r', linestyle='--')
ax.set_ylabel('Distance')
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'dendrogram_interaction_lrp_filtered1000.png'))
plt.show()


cluster_labels = fcluster(Z_x, t=cutoff, criterion='distance')
edges_cluster2 = pd.Series(list(edges_all_filtered.index))[cluster_labels==2]
nodes_cluster2 = list(set(edges_cluster2.str.split('-',expand = True).melt()['value'].to_list()))
nodes_cluster2 .sort()
edges_cluster2 = list(edges_cluster2)
edges_cluster2.sort()


cluster_labels = fcluster(Z_y, t=7, criterion='distance')
samples_cluster2 = pd.Series(samples)[cluster_labels==2].to_list()



