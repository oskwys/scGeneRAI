# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:28:42 2023

@author: d07321ow
"""

%matplotlib inline 
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
samples = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()

data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model')

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

df_clinical_features = f.add_cluster0(df_clinical_features)

# %%
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300'

topn = 1000

shared_nodes_matrix = pd.read_csv(os.path.join(path_to_data, 'shared_nodes_matrix_{}.csv'.format(topn)), index_col = 0)
shared_edges_matrix = pd.read_csv(os.path.join(path_to_data, 'shared_edges_matrix_{}.csv'.format(topn)), index_col = 0)
ars_matrix = pd.read_csv(os.path.join(path_to_data, 'ars_matrix_{}.csv'.format(topn)), index_col = 0)
amis_matrix = pd.read_csv(os.path.join(path_to_data, 'amis_matrix_{}.csv'.format(topn)), index_col = 0)


# %% clustermaps

#sns.heatmap(ars_matrix)

#sns.heatmap(amis_matrix)

#sns.heatmap(shared_nodes_matrix)


#sns.heatmap(shared_edges_matrix)

# %% dendrograms

labels_df = pd.DataFrame(samples, columns = ['samples'])

LRP_pd = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\LRP_individual_top1000.csv', index_col = 0)


# %% CLUSTERMAP ALL

df_clinical_features_ = LRP_pd.T.reset_index().iloc[:,0:1].merge(df_clinical_features, left_on = 'index', right_on = 'bcr_patient_barcode').set_index('index')
column_colors = f.map_subtypes_to_col_color(df_clinical_features_)


# %%% ars_matrix
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

data_to_dendrogram = ars_matrix
Z = linkage(data_to_dendrogram, method='ward')

cutoff = 30
fig, ax = plt.subplots(figsize=(10, 3))
dn = dendrogram(
    Z, 
    color_threshold=cutoff, 
    above_threshold_color='gray', 
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=30,  # show only the last 12 merged clusters
    #show_leaf_counts=True,  # show the number of samples in each cluster
    ax=ax
)

plt.axhline(y=cutoff, color='r', linestyle='--')
ax.set_ylabel('Distance')
ax.set_title('Adjusted Rand Score') 
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'dendrogram_ars'))
            
cluster_labels = fcluster(Z, t=cutoff, criterion='distance')
unique_labels, counts = np.unique(cluster_labels, return_counts=True)

labels_df['clusters_ARS'] = cluster_labels

sns.clustermap(ars_matrix, row_linkage=Z, col_linkage=Z , cmap = 'jet', col_colors=column_colors, colors_ratio=0.02)
plt.title('Adjusted Rand Score') 
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'clustermap_ars'))



# %%% amis_matrix
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

data_to_dendrogram = amis_matrix
Z = linkage(data_to_dendrogram, method='ward')

cutoff = 30
fig, ax = plt.subplots(figsize=(10, 3))
dn = dendrogram(
    Z, 
    color_threshold=cutoff, 
    above_threshold_color='gray', 
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=30,  # show only the last 12 merged clusters
    #show_leaf_counts=True,  # show the number of samples in each cluster
    ax=ax
)

plt.axhline(y=cutoff, color='r', linestyle='--')
ax.set_ylabel('Distance')
ax.set_title('Adjusted Mutual Info Score') 
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'dendrogram_amis'))
cluster_labels = fcluster(Z, t=cutoff, criterion='distance')
unique_labels, counts = np.unique(cluster_labels, return_counts=True)

labels_df['clusters_AMIS'] = cluster_labels

sns.clustermap(amis_matrix, row_linkage=Z, col_linkage=Z , cmap = 'jet', col_colors=column_colors, colors_ratio=0.02)
plt.title('Adjusted Mutual Info Score') 
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'clustermap_amis'))


# %%% shared_nodes_matrix
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

data_to_dendrogram = shared_nodes_matrix
Z = linkage(data_to_dendrogram, method='ward')

cutoff = 4000
fig, ax = plt.subplots(figsize=(10, 3))
dn = dendrogram(
    Z, 
    color_threshold=cutoff, 
    above_threshold_color='gray', 
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=30,  # show only the last 12 merged clusters
    #show_leaf_counts=True,  # show the number of samples in each cluster
    ax=ax
)

plt.axhline(y=cutoff, color='r', linestyle='--')
ax.set_ylabel('Distance')
ax.set_title('Number of shared nodes') 
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'dendrogram_shared_nodes'))
cluster_labels = fcluster(Z, t=cutoff, criterion='distance')
unique_labels, counts = np.unique(cluster_labels, return_counts=True)

labels_df['clusters_shared_nodes'] = cluster_labels

sns.clustermap(shared_nodes_matrix, row_linkage=Z, col_linkage=Z , cmap = 'jet', col_colors=column_colors, colors_ratio=0.02)
plt.title('Number of shared nodes') 
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'clustermap_shared_nodes'))

# %%% shared_edges_matrix
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

data_to_dendrogram = shared_edges_matrix
Z = linkage(data_to_dendrogram, method='ward')

cutoff = 9000
fig, ax = plt.subplots(figsize=(10, 3))
dn = dendrogram(
    Z, 
    color_threshold=cutoff, 
    above_threshold_color='gray', 
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=30,  # show only the last 12 merged clusters
    #show_leaf_counts=True,  # show the number of samples in each cluster
    ax=ax
)

plt.axhline(y=cutoff, color='r', linestyle='--')
ax.set_ylabel('Distance')
ax.set_title('Number of shared nodes') 
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'dendrogram_shared_edges'))
cluster_labels = fcluster(Z, t=cutoff, criterion='distance')
unique_labels, counts = np.unique(cluster_labels, return_counts=True)

labels_df['clusters_shared_edges'] = cluster_labels

sns.clustermap(shared_edges_matrix, row_linkage=Z, col_linkage=Z , cmap = 'jet', col_colors=column_colors, colors_ratio=0.02)

plt.title('Number of shared interactions') 
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'clustermap_shared_edges'))



# %%

labels_df = labels_df[labels_df['samples'].isin(list(LRP_pd.columns))]

labels_df.to_csv(os.path.join(path_to_save, 'cluster_labels_shared_edges.csv'))

# %%








from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

max_clusters = 15
silhouette_scores = []

for n_clusters in range(2, max_clusters):
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = clusterer.fit_predict(data_to_dendrogram)
    silhouette_avg = silhouette_score(data_to_dendrogram, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(5,5))
plt.plot(range(2, max_clusters), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.show()


from gap_statistic import OptimalK

# Use the gap statistic to determine the optimal number of clusters
optimalK = OptimalK(parallel_backend='none')
n_clusters = optimalK(data_to_dendrogram, cluster_array=np.arange(1, 15))

print('Optimal number of clusters', n_clusters)

# Plot the gap statistic values
plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
            optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()



import umap
reducer = umap.UMAP()
embedding = reducer.fit_transform(data_to_dendrogram)

X_umap = pd.DataFrame(embedding , columns  = ['umap_1','umap_2'])

fig, ax = plt.subplots(figsize = (8,8))
scatter = ax.scatter(X_umap['umap_1'], X_umap['umap_2'], cmap='viridis', s=10)


ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
title = 'Threshold ' + threshold + '\n' + 'n_neighbors ' + n_neighbors + '\n' + 'min_dist ' + min_dist
ax.set_title(title)


