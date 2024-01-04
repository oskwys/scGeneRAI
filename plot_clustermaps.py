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

topn = 1000

LRP_pd = pd.read_csv(os.path.join(path_to_save, 'LRP_individual_top{}.csv'.format(topn)), index_col = 0)


# %% CLUSTERMAP ALL

df_clinical_features_ = LRP_pd.T.reset_index().iloc[:,0:1].merge(df_clinical_features, left_on = 'index', right_on = 'bcr_patient_barcode').set_index('index')
column_colors = f.map_subtypes_to_col_color(df_clinical_features_)


g = sns.clustermap(LRP_pd, method = 'ward', vmin = 0.00, vmax = 0.1, cmap = 'jet', yticklabels = False, xticklabels = False,
               #mask = LRP_pd<0.01, 
               col_colors=column_colors)
g.ax_heatmap.set_xlabel('Samples')
g.ax_heatmap.set_ylabel('Interactions')

plt.savefig(os.path.join(path_to_save, 'clustermap_top{}.png'.format(topn)))

fig,ax = plt.subplots()
ax.hist(LRP_pd.values.ravel(), bins = 30)
LRP_pd.melt().describe()

# %% dendrogram and clusters
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

data_to_dendrogram = LRP_pd.T.values
Z = linkage(data_to_dendrogram, method='ward')

# Define the cutoff threshold to determine the clusters
cutoff = 3

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


dendro_colors = dn['color_list']
color_palette = dict(zip(np.unique(dendro_colors), sns.color_palette()))





# %%% UMAP
import umap
data_to_umap = LRP_pd.copy()

# Map the cluster labels to colors from the dendrogram
color_mapper = {1:'orange', 2:'green', 3:'red', 4:'purple', 5:'brown'}


             
reducer = umap.UMAP()
embedding = reducer.fit_transform(data_to_umap.T)
X_umap = pd.DataFrame(embedding , columns  = ['umap_1','umap_2'])

fig, ax = plt.subplots(figsize = (8,8))
sns.scatterplot(x = X_umap['umap_1'], y=X_umap['umap_2'], hue = cluster_labels, s=50,
                alpha=.7,
                linewidth=1,    # Size of the edge
                palette = color_mapper,
                edgecolor='white')
#scatter = ax.scatter(X_umap['umap_1'], X_umap['umap_2'], cmap='viridis', s=10)

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.axis('off')
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\umap_default.png')




sklearn.metrics.adjusted_rand_score




n_neighbors_list = [5, 10, 15, 20, 25, 30]
min_dist_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5]


for n_neighbors in n_neighbors_list:
    for min_dist in min_dist_list:
            
                        
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.05)
        embedding = reducer.fit_transform(data_to_umap.T)
        X_umap = pd.DataFrame(embedding , columns  = ['umap_1','umap_2'])
        
        fig, ax = plt.subplots(figsize = (8,8))
        sns.scatterplot(x = X_umap['umap_1'], y=X_umap['umap_2'], hue = cluster_labels, s=50,
                        alpha=.7,
                        linewidth=1,    # Size of the edge
                        palette = color_mapper,
                        edgecolor='white')
        #scatter = ax.scatter(X_umap['umap_1'], X_umap['umap_2'], cmap='viridis', s=10)
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.axis('off')
        ax.set_title(str(n_neighbors) + ' - ' +str(min_dist))
        plt.tight_layout()
        plt.savefig(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\umap_{}_{}.png'.format(n_neighbors, min_dist))


df_clinical_features_ = LRP_pd.T.reset_index().iloc[:,0:1].merge(df_clinical_features, left_on = 'index', right_on = 'bcr_patient_barcode').set_index('index')
column_colors = f.map_subtypes_to_col_color(df_clinical_features_)

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
axes = axes.flatten()  # Flatten axes to easily iterate over

# Iterate over subplots and color lists to create UMAP plots
i=0
for ax, colors in zip(axes, column_colors):
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors.values)
    ax.set_title(titles[i])
    i+=1
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle(pathway, fontsize=20)    
plt.tight_layout()


# %%% save cluster labels

cluster_lables_pd = pd.DataFrame()
cluster_lables_pd ['samples'] = list(LRP_pd.columns)
cluster_lables_pd ['cluster_labels'] = cluster_labels
cluster_lables_pd.to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\cluster_labels.csv')



# %% CLUSTERMAP PAM50

# load pam50 genes all

pam50_genes = pd.read_excel(os.path.join(path_to_save, 'pam50_genes_all.xlsx'))['genes'].to_list()
LRP_pd_temp = LRP_pd.iloc[:,:2].copy().reset_index()
LRP_pd_temp['source']  = LRP_pd_temp['index'].str.split(' - ', expand = True)[0]
LRP_pd_temp['target']= LRP_pd_temp['index'].str.split(' - ', expand = True)[1]
LRP_pd_temp['source'] = LRP_pd_temp['source'].str.split('_', expand=True)[0]
LRP_pd_temp['target'] = LRP_pd_temp['target'].str.split('_', expand=True)[0]

indices


indices = [index for index, row in LRP_pd_temp.iterrows() 
           if any(gene == row['source'] for gene in pam50_genes) 
           or any(gene == row['target'] for gene in pam50_genes)]

LRP_pd_pam50 = LRP_pd.iloc[indices, :-2]


g = sns.clustermap(LRP_pd_pam50, method = 'ward', vmin = 0.00, vmax = 0.05, cmap = 'jet', yticklabels = False, xticklabels = False,
               #mask = LRP_pd<0.01, 
               col_colors=column_colors)
g.ax_heatmap.set_xlabel('Samples')
g.ax_heatmap.set_ylabel('Interactions')

plt.savefig(os.path.join(path_to_save, 'clustermap_top{}_pam50all.png'.format(topn)))
fig,ax = plt.subplots()
ax.hist(LRP_pd_pam50.values.ravel(), bins = 30)


# load pam50 genes 37
pam50_genes = pd.read_excel(os.path.join(path_to_save, 'pam50_genes.xlsx'))['gene_new_name'].to_list()


indices = [index for index, row in LRP_pd_temp.iterrows() 
           if any(gene == row['source'] for gene in pam50_genes) 
           or any(gene == row['target'] for gene in pam50_genes)]

LRP_pd_pam50 = LRP_pd.iloc[indices, :-2]


g = sns.clustermap(LRP_pd_pam50, method = 'ward', vmin = 0.00, vmax = 0.05, cmap = 'jet', yticklabels = False, xticklabels = False,
               #mask = LRP_pd<0.01, 
               col_colors=column_colors)
g.ax_heatmap.set_xlabel('Samples')
g.ax_heatmap.set_ylabel('Interactions')

plt.savefig(os.path.join(path_to_save, 'clustermap_top{}_pam50.png'.format(topn)))
fig,ax = plt.subplots()
ax.hist(LRP_pd_pam50.values.ravel(), bins = 30)




# %% Pam50 and input genes overlap

pam50_genes = set(pd.read_excel(os.path.join(path_to_save, 'pam50_genes_all.xlsx'))['genes'].to_list())

input_genes = list(data_to_model.columns)
input_genes = set([x.split('_')[0] for x in input_genes])

set.intersection(pam50_genes, input_genes)








