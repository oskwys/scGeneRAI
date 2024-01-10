# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:14:33 2024

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


path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\baselines'
# %% get samples

samples = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()

data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model')

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

df_clinical_features = f.add_cluster2(df_clinical_features)


samples_groups = f.get_samples_by_group(df_clinical_features)

# %% expression data

# %%%  clustermap
column_colors = f.map_subtypes_to_col_color(df_clinical_features)

column_colors.append(df_clinical_features['cluster2'].map({1:'red',0:'lightgray'}))



g = sns.clustermap(df_exp, method = 'ward', vmin = 0.00, cmap = 'jet', yticklabels = False, xticklabels = False,
               #mask = LRP_pd<0.01, 
               col_colors=column_colors)
g.ax_heatmap.set_xlabel('Samples')
g.ax_heatmap.set_ylabel('Gene expression')

plt.savefig(os.path.join(path_to_save, 'baselines_clustermap_expression_raw.png'))

fig,ax = plt.subplots()
ax.hist(df_exp.values.ravel(), bins = 30)

data_to_clustermap = data_to_model.iloc[:, 0:603]

row_colors = f.map_cluset2_genes(data_to_clustermap.columns).to_list()



g = sns.clustermap(data_to_clustermap.T, method = 'ward', vmin = 0.00, vmax = 10, cmap = 'jet', yticklabels = False, xticklabels = False,
               #mask = LRP_pd<0.01, 
               col_colors = column_colors,
               row_colors = row_colors)
g.ax_heatmap.set_xlabel('Samples')
g.ax_heatmap.set_ylabel('Genes')

plt.savefig(os.path.join(path_to_save, 'baselines_clustermap_expression_inputdata.png'))


# %%% UMAP
import umap
data_to_umap = data_to_model.iloc[:, 0:603]
#data_to_umap = df_exp

# Map the cluster labels to colors from the dendrogram
color_mapper = {1:'red', 0:'gray'}
cluster_labels = df_clinical_features['cluster2']

             
reducer = umap.UMAP()
embedding = reducer.fit_transform(data_to_umap.T)
X_umap = pd.DataFrame(embedding , columns  = ['umap_1','umap_2'])

fig, ax = plt.subplots(figsize = (8,8))
sns.scatterplot(x = X_umap['umap_1'], y=X_umap['umap_2'], hue = cluster_labels, 
                s=50,
                alpha=.7,
                linewidth=1,    # Size of the edge
                palette = color_mapper,
                edgecolor='white')
#scatter = ax.scatter(X_umap['umap_1'], X_umap['umap_2'], cmap='viridis', s=10)

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('UMAP 2D representation for 603 gene expressions')
#ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'baselines_umap_expression_inputdata.png'))


# %%% top correlations
data_to_corr = data_to_model.iloc[:, 0:603]

# %%%% pearson
corr_pearson = data_to_corr.corr(method = 'pearson').abs()


data_to_dendrogram = corr_pearson.values
Z = linkage(data_to_dendrogram, method='ward')

cutoff = 18

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

row_colors = f.map_cluset2_genes(data_to_corr.columns).to_list()

g = sns.clustermap(corr_pearson, method = 'ward', vmin = 0.00, vmax = 1, cmap = 'jet', yticklabels = False, xticklabels = False,
               row_linkage = Z, col_linkage = Z, 
               col_colors = row_colors,
               row_colors = row_colors)
g.ax_heatmap.set_xlabel('Genes')
g.ax_heatmap.set_ylabel('Genes')

plt.savefig(os.path.join(path_to_save, 'baselines_clustermap_expression_inputdata_pearson_corr.png'))



genes = list(corr_pearson.index)
df_genes = pd.DataFrame()
df_genes['genes'] = genes
df_genes['cluster'] = cluster_labels
df_genes.pivot(columns = 'cluster').to_excel(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\exp_spearmancorr_clusters.xlsx')




# %%%% pearson masked r < X
threshold = .5

row_colors = f.map_cluset2_genes(data_to_corr.columns).to_list()
corr_pearson_masked = corr_pearson.copy()
corr_pearson_masked [corr_pearson_masked < threshold] = 0
g = sns.clustermap(corr_pearson_masked, method = 'ward', vmin = 0.00, vmax = 1, cmap = 'jet', yticklabels = False, xticklabels = False,
               #mask = LRP_pd<0.01, 
               col_colors = row_colors,
               row_colors = row_colors)
g.ax_heatmap.set_xlabel('Genes')
g.ax_heatmap.set_ylabel('Genes')

plt.savefig(os.path.join(path_to_save, 'baselines_clustermap_expression_inputdata_pearson_corr_masked.png'))




# %%%% spearman
corr_spearman = data_to_corr.corr(method = 'spearman').abs()

row_colors = f.map_cluset2_genes(data_to_corr.columns).to_list()
#row_colors = column_colors

g = sns.clustermap(corr_spearman, method = 'ward', vmin = 0, vmax = 1, cmap = 'jet', yticklabels = False, xticklabels = False,
               #mask = LRP_pd<0.01, 
               col_colors = row_colors,
               row_colors = row_colors)
g.ax_heatmap.set_xlabel('Genes')
g.ax_heatmap.set_ylabel('Genes')

plt.savefig(os.path.join(path_to_save, 'baselines_clustermap_expression_inputdata_spearman_corr.png'))


# %%%% spearman masked r < X
threshold = .5

row_colors = f.map_cluset2_genes(data_to_corr.columns).to_list()
corr_spearman_masked = corr_spearman.copy()
corr_spearman_masked [corr_spearman_masked < threshold] = 0
g = sns.clustermap(corr_spearman_masked, method = 'ward', vmin = 0.00, vmax = 1, cmap = 'jet', yticklabels = False, xticklabels = False,
               #mask = LRP_pd<0.01, 
               col_colors = row_colors,
               row_colors = row_colors)
g.ax_heatmap.set_xlabel('Genes')
g.ax_heatmap.set_ylabel('Genes')

plt.savefig(os.path.join(path_to_save, 'baselines_clustermap_expression_inputdata_spearman_corr_masked.png'))

# %% LRP vs correlations

topn = 1000
LRP_pd = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\LRP_individual_top{}.csv'.format(topn), index_col = 0)

cluster2_samples = df_clinical_features.loc[df_clinical_features['cluster2'] == 1 ,'bcr_patient_barcode'].to_list()
nocluster2_samples = df_clinical_features.loc[df_clinical_features['cluster2'] == 0 ,'bcr_patient_barcode'].to_list()

LRP_mean_cluster2 = LRP_pd[cluster2_samples].mean(axis=1).reset_index()
LRP_mean_nocluster2 = LRP_pd[nocluster2_samples].mean(axis=1).reset_index()
LRP_mean = LRP_pd.mean(axis=1).reset_index()

corr_pearson_cluster2 = data_to_corr.loc[data_to_corr.index.isin(cluster2_samples), :].corr(method = 'pearson').abs()
corr_pearson_nocluster2 = data_to_corr.loc[data_to_corr.index.isin(nocluster2_samples), :].corr(method = 'pearson').abs()
corr_pearson = data_to_corr.corr(method = 'pearson').abs()

# %%% scatter plots lrp vs r
def get_lrp_vs_corr(LRP_mean, corr_df):
    
    LRP_mean = LRP_mean.rename(columns = {0:'LRP'})
    LRP_mean['pair'] = [' - '.join(list(np.sort(a))) for a in LRP_mean['index'].str.split(' - ')]#, expand = True)
    corr_melted = corr_df.reset_index().melt(id_vars="index", var_name="Column", value_name="r")
    corr_melted = corr_melted[corr_melted['index'] != corr_melted['Column']].reset_index(drop=True)
    corr_melted['pair'] = corr_melted['index'] + ' - ' + corr_melted['Column']
    corr_melted['pair'] = [' - '.join(list(np.sort(a))) for a in corr_melted['pair'].str.split(' - ')]
    corr_melted = corr_melted.drop_duplicates(subset = 'pair')
    corr_melted.sort_values('pair')
    
    lrp_r = LRP_mean[['LRP', 'pair']].merge(corr_melted[['r','pair']], on = 'pair', how = 'inner')
    return lrp_r

lrp_r_cluster2 = get_lrp_vs_corr(LRP_mean_cluster2, corr_pearson_cluster2)
lrp_r_nocluster2 = get_lrp_vs_corr(LRP_mean_nocluster2, corr_pearson_nocluster2)
lrp_r = get_lrp_vs_corr(LRP_mean, corr_pearson)

fig, ax = plt.subplots(figsize = (4,4))
sns.kdeplot(data = lrp_r, x = 'r', y='LRP',
                  thresh=.1,)
sns.scatterplot(data = lrp_r, x = 'r', y='LRP', #hue = cluster_labels, 
                s=10,
                alpha=.5,
                linewidth=1,    # Size of the edge
                #palette = color_mapper,
                edgecolor='white')
#scatter = ax.scatter(X_umap['umap_1'], X_umap['umap_2'], cmap='viridis', s=10)
ax.set_ylim([0, 0.08])
#ax.set_xlabel('UMAP 1')
#ax.set_ylabel('UMAP 2')
#ax.set_title('UMAP 2D representation for 603 gene expressions')
#ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'baselines_LRP_vs_corr.png'))


fig, axs = plt.subplots(1,2,figsize = (10,5))
sns.kdeplot(data = lrp_r_nocluster2, x = 'r', y='LRP',color= 'b',  ax=axs[0],
                  thresh=.1,)
sns.scatterplot(data = lrp_r_nocluster2, x = 'r', y='LRP', #hue = cluster_labels, 
                s=10,
                alpha=.5,color= 'b',                linewidth=1,    ax=axs[0],
                #palette = color_mapper,
                edgecolor='white')
sns.kdeplot(data = lrp_r_cluster2, x = 'r', y='LRP',color= 'r', ax=axs[1], 
                  thresh=.1)
sns.scatterplot(data = lrp_r_cluster2, x = 'r', y='LRP', #hue = cluster_labels, 
                s=10,
                alpha=.5,color= 'r',                linewidth=1,    ax=axs[1],
                #palette = color_mapper,
                edgecolor='white')
for ax in axs:
    ax.set_ylim([0, 0.08])
    ax.set_xlim([0,1])
axs[0].set_title('Samples not from cluster2')
axs[1].set_title('Samples from cluster2')
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'baselines_LRP_vs_corr_cluster2.png'))


# %%% stats

import pingouin as pg
x = lrp_r_cluster2['LRP']
y = lrp_r_cluster2['r']
pg.corr(x,y).round(3)

x = lrp_r_nocluster2['LRP']
y = lrp_r_nocluster2['r']
pg.corr(x,y).round(3)

x = lrp_r['LRP']
y = lrp_r['r']
pg.corr(x,y).round(3)


# %% r distribution
corr_melted = corr_pearson.reset_index().melt(id_vars="index", var_name="Column", value_name="r")
corr_melted = corr_melted[corr_melted['index'] != corr_melted['Column']].reset_index(drop=True)
corr_melted['pair'] = corr_melted['index'] + ' - ' + corr_melted['Column']
corr_melted['pair'] = [' - '.join(list(np.sort(a))) for a in corr_melted['pair'].str.split(' - ')]
corr_melted = corr_melted.drop_duplicates(subset = 'pair')
corr_melted = corr_melted.sort_values('r', ascending = False).reset_index(drop=True)

corr_melted['r'].plot()


