# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:26:39 2024

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

# %%

import community as community_louvain
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score

from cdlib import algorithms
import random
random.seed(42)
np.random.seed(42)

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

df_clinical_features = f.add_cluster2(df_clinical_features)


samples_groups = f.get_samples_by_group(df_clinical_features)
df_clinical_features=df_clinical_features.set_index('bcr_patient_barcode')

# %%
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\sensitivity'

topn = 1000

LRP_pd = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\LRP_individual_top{}.csv'.format(topn), index_col = 0)

edges_cluster2 = pd.read_excel(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\edges_cluster2.xlsx',engine = 'openpyxl', header = None)[0].to_list()
samples_cluster2 = pd.read_excel(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\samples_cluster2.xlsx',engine = 'openpyxl', header = None)[0].to_list()

column_colors = f.map_subtypes_to_col_color(df_clinical_features)
column_colors.append(df_clinical_features['cluster2'].map({1:'red',0:'lightgray'}))
column_colors = pd.DataFrame(column_colors).T
#column_colors = [x.values for x in column_colors]

edges_cluster2 = [list(np.sort(x.split('-'))[::-1]) for x in edges_cluster2]
edges_cluster2 = [' - '.join(x) for x in edges_cluster2]

is_cluster2_edge = pd.Series(LRP_pd_temp.index).isin(edges_cluster2).astype('int')
row_colors = pd.DataFrame(data = is_cluster2_edge.map({1:'red',0:'lightgray'}).values, index = LRP_pd_temp.index, columns = ['cluster2 edge'])

grid = np.arange(300,1000,50)

sensitivity_results = pd.DataFrame()

for topn_temp in grid:
    print(topn_temp)
    LRP_pd_temp = LRP_pd.copy()
    sensitivity_results_temp =  pd.DataFrame()
    
    for col in LRP_pd.columns:
        
        series_temp = LRP_pd_temp[col].copy()
        series_temp  = series_temp .sort_values( ascending = False)
        thres_temp = series_temp.values[:topn_temp][-1]
    
        LRP_pd_temp.loc[LRP_pd_temp[col] < thres_temp, col] = 0

    Z_row = linkage(LRP_pd_temp, method='ward')
    Z_col = linkage(LRP_pd_temp.T, method='ward')

    # CLUSTER MAP
    g = sns.clustermap(LRP_pd_temp,
                       row_linkage=Z_row, col_linkage=Z_col,
                       #method = 'ward', 
                       vmin = 0.00, vmax = 0.1,
                       cmap = 'jet', yticklabels = False, xticklabels = False,
                   mask = LRP_pd_temp==0, 
                   col_colors = column_colors,
                   row_colors = row_colors)
    g.ax_heatmap.set_xlabel('Samples')
    g.ax_heatmap.set_ylabel('Interactions')

    plt.savefig(os.path.join(path_to_save, 'clustermap_sensitivity_{}.png'.format(topn_temp)))
    
    # DENDROGRAM
    ## SAMPLES
    cutoff = 3
    fig, ax = plt.subplots(figsize=(12, 5))
    dn = dendrogram(
        Z_col, 
        color_threshold=cutoff, 
        above_threshold_color='gray', 
        #truncate_mode='lastp',  # show only the last p merged clusters
        #p=30,  # show only the last 12 merged clusters
        show_leaf_counts=True,  # show the number of samples in each cluster
        ax=ax
    )
    #plt.axhline(y=cutoff, color='r', linestyle='--')
    ax.set_ylabel('Distance')
    plt.savefig(os.path.join(path_to_save, 'dendrogram_samples_sensitivity_{}.png'.format(topn_temp)))
    
    cluster_labels = fcluster(Z_col, t=2, criterion='maxclust')
    samples_cluster2 = pd.Series(samples)[cluster_labels==2].to_list()
        
    ## Edges
    cutoff = 3
    fig, ax = plt.subplots(figsize=(12, 5))
    dn = dendrogram(
        Z_row, 
        color_threshold=cutoff, 
        above_threshold_color='gray', 
        #truncate_mode='lastp',  # show only the last p merged clusters
        #p=30,  # show only the last 12 merged clusters
        show_leaf_counts=True,  # show the number of samples in each cluster
        ax=ax
    )
    #plt.axhline(y=cutoff, color='r', linestyle='--')
    ax.set_ylabel('Distance')
    plt.savefig(os.path.join(path_to_save, 'dendrogram_edges_sensitivity_{}.png'.format(topn_temp)))
    
    cluster_labels = fcluster(Z_row, t=2, criterion='maxclust')
    edges_cluster2 = pd.Series(list(LRP_pd_temp.index))[cluster_labels==2]
    nodes_cluster2 = list(set(edges_cluster2.str.split('-',expand = True).melt()['value'].to_list()))
    nodes_cluster2 .sort()
    edges_cluster2 = list(edges_cluster2)
    edges_cluster2.sort()
    
    sensitivity_results_temp['topn'] = [topn_temp]
    sensitivity_results_temp['edges_cluster2'] = [edges_cluster2]
    sensitivity_results_temp['nodes_cluster2'] = [nodes_cluster2]
    sensitivity_results_temp['samples_cluster2'] = [samples_cluster2]
    
    sensitivity_results = pd.concat((sensitivity_results,sensitivity_results_temp))
    
# %% sensitivity based on kneepoint

sensitivity_results_knee = pd.DataFrame()

knee_add_grid = np.arange(-100, 101, 20)

for knee_add in knee_add_grid:
    print(knee_add)
    LRP_pd_temp = LRP_pd.copy()
    sensitivity_results_temp =  pd.DataFrame()
    
    for col in LRP_pd.columns:
        
        series_temp = LRP_pd_temp[col].copy()
        series_temp  = series_temp .sort_values( ascending = False)
                        
        kl = KneeLocator(series_temp.reset_index().index, series_temp.values, curve="convex", direction = 'decreasing', S=1, interp_method='polynomial')
        knee_x = int(kl.knee) + knee_add
        edges_nonzero = list(series_temp.index)[:knee_x]
                
        LRP_pd_temp.loc[~LRP_pd_temp.index.isin(edges_nonzero), col] = 0

    Z_row = linkage(LRP_pd_temp, method='ward')
    Z_col = linkage(LRP_pd_temp.T, method='ward')

    # CLUSTER MAP
    g = sns.clustermap(LRP_pd_temp,
                       row_linkage=Z_row, col_linkage=Z_col,
                       #method = 'ward', 
                       vmin = 0.00, vmax = 0.1,
                       cmap = 'jet', yticklabels = False, xticklabels = False,
                   mask = LRP_pd_temp==0, 
                   col_colors = column_colors,
                   row_colors = row_colors)
    g.ax_heatmap.set_xlabel('Samples')
    g.ax_heatmap.set_ylabel('Interactions')

    plt.savefig(os.path.join(path_to_save, 'clustermap_sensitivity_knee_{}.png'.format(knee_add)))
    
    # DENDROGRAM
    ## SAMPLES
    cutoff = 3
    fig, ax = plt.subplots(figsize=(12, 5))
    dn = dendrogram(
        Z_col, 
        color_threshold=cutoff, 
        above_threshold_color='gray', 
        #truncate_mode='lastp',  # show only the last p merged clusters
        #p=30,  # show only the last 12 merged clusters
        show_leaf_counts=True,  # show the number of samples in each cluster
        ax=ax
    )
    #plt.axhline(y=cutoff, color='r', linestyle='--')
    ax.set_ylabel('Distance')
    plt.savefig(os.path.join(path_to_save, 'dendrogram_samples_sensitivity_knee_{}.png'.format(knee_add)))
    
    cluster_labels = fcluster(Z_col, t=2, criterion='maxclust')
    samples_cluster2 = pd.Series(samples)[cluster_labels==2].to_list()
        
    ## Edges
    cutoff = 3
    fig, ax = plt.subplots(figsize=(12, 5))
    dn = dendrogram(
        Z_row, 
        color_threshold=cutoff, 
        above_threshold_color='gray', 
        #truncate_mode='lastp',  # show only the last p merged clusters
        #p=30,  # show only the last 12 merged clusters
        show_leaf_counts=True,  # show the number of samples in each cluster
        ax=ax
    )
    #plt.axhline(y=cutoff, color='r', linestyle='--')
    ax.set_ylabel('Distance')
    plt.savefig(os.path.join(path_to_save, 'dendrogram_edges_sensitivity_knee_{}.png'.format(knee_add)))
    
    cluster_labels = fcluster(Z_row, t=2, criterion='maxclust')
    edges_cluster2 = pd.Series(list(LRP_pd_temp.index))[cluster_labels==2]
    nodes_cluster2 = list(set(edges_cluster2.str.split('-',expand = True).melt()['value'].to_list()))
    nodes_cluster2 .sort()
    edges_cluster2 = list(edges_cluster2)
    edges_cluster2.sort()
    
    sensitivity_results_temp['knee_add'] = [knee_add]
    sensitivity_results_temp['mean_size'] = (LRP_pd_temp != 0).sum(axis=0).mean()
    sensitivity_results_temp['edges_cluster2'] = [edges_cluster2]
    sensitivity_results_temp['nodes_cluster2'] = [nodes_cluster2]
    sensitivity_results_temp['samples_cluster2'] = [samples_cluster2]
    
    sensitivity_results_knee = pd.concat((sensitivity_results_knee,sensitivity_results_temp))






















