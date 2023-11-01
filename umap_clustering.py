# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:30:01 2023

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

cluster0_sample_index = 845#samples_cluster0_index[2]


# %% load umaps
from collections import Counter

umap_files = files = [f for f in os.listdir(path_to_data) if f.startswith('umap_')]
#cluster_results = pd.DataFrame()



for file in umap_files:
    
    threshold = file.split('_')[1]
    n_neighbors = file.split('_')[2]
    min_dist = file.split('_')[3].replace('.csv', '')
    
    X_umap = pd.read_csv(os.path.join(path_to_data, file), index_col = 0)
    
    
    fig, ax = plt.subplots(figsize = (8,8))
    scatter = ax.scatter(X_umap['umap_1'], X_umap['umap_2'], cmap='viridis', s=10)
    
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    title = 'Threshold ' + threshold + '\n' + 'n_neighbors ' + n_neighbors + '\n' + 'min_dist ' + min_dist
    ax.set_title(title)
    
    plt.savefig(os.path.join(path_to_save, 'UMAP_{}_{}_{}_justumap.png'.format(threshold, n_neighbors, min_dist)), dpi = 400)        
    
    import hdbscan


    # Step 1: Create a synthetic dataset
    # This creates a dataset with 3 centers (as an example)
    X = X_umap.values

    # Step 2: Fit HDBSCAN on the dataset
    # The minimum cluster size can be adjusted depending on your dataset
    
    #### !!!!!! UNCOMMENT if clusters needed
    # min_cluster_sizes = [ 10, 15, 20, 25, 50]
    
    # for min_cluster_size in min_cluster_sizes:
    #     clusterer = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size).fit(X)
    
    #     # Step 3: Extract the labels
    #     # Labels are the cluster each data point belongs to, noise points are labeled -1
    #     labels = clusterer.labels_
    #     label_x = labels[cluster0_sample_index]
    #     print('lable x: ', label_x)
    #     unique_, counts_ = np.unique(labels, return_counts = True)
        
    #     if label_x > 0:
    #         #if len(unique_)>2:
    #             #if counts_[1] < counts_[2]:
                
    #         index_x = labels == label_x
            
            
    #         index0 = labels == 0
    #         labels[index_x] = 0
    #         labels[index0] = label_x
    #         print('changed')
    #         label_x = labels[cluster0_sample_index]
    #         print('lable x: ', label_x)        
                
    
    
    #     # Step 4: Plot the results
    #     # Create a scatter plot assigning each cluster a unique color
    #     unique_labels = set(labels)
        
    #     #cluster_results['labels_{}_{}_{}_{}'.format(threshold, n_neighbors, min_dist, min_cluster_size)] = labels
    
    #     fig, ax =  plt.subplots(figsize = (8,8))
    
    #     # Set up a color palette (one color for each label, plus one for noise points labeled -1)
    #     colors = plt.cm.Paired(np.linspace(0, 1, len(unique_labels)))
    
    #     # Plot each cluster using a separate color
    #     for k, col in zip(unique_labels, colors):
    #         if k == -1:
    #             # Black is used for noise.
    #             col = 'k'
    
    #         class_member_mask = (labels == k)
    
    #         xy = X[class_member_mask]
    #         ax.scatter(xy[:, 0], xy[:, 1], s=10, c=[col], marker=u'o', alpha=0.8, label=f'Cluster {k}')
    
    #     title = 'HDBSCAN \n Threshold ' + threshold + '\n' + 'n_neighbors ' + n_neighbors + '\n' + 'min_dist ' + min_dist + '\n min_cluster_size ' +str( min_cluster_size)
    #     ax.set_title(title)
        
    #     ax.legend(title='Clusters')
    #     ax.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(path_to_save, 'UMAP_{}_{}_{}_clustersize{}.png'.format(threshold, n_neighbors, min_dist, min_cluster_size)), dpi = 400)        
    #### !!!!!! UNCOMMENT if clusters needed

    
    for group in samples_groups.keys():
        
        keys = list(samples_groups[group].keys())
        index1 = pd.Series(samples).isin(samples_groups[group][keys[0]]).values
        index2 = pd.Series(samples).isin(samples_groups[group][keys[1]]).values
        x1 = X[index1, :] 
        x2 = X[index2, :] 
        x3 = X[ ~ (index1 + index2), :]
        
        fig, ax = plt.subplots(figsize = (8,8))
        
        ax.scatter(x1[:,0], x1[:,1], label = keys[0], c = 'r')
        ax.scatter(x2[:,0], x2[:,1], label = keys[1], c = 'b')
        ax.scatter(x3[:,0], x3[:,1], label = 'other', c = 'gray')
        ax.legend()
        ax.axis('off')
        title = 'HDBSCAN \n Threshold ' + threshold + '\n' + 'n_neighbors ' + n_neighbors + '\n' + 'min_dist ' + min_dist + '\n Group ' + group
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(path_to_save, 'UMAP_{}_{}_{}_{}.png'.format(group, threshold, n_neighbors, min_dist)), dpi = 400)        

sns.clustermap(cluster_results)
    
# %%
import hdbscan
from collections import Counter

umap_files = files = [f for f in os.listdir(path_to_data) if f.startswith('umap_')]


cluster_results = pd.DataFrame()
for i, file in enumerate(umap_files):
    print(i, file)
    threshold = file.split('_')[1]
    n_neighbors = file.split('_')[2]
    min_dist = file.split('_')[3].replace('.csv', '')
    
    X_umap = pd.read_csv(os.path.join(path_to_data, file), index_col = 0)
    
    
    


    # Step 1: Create a synthetic dataset
    # This creates a dataset with 3 centers (as an example)
    X = X_umap.values

    # Step 2: Fit HDBSCAN on the dataset
    # The minimum cluster size can be adjusted depending on your dataset
    min_cluster_sizes = [ 10, 15, 20, 25, 50]
    
    for min_cluster_size in min_cluster_sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, allow_single_cluster=False).fit(X)
    
        # Step 3: Extract the labels
        # Labels are the cluster each data point belongs to, noise points are labeled -1
        labels = clusterer.labels_
        label_x = labels[cluster0_sample_index]
        print('lable x: ', label_x)
        unique_, counts_ = np.unique(labels, return_counts = True)
        
        if label_x > 0:
            #if len(unique_)>2:
                #if counts_[1] < counts_[2]:
                
            index_x = labels == label_x
            
            
            index0 = labels == 0
            labels[index_x] = 0
            labels[index0] = label_x
            print('changed')
            label_x = labels[cluster0_sample_index]
            print('lable x: ', label_x)        
        cluster_results['labels_{}_{}_{}_{}'.format(threshold, n_neighbors, min_dist, min_cluster_size)] = labels
        
        
        
        
        
        
        
sns.clustermap(cluster_results, vmax = 3, vmin=-1, cmap = 'jet', xticklabels=False)
plt.plot()    
sns.clustermap(cluster_results, vmax = 2, vmin=-1, cmap = 'jet', col_cluster = False)#, xticklabels=False)
plt.plot()
    

# only columns where 845 is in cluster 0
cols_to_select = cluster_results.columns[cluster_results.loc[845,:] == 0]
sns.clustermap(cluster_results.loc[:, cols_to_select], vmax = 2, vmin=-1, cmap = 'jet', col_cluster = False)
sns.clustermap(cluster_results.loc[:, cols_to_select], vmax = 2, vmin=-1, cmap = 'jet', col_cluster = True)



# what are the samples in the cluster 0 ?
being_0 = (cluster_results.loc[:, cols_to_select] == 0 ).sum(axis=1)
being_0.sort_values().reset_index()[0] .plot()
samples_0 = being_0[being_0 > 600]
samples_0.plot()
samples_cluster0 = np.array(samples)[list(samples_0.index)]

pd.DataFrame(samples_cluster0, columns = ['samples']).to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\samples_cluster0.csv')

# %%









unique_, counts_ = np.unique(labels, return_counts = True)

new_labels = [str(a) for a in list(unique_[1:][np.argsort(counts_[1:])][::-1])]

new_labels = {}

new_labels = reassign_labels(labels)
np.unique(new_labels, return_counts = True)


a = cluster_results.columns[-2]

samples_cluster0 = pd.Series(samples)[(cluster_results[a] == 0).values].values
samples_cluster0_index = list(pd.Series(samples)[(cluster_results[a] == 0).values].index)
cluster0_sample_index = 845#samples_cluster0_index[2]


label_x = labels[cluster0_sample_index]




















    
    
    
    
    