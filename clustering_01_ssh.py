# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:58:12 2023

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
#path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'

path_to_save = '/home/d07321ow/scratch/results_LRP_BRCA/umaps'
#path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\networks'


#path_to_lrp_results = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\results'
path_to_lrp_results = '/home/d07321ow/scratch/results_LRP_BRCA/results'

data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(path_to_data)

# %% get samples
samples = f.get_samples_with_lrp(path_to_lrp_results)
print('Samples: ', len(samples))
print('Samples: ', len(set(samples)))
# %%% get sample goups

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

samples_groups = f.get_samples_by_group(df_clinical_features)



# %%% load LRP data
lrp_dict = {}
lrp_files = []

for file in os.listdir(path_to_lrp_results):
    if file.startswith("LRP"):
        lrp_files.append(file)
        
        
n = len(lrp_files)  

#network_data = pd.DataFrame()
start_time = datetime.now()

for i in range(n):
    
            
    file_name = lrp_files[i]
    sample_name = file_name.split('_')[2]
    print(i, file_name)
    data_temp = pd.read_pickle(os.path.join(path_to_lrp_results , file_name), compression='infer', storage_options=None)

    data_temp = f.remove_same_source_target(data_temp)
        
    #data_temp = data_temp.sort_values('LRP', ascending= False).reset_index(drop=True)
    #data_temp = add_edge_colmn(data_temp)
    
    #network_data = pd.concat((network_data , data_temp))
        
    lrp_dict[sample_name] = data_temp
end_time = datetime.now()

print(end_time - start_time)




# %% PCA and UMAP for all


lrp_array = np.zeros((data_temp.shape[0], len(samples)))
 
 
for index, (sample_name, data) in enumerate(lrp_dict.items()):
    print(index, sample_name)
    lrp_array[:, index]  = data['LRP'].values


lrp_array_mean = np.round(  np.mean(lrp_array,axis=1), 5)
lrp_array_std = np.round( np.std(lrp_array,axis=1), 5)
lrp_array_median = np.round( np.median(lrp_array,axis=1), 5)
lrp_array_q1 = np.round( np.quantile(lrp_array, .25, axis=1), 5)
lrp_array_q3 = np.round( np.quantile(lrp_array, .75, axis=1), 5)

diff_ = lrp_array_q3 - lrp_array_q1




lrp_array_diff_pd = pd.DataFrame(data = np.array([diff_, data['source_gene'], data['target_gene']]).T,  index= data.reset_index(drop=True).index, columns = ['LRP_variability','source_gene','target_gene']   ).sort_values('LRP_variability', ascending = False).reset_index()
    
lrp_array_diff_pd['LRP_variability'].plot()


# get 10% of highest mean LRP
thresholds = [0.2, 0.1, 0.05, 0.01] # %
for threshold in thresholds:
    lrp_array_pd_topn = lrp_array_diff_pd.iloc[:int(lrp_array_diff_pd.shape[0] * threshold/100), :]

    lrp_array_pd_topn = lrp_array[lrp_array_pd_topn['index'].values, :].T

    
    import umap
    # Perform UMAP
    reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='euclidean')
    X_umap = reducer.fit_transform(lrp_array_pd_topn)
    
    X_umap = pd.DataFrame(X_umap, columns = ['umap_1', 'umap_2'])
    X_umap.to_csv(os.path.join(path_to_save, 'umap_thres{}.csv'.format(str(threshold).replae('.',''))))
    
    
    
    
    
    
    
#     # Plot
#     fig, ax = plt.subplots(figsize = (8,8))
#     scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], cmap='viridis', s=10)
    
    
#     ax.set_xlabel('UMAP 1')
#     ax.set_ylabel('UMAP 2')
#     ax.set_title('2D UMAP of Iris Dataset')
    
#     plt.show()
    
    

# # %%% PCA
# from sklearn.decomposition import PCA

# # Perform PCA
# pca = PCA(n_components=3)  # Reducing to 2 components
# X_pca = pca.fit_transform(lrp_array_pd_topn)

# # Print the explained variance ratio
# explained_variance = pca.explained_variance_ratio_
# print(f"Explained variance ratio: {explained_variance}")

# # Create a 2D scatter plot
# fig, ax = plt.subplots(figsize = (8,8))
# scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], cmap='viridis', s=10)

# ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]:.2%})')
# ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]:.2%})')
# ax.set_title('2D PCA of Iris Dataset')

# # Display the plot
# plt.show()


# import umap
# # Perform UMAP
# reducer = umap.UMAP(n_neighbors=15, n_components=3, min_dist=0.1, metric='euclidean')
# X_umap = reducer.fit_transform(lrp_array_pd_topn)

# # Plot
# fig, ax = plt.subplots(figsize = (8,8))
# scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], cmap='viridis', s=10)


# ax.set_xlabel('UMAP 1')
# ax.set_ylabel('UMAP 2')
# ax.set_title('2D UMAP of Iris Dataset')

# plt.show()



# import hdbscan


# # Step 1: Create a synthetic dataset
# # This creates a dataset with 3 centers (as an example)
# X = X_pca

# # Step 2: Fit HDBSCAN on the dataset
# # The minimum cluster size can be adjusted depending on your dataset
# clusterer = hdbscan.HDBSCAN(min_cluster_size = 15).fit(X)

# # Step 3: Extract the labels
# # Labels are the cluster each data point belongs to, noise points are labeled -1
# labels = clusterer.labels_

# # Step 4: Plot the results
# # Create a scatter plot assigning each cluster a unique color
# unique_labels = set(labels)
# cluster_results = pd.DataFrame()
# cluster_results['samples'] = samples
# cluster_results['labels'] = labels

# fig, ax = plt.subplots()

# # Set up a color palette (one color for each label, plus one for noise points labeled -1)
# colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

# # Plot each cluster using a separate color
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black is used for noise.
#         col = 'k'

#     class_member_mask = (labels == k)

#     xy = X[class_member_mask]
#     ax.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker=u'o', alpha=0.8, label=f'Cluster {k}')

# ax.set_title('HDBSCAN clustering')
# ax.legend(title='Clusters')
# plt.show()



# clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)