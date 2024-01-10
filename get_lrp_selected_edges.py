# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:26:52 2023

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
#path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'
path_to_save = '/home/d07321ow/scratch/results_LRP_BRCA/networks'
#path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\networks'


#path_to_pathways = r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI\PATHWAYS'
path_to_pathways ='/home/d07321ow/scratch/scGeneRAI/PATHWAYS'

#path_to_lrp_results = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\results'
path_to_lrp_results = '/home/d07321ow/scratch/results_LRP_BRCA/results'

data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(path_to_data)

# %% get samples
samples = f.get_samples_with_lrp(path_to_lrp_results)
print('Samples: ', len(samples))
print('Samples: ', len(set(samples)))
# %%% get sample goups

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)
df_clinical_features = f.add_cluster2(df_clinical_features)



samples_groups = f.get_samples_by_group(df_clinical_features)


# %%
topn = 1000
#edges_to_select = pd.read_csv(os.path.join(path_to_save, 'edges_to_select_{}.csv'.format(topn)), index_col = 0)['edge'].to_list()
edges_to_select = pd.read_csv(os.path.join(path_to_save, 'edges_to_select_1000_noexpexp.csv'.format(topn)), index_col = 0)['edge'].to_list()


# %%% load LRP data
lrp_dict = {}
lrp_files = []

for file in os.listdir(path_to_lrp_results):
    if file.startswith("LRP"):
        lrp_files.append(file)
        
        
n = len(lrp_files)  

#network_data = pd.DataFrame()
start_time = datetime.now()

LRP_matrix = np.zeros((len(edges_to_select),n))
samples_to_pd = []
for i in range(n):
    
            
    file_name = lrp_files[i]
    sample_name = file_name.split('_')[2]
    print(i, file_name)
    data_temp = pd.read_pickle(os.path.join(path_to_lrp_results , file_name), compression='infer', storage_options=None)

    data_temp = f.remove_same_source_target(data_temp)
    if i==0:
        data_temp_0 = f.add_edge_colmn(data_temp)
        data_temp_0 = f.add_edge_colmn(data_temp_0)
        index_ = data_temp_0['edge'].isin(edges_to_select)
        edges = data_temp.loc[index_, 'edge'].values
        
        
    data_temp = data_temp[index_].reset_index(drop = True)
    LRP_matrix[:,i] = data_temp['LRP'].values
    samples_to_pd.append(sample_name)

end_time = datetime.now()

print(end_time - start_time)



LRP_pd = pd.DataFrame(LRP_matrix, columns = samples_to_pd, index = edges)
#LRP_pd.to_csv(os.path.join(path_to_save, 'LRP_individual_top{}.csv'.format(topn)))
LRP_pd.to_csv(os.path.join(path_to_save, 'LRP_individual_top1000_noexpexp.csv'))


#sns.clustermap(LRP_pd, method = 'ward')
