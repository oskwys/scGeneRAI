# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:06:02 2023

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
%matplotlib inline

# %%
data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model')

samples = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

df_clinical_features = f.add_cluster0(df_clinical_features)

samples_groups = f.get_samples_by_group(df_clinical_features)



# %% grid for pathways
path_to_read = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\networks'
path_to_read_univariable = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\univariable'
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\networks'


network_xlsx_files = files = [i for i in os.listdir(path_to_read) if ((i.startswith('network_')) and (i.endswith('.xlsx')))]
uni_xlsx_files = files = [i for i in os.listdir(path_to_read_univariable) if ((i.startswith('univariable_')) and (i.endswith('.xlsx')))]

unique_pathways = set([i.split('.')[0].split('_')[-1] for i in network_xlsx_files])
from matplotlib.gridspec import GridSpec

for key in samples_groups.keys():
    for pathway in unique_pathways:
    
        print(key, pathway)
        
        file_names_temp = [i for i in network_xlsx_files if ((pathway in i) & (key in i))]
        try:
            file_name_pos = [i for i in file_names_temp if 'pos' in i][0]
            file_name_neg = [i for i in file_names_temp if 'neg' in i][0]
             
            group_pos = key + '_pos'
            group_neg = key + '_neg'
            
            
        except:
            file_name_neg = [i for i in file_names_temp if '_no_' in i][0]
            file_name_pos = file_name_neg.replace('no_','')
            
            group_pos = 'TNBC'
            group_neg = 'no_TNBC'
            
            
        print(file_name_pos, file_name_neg)
    
        file_names_temp = [i for i in uni_xlsx_files if ((pathway in i) & (key in i))]
        try:
            file_name_pos_uni = [i for i in file_names_temp if 'pos' in i][0]
            file_name_neg_uni = [i for i in file_names_temp if 'neg' in i][0]
            
        except:
            file_name_pos_uni = [i for i in file_names_temp if '_no_' in i][0]
            file_name_neg_uni = file_name_neg.replace('no_','')

    
        # LOAD LRP    
        df_neg = pd.read_excel(os.path.join(path_to_read , file_name_neg), engine = 'openpyxl',index_col=0)
        df_neg.columns = ['index_','edge', 'source_gene','target_gene','edge_type'] + [i + '_neg' for i in list(df_neg.columns[5:])]
        df_pos = pd.read_excel(os.path.join(path_to_read , file_name_pos), engine = 'openpyxl',index_col=0)
        df_pos.columns = ['index_','edge', 'source_gene','target_gene','edge_type'] + [i + '_pos' for i in list(df_pos.columns[5:])]
        
        # LOAD UNIVARIABLE
        df_neg_uni = pd.read_excel(os.path.join(path_to_read_univariable , file_name_pos_uni), engine = 'openpyxl',index_col=0)
        df_pos_uni = pd.read_excel(os.path.join(path_to_read_univariable , file_name_neg_uni), engine = 'openpyxl',index_col=0)
        
                
        # compare POS
        df_pos_all = pd.merge(df_pos, df_pos_uni, on = ['edge', 'source_gene','target_gene','edge_type'])
        df_neg_all = pd.merge(df_neg, df_neg_uni, on = ['edge', 'source_gene','target_gene','edge_type'])
        # split by test / type of edge
        numeric_pos = df_pos_all['test'] == 'spearman'
        numcat_pos = df_pos_all['test'] == 'mwu'
        categorical_pos = ~(numeric_pos + numcat_pos)   
        
        numeric_neg = df_neg_all['test'] == 'spearman'
        numcat_neg = df_neg_all['test'] == 'mwu'
        categorical_neg = ~(numeric_neg + numcat_neg)          
        
        topn = 100
        figsize = (30,30)
        #fig, ax = plt.subplots(1, 5, figsize = figsize)
        fig = plt.figure(figsize=figsize) 
                # Define the GridSpec
        gs = GridSpec(3, 5, figure=fig, height_ratios=[3,1,1])  # 2 rows, 4 columns
        
        # Now specify the location of each subplot in the grid
        ax_1 = fig.add_subplot(gs[:, 0])  # First column, all rows
        ax_2 = fig.add_subplot(gs[:, 1])  # Second column, all rows
        ax_3 = fig.add_subplot(gs[:, 2])  # Third column, all rows
        ax_4 = fig.add_subplot(gs[:, 3])  # Second column, all rows
        ax_5 = fig.add_subplot(gs[:, 4])  # Third column, all rows
        ax_6 = fig.add_subplot(gs[0, 5])    # Last column, first row
        ax_7 = fig.add_subplot(gs[1, 5])    # Last column, second row

        # numeric
        df_pos_all_num = df_pos_all[numeric_pos].sort_values('mean_pos').reset_index(drop=True)
        df_neg_all_num = df_neg_all[numeric_neg].sort_values('mean_neg').reset_index(drop=True)
        
        
        
        
        
        df = pd.merge(df_neg, df_pos, on = ['edge', 'source_gene','target_gene','edge_type'])
        df['edge_'] = df['edge'].str.replace('_exp','').str.replace('_amp','').str.replace('_mut','').str.replace('_del','').str.replace('_exp','')
        


        # mean
        df = df.sort_values('mean_neg', ascending=False).reset_index(drop=True)
        df_topn = df.iloc[:topn, :]
        
        df_topn_melt = df_topn.melt(id_vars=['edge', 'source_gene','target_gene','edge_type'], value_vars=['mean_pos','mean_neg'], var_name='group', value_name='average_LRP')
        