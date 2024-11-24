# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:36:12 2023

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

# %%
path1 = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\results_0\results'
path2 = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\results_0_v2\results'

G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\results

path1 = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\results_0\results'
path2 = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\results_0_v2\results'


files = os.listdir(path1)
files_2 = os.listdir(path2)


path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_model'

df_clinical_features = pd.read_csv( os.path.join(path_to_data, 'CCE_clinical_features.csv') )


# %%
subtype

s = ['BRCA', 'LGG', 'UCEC', 'LUAD', 'HNSC', 'PRAD', 'LUSC', 'STAD', 'COAD',
       'SKCM', 'CESC', 'SARC', 'OV', 'PAAD', 'ESCA', 'GBM', 'READ', 'UVM',
       'UCS', 'CHOL']

subtype = 'LUAD'

ids_ = df_clinical_features.loc[df_clinical_features['acronym'] == subtype, 'bcr_patient_barcode'].values
matching_files = [s for s in files if any(xs in s for xs in ids_)]




for file in matching_files:
        
    df_1 = pd.read_pickle(os.path.join(path1 , file), compression='infer', storage_options=None) .sort_values('LRP', ascending= False).reset_index(drop=True)
    df_1['sample'] = 1
    
    df_2 = pd.read_pickle(os.path.join(path2 , file), compression='infer', storage_options=None) .sort_values('LRP', ascending= False).reset_index(drop=True)
    df_2['sample'] = 2
    
    
a = df_1.head(1000)
b = df_2.head(1000)
c= pd.concat((a,b), axis=1)


