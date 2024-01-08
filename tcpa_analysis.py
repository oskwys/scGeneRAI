# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:14:29 2024

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

 #%%
 
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\TCPA_data\tmp'
tcpa_data = pd.read_csv(os.path.join(path_to_data, 'TCGA-BRCA-L4.csv'))
tcpa_data['bcr_patient_barcode'] = tcpa_data['Sample_ID'].str[:12]
# cce data

samples = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()
data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model')
df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)
df_clinical_features = f.add_cluster2(df_clinical_features)

samples_groups = f.get_samples_by_group(df_clinical_features)
#df_clinical_features=df_clinical_features.set_index('bcr_patient_barcode')

samples_intersection = set(df_clinical_features['bcr_patient_barcode']).intersection(set(tcpa_data['bcr_patient_barcode']))





















