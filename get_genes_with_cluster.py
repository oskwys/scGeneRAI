# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 22:57:33 2023

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

# %% get genes_with_cluster


path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300'
genes_with_cluster75 = pd.read_excel(os.path.join(path, 'genes_with_cluster_threshold0.75.xlsx'.format(threshold)), engine = 'openpyxl', index_col=0)
genes_with_cluster75=genes_with_cluster75.rename(columns = {'cluster_label':'cluster_labels075'})
genes_with_cluster80 = pd.read_excel(os.path.join(path, 'genes_with_cluster_threshold0.8.xlsx'.format(threshold)), engine = 'openpyxl', index_col=0)
genes_with_cluster80=genes_with_cluster80.rename(columns = {'cluster_label':'cluster_labels080'})
genes_with_cluster85 = pd.read_excel(os.path.join(path, 'genes_with_cluster_threshold0.85.xlsx'.format(threshold)), engine = 'openpyxl', index_col=0)
genes_with_cluster85=genes_with_cluster85.rename(columns = {'cluster_label':'cluster_labels085'})
genes_with_cluster90 = pd.read_excel(os.path.join(path, 'genes_with_cluster_threshold0.9.xlsx'.format(threshold)), engine = 'openpyxl', index_col=0)
genes_with_cluster90=genes_with_cluster90.rename(columns = {'cluster_label':'cluster_labels090'})





genes_with_cluster = genes_with_cluster90.merge(genes_with_cluster85, on = 'genes', how='outer').merge(genes_with_cluster80, on = 'genes', how='outer').merge(genes_with_cluster75, on = 'genes', how='outer')
genes_with_cluster = genes_with_cluster.sort_values(['cluster_labels090','genes'])
genes_with_cluster.to_excel(os.path.join(path, 'genes_with_cluster.xlsx'))

# %% get sample


samples = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()

data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model')

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

df_clinical_features = f.add_cluster0(df_clinical_features)


threshold = 0.85
samples_with_community = pd.read_excel(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\samples_with_community_threshold{}.xlsx'.format(threshold), index_col=0)
samples_with_community = samples_with_community.rename(columns = {'has_community':'has_community_{}_cluster1'.format(threshold), 0:'bcr_patient_barcode'})
df_clinical_features_community = df_clinical_features.merge(samples_with_community)

df_clinical_features_community.to_excel(os.path.join(path, 'df_clinical_features_community.xlsx'))

df_clinical_features_community.groupby(['has_community_0.85_cluster1']).sum()
