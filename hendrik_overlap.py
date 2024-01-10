# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:39:25 2024

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
genes_cluster2 = pd.read_excel(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\nodes_cluster2.xlsx',engine = 'openpyxl', header = None)[0].to_list()
edges_cluster2 = pd.read_excel(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\edges_cluster2.xlsx',engine = 'openpyxl', header = None)[0].to_list()
samples_cluster2 = pd.read_excel(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\samples_cluster2.xlsx',engine = 'openpyxl', header = None)[0].to_list()

genes_cluster2 = [x.split('_')[0] for x in genes_cluster2]

# %% load Hendriks columns

path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\ReprRes_J_Lehtio_1411-v1.1.0\NBISweden-ReprRes_J_Lehtio_1411-652291d\Data'

df11 = pd.read_excel(os.path.join(path, 'Genelists-summary160606.xlsx'), engine = 'openpyxl', sheet_name = 0, dtype= 'str')
#df11['source'] = 'From proteinatlas n databases'
df12 = pd.read_excel(os.path.join(path, 'Genelists-summary160606.xlsx'), engine = 'openpyxl', sheet_name = 1, dtype= 'str')
#df12['source'] = 'signatures and mut'
df13 = pd.read_excel(os.path.join(path, 'Genelists-summary160606.xlsx'), engine = 'openpyxl', sheet_name = 2, dtype= 'str')
#df13['source'] = 'None-tumor'
df2 = pd.read_excel(os.path.join(path,'COSMIC_n_BC_drivers.xlsx'), engine = 'openpyxl')
#df2['source'] = 'COSMIC_n_BC_drivers'
df3 = pd.read_excel(os.path.join(path,'KEGG_n_Hallmark_genes_for_mRNA-protein_corr.xlsx'), engine = 'openpyxl')
#df3['source'] = 'KEGG_n_Hallmark_genes_for_mRNA-protein_corr'


df = pd.concat((df11, df12, df13, df2, df3),axis=1)

columns_ = len(df11.columns) * ['From proteinatlas n databases'] + len(df12.columns) * ['signatures and mut'] + len(df13.columns) * ['None-tumor'] + len(df2.columns) * ['COSMIC_n_BC_drivers'] + len(df3.columns) * ['KEGG_n_Hallmark_genes_for_mRNA-protein_corr']


dict_ = {}
for col in df.columns:
    
    col
    vals = df[col].dropna().values
    vals = list(vals)
    vals.sort()
    dict_[col] = vals


# %% compute overlap

dict_overlap = {}
for col in dict_.keys():
    
    vals = dict_[col]
    
    intersection = list(set(genes_cluster2).intersection(set(vals)))
    intersection .sort()
    overlap_size = len(intersection)
    overlap_ratio = overlap_size / len(genes_cluster2)
    
    overlap_ratio_to_col = overlap_size / len(vals)
    
    dict_overlap[col] = {}
    dict_overlap[col]['intersection'] = intersection
    dict_overlap[col]['overlap_size'] = overlap_size
    dict_overlap[col]['overlap_ratio'] = overlap_ratio
    dict_overlap[col]['columns_size'] = len(vals)
    dict_overlap[col]['overlap_ratio_to_col'] = overlap_ratio_to_col
    


overlap_df = pd.DataFrame(dict_overlap).T
overlap_df['Source'] = columns_
overlap_df.to_excel(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\overlap_hendrik_cluster2_genes.xlsx')

# %% compute overlap with expexp_lrp_mean_clusters

expexp_cluster_genes = pd.read_excel(os.path.join(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples', 'expexp_lrp_mean_clusters.xlsx'), engine = 'openpyxl', header =1)

dict_overlap = {}
for cluster_id in expexp_cluster_genes.columns[1:]:
    print(cluster_id)
    dict_overlap[cluster_id] = {}
    
    genes_cluster = expexp_cluster_genes[cluster_id].dropna().to_list()
    genes_cluster = [x.replace('_exp','') for x in genes_cluster]
    
    for col in dict_.keys():
        vals = dict_[col]
        
        intersection = list(set(genes_cluster).intersection(set(vals)))
        intersection.sort()
        overlap_size = len(intersection)
        overlap_ratio = overlap_size / len(genes_cluster)
        overlap_ratio_to_col = overlap_size / len(vals)
        
        dict_overlap[cluster_id][col] = {
            'intersection': intersection,
            'overlap_size': overlap_size,
            'overlap_ratio': overlap_ratio,
            'columns_size': len(vals),
            'overlap_ratio_to_col': overlap_ratio_to_col
        }

# Flatten the dictionary and create a DataFrame
rows = []
for cluster_id, nested_dict in dict_overlap.items():
    for col, data_dict in nested_dict.items():
        row = {
            'cluster_id': cluster_id,
            'column': col,
            **data_dict  # This unpacks all key-value pairs from data_dict into the row
        }
        rows.append(row)

overlap_df = pd.DataFrame(rows)
overlap_df.to_excel(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\overlap_hendrik_expexp.xlsx')

# %% compute overlap with exp_spearmancorr_clusters

exp_spearman_cluster_genes = pd.read_excel(os.path.join(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples', 'exp_spearmancorr_clusters.xlsx'), engine = 'openpyxl', header =1)

dict_overlap = {}
for cluster_id in exp_spearman_cluster_genes.columns[1:]:
    print(cluster_id)
    dict_overlap[cluster_id] = {}
    
    genes_cluster = expexp_cluster_genes[cluster_id].dropna().to_list()
    genes_cluster = [x.replace('_exp','') for x in genes_cluster]
    
    for col in dict_.keys():
        vals = dict_[col]
        
        intersection = list(set(genes_cluster).intersection(set(vals)))
        intersection.sort()
        overlap_size = len(intersection)
        overlap_ratio = overlap_size / len(genes_cluster)
        overlap_ratio_to_col = overlap_size / len(vals)
        
        dict_overlap[cluster_id][col] = {
            'intersection': intersection,
            'overlap_size': overlap_size,
            'overlap_ratio': overlap_ratio,
            'columns_size': len(vals),
            'overlap_ratio_to_col': overlap_ratio_to_col
        }

# Flatten the dictionary and create a DataFrame
rows = []
for cluster_id, nested_dict in dict_overlap.items():
    for col, data_dict in nested_dict.items():
        row = {
            'cluster_id': cluster_id,
            'column': col,
            **data_dict  # This unpacks all key-value pairs from data_dict into the row
        }
        rows.append(row)

overlap_df = pd.DataFrame(rows)
overlap_df.to_excel(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\overlap_hendrik_exp_spearman.xlsx')











