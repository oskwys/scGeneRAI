# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 22:46:44 2023

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
pd.options.mode.chained_assignment = None  # default='warn'

# %% load data
# %%%  get input data

path_to_data = '/home/d07321ow/scratch/scGeneRAI/data/data_BRCA'
#path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'

path_to_save = '/home/d07321ow/scratch/results_LRP_BRCA/univariable'
#path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\networks'

#path_to_pathways = r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI\PATHWAYS'
path_to_pathways ='/home/d07321ow/scratch/scGeneRAI/PATHWAYS'


data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(path_to_data)

# %% get samples
path_to_lrp_results = '/home/d07321ow/scratch/results_LRP_BRCA/results'

#path_to_lrp_results = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\results'

samples = f.get_samples_with_lrp(path_to_lrp_results)

# %%% get sample goups

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

df_clinical_features = f.add_cluster0(df_clinical_features)

samples_groups = f.get_samples_by_group(df_clinical_features) 


# %% gene from pathways

genes_pathways = pd.read_csv(os.path.join(path_to_pathways, 'genes_pathways_pancanatlas_matched_cce.csv'))
genes_pathways_set = set(genes_pathways['cce_match'])

genes_pathways_dict = {}

for pathway in genes_pathways['Pathway'].unique():
    
    genes_pathways_dict[pathway] = genes_pathways.loc[genes_pathways['Pathway'] == pathway, 'Gene'].to_list()

      
# %% Univariable baseline
import pingouin as pg

for group in list(samples_groups.keys()):
    print(group)
    
    for subgroup in samples_groups[group].keys():
        print(subgroup, ' ---  Univariable baseline')
        
        samples_to_univariable = samples_groups[group][subgroup]
        
        data_temp = data_to_model[data_to_model.index.isin(samples_to_univariable)]
    
        for pathway in genes_pathways['Pathway'].unique():
            print(pathway)
            genes = genes_pathways_dict[pathway]
            
            # Spearman
            num_cols = [item for item in data_temp.columns if '_exp' in item or '_mut' in item]
            num_cols = [item for item in num_cols if any(gene in item for gene in genes)]
            print(' - Spearman')
            corrs_spearman_exp = f.get_correlation_r(data_temp, num_cols, method = 'spearman')
            
            
            
            # MWU
            cat_cols = [item for item in data_temp.columns if '_exp' not in item and '_mut' not in item]
            cat_cols = [item for item in cat_cols if any(gene in item for gene in genes)]
            print(' - MWU')
            pval_matrix , cles_matrix, mwu_stats  = f.get_mannwhitneyu_matrix(data_temp[cat_cols], data_temp[num_cols], iters=100)

            mwu_stats['test'] = 'mwu'
            
            
            #  Chi2

            cat_cols = [item for item in data_temp.columns if '_exp' not in item and '_mut' not in item]
            cat_cols = [item for item in cat_cols if any(gene in item for gene in genes)]
            print(' - Chi2')
            print(' -- ', cat_cols)
            from itertools import combinations

            chi2_res = pd.DataFrame()

            # Loop over all possible pairs of genes using itertools and compute the chi2 test
            for gene1, gene2 in combinations(cat_cols, 2):
                print('--- Chi2 genes: ', gene1, gene2)
                chi2_res_temp = pd.DataFrame()

                expected, observed, stats = pg.chi2_independence(data_temp, x=gene1,y=gene2)
                warning = ''
                if observed.min().min() < 5:
                    warning = '<5'
                # Perform chi2 test using pingouin
                chi2_res_temp = stats.iloc[0,:].T
                chi2_res_temp['source_gene'] = gene1
                chi2_res_temp['target_gene'] = gene2
                chi2_res_temp['warning'] = warning
                chi2_res_temp = chi2_res_temp.rename({'pval':'p-val'})
                # Store the result
                chi2_res = pd.concat((chi2_res, pd.DataFrame(chi2_res_temp).T))
                
            
            chi2_res = f.add_edge_colmn(chi2_res).drop(columns = ['test','lambda','dof','warning'])
            #chi2_res  = chi2_res[chi2_res ['warning'] == ''].reset_index(drop=True)   
            chi2_res['test'] = 'chi2'
            print(chi2_res)
                        
            univariable_res = pd.concat((corrs_spearman_exp, mwu_stats, chi2_res))
            
            univariable_res['width'] = 0
            univariable_res.loc[univariable_res['test'] == 'spearman', 'width'] = univariable_res.loc[univariable_res['test'] == 'spearman', 'r'].abs() / .7
            univariable_res.loc[univariable_res['test'] == 'chi2', 'width'] = univariable_res.loc[univariable_res['test'] == 'chi2', 'cramer'] / .7
            univariable_res.loc[univariable_res['test'] == 'mwu', 'width'] = (univariable_res.loc[univariable_res['test'] == 'mwu', 'CLES'] - 0.5).abs() * 2


            univariable_res.to_excel(os.path.join(path_to_save, 'univariable_res_{}_{}.xlsx'.format(pathway, subgroup)))

        