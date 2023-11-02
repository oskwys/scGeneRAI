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

def combine_into_set(row):
    list_ = list(set([row['source_gene'], row['target_gene']]))
    list_.sort()
    return str(list_[0]) +' - ' +str(list_[1])




# %%
data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model')

samples = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

df_clinical_features = f.add_cluster0(df_clinical_features)

samples_groups = f.get_samples_by_group(df_clinical_features)



# %% grid for pathways
path_to_read = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\networks'
path_to_read_univariable = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\univariable'
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\univariable'

path_to_save_excels = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\univariable\merged'


network_xlsx_files = files = [i for i in os.listdir(path_to_read) if ((i.startswith('network_')) and (i.endswith('.xlsx')))]
uni_xlsx_files = files = [i for i in os.listdir(path_to_read_univariable) if ((i.startswith('univariable_')) and (i.endswith('.xlsx')))]

unique_pathways = set([i.split('.')[0].split('_')[-1] for i in network_xlsx_files])

barplot_mapper = {'exp-exp':'gray', 'exp-mut':'red', 'amp-exp':'orange', 'del-exp':'green', 'exp-fus':'blue', 'mut-mut':'red',
       'amp-mut':'red', 'del-mut':'red', 'fus-mut':'red', 'amp-amp':'orange', 'amp-del':'orange', 'amp-fus':'orange',
       'del-del':'green', 'del-fus':'green', 'fus-fus':'blue'}




from matplotlib.gridspec import GridSpec

for key in list(samples_groups.keys())[-1:]:
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
            file_name_neg_uni = [i for i in file_names_temp if '_no_' in i][0]
            file_name_pos_uni = file_name_neg_uni.replace('no_','')

    
        # LOAD LRP    
        df_neg = pd.read_excel(os.path.join(path_to_read , file_name_neg), engine = 'openpyxl',index_col=0)
        df_neg.columns = ['index_','edge', 'source_gene','target_gene','edge_type'] + [i + '_neg' for i in list(df_neg.columns[5:])]
        df_pos = pd.read_excel(os.path.join(path_to_read , file_name_pos), engine = 'openpyxl',index_col=0)
        df_pos.columns = ['index_','edge', 'source_gene','target_gene','edge_type'] + [i + '_pos' for i in list(df_pos.columns[5:])]
        
        df_pos['edge_'] = df_pos['edge'].str.replace('_exp','').str.replace('_amp','').str.replace('_mut','').str.replace('_del','').str.replace('_exp','')
        df_neg['edge_'] = df_neg['edge'].str.replace('_exp','').str.replace('_amp','').str.replace('_mut','').str.replace('_del','').str.replace('_exp','')
        
        # LOAD UNIVARIABLE
        df_neg_uni = pd.read_excel(os.path.join(path_to_read_univariable , file_name_pos_uni), engine = 'openpyxl',index_col=0).reset_index(drop=True)
        df_pos_uni = pd.read_excel(os.path.join(path_to_read_univariable , file_name_neg_uni), engine = 'openpyxl',index_col=0).reset_index(drop=True)
                
        df_pos['edge_set'] = df_pos.apply(combine_into_set, axis=1)
        df_neg['edge_set'] = df_neg.apply(combine_into_set, axis=1)
        df_pos_uni['edge_set'] = df_pos_uni.apply(combine_into_set, axis=1)
        df_neg_uni['edge_set'] = df_neg_uni.apply(combine_into_set, axis=1)
                
        # compare POS
        df_pos_all = pd.merge(df_pos, df_pos_uni.drop(columns = ['edge', 'source_gene','target_gene','edge_type']), on = ['edge_set'])
        df_neg_all = pd.merge(df_neg, df_neg_uni.drop(columns = ['edge', 'source_gene','target_gene','edge_type']), on = ['edge_set'])
        # split by test / type of edge
        numeric_pos = df_pos_all['test'] == 'spearman'
        numcat_pos = df_pos_all['test'] == 'mwu'
        categorical_pos = df_pos_all['test'] == 'chi2'
        
        numeric_neg = df_neg_all['test'] == 'spearman'
        numcat_neg = df_neg_all['test'] == 'mwu'
        categorical_neg = df_neg_all['test'] == 'chi2'     
        
        
        df_neg_all_numeric = df_neg_all[numeric_neg].reset_index(drop=True)
        df_neg_all_numeric.loc[df_neg_all_numeric['p-val'] > 0.05, 'r'] = 0
        df_pos_all_numeric = df_pos_all[numeric_pos].reset_index(drop=True)
        df_pos_all_numeric.loc[df_pos_all_numeric['p-val'] > 0.05, 'r'] = 0
        
        df_neg_all_numcat = df_neg_all[numcat_neg].reset_index(drop=True)
        df_neg_all_numcat.loc[df_neg_all['p-val'] > 0.05, 'CLES'] = 0.5
        df_pos_all_numcat = df_pos_all[numcat_pos].reset_index(drop=True)
        df_pos_all_numcat.loc[df_pos_all_numcat['p-val'] > 0.05, 'CLES'] = 0.5
        
        df_neg_all_categorical = df_neg_all[categorical_neg].reset_index(drop=True)
        df_neg_all_categorical.loc[(df_neg_all_categorical['p-val'] > 0.05) | (df_neg_all_categorical['warning'] == '<5'), 'cramer'] = 0
        df_pos_all_categorical = df_pos_all[categorical_pos].reset_index(drop=True)
        df_pos_all_categorical.loc[(df_pos_all_categorical['p-val'] > 0.05) | (df_pos_all_categorical['warning'] == '<5'), 'cramer'] = 0
        
        
        df_neg_all_numeric.to_csv(os.path.join(path_to_save_excels,  'merged_numeric_'+ file_name_neg_uni.replace('.xlsx', '.csv')))
        df_pos_all_numeric.to_csv(os.path.join(path_to_save_excels,  'merged_numeric_'+ file_name_pos_uni.replace('.xlsx', '.csv')))
        
        df_neg_all_numcat.to_csv(os.path.join(path_to_save_excels,  'merged_numcat_'+ file_name_neg_uni.replace('.xlsx', '.csv')))
        df_pos_all_numcat.to_csv(os.path.join(path_to_save_excels,  'merged_numcat_'+ file_name_pos_uni.replace('.xlsx', '.csv')))
        
        df_neg_all_categorical.to_csv(os.path.join(path_to_save_excels,  'merged_categorical_'+ file_name_neg_uni.replace('.xlsx', '.csv')))
        df_pos_all_categorical.to_csv(os.path.join(path_to_save_excels,  'merged_categorical_'+ file_name_pos_uni.replace('.xlsx', '.csv')))
        
        
        #topn = 100
        figsize = (50,50)
        #fig, ax = plt.subplots(1, 5, figsize = figsize)
        fig = plt.figure(figsize=figsize) 
                # Define the GridSpec
        gs = GridSpec(3, 4, figure=fig, height_ratios=[4,2,1])  # 2 rows, 4 columns
        
        # Now specify the location of each subplot in the grid
        ax_1 = fig.add_subplot(gs[0, 0])  # First column, all rows
        ax_2 = fig.add_subplot(gs[0, 1])  # Second column, all rows
        ax_3 = fig.add_subplot(gs[0, 2])  # Third column, all rows
        ax_4 = fig.add_subplot(gs[0, 3])  # Second column, all rows
        #ax_5 = fig.add_subplot(gs[0, 4])  # Third column, all rows
        
        ax_6 = fig.add_subplot(gs[1, 0])  # First column, all rows
        ax_7 = fig.add_subplot(gs[1, 1])  # Second column, all rows
        ax_8 = fig.add_subplot(gs[1, 2])  # Third column, all rows
        ax_9 = fig.add_subplot(gs[1, 3])  # Second column, all rows
        
        ax_11 = fig.add_subplot(gs[2, 0])  # First column, all rows
        ax_12 = fig.add_subplot(gs[2, 1])  # Second column, all rows
        ax_13 = fig.add_subplot(gs[2, 2])  # Third column, all rows
        ax_14 = fig.add_subplot(gs[2, 3])  # Second column, all rows

        ##################
        # numeric
      
        df_pos_all_numeric = df_pos_all_numeric.sort_values('mean_pos', ascending = False).reset_index(drop=True)
        df_neg_all_numeric = df_neg_all_numeric.sort_values('mean_neg', ascending = False).reset_index(drop=True)
        
        ax=ax_1
        sns.barplot(df_pos_all_numeric , y = 'edge_set', x = 'mean_pos', hue = 'edge_type', palette = barplot_mapper, ax=ax, orient = 'h', dodge=False, errorbar=None)
        ax.set_yticklabels(df_pos_all_numeric[ 'edge_'])
        ax.grid()
        ax.set_ylabel(None)
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax.axvline(0.002, color = 'magenta', linestyle = '--')
        # ax2 = ax.twinx()
        # ax2.set_ylim(ax.get_ylim())
        # ax2.set_yticks(ax.get_yticks())
        # ax2.set_yticklabels(df_topn_melt.loc[:df_topn.shape[0]-1, 'edge_type'])
        #ax.legend([group_pos,group_neg])
        ax.set_title(key + ' POSITIVE\nLRP mean')
        
        
        ax=ax_2
        sns.barplot(df_pos_all_numeric , y = 'edge_set', x = 'r', hue = 'edge_type', palette = barplot_mapper, ax=ax, orient = 'h', dodge=False, errorbar=None)
        ax.set_yticklabels([])
        ax.set_xlim([-1,1])
        ax.set_ylabel(None)
        ax.grid()
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(df_pos_all_numeric.loc[:df_pos_all_numeric.shape[0]-1, 'edge_type'])
        ax.set_title(key + ' POSITIVE\nSpearman $r$')
        
        
        ax=ax_3
        sns.barplot(df_neg_all_numeric , y = 'edge_set', x = 'mean_neg', hue = 'edge_type', palette = barplot_mapper, ax=ax, orient = 'h', dodge=False, errorbar=None)
        ax.set_yticklabels(df_neg_all_numeric[ 'edge_'])
        ax.grid()
        ax.set_ylabel(None)
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax.axvline(0.002, color = 'magenta', linestyle = '--')
        # ax2 = ax.twinx()
        # ax2.set_ylim(ax.get_ylim())
        # ax2.set_yticks(ax.get_yticks())
        # ax2.set_yticklabels(df_topn_melt.loc[:df_topn.shape[0]-1, 'edge_type'])
        #ax.legend([group_pos,group_neg])
        ax.set_title(key + ' NEGATIVE\nLRP mean')
        
        
        ax=ax_4
        sns.barplot(df_neg_all_numeric , y = 'edge_set', x = 'r', hue = 'edge_type', palette = barplot_mapper, ax=ax, orient = 'h', dodge=False, errorbar=None)
        ax.set_yticklabels([])
        ax.set_xlim([-1,1])
        ax.set_ylabel(None)
        ax.grid()
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(df_neg_all_numeric.loc[:df_neg_all_numeric.shape[0]-1, 'edge_type'])
        ax.set_title(key + ' NEGATIVE\nSpearman $r$')
        
        ##################
        # MWU
      
        df_pos_all_numcat = df_pos_all_numcat.sort_values('mean_pos', ascending = False).reset_index(drop=True)
        df_neg_all_numcat = df_neg_all_numcat.sort_values('mean_neg', ascending = False).reset_index(drop=True)
        
        df_pos_all_numcat['CLES'] = df_pos_all_numcat['CLES'] - 0.5
        df_neg_all_numcat['CLES'] = df_neg_all_numcat['CLES'] - 0.5
        
        ax=ax_6
        sns.barplot(df_pos_all_numcat , y = 'edge_set', x = 'mean_pos', hue = 'edge_type', palette = barplot_mapper, ax=ax, orient = 'h', dodge=False, errorbar=None)
        ax.set_yticklabels(df_pos_all_numcat[ 'edge_'])
        ax.grid()
        ax.set_ylabel(None)
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax.axvline(0.002, color = 'magenta', linestyle = '--')
        # ax2 = ax.twinx()
        # ax2.set_ylim(ax.get_ylim())
        # ax2.set_yticks(ax.get_yticks())
        # ax2.set_yticklabels(df_topn_melt.loc[:df_topn.shape[0]-1, 'edge_type'])
        #ax.legend([group_pos,group_neg])
        ax.set_title(key + ' POSITIVE\nLRP mean')
        
        
        ax=ax_7
        sns.barplot(df_pos_all_numcat , y = 'edge_set', x = 'CLES', hue = 'edge_type', palette = barplot_mapper, ax=ax, orient = 'h', dodge=False, errorbar=None)
        ax.set_yticklabels([])
        ax.set_xlim([-.5, .5])
        ax.set_ylabel(None)
        ax.grid()
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(df_pos_all_numcat.loc[:df_pos_all_numcat.shape[0]-1, 'edge_type'])
        ax.set_title(key + ' POSITIVE\nMWU $CLES - 0.5$')
        
        
        ax=ax_8
        sns.barplot(df_neg_all_numcat , y = 'edge_set', x = 'mean_neg', hue = 'edge_type', palette = barplot_mapper, ax=ax, orient = 'h', dodge=False, errorbar=None)
        ax.set_yticklabels(df_neg_all_numcat[ 'edge_'])
        ax.grid()
        ax.set_ylabel(None)
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax.axvline(0.002, color = 'magenta', linestyle = '--')
        # ax2 = ax.twinx()
        # ax2.set_ylim(ax.get_ylim())
        # ax2.set_yticks(ax.get_yticks())
        # ax2.set_yticklabels(df_topn_melt.loc[:df_topn.shape[0]-1, 'edge_type'])
        #ax.legend([group_pos,group_neg])
        ax.set_title(key + ' NEGATIVE\nLRP mean')
        
        
        ax=ax_9
        sns.barplot(df_neg_all_numcat , y = 'edge_set', x = 'CLES', hue = 'edge_type', palette = barplot_mapper, ax=ax, orient = 'h', dodge=False, errorbar=None)
        ax.set_yticklabels([])
        ax.set_xlim([-.5, .5])
        ax.set_ylabel(None)
        ax.grid()
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(df_neg_all_numcat.loc[:df_neg_all_numcat.shape[0]-1, 'edge_type'])
        ax.set_title(key + ' NEGATIVE\nMWU $CLES$')
        
        
        
        ##################
        # CHI2
      
        df_pos_all_categorical = df_pos_all_categorical.sort_values('mean_pos', ascending = False).reset_index(drop=True)
        df_neg_all_categorical = df_neg_all_categorical.sort_values('mean_neg', ascending = False).reset_index(drop=True)
        
        ax=ax_11
        sns.barplot(df_pos_all_categorical , y = 'edge_set', x = 'mean_pos', hue = 'edge_type', palette = barplot_mapper, ax=ax, orient = 'h', dodge=False, errorbar=None)
        ax.set_yticklabels(df_pos_all_categorical[ 'edge_'])
        ax.grid()
        ax.set_ylabel(None)
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax.axvline(0.002, color = 'magenta', linestyle = '--')
        # ax2 = ax.twinx()
        # ax2.set_ylim(ax.get_ylim())
        # ax2.set_yticks(ax.get_yticks())
        # ax2.set_yticklabels(df_topn_melt.loc[:df_topn.shape[0]-1, 'edge_type'])
        #ax.legend([group_pos,group_neg])
        ax.set_title(key + ' POSITIVE\nLRP mean')
        
        
        ax=ax_12
        sns.barplot(df_pos_all_categorical , y = 'edge_set', x = 'cramer', hue = 'edge_type', palette = barplot_mapper, ax=ax, orient = 'h', dodge=False, errorbar=None)
        ax.set_yticklabels([])
        ax.set_xlim([0,1])
        ax.set_ylabel(None)
        ax.grid()
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(df_pos_all_categorical.loc[:df_pos_all_categorical.shape[0]-1, 'edge_type'])
        ax.set_title(key + ' POSITIVE\nChi2 $Cramer$')
        
        
        ax=ax_13
        sns.barplot(df_neg_all_categorical , y = 'edge_set', x = 'mean_neg', hue = 'edge_type', palette = barplot_mapper, ax=ax, orient = 'h', dodge=False, errorbar=None)
        ax.set_yticklabels(df_neg_all_categorical[ 'edge_'])
        ax.grid()
        ax.set_ylabel(None)
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax.axvline(0.002, color = 'magenta', linestyle = '--')
        # ax2 = ax.twinx()
        # ax2.set_ylim(ax.get_ylim())
        # ax2.set_yticks(ax.get_yticks())
        # ax2.set_yticklabels(df_topn_melt.loc[:df_topn.shape[0]-1, 'edge_type'])
        #ax.legend([group_pos,group_neg])
        ax.set_title(key + ' NEGATIVE\nLRP mean')
        
        
        ax=ax_14
        sns.barplot(df_neg_all_categorical , y = 'edge_set', x = 'cramer', hue = 'edge_type', palette = barplot_mapper, ax=ax, orient = 'h', dodge=False, errorbar=None)
        ax.set_yticklabels([])
        ax.set_xlim([0,1])
        ax.set_ylabel(None)
        ax.grid()
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(df_neg_all_categorical.loc[:df_neg_all_categorical.shape[0]-1, 'edge_type'])
        ax.set_title(key + ' NEGATIVE\nChi2 $Cramer$')
        
        plt.tight_layout()
        plt.savefig(os.path.join(path_to_save, 'gridplot_{}_{}.png'.format(key, pathway)), dpi = 200)
        plt.savefig(os.path.join(path_to_save, 'gridplot_{}_{}.pdf'.format(key, pathway)), format = 'pdf')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        