# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 00:00:53 2024

@author: owysocky
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
from matplotlib.gridspec import GridSpec
import math
import textwrap
# %% samples
path_to_data = r'G:\My Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'
data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(path_to_data)

samples = f.get_samples_with_lrp(path_to_lrp_results)

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)
df_clinical_features = f.add_cluster2(df_clinical_features)

samples_groups = f.get_samples_by_group(df_clinical_features)



# %% analyse MWU LRP
path_to_data = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\mwu'
path_to_lrp_mean = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'
path_to_save = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\mwu\plots'
path_to_save_csv = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\lrp_top_interactions\data_to_plots'
import os

df_lrp_mean = pd.read_csv(os.path.join(path_to_lrp_mean, 'lrp_mean_list.csv'), index_col=0)
df_lrp_mean['LRP_mean'].plot()

q75 = df_lrp_mean['LRP_mean'].quantile(.95)


fig, ax = plt.subplots(figsize = (5,3))
sns.histplot(df_lrp_mean['LRP_mean'],ax=ax)
ax.set_xlim([0, None])
ax.axvline(q75, linestyle = '--', color = 'black')
plt.tight_layout()    
plt.savefig(os.path.join(path_to_save , 'histogram_LRPmean'.format(group) + '.pdf'), format = 'pdf')
plt.show()

fig, ax = plt.subplots(figsize = (5,3))
sns.histplot(df_lrp_mean['LRP_mean'], ax=ax)
ax.set_xscale('log')
ax.axvline(q75, linestyle = '--', color = 'black')
plt.tight_layout()    
plt.savefig(os.path.join(path_to_save , 'histogram_LRPmean_zoom'.format(group) + '.pdf'), format = 'pdf')
plt.show()

index_ = df_lrp_mean['LRP_mean'] > q75

for group in list(samples_groups.keys()):
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)
    
    df = pd.read_csv(os.path.join(path_to_data, 'mwu_edges_LRP_{}.csv'.format(group)), index_col=0)
    df_lrp_mean1 = pd.read_csv(os.path.join(path_to_lrp_mean, 'lrp_mean_{}.csv'.format(subgroup1)), index_col=0)
    df_lrp_mean2 = pd.read_csv(os.path.join(path_to_lrp_mean, 'lrp_mean_{}.csv'.format(subgroup2)), index_col=0)

    #df['temp'] = temp['edge'].values
    #df['LRP_mean'] = df_lrp_mean['LRP_mean']
    
    fig, ax = plt.subplots(figsize = (5,3))
    sns.histplot(data=df_lrp_mean1, x="LRP", color = 'red',ax=ax, alpha = .3, label = subgroup1)
    sns.histplot(data=df_lrp_mean2, x="LRP", color = 'blue',ax=ax, alpha = .3, label = subgroup2)
    ax.set_xlim([0, None])
    ax.legend()
    plt.tight_layout()    
    plt.savefig(os.path.join(path_to_save , 'histogram_LRPmean_{}'.format(group) + '.pdf'), format = 'pdf')
    plt.show()
    
    fig, ax = plt.subplots(figsize = (5,3))
    sns.histplot(data=df_lrp_mean1, x="LRP", color = 'red',ax=ax, alpha = .3, label = subgroup1)
    sns.histplot(data=df_lrp_mean2, x="LRP", color = 'blue',ax=ax, alpha = .3, label = subgroup2)
    ax.set_xscale('log')
    ax.legend()
    plt.tight_layout()    
    plt.savefig(os.path.join(path_to_save , 'histogram_LRPmean_{}_zoom'.format(group) + '.pdf'), format = 'pdf')
    plt.show()

    
# Create a figure and axis object
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='CLES', y='p-val', ax=ax)
ax.set_yscale('log')
ax.invert_yaxis()



sns.displot(data = df, x ='CLES')

fig, ax = plt.subplots()
sns.histplot(data = df, x ='p-val',ax=ax)
ax.set_xscale('log')



for group in list(samples_groups.keys()):
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)
    
    df = pd.read_csv(os.path.join(path_to_data, 'mwu_edges_LRP_{}.csv'.format(group)), index_col=0).loc[index_,:].reset_index(drop=True)


    fig, ax = plt.subplots(figsize = (5,3))
    sns.histplot(df['CLES'], ax=ax)

    fig, ax = plt.subplots(figsize = (5,3))
    sns.histplot(np.log(df['p-val']), ax=ax, bins = 1000)
    #ax.set_xlim([0,0.001])
    ax.set_xscale('log')
    ax.axvline(q75, linestyle = '--', color = 'black')
    plt.tight_layout()    
    plt.savefig(os.path.join(path_to_save , 'histogram_LRPmean_zoom'.format(group) + '.pdf'), format = 'pdf')
    plt.show()



    fig, ax = plt.subplots()
    sns.scatterplot(x=df_lrp_mean.loc[index_, 'LRP_mean'], y=df['CLES'], ax=ax, alpha = 0.2, s = 3)
    ax.axhline(0.001, linestyle = '--', color = 'black')
    ax.set_yscale('log')
    ax.invert_yaxis()
    



# %%% analyse nodes MWU LRP
path_to_data = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\mwu'
path_to_lrp_mean = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'
path_to_save = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\mwu\plots'
path_to_save_csv = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\lrp_top_interactions\data_to_plots'
import os


index_ = df_lrp_mean['LRP_mean'] > q75

for group in list(samples_groups.keys()):
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)
    
    df = pd.read_csv(os.path.join(path_to_data, 'mwu_sum_LRP_genes_{}.csv'.format(group)), index_col=0)
    df_lrp_mean1 = pd.read_csv(os.path.join(path_to_data, 'mwu_sum_LRP_genes_{}.csv'.format(subgroup1)), index_col=0)
    df_lrp_mean2 = pd.read_csv(os.path.join(path_to_data, 'mwu_sum_LRP_genes_{}.csv'.format(subgroup2)), index_col=0)

    df['temp'] = temp['edge'].values
    df['LRP_mean'] = df_lrp_mean['LRP_mean']
    
    fig, ax = plt.subplots(figsize = (5,3))
    sns.histplot(data=df_lrp_mean1, x="LRP", color = 'red',ax=ax, alpha = .3, label = subgroup1)
    sns.histplot(data=df_lrp_mean2, x="LRP", color = 'blue',ax=ax, alpha = .3, label = subgroup2)
    ax.set_xlim([0, None])
    ax.legend()
    plt.tight_layout()    
    plt.savefig(os.path.join(path_to_save , 'histogram_LRPmean_{}'.format(group) + '.pdf'), format = 'pdf')
    plt.show()
    
    fig, ax = plt.subplots(figsize = (5,3))
    sns.histplot(data=df_lrp_mean1, x="LRP", color = 'red',ax=ax, alpha = .3, label = subgroup1)
    sns.histplot(data=df_lrp_mean2, x="LRP", color = 'blue',ax=ax, alpha = .3, label = subgroup2)
    ax.set_xscale('log')
    ax.legend()
    plt.tight_layout()    
    plt.savefig(os.path.join(path_to_save , 'histogram_LRPmean_{}_zoom'.format(group) + '.pdf'), format = 'pdf')
    plt.show()

    
# Create a figure and axis object
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='CLES', y='p-val', ax=ax)
ax.set_yscale('log')
ax.invert_yaxis()



sns.displot(data = df, x ='CLES')

fig, ax = plt.subplots()
sns.histplot(data = df, x ='p-val',ax=ax)
ax.set_xscale('log')



for group in list(samples_groups.keys()):
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)
    
    df = pd.read_csv(os.path.join(path_to_data, 'mwu_edges_LRP_{}.csv'.format(group)), index_col=0)
    df_lrp_mean1 = pd.read_csv(os.path.join(path_to_lrp_mean, 'lrp_mean_{}.csv'.format(subgroup1)), index_col=0)
    df_lrp_mean2 = pd.read_csv(os.path.join(path_to_lrp_mean, 'lrp_mean_{}.csv'.format(subgroup2)), index_col=0)
    df['temp'] = temp['edge'].values
    df['LRP_mean'] = df_lrp_mean['LRP_mean']
    df['LRP_mean_subgroup1'] = df_lrp_mean1['LRP']
    df['LRP_mean_subgroup2'] = df_lrp_mean2['LRP']
    
#    df =df.loc[index_,:].reset_index(drop=True)

    fig, ax = plt.subplots(figsize = (5,3))
    sns.histplot(df['CLES'], ax=ax)

    fig, ax = plt.subplots(figsize = (5,3))
    sns.histplot(np.log(df['p-val']), ax=ax, bins = 1000)
    #ax.set_xlim([0,0.001])
    ax.set_xscale('log')
    ax.axvline(q75, linestyle = '--', color = 'black')
    plt.tight_layout()    
    plt.savefig(os.path.join(path_to_save , 'histogram_LRPmean_zoom'.format(group) + '.pdf'), format = 'pdf')
    plt.show()



    fig, ax = plt.subplots()
    sns.scatterplot(x=df_lrp_mean.loc[index_, 'LRP_mean'], y=df['CLES'], ax=ax, alpha = 0.2, s = 3)
    ax.axhline(0.001, linestyle = '--', color = 'black')
    ax.set_yscale('log')
    ax.invert_yaxis()
    
    # get the lowest p-val
    top_n = 10000
    topn_df = df.sort_values('p-val', ascending = True).iloc[:top_n,:]
    
    
fig, ax = plt.subplots(figsize = (8,8))
sns.scatterplot(x=df['LRP_mean_subgroup1'], y=df['LRP_mean_subgroup2'],ax=ax, alpha = 0.1, s=1)
ax.plot([0,.01],[0,.01])















