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

samples = pd.read_csv(r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)
df_clinical_features = f.add_cluster2(df_clinical_features)

samples_groups = f.get_samples_by_group(df_clinical_features)

for column in df_clinical_features.columns:
    print(f"Value counts for column '{column}':")
    print(df_clinical_features[column].value_counts())
    print("\n")

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
    



# %%% paths 
path_to_data = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\mwu'
path_to_lrp_mean = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'
path_to_save = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\mwu\plots'
path_to_save_csv = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\mwu'
import os
# %% analyse nodes MWU LRP

for group in list(samples_groups.keys()):
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)
    
    df_lrp = pd.read_csv(os.path.join(path_to_lrp_mean, 'LRP_sum_mean_{}.csv'.format(group)), index_col=0)
    df_lrp.columns
    df_lrp = df_lrp[[df_lrp.columns[1], df_lrp.columns[0], df_lrp.columns[2]]]
    
    col1 = df_lrp.columns[1]
    col2 = df_lrp.columns[2]
    
    # DISTRIBUTION PLOT
    fig, ax = plt.subplots(figsize = (5,3))
    sns.histplot(data=df_lrp, x=col1, color = 'red',ax=ax, alpha = .3, label = subgroup1, bins = 20, kde=True)
    sns.histplot(data=df_lrp, x=col2, color = 'blue',ax=ax, alpha = .3, label = subgroup2, bins = 20, kde=True)
    #ax.set_xlim([0, None])
    ax.legend()
    plt.tight_layout()    
    plt.savefig(os.path.join(path_to_save , 'histogram_LRP_sum_mean_{}'.format(group) + '.pdf'), format = 'pdf')
    plt.show()
    
    
    df = pd.read_csv(os.path.join(path_to_data, 'mwu_sum_LRP_genes_{}.csv'.format(group)), index_col=0)
    df['genes'] = df_lrp['gene']
    
    
       
    # PLOT HISTOGRAM CLES
    fig, ax = plt.subplots(figsize = (5,3))
    sns.histplot(df['CLES'], ax=ax, bins = 50)
    #ax.set_xlim([0,0.001])
    ax.set_title('CLES from MWU test - LRP sum mean - {}'.format(group))
    plt.tight_layout()    
    plt.savefig(os.path.join(path_to_save , 'histogram_LRP_sum_CLES_{}'.format(group) + '.pdf'), format = 'pdf')
    plt.show()



    # PLOT HISTOGRAM pval
    fig, ax = plt.subplots(figsize = (5,3))
    sns.histplot(df['p-val'], ax=ax, bins = 50)
    #ax.set_xlim([0,0.001])
    ax.set_title('pval from MWU test - LRP sum mean - {}'.format(group))
    ax.axvline(0.01, linestyle = '--', color = 'gray')
    plt.tight_layout()    
    plt.savefig(os.path.join(path_to_save , 'histogram_LRP_sum_pval_{}'.format(group) + '.pdf'), format = 'pdf')
    plt.show()


    df_lrp['type'] = df_lrp['gene'].str.split('_', expand=True)[1]
    colors = pd.DataFrame(df['p-val'].copy())
    colors['sig'] = 0
    colors.loc[colors['p-val'] < 0.01, 'sig'] = 1
    colors['color'] = colors['sig'].map({0:'blue', 1:'red'})
    
    
    # PLOT CLES vs LRP
    fig, axs = plt.subplots(1,2,figsize = (10,4))
    ax=axs[0]
    sns.scatterplot(x=df_lrp[col1], y=df['CLES'], hue = df_lrp['type'], ax=ax, alpha = 0.5, s = 10)
    ax.axhline(0.6, linestyle = '--', color = 'black', linewidth= 0.5)
    ax.axhline(0.4, linestyle = '--', color = 'black', linewidth= 0.5)
    ax.get_legend().remove()
    ax.set_title(subgroup1)
    
    ax=axs[1]
    sns.scatterplot(x=df_lrp[col2], y=df['CLES'], hue = df_lrp['type'], ax=ax, alpha = 0.5, s = 10)
    ax.axhline(0.6, linestyle = '--', color = 'black', linewidth= 0.5)
    ax.axhline(0.4, linestyle = '--', color = 'black', linewidth= 0.5)
    ax.set_title(subgroup2)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout() 
    plt.savefig(os.path.join(path_to_save , 'histogram_LRP_sum_vs_CLES_{}'.format(group) + '.pdf'), format = 'pdf')
    
    
    # PLOT LOWESS LRP COL1 vs COL2
    fig, ax = plt.subplots(figsize = (4,4))
    sns.regplot(x=df_lrp[col1], y=df_lrp[col2], ax=ax, line_kws={'color': 'red'}, lowess=True, scatter=False)
    scatter = sns.scatterplot(x=df_lrp[col1], y=df_lrp[col2], s=5, hue=colors['sig'], palette={0:'blue', 1:'red'}, alpha = .5)
    legend_labels = ['pval >= 0.01', 'pval < 0.01']
    handles, _ = scatter.get_legend_handles_labels()
    scatter.legend(handles, legend_labels, title=None)
    #sns.regplot(x=df_lrp[col1], y=df_lrp[col2], ax=ax, line_kws={'color': 'red'}, lowess=True, scatter_kws={'s': 3, 'cmap':colors['color'].values}, )
    #ax.set_xlim([0,0.001])
    ax.set_title('pval from MWU test\nLRP sum mean \n{}'.format(group))
    ax.plot([0.5,3],[0.5,3], linestyle = '--', color = 'gray')
    plt.tight_layout()    
    plt.savefig(os.path.join(path_to_save , 'lowess_LRP_sum_mean_{}'.format(group) + '.pdf'), format = 'pdf')
    plt.show()
    
    

    # PLOT PAIRPLOT expression
    df_exp_temp = pd.DataFrame(df_exp.mean(), columns = ['expression']).reset_index()
    df_exp_temp = df_exp_temp.rename(columns = {'index':'genes'})
    
    df_exp_temp = df_exp_temp.merge(df, on = 'genes', how = 'left').merge(df_lrp, left_on = 'genes', right_on = 'gene', how = 'inner')
    

    # Create a PairGrid instance
    g = sns.PairGrid(df_exp_temp[['expression', col1, col2, 'CLES']], diag_sharey=False)
    g.map_lower(sns.regplot, scatter_kws={'s': 3}, line_kws={'color': 'red'}, lowess=True)
    g.map_upper(sns.kdeplot)
    g.map_diag(sns.histplot)
    plt.tight_layout() 
    plt.savefig(os.path.join(path_to_save , 'pairplot_LRP_sum_vs_CLES_vs_expr_{}'.format(group) + '.pdf'), format = 'pdf')
     
    
    
# %%% identify top N in each group
df_lrp_all = pd.read_csv(os.path.join(path_to_lrp_mean, 'LRP_sum_mean.csv'), index_col=0)
df_lrp_all.columns = samples
top_n = 5
types = ['amp','del','mut','fus']
for group in list(samples_groups.keys())[:-1]:
    print(group)

  
        
    

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)
    if group == 'progesterone_receptor':
        subgroup_label1 = 'PR+'
        subgroup_label2 = 'PR-'
    elif group == 'estrogen_receptor':
        subgroup_label1 = 'ER+'
        subgroup_label2 = 'ER-'
    elif group == 'her2':
        subgroup_label1 = 'HER2+'
        subgroup_label2 = 'HER2-'       
    else:
        subgroup_label1 = subgroup1
        subgroup_label2 = subgroup2  
        
    df_lrp = pd.read_csv(os.path.join(path_to_lrp_mean, 'LRP_sum_mean_{}.csv'.format(group)), index_col=0)
    df_lrp.columns
    df_lrp = df_lrp[[df_lrp.columns[1], df_lrp.columns[0], df_lrp.columns[2]]]
    
    df_mwu = pd.read_csv(os.path.join(path_to_data, 'mwu_sum_LRP_genes_{}.csv'.format(group)), index_col=0)
    df_mwu['gene'] = df_lrp['gene']
    df_mwu['type'] = df_mwu['gene'].str.split('_', expand=True)[1]
    #df_mwu = df_mwu.sort_values(['type','CLES'],ascending = False)
    df_mwu = df_mwu.sort_values(['type','CLES'],ascending = True)
    
    df_mwu_topn = df_mwu.groupby('type').head(top_n).reset_index(drop=True)
    df_mwu_topn = df_mwu_topn[df_mwu_topn['p-val'] < 0.01]
    
    col1 = df_lrp.columns[1]
    col2 = df_lrp.columns[2]
    
    samples1 = samples_groups[group][subgroup1]
    samples2 = samples_groups[group][subgroup2]
    
    for type_ in types:
        temp = df_mwu_topn[df_mwu_topn['type'] == type_].sort_values('CLES')
        genes_temp = temp['gene'].to_list()
        x1 = df_lrp_all.loc[ genes_temp, samples1].T.reset_index()
        x1['subgroup'] = subgroup_label1
        
        x2 = df_lrp_all.loc[ genes_temp, samples2].T.reset_index()
        x2['subgroup'] = subgroup_label2
        
        x = pd.concat((x1,x2)).melt(id_vars = ['index','subgroup'])
        
        fig,ax = plt.subplots(figsize = (len(genes_temp) , 3))
        sns.violinplot(data=x, x="gene", y="value", hue="subgroup",  split=True, gap=.1, inner="quart",ax=ax)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylabel('$LRP_{sum}$')
        ax.set_xlabel(None)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        for i, gene in enumerate(genes_temp):
            cles_value = temp.loc[temp['gene'] == gene, 'CLES'].values[0]
            ax.text(i, ax.get_ylim()[0]+.02, f'{cles_value:.2f}', horizontalalignment='center', size='small', color='black', weight='semibold')

        plt.tight_layout() 
        plt.savefig(os.path.join(path_to_save , 'violinplot_LRP_sum_{}_{}_l'.format(group,type_) + '.pdf'), format = 'pdf')
         
        
# select only CLES > X
CLES_threshold = 0.75        
for group in list(samples_groups.keys())[:-1]:
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)
       
    df_lrp = pd.read_csv(os.path.join(path_to_lrp_mean, 'LRP_sum_mean_{}.csv'.format(group)), index_col=0)
    df_lrp.columns
    df_lrp = df_lrp[[df_lrp.columns[1], df_lrp.columns[0], df_lrp.columns[2]]]
    
    df_mwu = pd.read_csv(os.path.join(path_to_data, 'mwu_sum_LRP_genes_{}.csv'.format(group)), index_col=0)
    df_mwu['gene'] = df_lrp['gene']
    df_mwu['type'] = df_mwu['gene'].str.split('_', expand=True)[1]
    df_mwu = df_mwu.sort_values(['type','CLES'],ascending = False)
    
    df_mwu_topn = df_mwu.groupby('type').head(50).reset_index(drop=True)
    df_mwu_topn = df_mwu_topn[df_mwu_topn['p-val'] < 0.01]
    df_mwu_topn = df_mwu_topn[df_mwu_topn['CLES'] >= CLES_threshold]
    
    col1 = df_lrp.columns[1]
    col2 = df_lrp.columns[2]
    
    samples1 = samples_groups[group][subgroup1]
    samples2 = samples_groups[group][subgroup2]
    
    for type_ in types:
        temp = df_mwu_topn[df_mwu_topn['type'] == type_].sort_values('CLES')
        if temp.shape[0] > 0:
            genes_temp = temp['gene'].to_list()
            x1 = df_lrp_all.loc[ genes_temp, samples1].T.reset_index()
            x1['subgroup'] = subgroup1
            
            x2 = df_lrp_all.loc[ genes_temp, samples2].T.reset_index()
            x2['subgroup'] = subgroup2
            
            x = pd.concat((x1,x2)).melt(id_vars = ['index','subgroup'])
            
            fig,ax = plt.subplots(figsize = (len(genes_temp) , 3))
            sns.violinplot(data=x, x="gene", y="value", hue="subgroup",  split=True, gap=.1, inner="quart",ax=ax)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_ylabel('$LRP_{sum}$')
            ax.set_xlabel(None)
            for i, gene in enumerate(genes_temp):
                cles_value = temp.loc[temp['gene'] == gene, 'CLES'].values[0]
                ax.text(i, ax.get_ylim()[0], f'{cles_value:.2f}', horizontalalignment='center', size='small', color='black', weight='semibold')
    
            plt.tight_layout() 
            plt.savefig(os.path.join(path_to_save , 'violinplot_LRP_sum_{}_{}_cles{}'.format(group,type_,str(CLES_threshold).replace('.','')) + '.pdf'), format = 'pdf')
             
                           
    
        
        
        
    
    df_lrp['type'] = df_lrp['gene'].str.split('_', expand=True)[1]
    
    df_lrp = df_lrp.sort_values(['type','LRP_sum_mean_cluster2_pos'])
    
    df_lrp.groupby('type').head(10)
    
# %% LRP sum mean clustermap

df_lrp = pd.read_csv(os.path.join(path_to_lrp_mean, 'LRP_sum_mean.csv'), index_col=0)
df_lrp.columns = samples

# col colors
df_clinical_features_ = df_lrp.T.reset_index().iloc[:,0:1].merge(df_clinical_features, left_on = 'index', right_on = 'bcr_patient_barcode').set_index('index')
column_colors = pd.DataFrame(f.map_subtypes_to_col_color(df_clinical_features_)).T
column_colors = column_colors.rename(columns = {'Estrogen_receptor':'ER','Progesterone_receptor':'PR'})

# row colors
def map_color(string):
    if 'mut' in string:
        return 'red'
    elif 'amp' in string:
        return 'blue'
    elif 'exp' in string:
        return 'gray'
    elif 'del' in string:
        return 'green'
    else:
        return 'black'
    
row_colors = df_lrp.reset_index().iloc[:,0].apply(map_color)
row_colors.index = df_lrp.index
row_colors = row_colors.rename(index = 'Type')

data_to_dendrogram = df_lrp.T.values
Z_col = linkage(data_to_dendrogram, method='ward')
Z_row = linkage(data_to_dendrogram.T, method='ward')


g=sns.clustermap(df_lrp, vmax = 10, method = 'ward', cmap = 'jet',col_linkage=Z_col, row_linkage=Z_row,
                 yticklabels = False, xticklabels = False,
                 col_colors=column_colors,
                 row_colors=row_colors)
g.ax_heatmap.set_xlabel('Samples')
g.ax_heatmap.set_ylabel('Genes')


# Define the cutoff threshold to determine the clusters
cutoff = 150
fig, ax = plt.subplots(figsize=(12, 5))
dn = dendrogram(
    Z, 
    color_threshold=cutoff, 
    above_threshold_color='gray', 
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=30,  # show only the last 12 merged clusters
    show_leaf_counts=True,  # show the number of samples in each cluster
    ax=ax
)

# Draw a line to signify the cutoff on the dendrogram
plt.axhline(y=cutoff, color='r', linestyle='--')

# Label the axes
ax.set_xlabel('Cluster Size/ID')
ax.set_ylabel('Distance')
 
cluster_labels = fcluster(Z, t=cutoff, criterion='distance')
unique_labels, counts = np.unique(cluster_labels, return_counts=True)


sns.clustermap(df, method = 'ward', mask = (df == 0) , cmap = 'jet', 
               row_linkage = Z, col_linkage = Z,
               yticklabels = False, xticklabels = False,)


genes = list(df.index)
df_genes = pd.DataFrame()
df_genes['genes'] = genes
df_genes['cluster'] = cluster_labels


# %%
    , df_mut, df_amp, df_del, df_fus,
    
    
    
    
    
    
    
    
    
    
    
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















