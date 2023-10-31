# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:53:47 2023

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

cluster0_samples = ['TCGA-B6-A0RT', 'TCGA-E9-A243', 'TCGA-AO-A0JC', 'TCGA-LL-A5YO',
       'TCGA-E9-A22D', 'TCGA-E2-A1B5', 'TCGA-AR-A1AX', 'TCGA-EW-A1OV',
       'TCGA-C8-A12V', 'TCGA-AR-A252', 'TCGA-BH-A0H5', 'TCGA-AQ-A7U7',
       'TCGA-A2-A0EQ', 'TCGA-AO-A128', 'TCGA-E2-A1II', 'TCGA-BH-A1F0',
       'TCGA-E9-A248', 'TCGA-A8-A08H', 'TCGA-EW-A1P7', 'TCGA-A2-A0CR',
       'TCGA-D8-A73U', 'TCGA-OL-A66I', 'TCGA-A2-A0CL', 'TCGA-E9-A2JT',
       'TCGA-A2-A25F', 'TCGA-A2-A0YK', 'TCGA-GM-A2DO', 'TCGA-AR-A1AJ',
       'TCGA-GM-A2DI', 'TCGA-PE-A5DE', 'TCGA-E2-A108', 'TCGA-AR-A0TS',
       'TCGA-AR-A0TT', 'TCGA-S3-AA15', 'TCGA-A2-A04Q', 'TCGA-A2-A0ST',
       'TCGA-AC-A2FB', 'TCGA-AR-A1AW', 'TCGA-A8-A0A7', 'TCGA-AR-A1AO',
       'TCGA-A2-A0EP', 'TCGA-BH-A209', 'TCGA-EW-A1IZ', 'TCGA-S3-AA17',
       'TCGA-E2-A1B6', 'TCGA-E9-A1NE', 'TCGA-BH-A0W5']

df_clinical_features['cluster0'] = 0
df_clinical_features.loc[df_clinical_features['bcr_patient_barcode'].isin(cluster0_samples), 'cluster0'] = 1

samples_groups = f.get_samples_by_group(df_clinical_features)



# %%
path_to_read = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\networks'
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\networks'


network_xlsx_files = files = [i for i in os.listdir(path_to_read) if ((i.startswith('network_')) and (i.endswith('.xlsx')))]

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
    
            
        df_neg = pd.read_excel(os.path.join(path_to_read , file_name_neg), engine = 'openpyxl',index_col=0)
        df_neg.columns = ['index_','edge', 'source_gene','target_gene','edge_type'] + [i + '_neg' for i in list(df_neg.columns[5:])]
        df_pos = pd.read_excel(os.path.join(path_to_read , file_name_pos), engine = 'openpyxl',index_col=0)
        df_pos.columns = ['index_','edge', 'source_gene','target_gene','edge_type'] + [i + '_pos' for i in list(df_pos.columns[5:])]
        
        
        df = pd.merge(df_neg, df_pos, on = ['edge', 'source_gene','target_gene','edge_type'])
        df['edge_'] = df['edge'].str.replace('_exp','').str.replace('_amp','').str.replace('_mut','').str.replace('_del','').str.replace('_exp','')
        topn = 100
        figsize = (30,20)
        #fig, ax = plt.subplots(1, 5, figsize = figsize)
        fig = plt.figure(figsize=figsize) 
                # Define the GridSpec
        gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 3])  # 2 rows, 4 columns
        
        # Now specify the location of each subplot in the grid
        ax_1 = fig.add_subplot(gs[:, 0])  # First column, all rows
        ax_2 = fig.add_subplot(gs[:, 1])  # Second column, all rows
        ax_3 = fig.add_subplot(gs[:, 2])  # Third column, all rows
        ax_4 = fig.add_subplot(gs[0, 3])    # Last column, first row
        ax_5 = fig.add_subplot(gs[1, 3])    # Last column, second row



        # mean
        df = df.sort_values('mean_neg', ascending=False).reset_index(drop=True)
        df_topn = df.iloc[:topn, :]
        
        df_topn_melt = df_topn.melt(id_vars=['edge', 'source_gene','target_gene','edge_type'], value_vars=['mean_pos','mean_neg'], var_name='group', value_name='average_LRP')
        
        ax=ax_1
        sns.barplot(df_topn_melt , y = 'edge', x = 'average_LRP', hue='group', palette = {'mean_neg':'gray','mean_pos':'red'}, ax=ax, orient = 'h')
        ax.set_yticklabels(df_topn[ 'edge_'])
        ax.grid()
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(df_topn_melt.loc[:df_topn.shape[0]-1, 'edge_type'])
        #ax.legend([group_pos,group_neg])
        ax.set_title('Mean')
        #plt.tight_layout()
        #plt.savefig(os.path.join(path_to_save, 'lrp_mean_{}_{}_{}.png'.format(pathway, key,topn)))
        
        
        # median
        df = df.sort_values('median_neg', ascending=False).reset_index(drop=True)
        df_topn = df.iloc[:topn, :]
        
        df_topn_melt = df_topn.melt(id_vars=['edge', 'source_gene','target_gene','edge_type'], value_vars=['median_pos','median_neg'], var_name='group', value_name='median_LRP')
        #fig, ax = plt.subplots(figsize = figsize)
        
        ax =ax_2
        sns.barplot(df_topn_melt , y = 'edge', x = 'median_LRP', hue='group',palette = {'median_neg':'gray','median_pos':'red'}, ax=ax, orient = 'h')
        ax.set_yticklabels(df_topn[ 'edge_'])
        ax.grid()
        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(df_topn_melt.loc[:df_topn.shape[0]-1, 'edge_type'])
        #ax.legend(labels = [group_pos,group_neg])
        ax.set_title('Median')
        #plt.tight_layout()
        #plt.savefig(os.path.join(path_to_save, 'lrp_median_{}_{}_{}.png'.format(pathway, key,topn)))
        
        # diff
        df['diff'] = df['mean_pos'] - df['mean_neg']
        df['diff_abs'] = df['diff'].abs()
        df = df.sort_values('diff_abs', ascending=False).reset_index(drop=True)
        neg_diff_q25 = np.quantile(df.loc[df['diff'] < 0, 'diff'].values, .25)
        pos_diff_q75 = np.quantile(df.loc[df['diff'] > 0, 'diff'].values, .75)
        
        df_topn = df.iloc[:topn, :]
        
        #fig, ax = plt.subplots(figsize = figsize)
        ax=ax_3
        node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}

        barplot_mapper = {'exp-exp':'gray', 'exp-mut':'red', 'amp-exp':'orange', 'del-exp':'green', 'exp-fus':'blue', 'mut-mut':'red',
               'amp-mut':'red', 'del-mut':'red', 'fus-mut':'red', 'amp-amp':'orange', 'amp-del':'orange', 'amp-fus':'orange',
               'del-del':'green', 'del-fus':'green', 'fus-fus':'blue'}
        ax.axvline(neg_diff_q25, linestyle = '--', color = 'magenta', label = 'q25')
        ax.axvline(pos_diff_q75, linestyle = '--', color = 'magenta', label = 'q75')                          
        sns.barplot(df_topn , y = 'edge', x = 'diff',hue = 'edge_type',  ax=ax, palette = barplot_mapper ,orient = 'h', dodge = False)
        ax.set_yticklabels(df_topn[ 'edge_'])
        ax.grid()

        ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
        ax.set_axisbelow(True)
        # ax2 = ax.twinx()
        # ax2.set_ylim(ax.get_ylim())
        # ax2.set_yticks(ax.get_yticks())
        # ax2.set_yticklabels(df_topn[ 'edge_type'])
        #plt.tight_layout()
        #plt.savefig(os.path.join(path_to_save, 'lrp_diff_{}_{}_{}.png'.format(pathway, key,topn)))
        # network
        
        network_pos = df.sort_values('mean_pos', ascending=False).reset_index(drop=True)
        network_pos = network_pos.iloc[:topn, :]
        network_pos['LRP'] = network_pos['mean_pos']
        
        network_neg = df.sort_values('mean_neg', ascending=False).reset_index(drop=True)
        network_neg = network_neg.iloc[:topn, :]
        network_neg['LRP'] = network_neg['mean_neg']
        
        
        name_to_save_pos = 'lrp_mean_network_{}_{}_{}_pos'.format(pathway, key, topn)
        name_to_save_neg = 'lrp_mean_network_{}_{}_{}_neg'.format(pathway, key, topn)
        title_pos = '{} {} {} POSITIVE'.format(pathway, key, topn)
        title_neg = '{} {} {} NEGATIVE'.format(pathway, key, topn)
        f.plot_network_(network_pos, path_to_save, 'kamada_kawai_layout', None, title_pos, name_to_save_pos, ax_4)
        f.plot_network_(network_neg, path_to_save, 'kamada_kawai_layout', None, title_neg, name_to_save_neg, ax_5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(path_to_save, 'gridplot_{}_{}_{}.png'.format(pathway, key,topn)))
        
       
        
       
        
# %% Grids for all genes

path_to_read = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\networks'
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\networks'


network_csv_files = files = [i for i in os.listdir(path_to_read) if ((i.startswith('network_')) and (i.endswith('.csv')))]


from matplotlib.gridspec import GridSpec

for key in samples_groups.keys():
    

    print(key)
    
    file_names_temp = [i for i in network_csv_files if key in i]
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

        
    df_neg = pd.read_csv(os.path.join(path_to_read , file_name_neg),index_col=0)
    df_neg.columns = ['index_','edge', 'source_gene','target_gene','edge_type'] + [i + '_neg' for i in list(df_neg.columns[5:])]
    df_pos = pd.read_csv(os.path.join(path_to_read , file_name_pos), index_col=0)
    df_pos.columns = ['index_','edge', 'source_gene','target_gene','edge_type'] + [i + '_pos' for i in list(df_pos.columns[5:])]
    
    
    df = pd.merge(df_neg, df_pos, on = ['edge', 'source_gene','target_gene','edge_type'])
    df['edge_'] = df['edge'].str.replace('_exp','').str.replace('_amp','').str.replace('_mut','').str.replace('_del','').str.replace('_exp','')
    topn = 100
    figsize = (30,20)
    #fig, ax = plt.subplots(1, 5, figsize = figsize)
    fig = plt.figure(figsize=figsize) 
            # Define the GridSpec
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 3])  # 2 rows, 4 columns
    
    # Now specify the location of each subplot in the grid
    ax_1 = fig.add_subplot(gs[:, 0])  # First column, all rows
    ax_2 = fig.add_subplot(gs[:, 1])  # Second column, all rows
    ax_3 = fig.add_subplot(gs[:, 2])  # Third column, all rows
    ax_4 = fig.add_subplot(gs[0, 3])    # Last column, first row
    ax_5 = fig.add_subplot(gs[1, 3])    # Last column, second row



    # mean
    df = df.sort_values('mean_neg', ascending=False).reset_index(drop=True)
    df_topn = df.iloc[:topn, :]
    
    df_topn_melt = df_topn.melt(id_vars=['edge', 'source_gene','target_gene','edge_type'], value_vars=['mean_pos','mean_neg'], var_name='group', value_name='average_LRP')
    
    ax=ax_1
    sns.barplot(df_topn_melt , y = 'edge', x = 'average_LRP', hue='group', palette = {'mean_neg':'gray','mean_pos':'red'}, ax=ax, orient = 'h')
    ax.set_yticklabels(df_topn[ 'edge_'])
    ax.grid()
    ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
    ax.set_axisbelow(True)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels(df_topn_melt.loc[:df_topn.shape[0]-1, 'edge_type'])
    #ax.legend([group_pos,group_neg])
    ax.set_title('Mean')
    #plt.tight_layout()
    #plt.savefig(os.path.join(path_to_save, 'lrp_mean_{}_{}_{}.png'.format(pathway, key,topn)))
    
    
    # median
    df = df.sort_values('median_neg', ascending=False).reset_index(drop=True)
    df_topn = df.iloc[:topn, :]
    
    df_topn_melt = df_topn.melt(id_vars=['edge', 'source_gene','target_gene','edge_type'], value_vars=['median_pos','median_neg'], var_name='group', value_name='median_LRP')
    #fig, ax = plt.subplots(figsize = figsize)
    
    ax =ax_2
    sns.barplot(df_topn_melt , y = 'edge', x = 'median_LRP', hue='group',palette = {'median_neg':'gray','median_pos':'red'}, ax=ax, orient = 'h')
    ax.set_yticklabels(df_topn[ 'edge_'])
    ax.grid()
    ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
    ax.set_axisbelow(True)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels(df_topn_melt.loc[:df_topn.shape[0]-1, 'edge_type'])
    #ax.legend(labels = [group_pos,group_neg])
    ax.set_title('Median')
    #plt.tight_layout()
    #plt.savefig(os.path.join(path_to_save, 'lrp_median_{}_{}_{}.png'.format(pathway, key,topn)))
    
    # diff
    df['diff'] = df['mean_pos'] - df['mean_neg']
    df['diff_abs'] = df['diff'].abs()
    df = df.sort_values('diff_abs', ascending=False).reset_index(drop=True)
    neg_diff_q25 = np.quantile(df.loc[df['diff'] < 0, 'diff'].values, .25)
    pos_diff_q75 = np.quantile(df.loc[df['diff'] > 0, 'diff'].values, .75)
    
    df_topn = df.iloc[:topn, :]
    
    #fig, ax = plt.subplots(figsize = figsize)
    ax=ax_3
    node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}

    barplot_mapper = {'exp-exp':'gray', 'exp-mut':'red', 'amp-exp':'orange', 'del-exp':'green', 'exp-fus':'blue', 'mut-mut':'red',
           'amp-mut':'red', 'del-mut':'red', 'fus-mut':'red', 'amp-amp':'orange', 'amp-del':'orange', 'amp-fus':'orange',
           'del-del':'green', 'del-fus':'green', 'fus-fus':'blue'}
    ax.axvline(neg_diff_q25, linestyle = '--', color = 'magenta', label = 'q25')
    ax.axvline(pos_diff_q75, linestyle = '--', color = 'magenta', label = 'q75')                          
    sns.barplot(df_topn , y = 'edge', x = 'diff',hue = 'edge_type',  ax=ax, palette = barplot_mapper ,orient = 'h', dodge = False)
    ax.set_yticklabels(df_topn[ 'edge_'])
    ax.grid()

    ax.legend(bbox_to_anchor=(.8, 0.00),loc='lower center' )
    ax.set_axisbelow(True)
    # ax2 = ax.twinx()
    # ax2.set_ylim(ax.get_ylim())
    # ax2.set_yticks(ax.get_yticks())
    # ax2.set_yticklabels(df_topn[ 'edge_type'])
    #plt.tight_layout()
    #plt.savefig(os.path.join(path_to_save, 'lrp_diff_{}_{}_{}.png'.format(pathway, key,topn)))
    # network
    
    network_pos = df.sort_values('mean_pos', ascending=False).reset_index(drop=True)
    network_pos = network_pos.iloc[:topn, :]
    network_pos['LRP'] = network_pos['mean_pos']
    
    network_neg = df.sort_values('mean_neg', ascending=False).reset_index(drop=True)
    network_neg = network_neg.iloc[:topn, :]
    network_neg['LRP'] = network_neg['mean_neg']
    
    
    name_to_save_pos = 'lrp_mean_network_{}_{}_{}_pos'.format(pathway, key, topn)
    name_to_save_neg = 'lrp_mean_network_{}_{}_{}_neg'.format(pathway, key, topn)
    title_pos = '{} {} {} POSITIVE'.format(pathway, key, topn)
    title_neg = '{} {} {} NEGATIVE'.format(pathway, key, topn)
    f.plot_network_(network_pos, path_to_save, 'kamada_kawai_layout', None, title_pos, name_to_save_pos, ax_4)
    f.plot_network_(network_neg, path_to_save, 'kamada_kawai_layout', None, title_neg, name_to_save_neg, ax_5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, 'gridplot_{}_{}.png'.format(key,topn)))
    
   
    
       
        
       
        
       
        
       
        
       
        
# %%
node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}
labels = list(node_color_mapper.keys())
colors = list(node_color_mapper.values())

# Plot legend
for label, color in zip(labels, colors):
    plt.plot([], [], 'o', color=color, label=label)

plt.legend(loc='best')

        
       
        
# %%
importlib.reload(f)
edges, path_to_save, layout=None, pos=None,  title='', name_to_save = 'network'

edges = network_pos
pos = None
layout = 'kamada_kawai_layout'
G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
degrees = np.array(list(nx.degree_centrality(G).values()))
degrees_norm = degrees / np.max(degrees)
# nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*100, node_size = network.degree)

if pos is not None:
    print('using POS')
else:
    if layout == None:
        pos = nx.spring_layout(G)

    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout == 'kamada_kawai_layout':
        pos = nx.kamada_kawai_layout(G)


widths = edges['LRP'] / edges['LRP'].max() * 10

edges_from_G = [i[0] + ' - ' + i[1] for i in list(G.edges)]
edge_colors = list(color_mapper(edges_from_G).values())

node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}
node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper).values

fig, ax = plt.subplots(figsize=(10, 10))
nx.draw(G, with_labels=False,
        node_color=node_colors,
        width=widths,
        pos=pos, # font_size=0,
        # cmap = colors,
        edge_color=edge_colors,
        ax=ax,
        # node_size = degrees,
        node_size=degrees_norm * 2000)

labels = {node: node.split('_')[0] for node in list(G.nodes)}
nx.draw_networkx_labels(G, pos, labels, font_size=25, ax=ax)

    ax.set_title(title)
    plt.tight_layout()


def color_mapper(input_list):
    # Define the color for each specific keyword
    keyword_to_color = {

        'del': 'green',  # assuming 'green' is the color for 'del'
        'amp': 'orange',  # assuming 'blue' is the color for 'amp'
        'fus': 'blue',  # assuming 'orange' is the color for 'fus'
        'mut': 'red',  # assuming 'red' is the color for 'mut'
    }

    # This dictionary will store the items with their corresponding color
    color_map = {}

    # Iterate through each item in the input list
    for item in input_list:
        # Default color if no keyword is matched, it's set to 'black' here
        color = 'gray'

        # Check each keyword to see if it exists in the item
        for keyword, assigned_color in keyword_to_color.items():
            if '_' + keyword in item:  # the underscore ensures we are checking, e.g., '_mut' and not just 'mut'
                color = assigned_color
                break  # if we found a keyword, we don't need to check the others for this item

        # Add the item and its color to the dictionary
        color_map[item] = color

    return color_map



def color_mapper(input_list):
    # Define the color for each specific keyword
    keyword_to_color = {

        'del': 'green',  # assuming 'green' is the color for 'del'
        'amp': 'orange',  # assuming 'blue' is the color for 'amp'
        'fus': 'blue',  # assuming 'orange' is the color for 'fus'
        'mut': 'red',  # assuming 'red' is the color for 'mut'
    }

    # This dictionary will store the items with their corresponding color
    color_map = {}

    # Iterate through each item in the input list
    for item in input_list:
        # Default color if no keyword is matched, it's set to 'black' here
        color = 'gray'

        # Check each keyword to see if it exists in the item
        for keyword, assigned_color in keyword_to_color.items():
            if '_' + keyword in item:  # the underscore ensures we are checking, e.g., '_mut' and not just 'mut'
                color = assigned_color
                break  # if we found a keyword, we don't need to check the others for this item

        # Add the item and its color to the dictionary
        color_map[item] = color

    return color_map






import networkx as nx
def plot_network_(edges, path_to_save, layout=None, pos=None,  title='', name_to_save = 'network'):

    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    # nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*100, node_size = network.degree)

    if pos is not None:
        print('using POS')
    else:
        if layout == None:
            pos = nx.spring_layout(G)

        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        elif layout == 'kamada_kawai_layout':
            pos = nx.kamada_kawai_layout(G)


    widths = edges['LRP'] / edges['LRP'].max() * 10

    edges_from_G = [i[0] + ' - ' + i[1] for i in list(G.edges)]
    edge_colors = list(color_mapper(edges_from_G).values())

    node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}
    node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper).values

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(G, with_labels=True,
            node_color=node_colors,
            width=widths,
            pos=pos, font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 2000)

    labels = {node: node.split('_')[0] for node in list(G.nodes)}
    nx.draw_networkx_labels(G, pos, labels, font_size=25, ax=ax)

    ax.set_title(title)
    plt.tight_layout()

    plt.savefig(os.path.join(path_to_save , name_to_save + '.svg'), format = 'svg')
    plt.savefig(os.path.join(path_to_save , name_to_save + '.png'), dpi = 300)



































edges = network_topn_pos

G_average = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
H = nx.Graph()
H.add_nodes_from(sorted(G_average.nodes(data=True)))
H.add_edges_from(G_average.edges(data=True))

pos = nx.shell_layout(H)

def plot_network_(edges, color_values, top_n, subtype, i, file, path_to_save, node_size=100, layout=None, pos = None, sample_id=''):
    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
    degrees = np.array(list(nx.degree_centrality(G).values())) 
    degrees_norm = degrees / np.max(degrees)
    #nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*100, node_size = network.degree)
    
        
    if pos is not None:
        print('using POS')
    else:
        if layout == None:
            pos = nx.spring_layout(G)
            
        elif layout== 'spectral':
            pos = nx.spectral_layout(G)
        elif layout== 'kamada_kawai_layout':
            pos = nx.kamada_kawai_layout(G)
            
    #colors = get_node_colors(G)
    #pos = nx.rescale_layout_dict(pos, scale = 100)
    
    nodes = list(G.nodes)
    values = color_values
    node_colors = []
    minima = np.min(values)
    maxima = np.max(values)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
    for v in values:
        node_colors.append(mapper.to_rgba(v))
    
    
    
    widths = edges['LRP'] / edges['LRP'].max() * 10
    
    edge_color_mapper = {'exp-exp':'gray', 'del-exp':'green', 'amp-exp':'orange', 'exp-mut':'red'} 
    edges_from_G = [i[0] + ' - ' + i[1] for i in list(G.edges)]
        
    #edges = edges.merge(  pd.Series(edges_from_G).reset_index(), left_on = 'edge', right_on = 0)
    edge_colors = list(color_mapper(edges_from_G).values())
    #edge_colors = edges
    
    node_color_mapper = {'exp':'gray', 'mut':'red', 'amp':'orange', 'del':'green','fus':'b'}
    node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper)
    #node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper)
    #cm.Greys((widths  - np.min(widths) )/ (np.max(widths) - np.min(widths)))
    
    fig,ax = plt.subplots(figsize=  (10,10))
    nx.draw(G, with_labels=False, 
            node_color=node_colors,
            width = widths,
            pos = pos,
            font_size = 0,
            #cmap = colors, 
            edge_color = edge_colors , 
            ax=ax,  
            #node_size = degrees,
            node_size = degrees_norm * 2000)
    
    labels = {node: node.split('_')[0] for node in list(G.nodes)}

    nx.draw_networkx_labels(G, pos, labels, font_size=25, ax=ax)

    ax.set_title(sample_id)
    plt.tight_layout()
    
    
    
    #plt.savefig(os.path.join(path_to_save , 'network_{}_{}_{}_{}.svg'.format(file, top_n, subtype, i)), format = 'svg')
    #plt.savefig(os.path.join(path_to_save , 'network_{}_{}_{}_{}.png'.format(file, top_n, subtype, i)), dpi = 300)
pos = nx.nx_agraph.graphviz_layout(G_average)
    




def color_mapper(input_list):
    # Define the color for each specific keyword
    keyword_to_color = {
        
        'del': 'green',    # assuming 'green' is the color for 'del'
        'amp': 'orange',     # assuming 'blue' is the color for 'amp'
        'fus': 'blue',   # assuming 'orange' is the color for 'fus'
        'mut': 'red',      # assuming 'red' is the color for 'mut'
    }

    # This dictionary will store the items with their corresponding color
    color_map = {}

    # Iterate through each item in the input list
    for item in input_list:
        # Default color if no keyword is matched, it's set to 'black' here
        color = 'gray'
        
        # Check each keyword to see if it exists in the item
        for keyword, assigned_color in keyword_to_color.items():
            if '_' + keyword in item:  # the underscore ensures we are checking, e.g., '_mut' and not just 'mut'
                color = assigned_color
                break  # if we found a keyword, we don't need to check the others for this item

        # Add the item and its color to the dictionary
        color_map[item] = color

    return color_map












