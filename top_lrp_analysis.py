# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 08:27:22 2024

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

# %%
path_to_data = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\lrp_top_interactions'
path_to_save = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\lrp_top_interactions\plots'
path_to_save_csv = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\lrp_top_interactions\data_to_plots'
import os
stat = 'mean'
csv_files = [f for f in os.listdir(path_to_data) if f.endswith('.csv') and stat in f]

dfs = {}
for file in csv_files:
    print(file)
    dfs[file] = pd.read_csv(os.path.join(path_to_data,file), index_col = 0 )


edge_types = []
groups = []
stats = []
for file in csv_files:
    edge_type = file.replace('.csv', '').split('_')[2]
    stat= file.replace('.csv', '').split('_')[-1]
    group = '_'.join(file.replace('.csv', '').split('_')[3:-1])
    
    edge_types.append(edge_type)
    groups.append(group)

edge_types = list(set(edge_types))
add_names = list(set(add_names))


# %% gene from pathways

path_to_pathways = r'C:\Users\owysocky\Documents\GitHub\scGeneRAI\PATHWAYS'

genes_pathways = pd.read_csv(os.path.join(path_to_pathways, 'genes_pathways_pancanatlas_matched_cce.csv'))
genes_pathways_set = set(genes_pathways['cce_match'])

genes_pathways_dict = {}

for pathway in genes_pathways['Pathway'].unique():
    
    genes_pathways_dict[pathway] = genes_pathways.loc[genes_pathways['Pathway'] == pathway, 'Gene'].to_list()

    
    

def compute_overlap(X, genes_pathways_dict):
    # Process X by splitting each gene by '_' and taking the first part
    processed_X = {gene.split('_')[0] for gene in X}
    
    # Dictionary to store overlap results
    overlap_genes = {}
    overlap_n = {}
    overlap_ratio = {}
    
    # Calculate overlap for each pathway
    for pathway, genes in genes_pathways_dict.items():
        overlap = processed_X.intersection(genes)
        overlap_genes[pathway] = overlap
        overlap_n[pathway] = len(overlap)
        overlap_ratio[pathway] = len(overlap) / len(genes)
    
    return overlap_genes, overlap_n, overlap_ratio




# %% load hendriks columns

path = r'G:\My Drive\SAFE_AI\CCE_DART\KI_dataset\ReprRes_J_Lehtio_1411-v1.1.0\NBISweden-ReprRes_J_Lehtio_1411-652291d\Data'

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
    
    
# %% plot gridplots


quantile_ = 0.99

top_nodes_dict = {}
overlap_df_all_keys = pd.DataFrame()
for key, df in dfs.items():
    # --- Your existing code to process df and create graph G ---
    print(key)
    top_x_percent = df['LRP'].quantile(quantile_)
    top_x_percent
    edges = df[df['LRP'] >= top_x_percent].copy()
    edges = df.iloc[:250].copy()

    combined_genes = pd.concat([edges['source_gene'], edges['target_gene']])
    unique_values = combined_genes.nunique()
    unique_values_list = combined_genes.unique()
    
    
    # creat Graph G
    edges['LRP_norm'] = edges['LRP'] / edges['LRP'].max()
    
    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP_norm')
        
    # get overlaps with PATHWAYs
    genes_to_dict = list(G.nodes)
    genes_to_dict .sort()
    top_nodes_dict[key] = {}
    top_nodes_dict[key]['genes_nodes'] = {'genes': genes_to_dict}
    overlap_genes_dict, overlap_n_dict, overlap_ratio_dict = compute_overlap(genes_to_dict, genes_pathways_dict)
    
    top_nodes_dict[key]['overlap_genes'] = overlap_genes_dict
    top_nodes_dict[key]['overlap_n'] = overlap_n_dict
    top_nodes_dict[key]['overlap_ratio'] = overlap_ratio_dict
    
    # get overlap with Hendrik
    genes_ = [x.split('_')[0] for x in genes_to_dict]
    dict_overlap = {}
    for col in dict_.keys():
        vals = dict_[col]
        
        intersection = list(set(genes_).intersection(set(vals)))
        intersection.sort()
        overlap_size = len(intersection)
        overlap_ratio = overlap_size / len(genes_)
        overlap_ratio_to_col = overlap_size / len(vals)
        
        dict_overlap[col] = {
            'intersection': intersection,
            'overlap_size': overlap_size,
            'overlap_ratio': overlap_ratio,
            'columns_size': len(vals),
            'overlap_ratio_to_col': overlap_ratio_to_col
        }

    # Flatten the dictionary and create a DataFrame
    rows = []
    #for cluster_id, nested_dict in dict_overlap.items():
    for col, data_dict in dict_overlap.items():
        row = {
            
            'column': col,
            **data_dict  # This unpacks all key-value pairs from data_dict into the row
        }
        rows.append(row)
    overlap_df = pd.DataFrame(rows)
    overlap_df['key'] = key
    overlap_df_all_keys = pd.concat((overlap_df_all_keys , overlap_df))

    # G properties     
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = edges['LRP'] / edges['LRP'].max() 

    edges_from_G = [i[0] + ' - ' + i[1] for i in list(G.edges)]
    edge_colors = list(f.color_mapper(edges_from_G).values())
    edge_colors = plt.cm.jet(widths)  # 'viridis' is a colormap, you can choose any

    node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}
    node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper)

    pos = nx.spring_layout(G, weight='LRP_norm')
    pos = nx.circular_layout(G)
    
    # Increase this value to move labels further away from nodes
    label_shift = 0.1
    pos_labels = {node: (coordinates[0] * (1 + label_shift), coordinates[1] * (1 + label_shift)) for node, coordinates in pos.items()}
    processed_labels = {node: node.split('_')[0] for node in G.nodes()}


    
    # # --- Word Cloud Preparation ---
    # all_node_labels = [node for node in G.nodes]
    # all_node_labels = [x.split('_')[0] for x in all_node_labels ]
    # text = ' '.join(all_node_labels)
   
    # --- Word Cloud Preparation ---
    # all_edge_labels = [list(edge) for edge in G.edges]
    # all_edge_labels = [item for sublist in all_edge_labels for item in sublist]
    # all_edge_labels = [x.split('_')[0] for x in all_edge_labels ]
    # text = ' '.join(all_edge_labels)


    # Creating the figure and GridSpec layout
    fig = plt.figure(figsize=(30, 15))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[3,1,1], height_ratios=[1,2,2])

    # Plot 1: Network Graph in full height on the left
    ax1 = fig.add_subplot(gs[:, 0])
    nx.draw(G, with_labels=False,
            node_color=node_colors,
            width=widths,
            pos=pos,
            edge_color=edge_colors,
            alpha=0.5,
            ax=ax1,
            node_size=degrees_norm * 200)
    #nx.draw_networkx_labels(G, pos_labels, labels=processed_labels, font_size=8)  # Draw processed labels with adjusted font size
    # Rotate labels according to their position in the circular layout

    # Factor to move labels outside the circle
    label_shift = 1.05
    # Rotate and position labels outside the circle
    for node, (x, y) in pos.items():
        angle = math.atan2(y, x)
        rotation = math.degrees(angle)
        text_rotation = rotation if x > 0 else rotation + 180
        x_shifted, y_shifted = x * label_shift, y * label_shift
        plt.text(x_shifted, y_shifted, s=node.split('_')[0], rotation=text_rotation, 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=8)
  
    ls = list(node_color_mapper.keys())
    cl = list(node_color_mapper.values())
    for label, color in zip(ls, cl):
        ax1.plot([], [], 'o', color=color, label=label)
    ax1.legend(title = 'Nodes', loc='best')

    ax1.set_title(f'Network Graph for {key}')

    # Plot 2: Distribution Plot
    ax2 = fig.add_subplot(gs[0, 1])
    sns.distplot(df['LRP'], hist=False, kde=True, kde_kws={'linewidth': 3}, ax=ax2)
    #ax2.axvline(df['LRP'].quantile(quantile_), color='r', linestyle='--')
    ax2.axvline(edges['LRP'].min(), color='r', linestyle='--')
    ax2.set_title(f'Distribution plot for {key}')
    ax2.set_xlabel('LRP')
    ax2.set_ylabel('Density')

    # Plot 3: Bar Plot for Top 20 Edges
    top_20_edges = edges.nlargest(20, 'LRP')[::-1]
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.barh( top_20_edges['source_gene'] + ' - ' + top_20_edges['target_gene'], top_20_edges['LRP'],)
    ax3.set_title('Top 20 Edges by LRP')
    #ax3.tick_params(axis='x', rotation=90)

    # Plot 4: Bar Plot of Top 20 Nodes
    strength = {node: sum(weight for _, _, weight in G.edges(node, data='LRP_norm')) for node in G.nodes()}
    strength = pd.DataFrame.from_dict(strength, orient='index').sort_values(by = 0, ascending = False)
    strength ['strenght_norm'] = strength [0] / strength [0].max()
    strength_20 =strength [:20][::-1]
    ax4 = fig.add_subplot(gs[1, 1])
    #top_20_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:20][::-1]
    #node_names, degrees = zip(*top_20_nodes)
    colors = [node_color_mapper.get(node.split('_')[1], 'black') for node in strength_20.index]

    ax4.barh(  strength_20.index, strength_20['strenght_norm'], color=colors)
    ax4.set_title('Top 20 Nodes by Weighted Degree')
    #ax4.tick_params(axis='x', rotation=90)
    
    ax5 = fig.add_subplot(gs[1, 2])
    a = pd.DataFrame.from_dict(overlap_n_dict, orient='index')
    ax5.barh(  a.index, a[0])
    ax5.set_title('Number of genes overlaping with genes from pathways')
    
    ax6 = fig.add_subplot(gs[2, 2])
    a = pd.DataFrame.from_dict(overlap_ratio_dict, orient='index')
    ax6.barh(   a.index, a[0])
    ax6.set_title('Ratio of overlap with genes from pathways')
    
    # overlap_ratio
    ax7 = fig.add_subplot(gs[0, 2])
    a = overlap_df.sort_values(by = 'overlap_size',ascending=False)[:10][::-1]
    ax7.barh(   a['column'], a['overlap_ratio'])
    ax7.set_title("Ratio of overlap with genes from Hendrik's columns")
    yticks = plt.gca().get_yticklabels()
    plt.gca().set_yticklabels([textwrap.fill(label.get_text(), 15) for label in yticks])

    
    
    # # Plot 5: Word Cloud
    # ax5 = fig.add_subplot(gs[1, 2])
    # wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)
    # ax5.imshow(wordcloud, interpolation='bilinear')
    # ax5.axis('off')
    # ax5.set_title('Word Cloud of Node Labels')
    
    plt.tight_layout()
    
    
    #plt.savefig(os.path.join(path_to_save , 'gridplot_' + key.replace('.csv','') + '.png'), dpi = 300)
    plt.savefig(os.path.join(path_to_save , 'gridplot_' + key.replace('.csv','') + '.pdf'), format = 'pdf')
    plt.show()

    plot_data = {

        "top_20_edges":  edges.nlargest(20, 'LRP')[::-1],
        "edges": edges,
                         
        "top_20_nodes_weighted_degree":  strength_20,
        "nodes_weighted_degree":  strength,
        
        "overlap_n_pathways": pd.DataFrame.from_dict(overlap_n_dict, orient='index')
        ,
        "overlap_ratio_pathways":  pd.DataFrame.from_dict(overlap_ratio_dict, orient='index')
        ,
        "genes_overlap_with_Hendrik": overlap_df.sort_values(by='overlap_size', ascending=False)[:10][::-1]
        
    }


    for key_plot, df in plot_data.items():
        file_path = os.path.join(path_to_save_csv, key.replace('.csv', '') + '_' + key_plot + '.csv')
        df.to_csv(file_path)


# %% compare groups - top nodes

path_to_save_csv

csv_files = [f for f in os.listdir(path_to_save_csv) if f.endswith('20_nodes_weighted_degree.csv')]


edge_types = []
groups = []
groups_both = []
for file in csv_files:
    edge_type = file.replace('.csv', '').split('_')[2]
    stat= file.replace('.csv', '').split('_')[-1]
    group = '_'.join(file.replace('.csv', '').split('_')[3:-1]).replace('_mean_top_20_nodes_weighted', '')
    if 'TNBC' in group:
        group_ = group[-4:]
    else:
        group_ = '_'.join(group.split('_')[:-1])
    
    edge_types.append(edge_type)
    groups.append(group)
    groups_both.append(group_)

files_dict = pd.DataFrame([csv_files, edge_types, groups, groups_both]).T
files_dict .columns = ['file','edge_type','subgroup', 'group']

df = files_dict
# Iterate over each unique edge type
for edge_type in df['edge_type'].unique():
    for group in df['group'].unique():
        if group == 'all':
            pass
        else:# Filter DataFrame for the current edge type
            filtered_df = df[(df['edge_type'] == edge_type) & (df['group'] == group)]
            print(edge_type, group)
            # Create a figure for the current edge type
            fig, axs = plt.subplots(ncols=2, figsize=(5,8),  sharey=True)
            
            # Plot each group in a separate subplot
            data_temp = pd.DataFrame()
            data_indextop20 = []
            for _, row in filtered_df.iterrows():
                # Load data from CSV
                csv_path = os.path.join(path_to_save_csv, row['file'].replace('top_20_nodes_weighted_degree','nodes_weighted_degree'))
                data = pd.read_csv(csv_path, index_col = 0)
                data['file'] = row['file']
                data_indextop20 += (list(data.index[:20]))
                data_temp = pd.concat((data_temp, data))
                
            data_indextop20 = list(set(data_indextop20))
            data_indextop20.sort()
            
            ax = axs[0]
            # Plot horizontal bar plot
            subgroup = filtered_df['subgroup'].unique()[0]
            data_to_plot = data_temp.reset_index().loc[(data_temp.reset_index()['index'].isin(data_indextop20)) & (data_temp.reset_index()['file'].str.contains(subgroup)),:]
            data_to_plot = data_to_plot.sort_values('strenght_norm')
            ax.barh(data_to_plot['index'], data_to_plot['strenght_norm'])
            ax.set_title(subgroup)
            ax.set_ylabel('Nodes')
            ax.set_xlabel('Weighted node degree\nnormalized')
            print(data_to_plot.shape)
            data_to_plot[['index', 'strenght_norm']].to_csv(os.path.join(path_to_save_csv, 'barplot_data_' + subgroup +'_'+ edge_type + '.csv'))
            
            ax = axs[1]
            # Plot horizontal bar plot
            subgroup = filtered_df['subgroup'].unique()[1]
            data_to_plot = data_temp.reset_index().loc[(data_temp.reset_index()['index'].isin(data_indextop20)) & (data_temp.reset_index()['file'].str.contains(subgroup)),:]
            #data_to_plot = data_to_plot.sort_values('index')
            ax.barh(data_to_plot['index'], data_to_plot['strenght_norm'])
            ax.set_title(subgroup)
            ax.set_ylabel('Nodes')
            ax.set_xlabel('Weighted node degree\nnormalized')
            print(data_to_plot.shape)
            data_to_plot[['index', 'strenght_norm']].to_csv(os.path.join(path_to_save_csv, 'barplot_data_' + subgroup + '_'+ edge_type + '.csv'))
        
            # Adjust layout and show plot
            plt.tight_layout()
            plt.suptitle(f'Edge Type: {edge_type}', fontsize=16)
            plt.subplots_adjust(top=0.90)
            plt.savefig(os.path.join(path_to_save , 'barplots_' + group + edge_type + '.pdf'), format = 'pdf')
            plt.show()


# %%% plot heatmap top weighted degree nodes
edge_types_selected =  ['amp', 'del', 'exp','fus','mut']
groups_selected = list(df['group'].unique())
groups_selected.remove('all')

to_reasoning_df = pd.DataFrame()

fig , axs = plt.subplots(len(groups_selected),5,figsize = (12,35))
for type_i, edge_type in enumerate(edge_types_selected):
    for group_i,group in enumerate(groups_selected):

        filtered_df = files_dict[(files_dict['edge_type'] == edge_type) & (files_dict['group'] == group)]
        print(edge_type, group)
        print(filtered_df['file'].values[0])
        print(filtered_df['file'].values[1])
        ax = axs[group_i, type_i]    
        subgroup1 = filtered_df['subgroup'].unique()[0]
        subgroup2 = filtered_df['subgroup'].unique()[1]
        data_temp1 = pd.read_csv(os.path.join(path_to_save_csv, 'barplot_data_' + subgroup1 + '_'+ edge_type + '.csv'))
        data_temp2 = pd.read_csv(os.path.join(path_to_save_csv, 'barplot_data_' + subgroup2 + '_'+ edge_type + '.csv'))
        data_heatmap = data_temp1[['index', 'strenght_norm']].merge(data_temp2[['index', 'strenght_norm']], on = 'index', how = 'outer').fillna(0)
        data_heatmap.columns = ['gene', subgroup1, subgroup2]
        data_heatmap['diff'] = (data_heatmap[subgroup1] - data_heatmap[subgroup2]).abs()
        data_heatmap['sig_diff'] = 0.0
        data_heatmap.loc[data_heatmap['diff'] > .5, 'sig_diff'] = .5
        data_heatmap.loc[data_heatmap['diff'] > .75, 'sig_diff'] = 1
        data_heatmap = data_heatmap.drop(columns = 'diff')
        data_heatmap = data_heatmap.set_index('gene')
                
        print(data_heatmap.shape)
        sns.heatmap(data_heatmap, cmap = 'Reds',ax=ax, vmin = 0, vmax = 1, cbar = False, yticklabels=True, linewidth = 0.1)
        ax.set_ylabel(None)
        ax.set_title(group + '\n' + edge_type)
        xticklabels = ax.get_xticklabels()
        new_xticklabels = ['pos' if 'pos' in label.get_text() and 'TNBC' not in label.get_text() 
                           else 'neg' if 'neg' in label.get_text() and 'TNBC' not in label.get_text() 
                           else label.get_text() 
                           for label in xticklabels]
        ax.set_xticklabels(new_xticklabels, rotation=45) 
        
        to_reasoning = pd.DataFrame(data_heatmap.loc[data_heatmap['sig_diff'] > 0 , subgroup1] - data_heatmap.loc[data_heatmap['sig_diff'] > 0 , subgroup2])
        to_reasoning['case'] = edge_type + ' ' + group + ' : ' + subgroup1 + ' - ' + subgroup2
        to_reasoning['edge_type'] = edge_type
        to_reasoning['group'] = group
        
        to_reasoning_df  = pd.concat((to_reasoning_df , to_reasoning))    
        
        
plt.tight_layout()
plt.savefig(os.path.join(path_to_save , 'heatmap_weighted_degree_node' +  '.pdf'), format = 'pdf')
plt.show()

to_reasoning_df.to_excel(os.path.join(path_to_save_csv, 'to_reasoning_weighted_degree_node.xlsx'))


# %% compare groups - top edges

path_to_save_csv

csv_files = [f for f in os.listdir(path_to_save_csv) if f.endswith('top_20_edges.csv')]


edge_types = []
groups = []
groups_both = []
for file in csv_files:
    edge_type = file.replace('.csv', '').split('_')[2]
    stat= file.replace('.csv', '').split('_')[-1]
    group = '_'.join(file.replace('.csv', '').split('_')[3:-1]).replace('_mean_top_20', '')
    if 'TNBC' in group:
        group_ = group[-4:]
    else:
        group_ = '_'.join(group.split('_')[:-1])
    
    edge_types.append(edge_type)
    groups.append(group)
    groups_both.append(group_)

files_dict = pd.DataFrame([csv_files, edge_types, groups, groups_both]).T
files_dict .columns = ['file','edge_type','subgroup', 'group']

df = files_dict
# Iterate over each unique edge type
for edge_type in df['edge_type'].unique():
    for group in df['group'].unique():
        if group == 'all':
            pass
        else:# Filter DataFrame for the current edge type
            filtered_df = df[(df['edge_type'] == edge_type) & (df['group'] == group)]
            print(edge_type, group)
            # Create a figure for the current edge type
            fig, axs = plt.subplots(ncols=2, figsize=(5,8),  sharey=True)
            
            # Plot each group in a separate subplot
            data_temp = pd.DataFrame()
            data_indextop20 = []
            for _, row in filtered_df.iterrows():
                # Load data from CSV
                csv_path = os.path.join(path_to_save_csv, row['file'].replace('top_20_edges','edges'))
                data = pd.read_csv(csv_path, index_col = 0)
                data['file'] = row['file']
                data_indextop20 += (list(data.loc[:20, 'edge']))
                data_temp = pd.concat((data_temp, data))
                
            data_indextop20 = list(set(data_indextop20))
            data_indextop20.sort()
            data_temp = data_temp.reset_index(drop=True)
            
            ax = axs[0]
            # Plot horizontal bar plot
            subgroup = filtered_df['subgroup'].unique()[0]
            data_to_plot = data_temp.loc[(data_temp['edge'].isin(data_indextop20)) & (data_temp['file'].str.contains(subgroup)),:]
            data_to_plot = data_to_plot.sort_values('LRP_norm')
            ax.barh(data_to_plot['edge'], data_to_plot['LRP_norm'])
            ax.set_title(subgroup)
            ax.set_ylabel('Edges')
            ax.set_xlabel('LRP normalized')
            print(data_to_plot.shape)
            data_to_plot[['edge', 'LRP_norm']].to_csv(os.path.join(path_to_save_csv, 'barplot_data_edges_' + subgroup +'_'+ edge_type + '.csv'))
            
            ax = axs[1]
            # Plot horizontal bar plot
            subgroup = filtered_df['subgroup'].unique()[1]
            data_to_plot = data_temp.loc[(data_temp['edge'].isin(data_indextop20)) & (data_temp['file'].str.contains(subgroup)),:]
            data_to_plot = data_to_plot.sort_values('LRP_norm')
            ax.barh(data_to_plot['edge'], data_to_plot['LRP_norm'])
            ax.set_title(subgroup)
            ax.set_ylabel('Edges')
            ax.set_xlabel('LRP normalized')
            print(data_to_plot.shape)
            data_to_plot[['edge', 'LRP_norm']].to_csv(os.path.join(path_to_save_csv, 'barplot_data_edges_' + subgroup +'_'+ edge_type + '.csv'))
            
            # Adjust layout and show plot
            plt.tight_layout()
            plt.suptitle(f'Edge Type: {edge_type}', fontsize=16)
            plt.subplots_adjust(top=0.90)
            plt.savefig(os.path.join(path_to_save , 'barplots_edges_' + group + edge_type + '.pdf'), format = 'pdf')
            plt.show()


# %%% plot heatmap top edges
edge_types_selected =  ['amp', 'del', 'exp','fus','mut']
groups_selected = list(df['group'].unique())
groups_selected.remove('all')

to_reasoning_df = pd.DataFrame()

fig , axs = plt.subplots(len(groups_selected),5,figsize = (12,35))
for type_i, edge_type in enumerate(edge_types_selected):
    for group_i,group in enumerate(groups_selected):

        filtered_df = files_dict[(files_dict['edge_type'] == edge_type) & (files_dict['group'] == group)]
        print(edge_type, group)
        print(filtered_df['file'].values[0])
        print(filtered_df['file'].values[1])
        ax = axs[group_i, type_i]    
        subgroup1 = filtered_df['subgroup'].unique()[0]
        subgroup2 = filtered_df['subgroup'].unique()[1]
        data_temp1 = pd.read_csv(os.path.join(path_to_save_csv, 'barplot_data_edges_' + subgroup1 + '_'+ edge_type + '.csv'))
        data_temp2 = pd.read_csv(os.path.join(path_to_save_csv, 'barplot_data_edges_' + subgroup2 + '_'+ edge_type + '.csv'))
        data_heatmap = data_temp1[['edge', 'LRP_norm']].merge(data_temp2[['edge', 'LRP_norm']], on = 'edge', how = 'outer').fillna(0)
        data_heatmap.columns = ['edge', subgroup1, subgroup2]
        data_heatmap.loc[data_heatmap[subgroup1] > 0, subgroup1] = 1
        data_heatmap.loc[data_heatmap[subgroup2] > 0, subgroup2] = 1

        data_heatmap = data_heatmap.set_index('edge')
                
        print(data_heatmap.shape)
        sns.heatmap(data_heatmap, cmap = 'Reds',ax=ax, vmin = 0, vmax = 1, cbar = False, yticklabels=True, linewidth = 0.1, square = True)
        ax.set_ylabel(None)
        ax.set_title(group + '\n' + edge_type)
        xticklabels = ax.get_xticklabels()
        new_xticklabels = ['pos' if 'pos' in label.get_text() and 'TNBC' not in label.get_text() 
                           else 'neg' if 'neg' in label.get_text() and 'TNBC' not in label.get_text() 
                           else label.get_text() 
                           for label in xticklabels]
        ax.set_xticklabels(new_xticklabels, rotation=45) 
        
        if (data_heatmap[subgroup1] != data_heatmap[subgroup2]).sum() ==0:
            pass
        else:
            to_reasoning = pd.DataFrame(data_heatmap.loc[data_heatmap[subgroup1] != data_heatmap[subgroup2], :] )
            to_reasoning.columns = ['subgroup1','subgroup2']
            to_reasoning['case'] = edge_type + ' ' + group + ' : ' + subgroup1 + ' - ' + subgroup2
            to_reasoning['edge_type'] = edge_type
            to_reasoning['group'] = group
            #to_reasoning['conclusion'] = 
            
            to_reasoning_df  = pd.concat((to_reasoning_df , to_reasoning))    
        
        
plt.tight_layout()
#plt.subplots_adjust(wspace=0.2)
plt.savefig(os.path.join(path_to_save , 'heatmap_edges' +  '.pdf'), format = 'pdf')
plt.show()

to_reasoning_df.to_excel(os.path.join(path_to_save_csv, 'to_reasoning_edges.xlsx'))


# %% compare groups - pathway overalps

path_to_save_csv

csv_files = [f for f in os.listdir(path_to_save_csv) if f.endswith('overlap_ratio_pathways.csv')]


edge_types = []
groups = []
groups_both = []
for file in csv_files:
    edge_type = file.replace('.csv', '').split('_')[2]
    stat= file.replace('.csv', '').split('_')[-1]
    group = '_'.join(file.replace('.csv', '').split('_')[3:-1]).replace('_mean_top_20', '')
    if 'TNBC' in group:
        group_ = group[-4:]
    else:
        group_ = '_'.join(group.split('_')[:-1])
    
    edge_types.append(edge_type)
    groups.append(group)
    groups_both.append(group_)

files_dict = pd.DataFrame([csv_files, edge_types, groups, groups_both]).T
files_dict .columns = ['file','edge_type','subgroup', 'group']

df = files_dict
# Iterate over each unique edge type
for edge_type in df['edge_type'].unique():
    for group in df['group'].unique():
        if group == 'all':
            pass
        else:# Filter DataFrame for the current edge type
            filtered_df = df[(df['edge_type'] == edge_type) & (df['group'] == group)]
            print(edge_type, group)
            # Create a figure for the current edge type
            fig, axs = plt.subplots(ncols=2, figsize=(5,8),  sharey=True)
            
            # Plot each group in a separate subplot
            data_temp = pd.DataFrame()
            data_indextop20 = []
            for _, row in filtered_df.iterrows():
                # Load data from CSV
                csv_path = os.path.join(path_to_save_csv, row['file'].replace('top_20_edges','edges'))
                data = pd.read_csv(csv_path, index_col = 0)
                data['file'] = row['file']
                data_indextop20 += (list(data.loc[:20, 'edge']))
                data_temp = pd.concat((data_temp, data))
                
            data_indextop20 = list(set(data_indextop20))
            data_indextop20.sort()
            data_temp = data_temp.reset_index(drop=True)
            
            ax = axs[0]
            # Plot horizontal bar plot
            subgroup = filtered_df['subgroup'].unique()[0]
            data_to_plot = data_temp.loc[(data_temp['edge'].isin(data_indextop20)) & (data_temp['file'].str.contains(subgroup)),:]
            data_to_plot = data_to_plot.sort_values('LRP_norm')
            ax.barh(data_to_plot['edge'], data_to_plot['LRP_norm'])
            ax.set_title(subgroup)
            ax.set_ylabel('Edges')
            ax.set_xlabel('LRP normalized')
            print(data_to_plot.shape)
            data_to_plot[['edge', 'LRP_norm']].to_csv(os.path.join(path_to_save_csv, 'barplot_data_edges_' + subgroup +'_'+ edge_type + '.csv'))
            
            ax = axs[1]
            # Plot horizontal bar plot
            subgroup = filtered_df['subgroup'].unique()[1]
            data_to_plot = data_temp.loc[(data_temp['edge'].isin(data_indextop20)) & (data_temp['file'].str.contains(subgroup)),:]
            data_to_plot = data_to_plot.sort_values('LRP_norm')
            ax.barh(data_to_plot['edge'], data_to_plot['LRP_norm'])
            ax.set_title(subgroup)
            ax.set_ylabel('Edges')
            ax.set_xlabel('LRP normalized')
            print(data_to_plot.shape)
            data_to_plot[['edge', 'LRP_norm']].to_csv(os.path.join(path_to_save_csv, 'barplot_data_edges_' + subgroup +'_'+ edge_type + '.csv'))
            
            # Adjust layout and show plot
            plt.tight_layout()
            plt.suptitle(f'Edge Type: {edge_type}', fontsize=16)
            plt.subplots_adjust(top=0.90)
            plt.savefig(os.path.join(path_to_save , 'barplots_edges_' + group + edge_type + '.pdf'), format = 'pdf')
            plt.show()


# %%% plot heatmap top edges
edge_types_selected =  ['amp', 'del', 'exp','fus','mut']
groups_selected = list(df['group'].unique())
groups_selected.remove('all')

to_reasoning_df = pd.DataFrame()

fig , axs = plt.subplots(len(groups_selected),5,figsize = (12,35))
for type_i, edge_type in enumerate(edge_types_selected):
    for group_i,group in enumerate(groups_selected):

        filtered_df = files_dict[(files_dict['edge_type'] == edge_type) & (files_dict['group'] == group)]
        print(edge_type, group)
        print(filtered_df['file'].values[0])
        print(filtered_df['file'].values[1])
        ax = axs[group_i, type_i]    
        subgroup1 = filtered_df['subgroup'].unique()[0]
        subgroup2 = filtered_df['subgroup'].unique()[1]
        data_temp1 = pd.read_csv(os.path.join(path_to_save_csv, 'barplot_data_edges_' + subgroup1 + '_'+ edge_type + '.csv'))
        data_temp2 = pd.read_csv(os.path.join(path_to_save_csv, 'barplot_data_edges_' + subgroup2 + '_'+ edge_type + '.csv'))
        data_heatmap = data_temp1[['edge', 'LRP_norm']].merge(data_temp2[['edge', 'LRP_norm']], on = 'edge', how = 'outer').fillna(0)
        data_heatmap.columns = ['edge', subgroup1, subgroup2]
        data_heatmap.loc[data_heatmap[subgroup1] > 0, subgroup1] = 1
        data_heatmap.loc[data_heatmap[subgroup2] > 0, subgroup2] = 1

        data_heatmap = data_heatmap.set_index('edge')
                
        print(data_heatmap.shape)
        sns.heatmap(data_heatmap, cmap = 'Reds',ax=ax, vmin = 0, vmax = 1, cbar = False, yticklabels=True, linewidth = 0.1, square = True)
        ax.set_ylabel(None)
        ax.set_title(group + '\n' + edge_type)
        xticklabels = ax.get_xticklabels()
        new_xticklabels = ['pos' if 'pos' in label.get_text() and 'TNBC' not in label.get_text() 
                           else 'neg' if 'neg' in label.get_text() and 'TNBC' not in label.get_text() 
                           else label.get_text() 
                           for label in xticklabels]
        ax.set_xticklabels(new_xticklabels, rotation=45) 
        
        if (data_heatmap[subgroup1] != data_heatmap[subgroup2]).sum() ==0:
            pass
        else:
            to_reasoning = pd.DataFrame(data_heatmap.loc[data_heatmap[subgroup1] != data_heatmap[subgroup2], :] )
            to_reasoning.columns = ['subgroup1','subgroup2']
            to_reasoning['case'] = edge_type + ' ' + group + ' : ' + subgroup1 + ' - ' + subgroup2
            to_reasoning['edge_type'] = edge_type
            to_reasoning['group'] = group
            #to_reasoning['conclusion'] = 
            
            to_reasoning_df  = pd.concat((to_reasoning_df , to_reasoning))    
        
        
plt.tight_layout()
#plt.subplots_adjust(wspace=0.2)
plt.savefig(os.path.join(path_to_save , 'heatmap_edges' +  '.pdf'), format = 'pdf')
plt.show()

to_reasoning_df.to_excel(os.path.join(path_to_save_csv, 'to_reasoning_edges.xlsx'))




# %% plot heatmaps overlap
overlap_ns = pd.DataFrame()
for key, df in dfs.items():
    # --- Your existing code to process df and create graph G ---
    print(key)
    edge_type = key.replace('.csv', '').split('_')[2]
    stat= key.replace('.csv', '').split('_')[-1]
    group = '_'.join(key.replace('.csv', '').split('_')[3:-1])
    
    temp_df = pd.DataFrame()
    #temp_df[group + ' - ' + edge_type] = top_nodes_dict[key]['overlap_n'].values()
    temp_df[edge_type + ' - ' + group] = top_nodes_dict[key]['overlap_n'].values()
    temp_df['subgroup'] = top_nodes_dict[key]['overlap_n'].keys()
    temp_df = temp_df.set_index('subgroup')
    temp_df = temp_df.T
    overlap_ns = pd.concat((overlap_ns , temp_df))
overlap_ns = overlap_ns.sort_index()
fig, ax = plt.subplots(figsize = (5,20))
sns.heatmap(overlap_ns, ax=ax, yticklabels= True, cmap  = 'jet')    
    
overlap_ratios = pd.DataFrame()
for key, df in dfs.items():
    # --- Your existing code to process df and create graph G ---
    print(key)
    edge_type = key.replace('.csv', '').split('_')[2]
    stat= key.replace('.csv', '').split('_')[-1]
    group = '_'.join(key.replace('.csv', '').split('_')[3:-1])
    
    temp_df = pd.DataFrame()
    #temp_df[group + ' - ' + edge_type] = top_nodes_dict[key]['overlap_n'].values()
    temp_df[edge_type + ' - ' + group] = top_nodes_dict[key]['overlap_ratio'].values()
    temp_df['subgroup'] = top_nodes_dict[key]['overlap_ratio'].keys()
    temp_df = temp_df.set_index('subgroup')
    temp_df = temp_df.T
    overlap_ratios = pd.concat((overlap_ratios , temp_df))
overlap_ratios = overlap_ratios.sort_index()
fig, ax = plt.subplots(figsize = (5,20))
sns.heatmap(overlap_ratios, ax=ax, yticklabels= True, xticklabels= True, cmap  = 'Reds', linewidth = 0.1, cbar=False)    
plt.tight_layout()
plt.savefig(os.path.join(path_to_save , 'heatmap_ratios' + '.pdf'), format = 'pdf')
plt.show()

 #%%










    
top_nodes_dict = {}

for key, df in dfs.items():
    # --- Your existing code to process df and create graph G ---
    print(key)
    top_x_percent = df['LRP'].quantile(quantile_)
    edges = df[df['LRP'] >= top_x_percent].copy()

    genes_to_dict = list(G.nodes)
    genes_to_dict .sort()
    top_nodes_dict[key] = {}
    top_nodes_dict[key]['genes_nodes'] = {'genes':genes_to_dict}

    overlap_genes, overlap_n, overlap_ratio = compute_overlap(genes_to_dict, genes_pathways_dict)
    top_nodes_dict[key] = {}
    
    top_nodes_dict[key]['overlap_genes'] = overlap_genes
    top_nodes_dict[key]['overlap_n'] = overlap_n
    top_nodes_dict[key]['overlap_ratio'] = overlap_ratio
        
# Convert to DataFrame

# Creating a DataFrame for each top-level key
dataframes = {}
for key in top_nodes_dict:
    df = pd.DataFrame(list(top_nodes_dict[key].items()), columns=['Pathway', key])
    dataframes[key] = df

# Accessing DataFrames
overlap_n_df = dataframes['overlap_n']
overlap_ratio_df = dataframes['overlap_ratio']









# Example usage
genes_pathways = pd.DataFrame({
    'Pathway': ['Pathway1', 'Pathway1', 'Pathway2', 'Pathway2'],
    'Gene': ['Gene1_abc', 'Gene2_xyz', 'Gene3_def', 'Gene2_xyz']
})
genes_pathways_dict = {}
for pathway in genes_pathways['Pathway'].unique():
    genes_pathways_dict[pathway] = genes_pathways.loc[genes_pathways['Pathway'] == pathway, 'Gene'].to_list()

# Example set X
X = {'CTNNB1', 'RPS6KA3'}

# Compute overlap
overlap_genes, overlap_n, overlap_ratio = compute_overlap(X, genes_pathways_dict)
print(overlap_results)

    
    
    
    
    
    
    
    
    