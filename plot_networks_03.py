# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:41:23 2023

@author: d07321ow
"""

#%matplotlib inline 
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

# %% load data

path_to_data = '/home/d07321ow/scratch/scGeneRAI/data/data_BRCA'
#path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'

path_to_save = '/home/d07321ow/scratch/results_LRP_BRCA/networks'
#path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\networks'


#path_to_pathways = r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI\PATHWAYS'
path_to_pathways ='/home/d07321ow/scratch/scGeneRAI/PATHWAYS'

#path_to_lrp_results = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\results'
path_to_lrp_results = '/home/d07321ow/scratch/results_LRP_BRCA/results'

data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(path_to_data)

# %% get samples

samples = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()

data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model')

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

df_clinical_features = f.add_cluster0(df_clinical_features)


samples_groups = f.get_samples_by_group(df_clinical_features)


# %%
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'

topn = 1000

LRP_pd = pd.read_csv(os.path.join(path_to_save, 'LRP_individual_top{}.csv'.format(topn)), index_col = 0)


# %% GLOBAL NETWORK

# %%% all interactions
edges = LRP_pd.mean(axis=1).reset_index()

edges['source_gene']  = edges['index'].str.split(' - ', expand = True)[0]
edges['target_gene']= edges['index'].str.split(' - ', expand = True)[1]
#edges['source_gene'] = edges['source_gene'].str.split('_', expand=True)[0]
#edges['target_gene'] = edges['target_gene'].str.split('_', expand=True)[0]
edges  = edges.rename(columns = {edges.columns[1]:'LRP'})
edges['LRP_norm'] = edges['LRP'] / edges['LRP'].max()
edges = edges.sort_values('LRP', ascending = False).reset_index(drop=True)

G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP_norm')
degrees = np.array(list(nx.degree_centrality(G).values()))
degrees_norm = degrees / np.max(degrees)
# nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*100, node_size = network.degree)

    

widths = edges['LRP'] / edges['LRP'].max() 

edges_from_G = [i[0] + ' - ' + i[1] for i in list(G.edges)]
edge_colors = list(f.color_mapper(edges_from_G).values())

node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}
if ax == None:
    fig, ax = plt.subplots(figsize=(10, 10))
    
    
    
pos = nx.spring_layout(G, weight='LRP_norm')

fig, ax = plt.subplots(figsize=(20,20))   
    # Plot legend
ls = list(node_color_mapper.keys())
cl = list(node_color_mapper.values())
for label, color in zip(ls, cl):
    ax.plot([], [], 'o', color=color, label=label)
ax.legend(title = 'Nodes', loc='best')

node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper)
    
nx.draw(G, with_labels=False,
        node_color=node_colors,
        width=widths,
        pos=pos, # font_size=0,
        # cmap = colors,
        edge_color=edge_colors,
        ax=ax,
        # node_size = degrees,
        node_size=degrees_norm * 2000)


ax.set_title('Top1000, all 5300 interactions, mean LRP')
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\global_network1000.png')


# %%% interaction > knee point

lrps = np.sort(LRP_pd.values.ravel())[::-1]
plt.plot(lrps)

kf = KneeFinder(range(len(lrps)), lrps)
knee_x, knee_y = kf.find_knee()
kf.plot()




edges_global = LRP_pd.mean(axis=1).reset_index()
edges_global['source_gene']  = edges_global['index'].str.split(' - ', expand = True)[0]
edges_global['target_gene']= edges_global['index'].str.split(' - ', expand = True)[1]
#edges['source_gene'] = edges['source_gene'].str.split('_', expand=True)[0]
#edges['target_gene'] = edges['target_gene'].str.split('_', expand=True)[0]
edges_global  = edges_global.rename(columns = {edges_global.columns[1]:'LRP'})
edges_global['LRP_norm'] = edges_global['LRP'] / edges_global['LRP'].max()
edges_global = edges_global.sort_values('LRP', ascending = False).reset_index(drop=True)
kf = KneeFinder(edges_global.index, edges_global['LRP'])
knee_x_global, knee_y = kf.find_knee()
kf.plot()
#knee_x=2000
edges_global = edges_global.iloc[:int(knee_x_global), :]

path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300'
mame_to_save = 'global_network1000_knee.png'
title = 'Top1000, {} interactions, filted using knee point, mean LRP'.format(edges.shape[0])
#plot_global_network(edges, path_to_save, name_to_save, title)

#def plot_global_network(edges, path_to_save, name_to_save, title):
    
    #edges = edges.loc[edges['LRP'] > knee_y, :].reset_index(drop=True)
        
G = nx.from_pandas_edgelist(edges_global, source='source_gene', target='target_gene', edge_attr='LRP_norm')
degrees = np.array(list(nx.degree_centrality(G).values()))
degrees_norm = degrees / np.max(degrees)
# nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*100, node_size = network.degree)

    

widths = edges_global['LRP'] / edges_global['LRP'].max() 

edges_from_G = [i[0] + ' - ' + i[1] for i in list(G.edges)]
edge_colors = list(f.color_mapper(edges_from_G).values())

node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}
    
pos_global = nx.spring_layout(G, weight='LRP_norm')

fig, ax = plt.subplots(figsize=(20,20))   
    # Plot legend
ls = list(node_color_mapper.keys())
cl = list(node_color_mapper.values())
for label, color in zip(ls, cl):
    ax.plot([], [], 'o', color=color, label=label)
ax.legend(title = 'Nodes', loc='best')

node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper)
    
nx.draw(G, with_labels=False,
        node_color=node_colors,
        width=widths,
        pos=pos_global, # font_size=0,
        # cmap = colors,
        edge_color=edge_colors,
        ax=ax,
        # node_size = degrees,
        node_size=degrees_norm * 500)


ax.set_title(title)
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, name_to_save))



# %%% global network for individual cluster

cluster_lables_pd = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\cluster_labels.csv', index_col=0)

for cluster_i in range(5):
    col_index_ = (LRP_pd.columns).isin( cluster_lables_pd[cluster_lables_pd['cluster_labels'] == cluster_i]['samples'].to_list())
    
    print(cluster_i, np.sum(col_index_))
    
    edges = LRP_pd.loc[:,col_index_] .mean(axis=1).reset_index()
    edges['source_gene']  = edges['index'].str.split(' - ', expand = True)[0]
    edges['target_gene']= edges['index'].str.split(' - ', expand = True)[1]
    #edges['source_gene'] = edges['source_gene'].str.split('_', expand=True)[0]
    #edges['target_gene'] = edges['target_gene'].str.split('_', expand=True)[0]
    edges  = edges.rename(columns = {edges.columns[1]:'LRP'})
    edges['LRP_norm'] = edges['LRP'] / edges['LRP'].max()
    
    edges = edges.loc[edges['index'].isin(edges_global['index'].to_list()), :].reset_index(drop=True)
    
    path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300'
    name_to_save = 'global_network1000_knee_cluster_{}.png'.format(cluster_i)
    title = 'Top1000, {} interactions, filted using knee point, mean LRP\n CLUSTER {}'.format(edges.shape[0], cluster_i)
    
    
    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP_norm')
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = edges['LRP'] / edges['LRP'].max() 
    
    edges_from_G = [i[0] + ' - ' + i[1] for i in list(G.edges)]
    edge_colors = list(f.color_mapper(edges_from_G).values())
    edge_colors = plt.cm.jet(widths)  # 'viridis' is a colormap, you can choose any
    
    
    node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}
        
    #pos = nx.spring_layout(G, weight='LRP_norm')
    
    fig, ax = plt.subplots(figsize=(10,global_network1000_knee))   
        # Plot legend
    ls = list(node_color_mapper.keys())
    cl = list(node_color_mapper.values())
    for label, color in zip(ls, cl):
        ax.plot([], [], 'o', color=color, label=label)
    ax.legend(title = 'Nodes', loc='best')
    
    node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper)
        
    nx.draw(G, with_labels=False,
            node_color=node_colors,
            width=widths*5,
            pos=pos_global, # font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 500)
    
    
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, name_to_save))


# %% communities in individual networks


import community as community_louvain
from sklearn.metrics import adjusted_rand_score

def detect_communities(G):
    # Use the Louvain method for community detection
    partition = community_louvain.best_partition(G)
    return partition

def communities_to_labels(partition):
    # Convert a partition to cluster labels
    labels = []
    for node in partition:
        labels.append(partition[node])
    return labels

def compare_communities(labels1, labels2):
    # Calculate the Adjusted Rand Index
    score = adjusted_rand_score(labels1, labels2)
    return score


edges = LRP_pd.iloc[:,1].reset_index()

edges['source_gene']  = edges['index'].str.split(' - ', expand = True)[0]
edges['target_gene']= edges['index'].str.split(' - ', expand = True)[1]
#edges['source_gene'] = edges['source_gene'].str.split('_', expand=True)[0]
#edges['target_gene'] = edges['target_gene'].str.split('_', expand=True)[0]
edges  = edges.rename(columns = {edges.columns[1]:'LRP'})
edges['LRP_norm'] = edges['LRP'] / edges['LRP'].max()
edges = edges.sort_values('LRP', ascending = False).reset_index(drop=True)
kf = KneeFinder(edges.index, edges['LRP'])
knee_x, knee_y = kf.find_knee()
kf.plot()
#knee_x=2000
edges = edges.iloc[:int(knee_x), :]


path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300'
name_to_save = 'global_network1000_knee_cluster_{}.png'.format(cluster_i)
title = 'Top1000, {} interactions, filted using knee point, mean LRP\n CLUSTER {}'.format(edges.shape[0], cluster_i)


G1 = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP_norm')
degrees = np.array(list(nx.degree_centrality(G).values()))
degrees_norm = degrees / np.max(degrees)
widths = edges['LRP'] / edges['LRP'].max() 

edges_from_G = [i[0] + ' - ' + i[1] for i in list(G.edges)]
edge_colors = list(f.color_mapper(edges_from_G).values())
edge_colors = plt.cm.jet(widths)  # 'viridis' is a colormap, you can choose any


node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}
    
pos = nx.spring_layout(G, weight='LRP_norm')

fig, ax = plt.subplots(figsize=(10,10))   
    # Plot legend
ls = list(node_color_mapper.keys())
cl = list(node_color_mapper.values())
for label, color in zip(ls, cl):
    ax.plot([], [], 'o', color=color, label=label)
ax.legend(title = 'Nodes', loc='best')

node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper)
    
cmap = plt.cm.jet  # Choose a colormap

# Create a color map for communities
community_colors = cmap(np.linspace(0, 1, len(set(partition1.values()))))

# Map the community colors to the nodes
node_colors = [community_colors[partition1[node]] for node in G.nodes]



nx.draw(G, with_labels=False,
        node_color=node_colors,
        width=widths*5,
        pos=pos, # font_size=0,
        # cmap = colors,
        edge_color=edge_colors,
        ax=ax,
        # node_size = degrees,
        node_size=degrees_norm * 500)


ax.set_title(title)
plt.tight_layout()



# Detect communities in each graph
partition1 = detect_communities(G1)
partition2 = detect_communities(G2)

# Convert partitions to labels
labels1 = communities_to_labels(partition1)
labels2 = communities_to_labels(partition2)

# Compare the communities
score = compare_communities(labels1, labels2)




# %%

edges = LRP_pd.iloc[:, i].reset_index()

edges = edges.sort_values('LRP', ascending = False).reset_index(drop=True)
edges['LRP'].plot()

from kneefinder import KneeFinder
kf = KneeFinder(edges.index, edges['LRP'])
knee_x, knee_y = kf.find_knee()
# plotting to check the results
kf.plot()


# %%

edges = LRP_pd.mean(axis=1).reset_index()

for i in range(10):
    edges = LRP_pd.iloc[:, i].reset_index()
    
    
    edges['source_gene']  = edges['index'].str.split(' - ', expand = True)[0]
    edges['target_gene']= edges['index'].str.split(' - ', expand = True)[1]
    #edges['source_gene'] = edges['source_gene'].str.split('_', expand=True)[0]
    #edges['target_gene'] = edges['target_gene'].str.split('_', expand=True)[0]
    edges  = edges.rename(columns = {edges.columns[1]:'LRP'})
    edges['LRP_norm'] = edges['LRP'] / edges['LRP'].max()
    edges = edges.sort_values('LRP', ascending = False).reset_index(drop=True)
    kf = KneeFinder(edges.index, edges['LRP'])
    knee_x, knee_y = kf.find_knee()
    edges = edges.iloc[:int(knee_x), :]
    
    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP_norm')
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    # nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*100, node_size = network.degree)
    
        
    
    widths = edges['LRP'] / edges['LRP'].max() 
    
    edges_from_G = [i[0] + ' - ' + i[1] for i in list(G.edges)]
    edge_colors = list(f.color_mapper(edges_from_G).values())
    
    node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}
    if ax == None:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        
        
    pos = nx.spring_layout(G, weight='LRP_norm')
    
    fig, ax = plt.subplots(figsize=(30,30))   
        # Plot legend
    ls = list(node_color_mapper.keys())
    cl = list(node_color_mapper.values())
    for label, color in zip(ls, cl):
        ax.plot([], [], 'o', color=color, label=label)
    ax.legend(title = 'Nodes', loc='best')
    
    node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper)
    
    
        
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