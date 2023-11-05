# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:41:23 2023

@author: d07321ow
"""

%matplotlib inline 
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
from kneefinder import KneeFinder
from kneed import KneeLocator

# %%

import community as community_louvain
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score

from cdlib import algorithms

def detect_communities(G):
    # Use the Louvain method for community detection
    #partition = community_louvain.best_partition(G)
    G_ig = convert_networkx_to_igraph(G)
    partition = algorithms.leiden(G_ig).to_node_community_map()
    partition = {key: value[0] for key, value in partition.items()}

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


import igraph as ig
import community as community_louvain
from cdlib import algorithms
def convert_networkx_to_igraph(nx_graph):
    # Create a mapping from networkx nodes to sequential integers
    mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    # Create the igraph graph
    g_ig = ig.Graph(directed=nx_graph.is_directed())
    # Add vertices with the name attribute set as the original networkx node identifiers
    g_ig.add_vertices(len(mapping))
    g_ig.vs['name'] = list(nx_graph.nodes())
    # Add edges by converting networkx nodes to the integer identifiers used in igraph
    ig_edges = [(mapping[edge[0]], mapping[edge[1]]) for edge in nx_graph.edges()]
    g_ig.add_edges(ig_edges)
    # If the graph is weighted, transfer the weights
    if nx.is_weighted(nx_graph):
        weights = [nx_graph[u][v]['weight'] for u,v in nx_graph.edges()]
        g_ig.es['weight'] = weights
    return g_ig

from collections import defaultdict

# Function to find overlapping communities
def find_recurring_communities(community_maps):
    recurring_community_pairs = defaultdict(list)
    for i, comm_map_i in enumerate(community_maps[:-1]):
        for j, comm_map_j in enumerate(community_maps[i+1:], start=i+1):
            for comm_i, nodes_i in comm_map_i.items():
                for comm_j, nodes_j in comm_map_j.items():
                    overlap = set(nodes_i) & set(nodes_j)
                    if overlap:  # If there's a significant overlap, you can define 'significant' as needed
                        recurring_community_pairs[(i, comm_i)].append((j, comm_j))
    return recurring_community_pairs

def analyze_recurring_communities(recurring_communities, threshold=0.5):
    # This function assumes a significant overlap is at least 50% shared nodes by default
    analysis_results = {}
    for (i, comm_i), mappings in recurring_communities.items():
        for (j, comm_j) in mappings:
            overlap = len(set(community_maps[i][comm_i]) & set(community_maps[j][comm_j]))
            total_nodes = max(len(community_maps[i][comm_i]), len(community_maps[j][comm_j]))
            overlap_ratio = overlap / total_nodes
            
            if overlap_ratio >= threshold:
                analysis_results.setdefault((i, comm_i), []).append((j, comm_j, overlap_ratio))

    return analysis_results        
def rank_recurring_communities(recurring_community_analysis):
    # Create a list to hold all overlaps with their details
    overlap_details = []

    # Iterate over the analysis results to collect overlap information
    for (net_i, comm_i), mappings in recurring_community_analysis.items():
        for (net_j, comm_j, ratio) in mappings:
            overlap_details.append({
                'network_i': net_i,
                'community_i': comm_i,
                'network_j': net_j,
                'community_j': comm_j,
                'overlap_ratio': ratio
            })

    # Sort the list by overlap_ratio in descending order
    ranked_overlaps = sorted(overlap_details, key=lambda x: x['overlap_ratio'], reverse=True)

    return ranked_overlaps

def rank_by_frequency(recurring_community_analysis):
    frequency_dict = {}

    # Count the frequency of each community across all networks
    for (net_i, comm_i), mappings in recurring_community_analysis.items():
        if (net_i, comm_i) not in frequency_dict:
            frequency_dict[(net_i, comm_i)] = set()
        for (net_j, comm_j, _) in mappings:
            frequency_dict[(net_i, comm_i)].add(net_j)

    # Calculate the frequency as the number of unique networks in which the community recurs
    for key, value in frequency_dict.items():
        frequency_dict[key] = len(value)

    # Sort communities by their frequency in descending order
    sorted_communities = sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True)

    return sorted_communities


# Jaccard Similarity Function
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0
def dices_coefficient(set1, set2):
    intersection_size = len(set1.intersection(set2))
    sum_set_sizes = len(set1) + len(set2)
    if sum_set_sizes == 0:
        return 0.0  # Avoid division by zero
    return (2 * intersection_size) / sum_set_sizes
def overlap_coefficient(set1, set2):
    intersection_size = len(set1.intersection(set2))
    smaller_set_size = min(len(set1), len(set2))
    if smaller_set_size == 0:
        return 0.0  # Avoid division by zero
    return intersection_size / smaller_set_size
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
        node_size=degrees_norm * 1000,
        edgecolors='white',  # This adds the white border
    linewidths=0.5)


ax.set_title('Top1000, all 5300 interactions, mean LRP')
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\global_network1000.png')


# %%% interaction > knee point

# lrps = np.sort(LRP_pd.values.ravel())[::-1]
# plt.plot(lrps)

# kf = KneeFinder(range(len(lrps)), lrps)
# knee_x, knee_y = kf.find_knee()
# kf.plot()



path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300'
name_to_save = 'global_network1000_knee.png'
title = 'Top1000\nfilted using knee point\nmean LRP'

edges_global = LRP_pd.mean(axis=1).reset_index()
edges_global['source_gene']  = edges_global['index'].str.split(' - ', expand = True)[0]
edges_global['target_gene']= edges_global['index'].str.split(' - ', expand = True)[1]
#edges['source_gene'] = edges['source_gene'].str.split('_', expand=True)[0]
#edges['target_gene'] = edges['target_gene'].str.split('_', expand=True)[0]
edges_global  = edges_global.rename(columns = {edges_global.columns[1]:'LRP'})
edges_global['LRP_norm'] = edges_global['LRP'] / edges_global['LRP'].max()
edges_global = edges_global.sort_values('LRP', ascending = False).reset_index(drop=True)
# kf = KneeFinder(edges_global.index, edges_global['LRP'])
# knee_x_global, knee_y = kf.find_knee()
# kf.plot()


kl = KneeLocator(edges_global.index, edges_global['LRP'], curve="convex", direction = 'decreasing', S=1, interp_method='polynomial')
knee_x_global = int(kl.knee)
kl.plot_knee(figsize= (4,4), title = title + ' selected: ' + str(knee_x_global), ylabel = 'LRP')
plt.savefig(os.path.join(path_to_save, 'knee_' + name_to_save))
plt.show()
kl.plot_knee_normalized(figsize= (4,4), title = title)
plt.show()


#knee_x=2000
edges_global = edges_global.iloc[:int(knee_x_global), :]
title = title + '\nInteractions: ' + str(int(edges_global.shape[0]))
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
    
#pos_global = nx.spring_layout(G, weight='LRP_norm')
pos_global = nx.spring_layout(G, weight='LRP_norm')

fig, ax = plt.subplots(figsize=(10,10))   
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
        pos=pos_global,
        # font_size=0,
        # cmap = colors,
        edge_color=edge_colors,
        ax=ax,
        # node_size = degrees,
        node_size=degrees_norm * 500,
        edgecolors='white',  # This adds the white border
    linewidths=0.5)


ax.set_title(title)
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, name_to_save))


# %% examples of individual networks

kl = KneeLocator(edges.index, edges['LRP'], curve="convex", direction = 'decreasing', interp_method="polynomial",)
kl.plot_knee()

# %%% networks colored by types etc
for i in range(10):
    
    path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300'
    name_to_save = 'global_network1000_knee_example_{}.png'.format(i)
    title = 'Example {}\nFiltered using knee point\n'.format(i)
        
    edges = LRP_pd.iloc[:,i].reset_index()
    edges['source_gene']  = edges['index'].str.split(' - ', expand = True)[0]
    edges['target_gene']= edges['index'].str.split(' - ', expand = True)[1]
    #edges['source_gene'] = edges['source_gene'].str.split('_', expand=True)[0]
    #edges['target_gene'] = edges['target_gene'].str.split('_', expand=True)[0]
    edges  = edges.rename(columns = {edges.columns[1]:'LRP'})
    edges['LRP_norm'] = edges['LRP'] / edges['LRP'].max()
    edges = edges.sort_values('LRP', ascending = False).reset_index(drop=True)

    kf = KneeFinder(edges.index, edges['LRP'])
    knee_x, knee_y = kf.find_knee()
    #kf.plot()
    #plt.tight_layout()
    #plt.savefig(os.path.join(path_to_save, 'knee_' + name_to_save))
    #plt.show()
    
    kl = KneeLocator(edges.index, edges['LRP'], curve="convex", direction = 'decreasing', S=1, interp_method='polynomial')
    knee_x = int(kl.knee)
    kl.plot_knee(figsize= (4,3), title = title + '\nSelected: ' + str(knee_x), ylabel = 'LRP')
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, 'knee_' + name_to_save))
    plt.show()
    kl.plot_knee_normalized(figsize= (4,3), title = title)
    plt.show()
    
    edges = edges.iloc[:int(knee_x), :].reset_index(drop=True)
    title = title + '\nInteractions: ' + str(int(edges.shape[0]))

    
    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP_norm')
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = edges['LRP'] / edges['LRP'].max() 
    
    edges_from_G = [i[0] + ' - ' + i[1] for i in list(G.edges)]
    edge_colors = list(f.color_mapper(edges_from_G).values())
    #edge_colors = plt.cm.jet(widths)  # 'viridis' is a colormap, you can choose any
        
    node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}
        
    pos = nx.spring_layout(G, weight='LRP_norm')
    
    fig, ax = plt.subplots(figsize=(7,7))   
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
            pos=pos, # font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 500,
            edgecolors='white',  # This adds the white border
        linewidths=0.5)
    
    
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, name_to_save))
    plt.show()

# %%% netowrks colored by communiites
    


for i in range(10):
    
    path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300'
    name_to_save = 'global_network1000_knee_example_{}.png'.format(i)
    title = 'Example {}\nFiltered using knee point\n'.format(i)
        
    edges = LRP_pd.iloc[:,i].reset_index()
    edges['source_gene']  = edges['index'].str.split(' - ', expand = True)[0]
    edges['target_gene']= edges['index'].str.split(' - ', expand = True)[1]
    #edges['source_gene'] = edges['source_gene'].str.split('_', expand=True)[0]
    #edges['target_gene'] = edges['target_gene'].str.split('_', expand=True)[0]
    edges  = edges.rename(columns = {edges.columns[1]:'LRP'})
    edges['LRP_norm'] = edges['LRP'] / edges['LRP'].max()
    edges = edges.sort_values('LRP', ascending = False).reset_index(drop=True)

    kf = KneeFinder(edges.index, edges['LRP'])
    knee_x, knee_y = kf.find_knee()
    #kf.plot()
    #plt.tight_layout()
    #plt.savefig(os.path.join(path_to_save, 'knee_' + name_to_save))
    #plt.show()
    
    kl = KneeLocator(edges.index, edges['LRP'], curve="convex", direction = 'decreasing', S=1, interp_method='polynomial')
    knee_x = int(kl.knee)
    #kl.plot_knee(figsize= (4,3), title = title + '\nSelected: ' + str(knee_x), ylabel = 'LRP')
    #plt.tight_layout()
    #plt.savefig(os.path.join(path_to_save, 'knee_' + name_to_save))
    #plt.show()
    #kl.plot_knee_normalized(figsize= (4,3), title = title)
    #plt.show()
    
    edges = edges.iloc[:int(knee_x), :].reset_index(drop=True)
    title = title + '\nInteractions: ' + str(int(edges.shape[0]))

    
    
    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP_norm')
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = edges['LRP'] / edges['LRP'].max() 
    
    edge_colors = plt.cm.Greys(widths)  # 'viridis' is a colormap, you can choose any
        
    # NODE COLORS by community
    G_ig = convert_networkx_to_igraph(G)
    coms = algorithms.leiden(G_ig, weights = widths)

    communities_leiden = coms.to_node_community_map()
    communities_leiden = {key: value[0] for key, value in communities_leiden.items()}

    # Create a color map for the communities
    community_colors = {node: communities_leiden.get(node) for node in G.nodes()}

    # Now we can generate a unique color for each community
    unique_communities = list(set(community_colors.values()))
    community_color_map = plt.cm.jet(np.linspace(0, 1, len(unique_communities)))
    community_color_map = {com: community_color_map[i] for i, com in enumerate(unique_communities)}

    # Map the community colors to each node
    node_colors = [community_color_map[community_colors[node]] for node in G.nodes()]

        
    pos = nx.spring_layout(G, weight='LRP_norm')
    
    fig, ax = plt.subplots(figsize=(7,7))   
        # Plot legend
    community_labels = {com: f'Community {com}' for com in unique_communities}

    for community in unique_communities:
        ax.plot([], [], 'o', color=community_color_map[community], label=community_labels[community])
    ax.legend(title='Communities', loc='best')
        
    nx.draw(G, with_labels=False,
            node_color=node_colors,
            width=widths*5,
            pos=pos, # font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 500,
            edgecolors='white',  # This adds the white border
        linewidths=0.5)
    
    
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, name_to_save))
    plt.show()





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

edges = LRP_pd.iloc[:,0].reset_index()
edges['source_gene']  = edges['index'].str.split(' - ', expand = True)[0]
edges['target_gene']= edges['index'].str.split(' - ', expand = True)[1]
G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene')

nodes_all = list(G.nodes)
nodes_labaled_all = pd.DataFrame(nodes_all, columns=['node'])

G_list = []
partitions_list = []
n_edges = []
n_nodes = []
for i in range(988):
    print(i)
    edges = LRP_pd.iloc[:,i].reset_index()
        
    edges = LRP_pd.iloc[:,i].reset_index()
    edges['source_gene']  = edges['index'].str.split(' - ', expand = True)[0]
    edges['target_gene']= edges['index'].str.split(' - ', expand = True)[1]
    #edges['source_gene'] = edges['source_gene'].str.split('_', expand=True)[0]
    #edges['target_gene'] = edges['target_gene'].str.split('_', expand=True)[0]
    edges  = edges.rename(columns = {edges.columns[1]:'LRP'})
    edges['LRP_norm'] = edges['LRP'] / edges['LRP'].max()
    edges = edges.sort_values('LRP', ascending = False).reset_index(drop=True)
    
    kl = KneeLocator(edges.index, edges['LRP'], curve="convex", direction = 'decreasing', S=1, interp_method='polynomial')
    knee_x = int(kl.knee)
    
    edges = edges.iloc[:int(knee_x), :].reset_index(drop=True)
      
    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP_norm')
    partition = detect_communities(G)
    
    G_list.append(G)
    partitions_list.append(partition)
    
    labels = communities_to_labels(partition)
    edges_ = edges['index'].to_list()
    nodes_ = list(G.nodes)
    n_edges.append(len(edges_))
    n_nodes.append(len(nodes_))
    nodes_labaled = pd.DataFrame(np.array([nodes_, labels]).T, columns = ['node', 'community_{}'.format(i)])
    
    nodes_labaled_all = nodes_labaled_all.merge(nodes_labaled, on = 'node', how = 'left')
 
    
# plots with statistics

plt.hist(n_edges, bins=30)
plt.title('Histogram of the no. interactions\nin 988 networks after knee point threshold\nAverage ' + str(int(np.mean(n_edges))))
print('Average no. nodes: ', int(np.mean(n_edges)))
plt.show()

plt.hist(n_nodes, bins=30)
plt.title('Histogram of the no. nodes\nin 988 networks after knee point threshold\nAverage ' + str(int(np.mean(n_nodes))))
print('Average no. nodes: ', int(np.mean(n_nodes)))


# %%% communities in clusters

cluster_labels_df = pd.read_csv(os.path.join(path_to_save, 'cluster_labels_shared_edges.csv'), index_col = 0)

for cluster_id in np.sort(cluster_labels_df['clusters_shared_edges'].unique()):
    
    

    index_ = list((cluster_labels_df['clusters_shared_edges'] == cluster_id))
    print('Cluster: ', cluster_id, 'size: ', np.sum(index_))

    
    community_maps = []
    for partition in np.array(partitions_list)[index_]:
        community_map = {}
        for node, comm in partition.items():
            community_map.setdefault(comm, []).append(node)
        community_maps.append(community_map)
        


    recurring_communities = find_recurring_communities(community_maps)
        # Continuing from the previous step's recurring_communities dictionary

    
    # Analyze recurring communities with a threshold for significant overlap
    recurring_community_analysis = analyze_recurring_communities(recurring_communities, threshold=0.75)
    
    # Output the analysis results
    for (net_i, comm_i), mappings in recurring_community_analysis.items():
        print(f"Community {comm_i} in Network {net_i} recurs with the following communities:")
        for (net_j, comm_j, ratio) in mappings:
            print(f"- Community {comm_j} in Network {net_j} with an overlap ratio of {ratio:.2f}")
            


    
    
    # Rank the recurring communities by overlap
    ranked_recurring_communities = rank_recurring_communities(recurring_community_analysis)
    
    # Output the ranked results
    print("Ranked Recurring Communities (from highest overlap):")
    for overlap in ranked_recurring_communities:
        print(f"Network {overlap['network_i']} Community {overlap['community_i']} <--> "
              f"Network {overlap['network_j']} Community {overlap['community_j']} | "
              f"Overlap Ratio: {overlap['overlap_ratio']:.2f}")
    
    
    
    # Rank the recurring communities by frequency
    ranked_by_frequency = rank_by_frequency(recurring_community_analysis)
    
    # Output the ranked results
    print("Ranked Recurring Communities by Frequency:")
    for (network, community), frequency in ranked_by_frequency[:20]:
        print(f"Network {network} Community {community} recurs in {frequency} other networks.")
    
        
        
    
    #df = pd.DataFrame(ranked_by_frequency)
    df = pd.DataFrame(ranked_by_frequency, columns=['network_community', 'frequency'])
    
    # Split the 'network_community' tuple into separate columns
    df[['network', 'community']] = pd.DataFrame(df['network_community'].tolist(), index=df.index)
    #df[['network', 'community']] = pd.DataFrame(df['network_communities'].tolist(), index=df.index)
    # Drop the 'network_community' column as it's no longer needed
    df.drop('network_community', axis=1, inplace=True)
    # Reorder the columns
    df = df[['network', 'community', 'frequency']]
    #df['genes'] = df.apply(lambda row: community_maps[row['networks_communities'][0][0]][row['networks_communities'][0][1]], axis=1)
    df['genes'] = df.apply(lambda row: community_maps[row['network']][row['community']], axis=1)
    
    df['community_size'] = df['genes'].apply(len)
    
    df_temp = df.copy()
    df_temp['genes_lsit']= df_temp['genes'].astype('str')
    df_grouped = df_temp.groupby('genes_lsit').agg({'frequency': 'max', 'network': 'max','community':'max', 'community_size':'max', 'genes':'first'}).reset_index().sort_values('frequency', ascending=False).reset_index(drop=True)
    
    from itertools import combinations
    
    # First, you might want to select only the most frequent communities for comparison
    
    min_frequency = 5
    min_community_size = 3
    
    df_grouped_filtered = df_grouped[(df_grouped['frequency'] >= min_frequency) & (df_grouped['community_size'] >=min_community_size)]
    
    N = 606  # or however many you wish to compare
    top_communities = df_grouped_filtered.nlargest(N, 'frequency')
    
    # Compare each community with every other community in the top N
    # Use a DataFrame to store the similarities
    similarity_jaccard = pd.DataFrame(index=top_communities.index, columns=top_communities.index)
    similarity_dice = pd.DataFrame(index=top_communities.index, columns=top_communities.index)
    similarity_overlap = pd.DataFrame(index=top_communities.index, columns=top_communities.index)
    
    # Calculate the Jaccard similarity for each pair of communities
    for (idx1, row1), (idx2, row2) in combinations(top_communities.iterrows(), 2):
        community1_genes = set(row1['genes'])
        community2_genes = set(row2['genes'])
        similarity = jaccard_similarity(community1_genes, community2_genes)
        similarity_jaccard.at[idx1, idx2] = similarity
        similarity_jaccard.at[idx2, idx1] = similarity
        
        similarity = dices_coefficient(community1_genes, community2_genes)
        similarity_dice.at[idx1, idx2] = similarity
        similarity_dice.at[idx2, idx1] = similarity
        
        similarity = overlap_coefficient(community1_genes, community2_genes)
        similarity_overlap.at[idx1, idx2] = similarity
        similarity_overlap.at[idx2, idx1] = similarity
    
    # Fill diagonal with 1s for self-similarity
    np.fill_diagonal(similarity_jaccard.values, 1)
    np.fill_diagonal(similarity_dice.values, 1)
    np.fill_diagonal(similarity_overlap.values, 1)
    
    # Convert the DataFrame to a numeric type (it will be filled with strings otherwise)
    similarity_jaccard = similarity_jaccard.astype(float)
    similarity_dice = similarity_dice.astype(float)
    similarity_overlap = similarity_overlap.astype(float)
    
    sns.heatmap(similarity_jaccard, cmap = 'jet', vmin = 0, vmax = 1)
    sns.clustermap(similarity_jaccard, cmap = 'jet', vmin = 0, vmax = 1, method = 'ward')
    
    sns.heatmap(similarity_dice, cmap = 'jet', vmin = 0, vmax = 1)
    sns.clustermap(similarity_dice, cmap = 'jet', vmin = 0, vmax = 1, method = 'ward')
    
    sns.heatmap(similarity_overlap, cmap = 'jet', vmin = 0, vmax = 1)
    sns.clustermap(similarity_overlap, cmap = 'jet', vmin = 0, vmax = 1, method = 'ward')
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
# nodes_labaled_all.iloc[:, 1:] = nodes_labaled_all.iloc[:, 1:].fillna(-1).astype('int')
# nodes_labaled_all.iloc[:, 1:] = nodes_labaled_all.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# sns.heatmap(nodes_labaled_all.iloc[:, 1:])
# sns.clustermap(nodes_labaled_all.iloc[:, 1:], method = 'ward')

# a = nodes_labaled_all.iloc[:, 1]
# b = nodes_labaled_all.iloc[:, 2]


# # similarity based on communities

# shared_nodes = set(G_list[0].nodes) & set(G_list[1].nodes)
# labels1 = [partitions_list[0][node] for node in shared_nodes if node in partitions_list[0]]
# labels2 = [partitions_list[1][node] for node in shared_nodes if node in partitions_list[1]]

# # Calculate similarity metrics for the shared nodes
# ars = adjusted_rand_score(labels1, labels2)



# from itertools import combinations
# # Number of networks
# n_networks = len(G_list)
# # Create an n x n matrix filled with zeros (or NaN if you prefer)
# ars_matrix = np.zeros((n_networks, n_networks))
# amis_matrix = np.zeros((n_networks, n_networks))
# shared_nodes_matrix = np.zeros((n_networks, n_networks))
# shared_edges_matrix = np.zeros((n_networks, n_networks))

# # Iterate over all unique pairs of networks
# for (i, G1), (j, G2) in combinations(enumerate(G_list), 2):
#     print(i,j)
#     # Find the shared nodes between the two graphs
#     shared_nodes = set(G1.nodes) & set(G2.nodes)
#     # Fill the symmetric matrix positions
#     shared_nodes_matrix[i, j] = len(shared_nodes)
#     shared_nodes_matrix[j, i] = len(shared_nodes)
    
    
#     shared_edges = set(G1.edges) & set(G2.edges)
#     # Fill the symmetric matrix positions
#     shared_edges_matrix[i, j] = len(shared_edges)
#     shared_edges_matrix[j, i] = len(shared_edges)
    
    
#     # Get the community labels for the shared nodes in both networks
#     labels1 = [partitions_list[i][node] for node in shared_nodes if node in partitions_list[i]]
#     labels2 = [partitions_list[j][node] for node in shared_nodes if node in partitions_list[j]]

#     # Calculate the ARS between the two sets of labels
#     ars = adjusted_rand_score(labels1, labels2)
#     amis = adjusted_mutual_info_score(labels1, labels2)
    
#     # Fill the symmetric matrix positions
#     ars_matrix[i, j] = ars
#     ars_matrix[j, i] = ars
    
#     amis_matrix[i, j] = amis
#     amis_matrix[j, i] = amis

# sns.heatmap(ars_matrix)
# sns.clustermap(ars_matrix, method = 'ward')

# sns.heatmap(amis_matrix)
# sns.clustermap(amis_matrix, method = 'ward')



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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # %%
    
    
    
    import igraph as ig

    # G_ig = convert_networkx_to_igraph(G)
    # coms = algorithms.leiden(G_ig)

    # communities_louvain = community_louvain.best_partition(G)
    # communities_leiden = coms.to_node_community_map()
    # communities_leiden = {key: value[0] for key, value in communities_leiden.items()}


    edges = LRP_pd.iloc[:,100].reset_index()

    edges['source_gene']  = edges['index'].str.split(' - ', expand = True)[0]
    edges['target_gene']= edges['index'].str.split(' - ', expand = True)[1]
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


    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP_norm')
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
    partition1 = detect_communities(G)
    partition2 = detect_communities(G)

    # Convert partitions to labels
    labels1 = communities_to_labels(partition1)
    labels2 = communities_to_labels(partition2)

    # Compare the communities
    score = compare_communities(labels1, labels2)

    eigenvalues1, _ = np.linalg.eig(nx.adjacency_matrix(G1).todense())
    eigenvalues2, _ = np.linalg.eig(nx.adjacency_matrix(G2).todense())

    # Plot eigenvalues
    plt.scatter(range(len(eigenvalues1)), np.sort(eigenvalues1), label='Network 1')
    plt.scatter(range(len(eigenvalues2)), np.sort(eigenvalues2), label='Network 2')
    plt.legend()
    plt.show()
