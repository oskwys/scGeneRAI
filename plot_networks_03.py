# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:41:23 2023

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

import community as community_louvain
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score

from cdlib import algorithms
import random
random.seed(42)
np.random.seed(42)

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
        print('i: ', i)
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
        print(i, comm_i)
        for (j, comm_j) in mappings:
            overlap = len(set(community_maps[i][comm_i]) & set(community_maps[j][comm_j]))
            total_nodes = max(len(community_maps[i][comm_i]), len(community_maps[j][comm_j]))
            overlap_ratio = overlap / total_nodes
            
            if overlap_ratio >= threshold:
                print(overlap_ratio)
                #print(' ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ')
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
    overlap_dict = {}
    # Count the frequency of each community across all networks
    for (net_i, comm_i), mappings in recurring_community_analysis.items():
        if (net_i, comm_i) not in frequency_dict:
            frequency_dict[(net_i, comm_i)] = set()
            overlap_dict[(net_i, comm_i)] = []
        for (net_j, comm_j, overlap_j) in mappings:
            frequency_dict[(net_i, comm_i)].add(net_j)
            overlap_dict[(net_i, comm_i)].append(overlap_j)
            
    frequency_dict_ = frequency_dict.copy()
    # Calculate the frequency as the number of unique networks in which the community recurs
    for key, value in frequency_dict.items():
        frequency_dict[key] = len(value)

    # Sort communities by their frequency in descending order
    sorted_communities = sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_communities, frequency_dict_, overlap_dict


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
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300'

topn = 1000

LRP_pd = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\LRP_individual_top{}.csv'.format(topn), index_col = 0)


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

plot_graphs = True

for i in range(988):
    print(i)
    edges = LRP_pd.iloc[:,i].reset_index()
        
    edges = LRP_pd.iloc[:,i].reset_index()
    edges['source_gene']  = edges['index'].str.split(' - ', expand = True)[0]
    edges['target_gene']= edges['index'].str.split(' - ', expand = True)[1]
    #edges['source_gene'] = edges['source_gene'].str.split('_', expand=True)[0]
    #edges['target_gene'] = edges['target_gene'].str.split('_', expand=True)[0]
    edges  = edges.rename(columns = {edges.columns[1]:'LRP'})
    #edges['LRP'] = edges['LRP'] / edges['LRP'].max()
    edges['LRP_norm'] = edges['LRP'] / edges['LRP'].max()
    edges = edges.sort_values('LRP', ascending = False).reset_index(drop=True)
    
    kl = KneeLocator(edges.index, edges['LRP'], curve="convex", direction = 'decreasing', S=1, interp_method='polynomial')
    knee_x = int(kl.knee)
    
    edges = edges.iloc[:int(knee_x), :].reset_index(drop=True)
      
    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr=['LRP','LRP_norm'])
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


# %%% plot and save all 988 networks by clusters
cluster_labels_df = pd.read_csv(os.path.join(path_to_save, 'cluster_labels_shared_edges.csv'), index_col = 0)

pos_988 = []
for i in range(988):
    cluster_id = cluster_labels_df.loc[i, 'clusters_shared_edges']
    print(i, 'cluster ', cluster_id)
    
    G = G_list[i]
    partition = partitions_list[i]
    
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = list(nx.get_edge_attributes(G, 'LRP_norm').values())
    widths = [x*2 for x in widths]
    
    edge_colors = plt.cm.Greys(widths)  # 'viridis' is a colormap, you can choose any
        
    # NODE COLORS by community
    community_colors = {node: partition.get(node) for node in G.nodes()}

    # Now we can generate a unique color for each community
    unique_communities = list(set(community_colors.values()))
    community_color_map = plt.cm.jet(np.linspace(0, 1, len(unique_communities)))
    community_color_map = {com: community_color_map[i] for i, com in enumerate(unique_communities)}

    # Map the community colors to each node
    node_colors = [community_color_map[community_colors[node]] for node in G.nodes()]
            
    pos = nx.spring_layout(G, weight='LRP_norm')
    pos_988.append(pos)
    fig, ax = plt.subplots(figsize=(7,7))   
        # Plot legend
    community_labels = {com: f'Community {com}' for com in unique_communities}

    #for community in unique_communities:
     #   ax.plot([], [], 'o', color=community_color_map[community], label=community_labels[community])
    #ax.legend(title='Communities', loc='best')
        
    nx.draw(G, with_labels=False,
            node_color=node_colors,
            width=widths*10,
            pos=pos, # font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 500,
            edgecolors='white',  # This adds the white border
        linewidths=0.5)
    
    title = 'Cluster: ' + str(cluster_id) + ' sample_i: ' + str(i) + '\n' + samples[i]
    ax.set_title(title)
    plt.tight_layout()
    name_to_save = 'network_cluster' + str(cluster_id) + '_' + str(i) + '_' + samples[i]
    plt.savefig(os.path.join(path_to_save + '\\all_988_networks', name_to_save))
    plt.show()


# %%%  filtered interactions analysis

edges_dict= {}
for i, G in enumerate(G_list):
    print(i)
    edges = ['-'.join(list(set(list(x)))) for x in list(G.edges)]
    values = [x[2] for x in list(G.edges(data = 'LRP'))]
    temp = pd.DataFrame([edges, values]).T
    temp = temp.rename(columns = {0:'edge', 1:i})
    
    edges_dict[str(i)] = temp
    
edges_all_filtered
for i in range(988):
    print(i)
    if i ==0:
        edges_all_filtered = edges_dict[str(i)] 
        
    else:
        edges_all_filtered = edges_all_filtered.merge( edges_dict[str(i)], on = 'edge', how  ='outer')

edges_all_filtered = edges_all_filtered.fillna(0)
edges_all_filtered = edges_all_filtered.sort_values('edge')
edges_all_filtered.to_csv(os.path.join(path, 'LRP_individual_top1000_filtered.csv'))


edges_all_filtered = edges_all_filtered.set_index('edge')
sns.clustermap(edges_all_filtered, method = 'ward', mask = edges_all_filtered==0, vmax = 0.1)


Z_x = linkage(edges_all_filtered, method = 'ward')
Z_y = linkage(edges_all_filtered.T, method = 'ward')
#sns.clustermap(a, cmap = 'Reds', vmin = 0, col_colors=col_colors, yticklabels = True, figsize = (20,20))
fg = sns.clustermap(edges_all_filtered, method = 'ward', cmap = 'jet', vmax = 0.1, mask = edges_all_filtered==0,
                    yticklabels = False, figsize = (10,10), row_linkage = Z_x, col_linkage = Z_y,
                    xticklabels = False,
                    #col_cluster =False
                    )
fg.ax_heatmap.set_xlabel('Sample')
fg.ax_heatmap.set_ylabel('Interaction')

plt.title('Overlap: {}'.format(threshold))
plt.savefig(os.path.join(path_to_save, 'clustermap_lrp_1000_filtered.png'))
plt.savefig(os.path.join(path_to_save, 'clustermap_lrp_1000_filtered.pdf'), format= 'pdf')



dendrogram_cutoff = 7
cutoff = dendrogram_cutoff
fig, ax = plt.subplots(figsize=(10, 3))
dn = dendrogram(
    Z_y, 
    color_threshold=cutoff, 
    above_threshold_color='gray', 
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=30,  # show only the last 12 merged clusters
    #show_leaf_counts=True,  # show the number of samples in each cluster
    ax=ax
)

plt.axhline(y=cutoff, color='r', linestyle='--')
ax.set_ylabel('Distance')
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'dendrogram_samples_lrp_filtered1000.png'))
plt.show()

cluster_labels = fcluster(Z_y, t=cutoff, criterion='distance')
samples_cluster2 = pd.Series(samples)[cluster_labels==2]
samples_i_cluster2 = pd.Series(range(988))[cluster_labels==2].values




dendrogram_cutoff = 7
cutoff = dendrogram_cutoff
fig, ax = plt.subplots(figsize=(10, 3))
dn = dendrogram(
    Z_x, 
    color_threshold=cutoff, 
    above_threshold_color='gray', 
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=30,  # show only the last 12 merged clusters
    #show_leaf_counts=True,  # show the number of samples in each cluster
    ax=ax
)

plt.axhline(y=cutoff, color='r', linestyle='--')
ax.set_ylabel('Distance')
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'dendrogram_interaction_lrp_filtered1000.png'))
plt.show()


cluster_labels = fcluster(Z_x, t=cutoff, criterion='distance')
edges_cluster2 = pd.Series(list(edges_all_filtered.index))[cluster_labels==2]
nodes_cluster2 = list(set(edges_cluster2.str.split('-',expand = True).melt()['value'].to_list()))
nodes_cluster2 .sort()

# %%%% network stats
def calculate_combined_network_metrics(G, edge_subset, nodes_cluster2):
    # Function to calculate metrics for a given subgraph
    def calculate_metrics_for_subgraph(subgraph, prefix):
        degrees = [degree for _, degree in subgraph.degree()]
        average_degree = sum(degrees) / len(degrees) if degrees else 0

        subgraph_metrics = {
            f'{prefix}_average_node_degree': average_degree,
            f'{prefix}_degree_distribution': degrees,
            f'{prefix}_average_clustering': nx.average_clustering(subgraph),
            f'{prefix}_density': nx.density(subgraph),
            f'{prefix}_betweenness_centrality': nx.betweenness_centrality(subgraph),
            f'{prefix}_closeness_centrality': nx.closeness_centrality(subgraph),
            f'{prefix}_degree_centrality': nx.degree_centrality(subgraph)
        }

        if nx.is_connected(subgraph):
            subgraph_metrics[f'{prefix}_average_shortest_path_length'] = nx.average_shortest_path_length(subgraph)
        else:
            subgraph_metrics[f'{prefix}_average_shortest_path_length'] = 'Graph is not connected'

        try:
            partition = community_louvain.best_partition(subgraph)
            subgraph_metrics[f'{prefix}_modularity'] = community_louvain.modularity(partition, subgraph)
        except ValueError:
            subgraph_metrics[f'{prefix}_modularity'] = np.nan

        return subgraph_metrics

    # Edge pairs metrics
    edge_subset = [tuple(x.split('-')) for x in edge_subset]
    subgraph_edges = G.edge_subgraph(edge_subset).copy()
    remaining_edges = [edge for edge in G.edges() if edge not in edge_subset]
    subgraph_remaining = G.edge_subgraph(remaining_edges).copy()

    edge_metrics = calculate_metrics_for_subgraph(subgraph_edges, 'subset_edges')
    remaining_edge_metrics = calculate_metrics_for_subgraph(subgraph_remaining, 'remaining_edges')

    # Extract nodes from the edge subsets and check if they are in G
    nodes_in_subgraph_edges = {node for edge in edge_subset for node in edge if node in G}
    nodes_in_remaining_edges = set(G.nodes()) - nodes_in_subgraph_edges

    # Group centrality metrics
    group_metrics = {
        'group_closeness_centrality_cluster2': nx.group_closeness_centrality(G, list(nodes_in_subgraph_edges)),
        'group_closeness_centrality_rest': nx.group_closeness_centrality(G, list(nodes_in_remaining_edges)),
        'group_degree_centrality_cluster2': nx.group_degree_centrality(G, list(nodes_in_subgraph_edges)),
        'group_degree_centrality_rest': nx.group_degree_centrality(G, list(nodes_in_remaining_edges))
    }

    # Combine and return all metrics
    all_metrics = {**edge_metrics, **remaining_edge_metrics, **group_metrics}
    return all_metrics


metrics = calculate_combined_network_metrics(G, list(edges_cluster2), nodes_cluster2)



metrics_cluster2 = pd.DataFrame()
for i in list(samples_i_cluster2):
    print(i)
    G = G_list[i]
    metrics = calculate_combined_network_metrics(G, list(edges_cluster2), nodes_cluster2)
    metrics = pd.DataFrame().from_dict(metrics, orient ='index').T
    metrics['sample_i'] = i
    metrics_cluster2 = pd.concat((metrics_cluster2, metrics))
    
metrics_nocluster2 = pd.DataFrame()
for i in [item for item in range(988) if item not in list(samples_i_cluster2)]:
    print(i)
    G = G_list[i]
    metrics = calculate_combined_network_metrics(G, list(edges_cluster2), nodes_cluster2)
    metrics = pd.DataFrame().from_dict(metrics, orient ='index').T
    metrics['sample_i'] = i
    metrics_nocluster2 = pd.concat((metrics_nocluster2, metrics))

metrics_cluster2 = metrics_cluster2.reset_index(drop=True)
metrics_cluster2_melted = metrics_cluster2.melt(id_vars='sample_i')
metrics_cluster2_melted['samples'] = 'cluster2'

metrics_nocluster2 = metrics_nocluster2.reset_index(drop=True)
metrics_nocluster2_melted = metrics_nocluster2.melt(id_vars='sample_i')
metrics_nocluster2_melted['samples'] = 'not_in_cluster2'

metrics_all = pd.concat((metrics_cluster2_melted, metrics_nocluster2_melted))
metrics_all.to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\metrics_graph_cluster2.csv')


def plot_violin_metrics(str_contain = ''):
    
    temp = metrics_all[metrics_all['variable'].str.contains(str_contain)]
    temp['value']=temp['value'].astype('float')
    #sns.stripplot(data = temp, x = 'samples', hue = 'variable' ,y = 'value')
    fig,ax = plt.subplots(figsize = (5,4))
    sns.violinplot(data = temp, x = 'samples', hue = 'variable' ,y = 'value',inner="quart", palette = ['red','blue'],split=True,ax=ax, labels = ['cluster','other'])
    ax.set_ylabel(str_contain)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, title='Nodes', labels=['Cluster', 'Other'])
    ax.set_xticklabels(['Cluster', 'Other'])
    ax.set_ylim([0,None])
    ax.set_xlabel('Samples')
    plt.tight_layout()

plot_violin_metrics(str_contain = 'edges_average_node_degree')
plot_violin_metrics(str_contain = 'average_clustering')
plot_violin_metrics(str_contain = 'modularity')
plot_violin_metrics(str_contain = 'group_closeness_centrality')
plot_violin_metrics(str_contain = 'group_degree_centrality')



# %%%%% networks stats cluster2 with baseline
def calculate_group_metrics_with_baseline(G, genes_cluster, num_iterations=20):
    group_metrics = {}
    group_nodes = [node for node in genes_cluster if node in  G.nodes()]
    num_nodes_in_group = len(group_nodes)
    actual_metrics = compute_metrics_for_group(G, group_nodes)
    # Compute metrics for the complementary group (all other genes)
    complementary_nodes = [node for node in G if node not in group_nodes]
    complementary_metrics = compute_metrics_for_group(G, complementary_nodes)
    # Compute metrics for random groups
    random_metrics = {metric: [] for metric in actual_metrics}
    all_nodes = list(G.nodes())  # Convert G.nodes() to a list
    for _ in range(num_iterations):
        if num_nodes_in_group > len(all_nodes):
            continue
        random_nodes = random.sample(all_nodes, num_nodes_in_group)
        metrics = compute_metrics_for_group(G, random_nodes)
        for metric, value in metrics.items():
            random_metrics[metric].append(value)
    # Calculate average of random metrics
    average_random_metrics = {metric: np.mean(values) for metric, values in random_metrics.items()}
    # Store the metrics
    group_metrics = {
        'actual': actual_metrics,
        'complementary': complementary_metrics,
        'average_random': average_random_metrics
    }
    return group_metrics

def compute_metrics_for_group(G, nodes):
    # Calculate clustering coefficients
    clustering_coeffs = nx.clustering(G, nodes=nodes)
    avg_clustering = np.mean(list(clustering_coeffs.values())) if clustering_coeffs else 0

    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)
    avg_degree_centrality = np.mean([degree_centrality[node] for node in nodes if node in degree_centrality])

    # Calculate betweenness centrality
    #betweenness_centrality = nx.betweenness_centrality(G)
    #avg_betweenness_centrality = np.mean([betweenness_centrality[node] for node in nodes if node in betweenness_centrality])

    # Calculate closeness centrality
    #closeness_centrality = nx.closeness_centrality(G)
    #avg_closeness_centrality = np.mean([closeness_centrality[node] for node in nodes if node in closeness_centrality])

    # Calculate edge density
    subgraph = G.subgraph(nodes)
    edge_density = nx.density(subgraph)

    return {
        'number_of_nodes_in_graph': len(nodes),
        'average_clustering_coefficient': avg_clustering,
        'average_degree_centrality': avg_degree_centrality,
        #'average_betweenness_centrality': avg_betweenness_centrality,
        #'average_closeness_centrality': avg_closeness_centrality,
        'edge_density': edge_density
    }


metrics_cluster2 = pd.DataFrame()
for i in list(samples_i_cluster2):
    print(i)
    G = G_list[i]
    metrics = calculate_group_metrics_with_baseline(G, nodes_cluster2)
    data_actual = [{'Metric': metric, 'Value': value, 'Type': 'Actual'}
                              for metric, value in metrics['actual'].items()]
    data_complementary = [{'Metric': metric, 'Value': value, 'Type': 'Average'}                   
                   for metric, value in metrics['complementary'].items()]
    data_random = [{'Metric': metric, 'Value': value, 'Type': 'Average Random'}
                                      for metric, value in metrics['average_random'].items()]
    
    # Concatenate the data
    temp = pd.concat([pd.DataFrame(data_actual),pd.DataFrame(data_complementary), pd.DataFrame(data_random)], ignore_index=True)
    #temp.rename(columns={'index': 'Pathway'}, inplace=True)
    temp['sample_i'] = i
    # Melt the DataFrame
    metrics_cluster2 = pd.concat((metrics_cluster2, temp))
    
metrics_nocluster2 = pd.DataFrame()
for i in [item for item in range(988) if item not in list(samples_i_cluster2)]:
    print(i)
    G = G_list[i]
    metrics = calculate_group_metrics_with_baseline(G, nodes_cluster2)
    data_actual = [{'Metric': metric, 'Value': value, 'Type': 'Actual'}
                              for metric, value in metrics['actual'].items()]
    data_complementary = [{'Metric': metric, 'Value': value, 'Type': 'Average'}                   
                   for metric, value in metrics['complementary'].items()]
    data_random = [{'Metric': metric, 'Value': value, 'Type': 'Average Random'}
                                      for metric, value in metrics['average_random'].items()]
    
    # Concatenate the data
    temp = pd.concat([pd.DataFrame(data_actual),pd.DataFrame(data_complementary), pd.DataFrame(data_random)], ignore_index=True)
    #temp.rename(columns={'index': 'Pathway'}, inplace=True)
    temp['sample_i'] = i
    metrics_nocluster2 = pd.concat((metrics_nocluster2, temp))

metrics_cluster2 = metrics_cluster2.reset_index(drop=True)
metrics_cluster2['samples'] = 'cluster2'

metrics_nocluster2 = metrics_nocluster2.reset_index(drop=True)
metrics_nocluster2['samples'] = 'not_in_cluster2'

metrics_all = pd.concat((metrics_cluster2, metrics_nocluster2))
metrics_all.to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\metrics_graph_cluster2_baseline.csv')


def plot_violin_metrics(str_contain = ''):
    
    temp = metrics_all[metrics_all['variable'].str.contains(str_contain)]
    temp['value']=temp['value'].astype('float')
    #sns.stripplot(data = temp, x = 'samples', hue = 'variable' ,y = 'value')
    fig,ax = plt.subplots(figsize = (5,4))
    sns.violinplot(data = temp, x = 'samples', hue = 'variable' ,y = 'value',inner="quart", palette = ['red','blue'],split=True,ax=ax, labels = ['cluster','other'])
    ax.set_ylabel(str_contain)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, title='Nodes', labels=['Cluster', 'Other'])
    ax.set_xticklabels(['Cluster', 'Other'])
    ax.set_ylim([0,None])
    ax.set_xlabel('Samples')
    plt.tight_layout()

plot_violin_metrics(str_contain = 'edges_average_node_degree')
metrics_all

temp = metrics_all[(metrics_all['Metric'].str.contains('average_clustering_coefficient')) & (metrics_all['Type'] != 'Average')]
temp = metrics_all[(metrics_all['Metric']!='number_of_nodes_in_graph') &  (metrics_all['Type'] != 'Average')]
temp = metrics_all
# Get unique metrics
unique_metrics = temp['Metric'].unique()


# Determine the grid size for subplots (you can adjust the layout as needed)
n_metrics = len(unique_metrics)
n_cols = 4  # for example, 3 columns
n_rows = (n_metrics + n_cols - 1) // n_cols  # calculate rows needed

# Create a figure with multiple subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), constrained_layout=True)
for i, metric in enumerate(unique_metrics):
    ax = axes.flatten()[i]  # Get the current axis for the subplot
    temp_filtered = temp[temp['Metric'] == metric]
    sns.violinplot(data=temp_filtered, x='Type', y='Value', inner="quart",
                   palette=['red', 'blue', 'lightblue'], 
                   order=['Actual', 'Average Random', 'Average'],
                   split=True, ax=ax)
    
    ax.set_title(metric)  # Set the title to the metric name
    ax.set_ylabel(None)
    ax.set_ylim([0, None])
    ax.set_xlabel(None)
# Turn off any unused subplots
for j in range(i+1, n_rows*n_cols):
    axes.flatten()[j].axis('off')
plt.show()


# Create a figure with multiple subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), constrained_layout=True)
for i, metric in enumerate(unique_metrics):
    ax = axes.flatten()[i]  # Get the current axis for the subplot
    temp_filtered = temp[temp['Metric'] == metric]
    sns.violinplot(data=temp_filtered, x='Type', y='Value', inner="quart",hue='samples',
                   palette=['red', 'blue', 'lightblue'], 
                   order=['Actual', 'Average Random', 'Average'],
                   split=True, ax=ax)
    
    ax.set_title(metric)  # Set the title to the metric name
    ax.set_ylabel(None)
    ax.set_ylim([0, None])
    ax.set_xlabel(None)
# Turn off any unused subplots
for j in range(i+1, n_rows*n_cols):
    axes.flatten()[j].axis('off')
plt.show()




# %%%% plot networks


pos_selected = {}
# colored by all communities
for i in range(988):
    
    print(i)
    
    G = G_list[i]
    partition = partitions_list[i]
    
    
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = list(nx.get_edge_attributes(G, 'LRP_norm').values())
    widths = [x*2 for x in widths]
    
    edge_colors = plt.cm.Greys(widths)  # 'viridis' is a colormap, you can choose any
        
    # NODE COLORS by community
    community_colors = {node: partition.get(node) for node in G.nodes()}

    # Now we can generate a unique color for each community
    unique_communities = list(set(community_colors.values()))
    community_color_map = plt.cm.jet(np.linspace(0, 1, len(unique_communities)))
    community_color_map = {com: community_color_map[i] for i, com in enumerate(unique_communities)}

    # Map the community colors to each node
    node_colors = [community_color_map[community_colors[node]] for node in G.nodes()]
            
    pos = nx.spring_layout(G, weight='LRP_norm')
    pos_selected[str(i)] = pos
    fig, ax = plt.subplots(figsize=(7,7))   
        # Plot legend
    community_labels = {com: f'Community {com}' for com in unique_communities}

    #for community in unique_communities:
     #   ax.plot([], [], 'o', color=community_color_map[community], label=community_labels[community])
    #ax.legend(title='Communities', loc='best')
        
    nx.draw(G, with_labels=False,
            node_color=node_colors,
            width=widths*10,
            pos=pos, # font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 500,
            edgecolors='white',  # This adds the white border
        linewidths=0.5)
    
    title = 'sample_i: ' + str(i) + '\n' + samples[i]
    ax.set_title(title)
    plt.tight_layout()
    if i in samples_i_cluster2:
        name_to_save = 'network_filtered' + str(i) + '_' + samples[i] + '_allcoms_cluster2'
    else:
        name_to_save = 'network_filtered' + str(i) + '_' + samples[i] + '_allcoms_notincluster2'
    
    plt.savefig(os.path.join(path_to_save + '\\networks_knee_filtered', name_to_save))
    plt.show()
      
    
# colored only reccuring community with genes from selected cluster
for i in range(988):
    
    print(i)
    
    G = G_list[i]
    partition = partitions_list[i]
    
    
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = list(nx.get_edge_attributes(G, 'LRP_norm').values())
    widths = [x*2 for x in widths]
    
    edge_colors = ['red' if '-'.join(list(set(list(edge)))) in edges_cluster2.to_list() else 'gray' for edge in G.edges()]
        
    # NODE COLORS by SELECTED GENES
    # Create a set of nodes that are in the edges list
    nodes_in_edges = set()
    for edge in edges_cluster2:
        nodes_in_edges.update(edge.split('-'))
    
    # Set node colors
    node_colors = ['red' if node in nodes_in_edges else 'blue' for node in G.nodes()]
     
    pos = pos_selected[str(i)]
    
    fig, ax = plt.subplots(figsize=(7,7))   
        # Plot legend
    #community_labels = {com: f'Community {com}' for com in unique_communities}

    #for community in unique_communities:
     #   ax.plot([], [], 'o', color=community_color_map[community], label=community_labels[community])
    #ax.legend(title='Communities', loc='best')
        
    nx.draw(G, with_labels=False,
            node_color=node_colors,
            width=widths*10,
            pos=pos, # font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 500,
            edgecolors='white',  # This adds the white border
            linewidths=0.5)
    
    title = 'sample_i: ' + str(i) + '\n' + samples[i]
    ax.set_title(title)
    plt.tight_layout()
    
    if i in samples_i_cluster2:
        name_to_save = 'network_filtered' + str(i) + '_' + samples[i] + '_selectedinteraction'
    else:
        name_to_save = 'network_filtered' + str(i) + '_' + samples[i] + '_notinselectedinteraction'
    plt.savefig(os.path.join(path_to_save + '\\networks_knee_filtered', name_to_save))
    plt.show()
    
    
    
# %%%%% PLOT 20 random networks from the cluster2
random_n_sample = random.sample(list(samples_i_cluster2),20)
fig, axs = plt.subplots(5,4,figsize=(20,20)) 
axs = axs.flatten()
j=0  
for i in random_n_sample:
    
    G = G_list[i]
    partition = partitions_list[i]
    
    
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = list(nx.get_edge_attributes(G, 'LRP_norm').values())
    widths = [x*2 for x in widths]
    
    edge_colors = ['red' if '-'.join(list(set(list(edge)))) in edges_cluster2.to_list() else 'gray' for edge in G.edges()]
        
    # NODE COLORS by SELECTED GENES
    # Create a set of nodes that are in the edges list
    nodes_in_edges = set()
    for edge in edges_cluster2:
        nodes_in_edges.update(edge.split('-'))
    
    # Set node colors
    node_colors = ['red' if node in nodes_in_edges else 'blue' for node in G.nodes()]
     
    pos = pos_selected[str(i)]
    ax =axs[j]
    nx.draw(G, with_labels=False,
            node_color=node_colors,
            width=widths*10,
            pos=pos, # font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 500,
            edgecolors='white',  # This adds the white border
            linewidths=0.5)
    
    title = samples[i]
    ax.set_title(title)
    j+=1
    
plt.tight_layout()
name_to_save = 'network_filtered' + '_selectedinteraction20'
plt.savefig(os.path.join(path_to_save + '\\networks_knee_filtered', name_to_save))
plt.savefig(os.path.join(path_to_save + '\\networks_knee_filtered', name_to_save+'.pdf'), format = 'pdf')


# %%%%% PLOT 20 random networks NOT from the cluster2
random_n_sample = random.sample([item for item in range(988) if item not in list(samples_i_cluster2)]  ,20)
fig, axs = plt.subplots(5,4,figsize=(20,20)) 
axs = axs.flatten()
j=0  
for i in random_n_sample:
    
    G = G_list[i]
    partition = partitions_list[i]
    
    
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = list(nx.get_edge_attributes(G, 'LRP_norm').values())
    widths = [x*2 for x in widths]
    
    edge_colors = ['red' if '-'.join(list(set(list(edge)))) in edges_cluster2.to_list() else 'gray' for edge in G.edges()]
        
    # NODE COLORS by SELECTED GENES
    # Create a set of nodes that are in the edges list
    nodes_in_edges = set()
    for edge in edges_cluster2:
        nodes_in_edges.update(edge.split('-'))
    
    # Set node colors
    node_colors = ['red' if node in nodes_in_edges else 'blue' for node in G.nodes()]
     
    pos = pos_selected[str(i)]
    ax =axs[j]
    nx.draw(G, with_labels=False,
            node_color=node_colors,
            width=widths*10,
            pos=pos, # font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 500,
            edgecolors='white',  # This adds the white border
            linewidths=0.5)
    
    title = samples[i]
    ax.set_title(title)
    j+=1
    
plt.tight_layout()
name_to_save = 'network_filtered' + '_notinselectedinteraction20'
plt.savefig(os.path.join(path_to_save + '\\networks_knee_filtered', name_to_save))
plt.savefig(os.path.join(path_to_save + '\\networks_knee_filtered', name_to_save+'.pdf'), format = 'pdf')

# %%%% networks - nodes colored by Pathway
import matplotlib.patches as mpatches

path_to_pathways = r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI\PATHWAYS'
genes_pathways = pd.read_csv(os.path.join(path_to_pathways, 'genes_pathways_pancanatlas_matched_cce.csv'))
genes_pathways_set = set(genes_pathways['cce_match'])

genes_pathways_dict = {}

for pathway in genes_pathways['Pathway'].unique():
    
    genes_pathways_dict[pathway] = genes_pathways.loc[genes_pathways['Pathway'] == pathway, 'Gene'].to_list()

colormap = plt.cm.tab10  # This is an example colormap with 10 distinct colors
colors = [colormap(i) for i in range(len(genes_pathways_dict.keys()))]
gene_to_pathway = {gene: pathway for pathway, genes in genes_pathways_dict.items() for gene in genes}
legend_patches = [mpatches.Patch(color=color, label=pathway) for pathway, color in color_dict.items()]
color_dict = dict(zip(genes_pathways_dict.keys(), colors))
light_gray_rgba = (0.83, 0.83, 0.83, 0.7) 
alpha_value = 0.6  # Set transparency level (0.0 to 1.0)
color_dict = {key: (*color[0:3], alpha_value) for key, color in color_dict.items()}

# Assign colors to keys

# colored only reccuring community with genes from selected cluster
for i in range(988):
    
    print(i)
    
    G = G_list[i]
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = list(nx.get_edge_attributes(G, 'LRP_norm').values())
    widths = [x*2 for x in widths]
    

    pos = pos_selected[str(i)]
    nodes = [x.split('_')[0] for x in G.nodes()]
    node_colors = [color_dict[gene_to_pathway[node]] if node in gene_to_pathway and gene_to_pathway[node] in color_dict else light_gray_rgba for node in nodes]
    edge_colors = plt.cm.Greys(widths)
    
    fig, ax = plt.subplots(figsize=(7,7))   
    nx.draw(G, with_labels=False,
            node_color=node_colors,
            width=widths*10,
            pos=pos, # font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 500,
            edgecolors='white',  # This adds the white border
            linewidths=0.5)
    ax.legend(handles=legend_patches, title='Pathways', loc='best')

    title = 'sample_i: ' + str(i) + '\n' + samples[i]
    ax.set_title(title)
    plt.tight_layout()
    
    name_to_save = 'network_filtered' + str(i) + '_' + samples[i] + '_coloredbypathway'
    plt.savefig(os.path.join(path_to_save + '\\networks_color_by_pathway', name_to_save))
    plt.show()
    
# %%%%% graph stats for pathways

def calculate_group_metrics_with_baseline(G, genes_pathways_dict, num_iterations=20):
    # Invert genes_pathways_dict to map each gene to its pathway
    gene_to_pathway = {gene: pathway for pathway, genes in genes_pathways_dict.items() for gene in genes}
    
    # Initialize a dictionary to store the metrics for each group and their random baselines
    group_metrics = {}

    for pathway, genes in genes_pathways_dict.items():
        # Filter nodes in this group that are present in G
        
        nodes = [x.split('_')[0] for x in G.nodes()]
        group_nodes = [node for node in nodes if node in gene_to_pathway and gene_to_pathway[node] == pathway]
        group_nodes = [x+'_exp' for x in group_nodes]
        
        #group_nodes = [node for node in genes if node in G]
        num_nodes_in_group = len(group_nodes)

        if num_nodes_in_group == 0:
            continue  # Skip if no nodes for this pathway in G

        # Compute metrics for the actual group
        actual_metrics = compute_metrics_for_group(G, group_nodes)

        # Compute metrics for the complementary group (all other genes)
        complementary_nodes = [node for node in G if node not in group_nodes]

        complementary_metrics = compute_metrics_for_group(G, complementary_nodes)

        # Compute metrics for random groups
        random_metrics = {metric: [] for metric in actual_metrics}
        all_nodes = list(G.nodes())  # Convert G.nodes() to a list
        for _ in range(num_iterations):
            if num_nodes_in_group > len(all_nodes):
                # Handle case where group size is larger than graph size
                continue
            random_nodes = random.sample(all_nodes, num_nodes_in_group)
            metrics = compute_metrics_for_group(G, random_nodes)
            for metric, value in metrics.items():
                random_metrics[metric].append(value)

        # Calculate average of random metrics
        average_random_metrics = {metric: np.mean(values) for metric, values in random_metrics.items()}

        # Store the metrics
        group_metrics[pathway] = {
            'actual': actual_metrics,
            'complementary': complementary_metrics,
            'average_random': average_random_metrics
        }

    return group_metrics

def compute_metrics_for_group(G, nodes):
    # Calculate clustering coefficients
    clustering_coeffs = nx.clustering(G, nodes=nodes)
    avg_clustering = np.mean(list(clustering_coeffs.values())) if clustering_coeffs else 0

    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)
    avg_degree_centrality = np.mean([degree_centrality[node] for node in nodes if node in degree_centrality])

    # Calculate betweenness centrality
    #betweenness_centrality = nx.betweenness_centrality(G)
    #avg_betweenness_centrality = np.mean([betweenness_centrality[node] for node in nodes if node in betweenness_centrality])

    # Calculate closeness centrality
    #closeness_centrality = nx.closeness_centrality(G)
    #avg_closeness_centrality = np.mean([closeness_centrality[node] for node in nodes if node in closeness_centrality])

    # Calculate edge density
    subgraph = G.subgraph(nodes)
    edge_density = nx.density(subgraph)

    return {
        'number_of_nodes_in_graph': len(nodes),
        'average_clustering_coefficient': avg_clustering,
        'average_degree_centrality': avg_degree_centrality,
        #'average_betweenness_centrality': avg_betweenness_centrality,
        #'average_closeness_centrality': avg_closeness_centrality,
        'edge_density': edge_density
    }




metrics_by_pathways = pd.DataFrame()
for i in range(988):
    
    print(i)
    
    G = G_list[i]
    metrics = calculate_group_metrics_with_baseline(G, genes_pathways_dict)
    
    data_actual = [{'Pathway': pathway, 'Metric': metric, 'Value': value, 'Type': 'Actual'}
               for pathway, mets in metrics.items()
               for metric, value in mets['actual'].items()]
    data_complementary = [{'Pathway': pathway, 'Metric': metric, 'Value': value, 'Type': 'Average'}
                   for pathway, mets in metrics.items()
                   for metric, value in mets['complementary'].items()]
    data_random = [{'Pathway': pathway, 'Metric': metric, 'Value': value, 'Type': 'Average Random'}
                   for pathway, mets in metrics.items()
                   for metric, value in mets['average_random'].items()]
    
    # Concatenate the data
    temp = pd.concat([pd.DataFrame(data_actual),pd.DataFrame(data_complementary), pd.DataFrame(data_random)], ignore_index=True)
    #temp.rename(columns={'index': 'Pathway'}, inplace=True)
    temp['sample_i'] = i
    # Melt the DataFrame
    metrics_by_pathways = pd.concat((metrics_by_pathways, temp))
    
metrics_by_pathways_melted = metrics_by_pathways#metrics_by_pathways.melt(id_vars=['Pathway', 'sample_i'], var_name='Metric', value_name='Value')



temp = metrics_all[(metrics_all['Metric'].str.contains('average_clustering_coefficient')) & (metrics_all['Type'] != 'Average')]
temp = metrics_all[(metrics_all['Metric']!='number_of_nodes_in_graph') &  (metrics_all['Type'] != 'Average')]

fig, axes = plt.subplots(len(genes_pathways_dict.keys()), 4, figsize=(4 * 5, 10 * 3), constrained_layout=True)
axes = axes.flatten()
k=0
for pathway in genes_pathways_dict.keys():
    temp = metrics_by_pathways_melted[metrics_by_pathways_melted['Pathway'] == pathway]
    # Get unique metrics
    unique_metrics = temp['Metric'].unique()
    
    
    # Determine the grid size for subplots (you can adjust the layout as needed)
    n_metrics = len(unique_metrics)
    n_cols = 4  # for example, 3 columns
    n_rows = (n_metrics + n_cols - 1) // n_cols  # calculate rows needed
    
    # Create a figure with multiple subplots
    
    for i, metric in enumerate(unique_metrics):
        ax = axes[k]  # Get the current axis for the subplot
        temp_filtered = temp[temp['Metric'] == metric]
        sns.violinplot(data=temp_filtered, x='Type', y='Value', inner="quart",
                       palette=['red', 'blue', 'lightblue'], 
                       order=['Actual', 'Average Random', 'Average'],
                       split=True, ax=ax)
        
        ax.set_title(pathway + '\n' + metric)  # Set the title to the metric name
        ax.set_ylabel(None)
        ax.set_ylim([0, None])
        ax.set_xlabel(None)
        k +=1
    # Turn off any unused subplots
    for j in range(i+1, n_rows*n_cols):
        axes.flatten()[j].axis('off')

name_to_save = 'pathway_network_stats'
plt.savefig(os.path.join(path_to_save + '\\networks_color_by_pathway', name_to_save))
plt.show()





for metric in metrics_by_pathways_melted['Metric'].unique():
    
    fig,ax = plt.subplots(figsize = (10,6))
    sns.violinplot(data = metrics_by_pathways_melted[(metrics_by_pathways_melted['Metric']==metric)&(metrics_by_pathways_melted['Type']!='Average Random')], x = 'Pathway', hue = 'Type' ,y = 'Value',inner="quart", split=True, palette = 'tab10', ax=ax)
    ax.set_ylabel(None)
    ax.set_ylim([0,None])
    #ax.legend(handles=handles, title='Nodes', labels=['Cluster', 'Other'])
    ax.set_title(metric)
    ax.set_xlabel(None)
    plt.tight_layout()    
    
    fig,ax = plt.subplots(figsize = (10,6))
    sns.boxplot(data = metrics_by_pathways_melted[(metrics_by_pathways_melted['Metric']==metric)&(metrics_by_pathways_melted['Type']!='Average Random')], x = 'Pathway', hue = 'Type' ,y = 'Value', palette = 'tab10', ax=ax, showfliers=False)
    ax.set_ylabel(None)
    ax.set_ylim([0,None])
    #ax.legend(handles=handles, title='Nodes', labels=['Cluster', 'Other'])
    ax.set_title(metric)
    ax.set_xlabel(None)
    plt.tight_layout()   
    
    fig,ax = plt.subplots(figsize = (10,6))
    sns.boxplot(data = metrics_by_pathways_melted[(metrics_by_pathways_melted['Metric']==metric)&(metrics_by_pathways_melted['Type']!='Average Random')], x = 'Pathway', hue = 'Type' ,y = 'Value', palette = 'tab10', ax=ax, showfliers=False)
    ax.set_ylabel(None)
    ax.set_ylim([0,None])
    #ax.legend(handles=handles, title='Nodes', labels=['Cluster', 'Other'])
    ax.set_title(metric)
    ax.set_xlabel(None)
    plt.tight_layout() 
    
pivoted_df = metrics_by_pathways_melted.pivot_table(index=['Pathway', 'Metric', 'sample_i'], 
                                                    columns='Type', 
                                                    values='Value').reset_index()

# Renaming the columns for clarity if needed
pivoted_df.columns.name = None  # Remove the category name
pivoted_df = pivoted_df.rename_axis(None, axis=1)

# Create a figure with a 3x4 grid of subplots
fig, axes = plt.subplots(3, 4, figsize=(15, 10))  # Adjust the size as needed
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

for i, pathway in enumerate(genes_pathways_dict.keys()):
    ax = axes[i]
    df_pathway = pivoted_df[(pivoted_df['Metric'] == 'average_clustering_coefficient') & (pivoted_df['Pathway'] == pathway)]

    sns.scatterplot(data=df_pathway, 
                    x='Average Random',   # Replace with actual column name
                    y='Actual',          # Replace with actual column name
                    ax=ax,  alpha=.4,s=15)
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title(pathway)
    ax.set_xlabel('Average Random Clustering Coefficient')
    ax.set_ylabel('Actual Clustering Coefficient')

# Turn off remaining empty subplots
for j in range(i + 1, 12):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


# Create a figure with a 3x4 grid of subplots
fig, axes = plt.subplots(3, 4, figsize=(15, 10))  # Adjust the size as needed
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

for i, pathway in enumerate(genes_pathways_dict.keys()):
    ax = axes[i]
    df_pathway = pivoted_df[(pivoted_df['Metric'] == 'average_clustering_coefficient') & (pivoted_df['Pathway'] == pathway)]

    sns.scatterplot(data=df_pathway, 
                    x='Average',   # Replace with actual column name
                    y='Actual',          # Replace with actual column name
                    ax=ax, alpha=.4,s=15)
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title(pathway)
    ax.set_xlabel('Average Clustering Coefficient')
    ax.set_ylabel('Actual Clustering Coefficient')

# Turn off remaining empty subplots
for j in range(i + 1, 12):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


# Create a figure with a 3x4 grid of subplots
fig, axes = plt.subplots(3, 4, figsize=(15, 10))  # Adjust the size as needed
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

for i, pathway in enumerate(genes_pathways_dict.keys()):
    ax = axes[i]
    df_pathway = pivoted_df[(pivoted_df['Metric'] == 'edge_density') & (pivoted_df['Pathway'] == pathway)]

    sns.scatterplot(data=df_pathway, 
                    x='Average',   # Replace with actual column name
                    y='Actual',          # Replace with actual column name
                    ax=ax, alpha=.4,s=15)
    
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlim([0, .2])
    ax.set_ylim([0, .2])
    ax.set_title(pathway)
    ax.set_xlabel('Average edge_density ')
    ax.set_ylabel('Actual edge_density ')

# Turn off remaining empty subplots
for j in range(i + 1, 12):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


# %%%%% plot top 10 Clustering Coefficient from each pathway
filtered_samples = metrics_by_pathways[(metrics_by_pathways['Metric'] == 'number_of_nodes_in_graph') & 
                                       (metrics_by_pathways['Value'] > 5)]

# Select the 'sample_i' column
selected_sample_i = filtered_samples['sample_i']

df_clustering = metrics_by_pathways[(metrics_by_pathways['Metric'] == 'average_clustering_coefficient')&(metrics_by_pathways['Type'] == 'Actual')&(metrics_by_pathways['Pathway'].isin([ 'HIPPO', 'MYC', 'NOTCH', 'NRF2', 'PI3K', 'RTK-RAS', 'TGF-Beta', 'WNT'])) ]
df_clustering = df_clustering[df_clustering['sample_i'].isin(selected_sample_i)]

# Get top 10 highest for each pathway
top_10_highest = df_clustering.groupby('Pathway').apply(lambda x: x.nlargest(10, 'Value')).reset_index(drop=True)
# Get top 10 lowest for each pathway
top_10_lowest = df_clustering.groupby('Pathway').apply(lambda x: x.nsmallest(10, 'Value')).reset_index(drop=True)



# Function to draw a single graph
def draw_graph(G, pathway_genes, sample_i, ax):
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = [x*2 for x in nx.get_edge_attributes(G, 'LRP_norm').values()]
    edge_colors = plt.cm.Greys(widths)

    # Node colors: red for nodes in the pathway, light blue for others
    nodes = [x.split('_')[0] for x in G.nodes()]
    light_blue_rgba = (0.678, 0.847, 0.902, 0.5)  # Light blue with transparency
    node_colors = ['red' if node in pathway_genes else light_blue_rgba for node in nodes]

    pos = nx.spring_layout(G, weight='LRP_norm')
    nx.draw(G, with_labels=False, node_color=node_colors, width=widths*10, pos=pos,
            edge_color=edge_colors, ax=ax, node_size=degrees_norm * 500, edgecolors='white', linewidths=0.5)
    ax.set_title(f"Sample {sample_i}")

# Plotting top 10 highest and lowest for each pathway
for pathway in df_clustering['Pathway'].unique():
    pathway_genes = set(genes_pathways_dict[pathway])  # Genes in the current pathway

    # Get sample indices for highest and lowest clustering coefficients
    highest_samples = top_10_highest[top_10_highest['Pathway'] == pathway]['sample_i'].tolist()
    lowest_samples = top_10_lowest[top_10_lowest['Pathway'] == pathway]['sample_i'].tolist()

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # 2 rows, 10 columns
    axes = axes.flatten()

    # Plot top 10 highest
    for i, sample in enumerate(highest_samples):
        draw_graph(G_list[sample], pathway_genes, sample, axes[i])

    plt.suptitle(f"Pathway: {pathway} - top10 highest average_clustering_coefficient for genes from {pathway}")
    plt.tight_layout()
    
    name_to_save = 'top10_average_clustering_coefficient_' + pathway +'_highest'
    plt.savefig(os.path.join(path_to_save + '\\networks_color_by_pathway', name_to_save))
    plt.savefig(os.path.join(path_to_save + '\\networks_color_by_pathway', name_to_save+'.pdf'),format='pdf')
    plt.show()


    fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # 2 rows, 10 columns
    axes = axes.flatten()
    # Plot top 10 lowest
    for i, sample in enumerate(lowest_samples):
        draw_graph(G_list[sample], pathway_genes, sample, axes[i])

    plt.suptitle(f"Pathway: {pathway} - top10 lowest average_clustering_coefficient for genes from {pathway}")
    plt.tight_layout()
    
    name_to_save = 'top10_average_clustering_coefficient_' + pathway +'lowest'
    plt.savefig(os.path.join(path_to_save + '\\networks_color_by_pathway', name_to_save))
    plt.savefig(os.path.join(path_to_save + '\\networks_color_by_pathway', name_to_save+'.pdf'),format='pdf')
    plt.show()



    
for metric in metrics_by_pathways_melted['Metric'].unique():
    
    fig,ax = plt.subplots(figsize = (10,6))
    sns.violinplot(data = metrics_by_pathways_melted[(metrics_by_pathways_melted['Metric']==metric)&(metrics_by_pathways_melted['Type']!='Average')], x = 'Pathway', hue = 'Type' ,y = 'Value',inner="quart", split=True, palette = 'tab10', ax=ax)
    ax.set_ylabel(None)
    ax.set_ylim([0,None])
    #ax.legend(handles=handles, title='Nodes', labels=['Cluster', 'Other'])
    ax.set_title(metric)
    ax.set_xlabel(None)
    plt.tight_layout()  
# fig,ax = plt.subplots(figsize = (10,15))
# sns.stripplot(data = metrics_by_pathways_melted[metrics_by_pathways_melted['Metric']!='number_of_nodes_in_graph'], y = 'Metric', hue = 'Pathway' ,x = 'Value',palette = 'tab10', ax=ax, orient='h', dodge=True)
# ax.set_ylabel(None)
# #handles, labels = ax.get_legend_handles_labels()
# #ax.legend(handles=handles, title='Nodes', labels=['Cluster', 'Other'])
# #ax.set_ylim([0,None])
# ax.set_xlabel(None)
# plt.tight_layout()    


# fig,ax = plt.subplots(figsize = (10,5))
# sns.violinplot(data = metrics_by_pathways_melted[metrics_by_pathways_melted['Metric']=='number_of_nodes_in_graph'], y = 'Pathway' ,x = 'Value',inner="quart", palette = 'tab10', ax=ax, orient='h')
# ax.set_ylabel(None)
# #handles, labels = ax.get_legend_handles_labels()
# #ax.legend(handles=handles, title='Nodes', labels=['Cluster', 'Other'])
# #ax.set_ylim([0,None])
# ax.set_xlabel(None)
# plt.tight_layout()  

fig,ax = plt.subplots(figsize = (10,5))
sns.boxplot(data = metrics_by_pathways_melted[metrics_by_pathways_melted['Metric']=='number_of_nodes_in_graph'], x = 'Pathway' ,y = 'Value',palette = 'tab10', ax=ax, dodge=True)
ax.set_ylabel(None)
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles=handles, title='Nodes', labels=['Cluster', 'Other'])
ax.set_ylim([0,None])
ax.set_xlabel('number_of_nodes_in_graph')
plt.tight_layout()    
 
# %%% communities in all dataset


community_maps = []
for partition in np.array(partitions_list):
    community_map = {}
    for node, comm in partition.items():
        community_map.setdefault(comm, []).append(node)
    community_maps.append(community_map)
# %%%% recurring_communities 
recurring_communities = find_recurring_communities(community_maps)
    # Continuing from the previous step's recurring_communities dictionary



# %%%% recurring_community_analysis 
# Analyze recurring communities with a threshold for significant overlap
threshold = .85
recurring_community_analysis = analyze_recurring_communities(recurring_communities, threshold=threshold)

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
ranked_by_frequency, frequency_dict, overlap_dict = rank_by_frequency(recurring_community_analysis)

# Output the ranked results
print("Ranked Recurring Communities by Frequency:")
for (network, community), frequency in ranked_by_frequency[:20]:
    print(f"Network {network} Community {community} recurs in {frequency} other networks.")

    
    

#df = pd.DataFrame(ranked_by_frequency)
df = pd.DataFrame(ranked_by_frequency, columns=['network_community', 'frequency'])

sizes = [len(item[1]) for item in frequency_dict.items()]
networks_temp = [list(item) for item in frequency_dict.values()]
overlaps_temp = [item for item in overlap_dict.values()]
df_temp = pd.DataFrame([sizes, networks_temp, overlaps_temp]).T
df_temp.columns = ['frequency','sample_no', 'overlaps']
df_temp['frequency'] = df_temp['frequency'].astype('int')
df_temp = df_temp.sort_values('frequency', ascending=False)

df['networks_with_community'] = df_temp['sample_no'].reset_index(drop=True)
df['community_overlaps'] = df_temp['overlaps'].reset_index(drop=True)

# Split the 'network_community' tuple into separate columns
df[['network', 'community']] = pd.DataFrame(df['network_community'].tolist(), index=df.index)
#df[['network', 'community']] = pd.DataFrame(df['network_communities'].tolist(), index=df.index)
# Drop the 'network_community' column as it's no longer needed
df.drop('network_community', axis=1, inplace=True)
# Reorder the columns
df = df[['network', 'community', 'frequency', 'networks_with_community','community_overlaps']]
#df['genes'] = df.apply(lambda row: community_maps[row['networks_communities'][0][0]][row['networks_communities'][0][1]], axis=1)
df['genes'] = df.apply(lambda row: community_maps[row['network']][row['community']], axis=1)

df['genes']  = df['genes'] .apply(lambda row: list(np.sort(row)))

df['community_size'] = df['genes'].apply(len)

df_temp = df.copy()
df_temp['genes_list']= df_temp['genes'].astype('str')

df_grouped = df_temp.groupby('genes_list').agg({'frequency': 'max', 'network': 'max','community':'max', 'community_size':'max', 'genes':'first','networks_with_community':'first', 'community_overlaps':'first'}).reset_index().sort_values('frequency', ascending=False).reset_index(drop=True)

from itertools import combinations

# First, you might want to select only the most frequent communities for comparison

min_frequency = 10
min_community_size = 5

df_grouped_filtered = df_grouped[(df_grouped['frequency'] >= min_frequency) & (df_grouped['community_size'] >=min_community_size)]
df_grouped_filtered['networks_with_community'] = df_grouped_filtered.apply(    lambda row: row['networks_with_community'] + [row['network']], axis=1)


N = 1000  # or however many you wish to compare
top_communities = df_grouped_filtered.nlargest(N, 'frequency')
   
top_communities.to_excel(os.path.join(path_to_save, 'top_communities_full_dataset_all_{}.xlsx'.format(threshold)))
    
 
# %%%% genes vs communitites

def create_genes_vs_samples_dataframe(df):
    # Flatten the 'genes' column to get a unique set of genes
    all_genes = list(set([gene for sublist in df['genes'] for gene in sublist]))

    # Flatten the 'networks_with_community' column to get unique network numbers
    all_networks = list(set([network for sublist in df['networks_with_community'] for network in sublist]))
    #seen = set()
    #all_networks = [x for x in all_networks if not (x in seen or seen.add(x))]
    # Create a DataFrame with zeros
    gene_network_df = pd.DataFrame(0, index=all_genes, columns=all_networks)

    # Iterate over each row and update gene_network_df
    for _, row in df.iterrows():
        print(_)
        for gene in row['genes']:
            for network in row['networks_with_community']:
                if gene in all_genes and network in all_networks:
                    gene_network_df.at[gene, network] = 1

    return gene_network_df
a = create_genes_vs_samples_dataframe(df_grouped_filtered)
# normalized_data = (df['frequency'].values - df['frequency'].values.min()) / (df['frequency'].values.max() - df['frequency'].values.min())
# reds = plt.cm.Reds
# col_colors = reds(normalized_data)


Z = linkage(a, method = 'ward')
#sns.clustermap(a, cmap = 'Reds', vmin = 0, col_colors=col_colors, yticklabels = True, figsize = (20,20))
fg = sns.clustermap(a, method = 'ward', cmap = 'Reds', vmin = 0, #col_colors=col_colors,
                    yticklabels = True, figsize = (25,25), row_linkage = Z,
                    xticklabels = True,
                    #col_cluster =False
                    )
fg.ax_heatmap.set_xlabel('Sample ')
fg.ax_heatmap.set_ylabel('Nodes (genes)')

# Add grid lines
for i in range(a.shape[0]):
    fg.ax_heatmap.axhline(i, color='white', lw=0.2)
for i in range(a.shape[1]):
    fg.ax_heatmap.axvline(i, color='white', lw=0.2)
    
yticklabels = [x.get_text() for x in fg.ax_heatmap.get_yticklabels()]
plt.title('Overlap: {}'.format(threshold))
plt.savefig(os.path.join(path_to_save, 'clustermap_communities_genes_full_dataset_{}.png'.format(threshold)))
plt.savefig(os.path.join(path_to_save, 'clustermap_communities_genes_full_dataset_{}.pdf'.format(threshold)), format= 'pdf')


dendrogram_cutoff = 40
cutoff = dendrogram_cutoff
fig, ax = plt.subplots(figsize=(10, 3))
dn = dendrogram(
    Z, 
    color_threshold=cutoff, 
    above_threshold_color='gray', 
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=30,  # show only the last 12 merged clusters
    #show_leaf_counts=True,  # show the number of samples in each cluster
    ax=ax
)

plt.axhline(y=cutoff, color='r', linestyle='--')
ax.set_ylabel('Distance')
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'dendrogram_communities_genes_full_dataset_{}.png'.format(threshold)))
plt.show()

# %%%%% samples_with_community
cluster_labels = fcluster(Z, t=cutoff, criterion='distance')
unique_labels, counts = np.unique(cluster_labels, return_counts=True)
genes_with_cluster = pd.DataFrame(a.index, columns = ['genes'])
genes_with_cluster['cluster_label'] = cluster_labels
genes_with_cluster .to_excel(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\genes_with_cluster_threshold{}.xlsx'.format(threshold))
genes_with_cluster1 = genes_with_cluster[genes_with_cluster['cluster_label'] ==1]['genes'].to_list()
# samples that have the selected community


cumulative_set = set()
sizes = []
for i, row in df_grouped_filtered.iterrows():
    # Update the cumulative set with values from the current row
    
    overlap = len(set(row['genes']) & set(genes_with_cluster1))
    total_nodes = len(genes_with_cluster1)
    overlap_ratio = overlap / total_nodes
    
    if overlap_ratio > .5 :
    
        print(overlap_ratio, row['networks_with_community'], len(set(row['networks_with_community'])))
        cumulative_set.update(row['networks_with_community'])
        
        # Append the current size of the set to the sizes list
        sizes.append(len(cumulative_set))

fig,ax=plt.subplots(figsize = (5,3))
ax.plot(sizes)
ax.set_xlabel('No. communities included')
ax.set_ylabel('Cummulative no. samples')
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'pts_vs_communities_{}.png'.format(threshold)))
plt.show()

samples_selected = list(cumulative_set)
samples_selected .sort()

samples_with_community = pd.DataFrame(samples)
samples_with_community ['has_community']  = 0
samples_with_community .loc[samples_selected, 'has_community'] = 1
samples_with_community .to_excel(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\samples_with_community_threshold{}.xlsx'.format(threshold))

# # samples vs communitites

# num_columns = 989  # 0 to 988

# # Create a new DataFrame with zeros
# pivot_df = pd.DataFrame(0, index=df.index, columns=range(num_columns))

# def create_pts_networks_dataframe(df):
#     # Flatten the 'genes' column to get a unique set of genes
#     samples = np.arange(0,989,1)

#     # Flatten the 'networks_with_community' column to get unique network numbers
#     all_networks = list(set([network for sublist in df['networks_with_community'] for network in sublist]))

#     # Create a DataFrame with zeros
#     gene_network_df = pd.DataFrame(0, index=all_genes, columns=all_networks)

#     # Iterate over each row and update gene_network_df
#     for _, row in df.iterrows():
#         print(_)
#         for sample in samples:
#             for network in row['networks_with_community']:
#                 if gene in all_genes and network in all_networks:
#                     gene_network_df.at[gene, network] = 1
                    
# b = create_pts_networks_dataframe(df_grouped_filtered)
# normalized_data = (df['frequency'].values - df['frequency'].values.min()) / (df['frequency'].values.max() - df['frequency'].values.min())
# reds = plt.cm.Reds
# col_colors = reds(normalized_data)


# #sns.clustermap(a, cmap = 'Reds', vmin = 0, col_colors=col_colors, yticklabels = True, figsize = (20,20))
# fg = sns.clustermap(b, method = 'ward', cmap = 'Reds', vmin = 0, col_colors=col_colors,
#                     yticklabels = True, figsize = (20,20),
#                     col_cluster =False)
# fg.ax_heatmap.set_xlabel('Communities')
# fg.ax_heatmap.set_ylabel('Nodes (genes)')
# yticklabels = [x.get_text() for x in fg.ax_heatmap.get_yticklabels()]
# plt.title('Overlap: {}'.format(threshold))



# for index, row in df.iterrows():
#     for i , number in enumerate(row['networks_with_community']):
#         if 0 <= number < num_columns:
#             pivot_df.at[index, number] = row['community_overlaps'][i]
# pivot_df = pivot_df.T

# sns.clustermap(pivot_df, method = 'simple', cmap = 'jet', vmin = .5)
# sns.clustermap(pivot_df, method = 'average', cmap = 'jet', vmin = .5)
# sns.clustermap(pivot_df, method = 'ward', cmap = 'jet', vmin = .5, col_cluster =False)

# %%%% plot all networks with the selected genes and community
pos_selected = {}
# colored by all communities
for i in samples_selected:
    
    print(i)
    
    G = G_list[i]
    partition = partitions_list[i]
    
    
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = list(nx.get_edge_attributes(G, 'LRP_norm').values())
    widths = [x*2 for x in widths]
    
    edge_colors = plt.cm.Greys(widths)  # 'viridis' is a colormap, you can choose any
        
    # NODE COLORS by community
    community_colors = {node: partition.get(node) for node in G.nodes()}

    # Now we can generate a unique color for each community
    unique_communities = list(set(community_colors.values()))
    community_color_map = plt.cm.jet(np.linspace(0, 1, len(unique_communities)))
    community_color_map = {com: community_color_map[i] for i, com in enumerate(unique_communities)}

    # Map the community colors to each node
    node_colors = [community_color_map[community_colors[node]] for node in G.nodes()]
            
    pos = nx.spring_layout(G, weight='LRP_norm')
    pos_selected[str(i)] = pos
    fig, ax = plt.subplots(figsize=(7,7))   
        # Plot legend
    community_labels = {com: f'Community {com}' for com in unique_communities}

    #for community in unique_communities:
     #   ax.plot([], [], 'o', color=community_color_map[community], label=community_labels[community])
    #ax.legend(title='Communities', loc='best')
        
    nx.draw(G, with_labels=False,
            node_color=node_colors,
            width=widths*10,
            pos=pos, # font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 500,
            edgecolors='white',  # This adds the white border
        linewidths=0.5)
    
    title = 'sample_i: ' + str(i) + '\n' + samples[i]
    ax.set_title(title)
    plt.tight_layout()
    name_to_save = 'network_' + str(i) + '_' + samples[i] + '_allcoms'
    plt.savefig(os.path.join(path_to_save + '\\networks_with_selected_community', name_to_save))
    plt.show()
      
    
# colored only reccuring community with genes from selected cluster
for i in samples_selected:
    
    print(i)
    
    G = G_list[i]
    partition = partitions_list[i]
    
    
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = list(nx.get_edge_attributes(G, 'LRP_norm').values())
    widths = [x*2 for x in widths]
    
    edge_colors = plt.cm.Greys(widths)  # 'viridis' is a colormap, you can choose any
        
    # NODE COLORS by SELECTED GENES
    node_colors = ['blue' if node in genes_with_cluster1 else 'gray' for node in G.nodes()]
        
    pos = pos_selected[str(i)]
    
    fig, ax = plt.subplots(figsize=(7,7))   
        # Plot legend
    community_labels = {com: f'Community {com}' for com in unique_communities}

    #for community in unique_communities:
     #   ax.plot([], [], 'o', color=community_color_map[community], label=community_labels[community])
    #ax.legend(title='Communities', loc='best')
        
    nx.draw(G, with_labels=False,
            node_color=node_colors,
            width=widths*10,
            pos=pos, # font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 500,
            edgecolors='white',  # This adds the white border
        linewidths=0.5)
    
    title = 'sample_i: ' + str(i) + '\n' + samples[i]
    ax.set_title(title)
    plt.tight_layout()
    name_to_save = 'network_' + str(i) + '_' + samples[i] + '_selectedgenes'
    plt.savefig(os.path.join(path_to_save + '\\networks_with_selected_community', name_to_save))
    plt.show()
    
    
    
    
# colored only reccuring community with genes from selected cluster ALL SAMPLES
for i in range(989):
    if i in samples_selected:
        print(i, ' in SELECTED')
        pass
    else:
        print(i)
    
        G = G_list[i]
        partition = partitions_list[i]
        
        
        degrees = np.array(list(nx.degree_centrality(G).values()))
        degrees_norm = degrees / np.max(degrees)
        widths = list(nx.get_edge_attributes(G, 'LRP_norm').values())
        widths = [x*2 for x in widths]
        
        edge_colors = plt.cm.Greys(widths)  # 'viridis' is a colormap, you can choose any
            
        # NODE COLORS by SELECTED GENES
        node_colors = ['blue' if node in genes_with_cluster1 else 'gray' for node in G.nodes()]
            
        pos = nx.spring_layout(G, weight='LRP_norm')
        
        fig, ax = plt.subplots(figsize=(7,7))   
            # Plot legend
        community_labels = {com: f'Community {com}' for com in unique_communities}
    
        #for community in unique_communities:
         #   ax.plot([], [], 'o', color=community_color_map[community], label=community_labels[community])
        #ax.legend(title='Communities', loc='best')
            
        nx.draw(G, with_labels=False,
                node_color=node_colors,
                width=widths*10,
                pos=pos, # font_size=0,
                # cmap = colors,
                edge_color=edge_colors,
                ax=ax,
                # node_size = degrees,
                node_size=degrees_norm * 500,
                edgecolors='white',  # This adds the white border
            linewidths=0.5)
        
        title = 'sample_i: ' + str(i) + '\n' + samples[i]
        ax.set_title(title)
        plt.tight_layout()
        name_to_save = 'network_' + str(i) + '_' + samples[i] + '_selectedgenes_notinselected'
        plt.savefig(os.path.join(path_to_save + '\\networks_with_selected_community', name_to_save))
        plt.show()    





# %%% communities in clusters

cluster_labels_df = pd.read_csv(os.path.join(path_to_save, 'cluster_labels_shared_edges.csv'), index_col = 0)
top_communities_all = pd.DataFrame()
threshold = .85
cutoffs = [3,.2,3,2,3,3]
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
    recurring_community_analysis = analyze_recurring_communities(recurring_communities, threshold=threshold)
    
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
    df_temp['genes_list']= df_temp['genes'].astype('str')
    df_grouped = df_temp.groupby('genes_list').agg({'frequency': 'max', 'network': 'max','community':'max', 'community_size':'max', 'genes':'first'}).reset_index().sort_values('frequency', ascending=False).reset_index(drop=True)
    
    from itertools import combinations
    
    # First, you might want to select only the most frequent communities for comparison
    
    min_frequency = 3
    min_community_size = 3
    
    df_grouped_filtered = df_grouped[(df_grouped['frequency'] >= min_frequency) & (df_grouped['community_size'] >=min_community_size)]
    
    N = 1000  # or however many you wish to compare
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
    
    top_communities['cluster_id_size'] = np.sum(index_)
    top_communities['relative_frequency'] = top_communities['frequency'] / top_communities['cluster_id_size']
    top_communities['intersection_within_cluster']  = ['']*top_communities.shape[0]
    top_communities['union_within_cluster']  =  ['']*top_communities.shape[0]
    top_communities['intersection_within_cluster_size']  = 0
    top_communities['union_within_cluster_size']  =  0
    
     
    top_communities['cluster_id'] = cluster_id
    top_communities['threshold'] = threshold
    try:
        data_to_dendrogram = similarity_jaccard
        Z = linkage(data_to_dendrogram, method='ward')
        
        cutoff = cutoffs[cluster_id-1]
        fig, ax = plt.subplots(figsize=(10, 3))
        dn = dendrogram(
            Z, 
            color_threshold=cutoff, 
            above_threshold_color='gray', 
            #truncate_mode='lastp',  # show only the last p merged clusters
            #p=30,  # show only the last 12 merged clusters
            #show_leaf_counts=True,  # show the number of samples in each cluster
            ax=ax
        )
        
        plt.axhline(y=cutoff, color='r', linestyle='--')
        ax.set_ylabel('Distance')
        ax.set_title('Cluster {}\ncommunities_jaccard_score'.format(cluster_id) ) 
        plt.tight_layout()
        plt.savefig(os.path.join(path_to_save, 'dendrogram_communities_jaccard_score_cluster{}'.format(cluster_id)))
        plt.show()
    
        #sns.heatmap(similarity_jaccard, cmap = 'jet', vmin = 0, vmax = 1)
        sns.clustermap(similarity_jaccard, cmap = 'jet', vmin = 0, vmax = 1, method = 'ward')
        plt.title('cluster ' + str(cluster_id))
        plt.show()
        # sns.heatmap(similarity_dice, cmap = 'jet', vmin = 0, vmax = 1)
        # sns.clustermap(similarity_dice, cmap = 'jet', vmin = 0, vmax = 1, method = 'ward')
        
        # sns.heatmap(similarity_overlap, cmap = 'jet', vmin = 0, vmax = 1)
        # sns.clustermap(similarity_overlap, cmap = 'jet', vmin = 0, vmax = 1, method = 'ward')
     
           
        cluster_labels = fcluster(Z, t=cutoff, criterion='distance')
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    
        top_communities['cluster_label'] = cluster_labels
        
        for community_cluster in np.sort(unique_labels):
            
            communities_temp = top_communities[top_communities['cluster_label'] == community_cluster].reset_index(drop=True)
            genes_temp = communities_temp['genes'].to_list()
         
            genes_intersection = set(genes_temp[0]).intersection(*genes_temp[1:])
            genes_union = set().union(*genes_temp)
    
            top_communities.loc[top_communities['cluster_label'] == community_cluster, 'intersection_within_cluster']  = str(list(genes_intersection))
            top_communities.loc[top_communities['cluster_label'] == community_cluster, 'union_within_cluster']  = str(list(genes_union))
            
            top_communities.loc[top_communities['cluster_label'] == community_cluster, 'intersection_within_cluster_size']  = len(genes_intersection)
            top_communities.loc[top_communities['cluster_label'] == community_cluster, 'union_within_cluster_size']  = len(genes_union)
    except:
        top_communities['cluster_label'] = -1
        
    
    top_communities_all = pd.concat((top_communities_all , top_communities))

top_communities_all = top_communities_all.reset_index(drop=True)

    
top_communities_all.to_excel(os.path.join(path_to_save, 'top_communities_all.xlsx'))
    
 
    
 
    
top_communities_all_gr = top_communities_all.groupby(by = ['cluster_id','cluster_label']).first().reset_index()
    
top_communities_all_gr['intersection_within_cluster']
    
 
    
def count_communities_containing_genes(community_maps, genes_set):
   """
   Count how many communities the entire set of genes is contained within.
   
   Parameters:
   community_maps (list of dicts): A list where each element is a dictionary mapping community IDs to gene lists for each network.
   genes_set (set): The set of genes to check for across communities.

   Returns:
   int: The count of communities containing the entire set of genes.
   """
   count = 0
   for network_communities in community_maps:
       for community_genes in network_communities.values():
           # Check if all genes are in the current community
           if genes_set.issubset(set(community_genes)):
               count += 1
   return count

# Example usage:
#genes_set = set(['JAK2_exp', 'TNFAIP3_exp', 'JAK3_exp', 'PRF1_exp', 'CD79B_exp', 'CARD11_exp', 'CD74_exp', 'FCRL4_exp', 'SOCS1_exp', 'CYLD_exp', 'SYK_exp', 'IKZF1_exp', 'PTPRC_exp', 'CD28_exp', 'NFATC2_exp', 'CXCR4_exp', 'CD79A_exp', 'POU2AF1_exp', 'IL7R_exp', 'LCK_exp', 'B2M_exp', 'CCR4_exp', 'P2RY8_exp', 'CIITA_exp', 'BIRC3_exp', 'TNFRSF17_exp', 'BCL11B_exp', 'IRF4_exp', 'TCF7_exp', 'BTK_exp', 'CSF1R_exp', 'CCR7_exp', 'WAS_exp', 'STK4_exp', 'PDCD1LG2_exp'])
#gene_occurrences = count_communities_containing_genes(community_maps, genes_set)
import ast
gene_occurrences_list = []
gene_occurrences_outside_list = []
for cluster_id in np.sort(cluster_labels_df['clusters_shared_edges'].unique()):
    
    top_communities_all_gr_temp = top_communities_all_gr[top_communities_all_gr['cluster_id'] == cluster_id]
    
    index_ = list((cluster_labels_df['clusters_shared_edges'] == cluster_id))
    print('Cluster: ', cluster_id, 'size: ', np.sum(index_))

    community_maps = []
    for partition in np.array(partitions_list)[index_]:
        community_map = {}
        for node, comm in partition.items():
            community_map.setdefault(comm, []).append(node)
        community_maps.append(community_map)
        
    for i,row, in top_communities_all_gr_temp.iterrows():
        print(i, row['intersection_within_cluster'])
        genes_set = set(ast.literal_eval(row['intersection_within_cluster']))
       #['JAK2_exp', 'TNFAIP3_exp', 'JAK3_exp', 'PRF1_exp', 'CD79B_exp', 'CARD11_exp', 'CD74_exp', 'FCRL4_exp', 'SOCS1_exp', 'CYLD_exp', 'SYK_exp', 'IKZF1_exp', 'PTPRC_exp', 'CD28_exp', 'NFATC2_exp', 'CXCR4_exp', 'CD79A_exp', 'POU2AF1_exp', 'IL7R_exp', 'LCK_exp', 'B2M_exp', 'CCR4_exp', 'P2RY8_exp', 'CIITA_exp', 'BIRC3_exp', 'TNFRSF17_exp', 'BCL11B_exp', 'IRF4_exp', 'TCF7_exp', 'BTK_exp', 'CSF1R_exp', 'CCR7_exp', 'WAS_exp', 'STK4_exp', 'PDCD1LG2_exp'])
        gene_occurrences = count_communities_containing_genes(community_maps, genes_set)
        
        print(gene_occurrences)
        
        gene_occurrences_list.append(gene_occurrences)
        print(cluster_id, row['cluster_label'])
        
    community_maps = []
    index_ = list((cluster_labels_df['clusters_shared_edges'] != cluster_id))
    for partition in np.array(partitions_list)[index_]:
        community_map = {}
        for node, comm in partition.items():
            community_map.setdefault(comm, []).append(node)
        community_maps.append(community_map)
        
    for i,row, in top_communities_all_gr_temp.iterrows():
        print(i, row['intersection_within_cluster'])
        genes_set = set(ast.literal_eval(row['intersection_within_cluster']))
       #['JAK2_exp', 'TNFAIP3_exp', 'JAK3_exp', 'PRF1_exp', 'CD79B_exp', 'CARD11_exp', 'CD74_exp', 'FCRL4_exp', 'SOCS1_exp', 'CYLD_exp', 'SYK_exp', 'IKZF1_exp', 'PTPRC_exp', 'CD28_exp', 'NFATC2_exp', 'CXCR4_exp', 'CD79A_exp', 'POU2AF1_exp', 'IL7R_exp', 'LCK_exp', 'B2M_exp', 'CCR4_exp', 'P2RY8_exp', 'CIITA_exp', 'BIRC3_exp', 'TNFRSF17_exp', 'BCL11B_exp', 'IRF4_exp', 'TCF7_exp', 'BTK_exp', 'CSF1R_exp', 'CCR7_exp', 'WAS_exp', 'STK4_exp', 'PDCD1LG2_exp'])
        gene_occurrences = count_communities_containing_genes(community_maps, genes_set)
        
        print(gene_occurrences)
        
        gene_occurrences_outside_list.append(gene_occurrences)
        print(cluster_id, row['cluster_label'])    
        
        
        
        
        
top_communities_all_gr['gene_set_occurrences_within_cluster']     = gene_occurrences_list   
top_communities_all_gr['relative_gene_set_occurrences_within_cluster']     = top_communities_all_gr['gene_set_occurrences_within_cluster'] / top_communities_all_gr['cluster_id_size']   
top_communities_all_gr['gene_set_occurrences_outside_cluster']     = gene_occurrences_list   
top_communities_all_gr['relative_gene_set_occurrences_outside_cluster']     = top_communities_all_gr['gene_set_occurrences_outside_cluster'] / (988 - top_communities_all_gr['cluster_id_size']   )

top_communities_all_gr['cluster_id_comm_id'] = 'Cluster ' + top_communities_all_gr['cluster_id'].astype('str') + ' - community ' + top_communities_all_gr['cluster_label'].astype('str') 
top_communities_all_gr.to_excel(os.path.join(path_to_save, 'top_communities_all_gr.xlsx'))

to_plot = top_communities_all_gr.melt(id_vars='cluster_id_comm_id', value_vars=['relative_gene_set_occurrences_within_cluster','relative_gene_set_occurrences_outside_cluster'])#, var_name=None, value_name='value', col_level=None, ignore_index=True)
    
fig,ax=plt.subplots(figsize = (4,6))
sns.barplot(data=to_plot, y = 'cluster_id_comm_id', x = 'value', hue = 'variable',ax=ax, orient='h')
h, l = ax.get_legend_handles_labels()
labels = ['Inside cluster','Outside cluster']
ax.legend(h, labels, title="Relative occurances of community", bbox_to_anchor=(0.3, 1.1), loc = 'center')
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.set_xlim([0,1])
plt.tight_layout()



# %%% compute overlaps within cluster

shared_nodes_matrix = pd.read_csv(os.path.join(path_to_data, 'shared_nodes_matrix_{}.csv'.format(topn)), index_col = 0)
shared_edges_matrix = pd.read_csv(os.path.join(path_to_data, 'shared_edges_matrix_{}.csv'.format(topn)), index_col = 0)

# intracluster overlap

intracluster_nodes_overlap_mean = []
intracluster_nodes_overlaps = []
intracluster_edges_overlap_mean = []
intracluster_edges_overlaps = []

intracluster_edges_overlaps_ratio = []
intracluster_nodes_overlaps_ratio = []

nodes_mean_list = []
nodes_list = []
edges_mean_list = []
edges_list = []
for cluster_id in np.sort(cluster_labels_df['clusters_shared_edges'].unique()):

    index_ = list((cluster_labels_df['clusters_shared_edges'] == cluster_id))
    nodes_mean_list.append(np.mean(np.array(n_nodes)[index_]))
    nodes_list.append(np.array(n_nodes)[index_])
    
    edges_mean_list.append(np.mean(np.array(n_edges)[index_]))
    edges_list.append(np.array(n_edges)[index_])
                      
    print('Cluster: ', cluster_id, 'size: ', np.sum(index_))
    
    matrix_temp = shared_nodes_matrix.loc[index_,index_]
    
    upper_triangle_indices = np.triu_indices(n=matrix_temp.shape[0], k=1)

    # Select the elements from the matrix
    upper_triangle_elements = matrix_temp.values[upper_triangle_indices]
        
    intracluster_nodes_overlap_mean.append(np.mean(upper_triangle_elements))
    intracluster_nodes_overlaps.append(upper_triangle_elements)
    intracluster_nodes_overlaps_ratio.append(upper_triangle_elements / np.mean(np.array(n_nodes)[index_]))
                                             
                                             
    matrix_temp = shared_edges_matrix.loc[index_,index_]
    
    upper_triangle_indices = np.triu_indices(n=matrix_temp.shape[0], k=1)

    # Select the elements from the matrix
    upper_triangle_elements = matrix_temp.values[upper_triangle_indices]
        
    intracluster_edges_overlap_mean.append(np.mean(upper_triangle_elements))
    intracluster_edges_overlaps.append(upper_triangle_elements)    
    intracluster_edges_overlaps_ratio.append(upper_triangle_elements / np.mean(np.array(n_edges)[index_]))
    
                                             
    
    
    
values = []
array_ids = []
for i, arr in enumerate(intracluster_edges_overlaps, start=1):
    values.extend(arr)
    array_ids.extend([i]*len(arr))
intracluster_edges_overlaps_df = pd.DataFrame({'Overlaping_edges': values,'Cluster_id': array_ids})
values = []
array_ids = []
for i, arr in enumerate(intracluster_nodes_overlaps, start=1):
    values.extend(arr)
    array_ids.extend([i]*len(arr))
intracluster_nodes_overlaps_df = pd.DataFrame({'Overlaping_nodes': values,'Cluster_id': array_ids})

values = []
array_ids = []
for i, arr in enumerate(intracluster_edges_overlaps_ratio, start=1):
    values.extend(arr)
    array_ids.extend([i]*len(arr))
intracluster_edges_overlaps_ratio_df = pd.DataFrame({'Relative_Overlaping_edges': values,'Cluster_id': array_ids})
values = []
array_ids = []
for i, arr in enumerate(intracluster_nodes_overlaps_ratio, start=1):
    values.extend(arr)
    array_ids.extend([i]*len(arr))
intracluster_nodes_overlaps_ratio_df = pd.DataFrame({'Relative_Overlaping_nodes': values,'Cluster_id': array_ids})


values = []
array_ids = []
for i, arr in enumerate(edges_list, start=1):
    values.extend(arr)
    array_ids.extend([i]*len(arr))
edges_list_df = pd.DataFrame({'No_edges': values,'Cluster_id': array_ids})

values = []
array_ids = []
for i, arr in enumerate(nodes_list, start=1):
    values.extend(arr)
    array_ids.extend([i]*len(arr))
nodes_list_df = pd.DataFrame({'No_nodes': values,'Cluster_id': array_ids})





fig,ax = plt.subplots(figsize = (4,4))    
sns.boxplot(x='Cluster_id', y='No_edges', data=edges_list_df, fliersize=0, color='white',ax=ax)
ax.set_ylim([0, None])
ax.set_title('Number of Interactions in network')
ax.set_ylabel(None)
plt.tight_layout()

fig,ax = plt.subplots(figsize = (4,4))    
sns.boxplot(x='Cluster_id', y='No_nodes', data=nodes_list_df, fliersize=0, color='white',ax=ax)
ax.set_ylim([0, None])
ax.set_title('Number of nodes in network')
ax.set_ylabel(None)
plt.tight_layout()



fig,ax = plt.subplots(figsize = (4,4))    
sns.boxplot(x='Cluster_id', y='Relative_Overlaping_edges', data=intracluster_edges_overlaps_ratio_df, fliersize=0, color='white',ax=ax)
ax.set_ylim([0, 1])
ax.set_title('Intracluster Relative Overlaping Interactions')
ax.set_ylabel(None)
plt.tight_layout()

fig,ax = plt.subplots(figsize = (4,4))    
sns.boxplot(x='Cluster_id', y='Relative_Overlaping_nodes', data=intracluster_nodes_overlaps_ratio_df, fliersize=0, color='white',ax=ax)
ax.set_ylim([0, 1])
ax.set_title('Intracluster Relative Overlaping nodes')
ax.set_ylabel(None)
plt.tight_layout()

fig,ax = plt.subplots(figsize = (4,4))    
sns.boxplot(x='Cluster_id', y='Overlaping_edges', data=intracluster_edges_overlaps_df, fliersize=0, color='white',ax=ax)
ax.set_ylim([0, 700])
ax.set_title('Intracluster Overlaping Interactions')
ax.set_ylabel(None)
plt.tight_layout()

fig,ax = plt.subplots(figsize = (4,4))    
sns.boxplot(x='Cluster_id', y='Overlaping_nodes', data=intracluster_nodes_overlaps_df, fliersize=0, color='white',ax=ax)
ax.set_ylim([0, 250])
ax.set_title('Intracluster  Overlaping nodes')
ax.set_ylabel(None)
plt.tight_layout()

# %%% compute overlaps between clusters
intercluster_nodes_overlap_mean = []
intercluster_nodes_overlaps = []
intercluster_edges_overlap_mean = []
intercluster_edges_overlaps = []

intercluster_edges_overlaps_ratio = []
intercluster_nodes_overlaps_ratio = []

for cluster_id in np.sort(cluster_labels_df['clusters_shared_edges'].unique()):

    index_ = list((cluster_labels_df['clusters_shared_edges'] == cluster_id))
    index_neg = list((cluster_labels_df['clusters_shared_edges'] != cluster_id))
                      
    print('Cluster: ', cluster_id, 'size: ', np.sum(index_))
    
    matrix_temp = shared_nodes_matrix.loc[index_,index_neg]
    upper_triangle_elements = matrix_temp.values.ravel()
        
    intercluster_nodes_overlap_mean.append(np.mean(upper_triangle_elements))
    intercluster_nodes_overlaps.append(upper_triangle_elements)
    intercluster_nodes_overlaps_ratio.append(upper_triangle_elements / np.mean(np.array(n_nodes)[index_]))
                                             
                                             
    matrix_temp = shared_edges_matrix.loc[index_,index_neg]
    upper_triangle_elements =matrix_temp.values.ravel()
        
    intercluster_edges_overlap_mean.append(np.mean(upper_triangle_elements))
    intercluster_edges_overlaps.append(upper_triangle_elements)    
    intercluster_edges_overlaps_ratio.append(upper_triangle_elements / np.mean(np.array(n_edges)[index_]))
    
                                             
    
    
    
values = []
array_ids = []
for i, arr in enumerate(intercluster_edges_overlaps, start=1):
    values.extend(arr)
    array_ids.extend([i]*len(arr))
intercluster_edges_overlaps_df = pd.DataFrame({'Overlaping_edges': values,'Cluster_id': array_ids})
values = []
array_ids = []
for i, arr in enumerate(intercluster_nodes_overlaps, start=1):
    values.extend(arr)
    array_ids.extend([i]*len(arr))
intercluster_nodes_overlaps_df = pd.DataFrame({'Overlaping_nodes': values,'Cluster_id': array_ids})

values = []
array_ids = []
for i, arr in enumerate(intercluster_edges_overlaps_ratio, start=1):
    values.extend(arr)
    array_ids.extend([i]*len(arr))
intercluster_edges_overlaps_ratio_df = pd.DataFrame({'Relative_Overlaping_edges': values,'Cluster_id': array_ids})
values = []
array_ids = []
for i, arr in enumerate(intercluster_nodes_overlaps_ratio, start=1):
    values.extend(arr)
    array_ids.extend([i]*len(arr))
intercluster_nodes_overlaps_ratio_df = pd.DataFrame({'Relative_Overlaping_nodes': values,'Cluster_id': array_ids})



fig,ax = plt.subplots(figsize = (4,4))    
sns.boxplot(x='Cluster_id', y='Relative_Overlaping_edges', data=intercluster_edges_overlaps_ratio_df, fliersize=0, color='white',ax=ax)
ax.set_ylim([0, 1])
ax.set_title('Intercluster Relative Overlaping interactions')
ax.set_ylabel(None)
plt.tight_layout()

fig,ax = plt.subplots(figsize = (4,4))    
sns.boxplot(x='Cluster_id', y='Relative_Overlaping_nodes', data=intercluster_nodes_overlaps_ratio_df, fliersize=0, color='white',ax=ax)
ax.set_ylim([0, 1])
ax.set_title('Intercluster Relative Overlaping nodes')
ax.set_ylabel(None)
plt.tight_layout()

fig,ax = plt.subplots(figsize = (4,4))    
sns.boxplot(x='Cluster_id', y='Overlaping_edges', data=intercluster_edges_overlaps_df, fliersize=0, color='white',ax=ax)
ax.set_ylim([0, 700])
ax.set_title('Intercluster Overlaping Interactions')
ax.set_ylabel(None)
plt.tight_layout()

fig,ax = plt.subplots(figsize = (4,4))    
sns.boxplot(x='Cluster_id', y='Overlaping_nodes', data=intercluster_nodes_overlaps_df, fliersize=0, color='white',ax=ax)
ax.set_ylim([0, 250])
ax.set_title('Intercluster  Overlaping nodes')
ax.set_ylabel(None)
plt.tight_layout()

# %%%plot gene_sets in all networks in the cluster




cluster_labels_df = pd.read_csv(os.path.join(path_to_save, 'cluster_labels_shared_edges.csv'), index_col = 0)

cluster_id = 6
#gene_set = top_communities_all_gr.loc[top_communities_all_gr['cluster_id_comm_id'] == 'Cluster 5 - community 1', 'genes'].values[0]
gene_set = top_communities_all_gr.loc[top_communities_all_gr['cluster_id_comm_id'] == 'Cluster 6 - community 2', 'genes'].values[0]

indices = list(cluster_labels_df[cluster_labels_df['clusters_shared_edges'] == cluster_id].index)
        
for i in indices:

    G = G_list[i]
    
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = list(nx.get_edge_attributes(G, 'LRP_norm').values())
    widths = [x*2 for x in widths]
    
    edge_colors = plt.cm.Greys(widths)  # 'viridis' is a colormap, you can choose any
    
    
    # Check that the gene_set only contains nodes that are in the graph
    gene_set_temp = {node for node in gene_set if node in G}
    color_map = ['blue' if node in gene_set_temp else 'gray' for node in G.nodes()]
            
    pos = pos_988[i]
    
    fig, ax = plt.subplots(figsize=(7,7))   
    nx.draw(G, with_labels=False,
            node_color=color_map,
            width=widths*10,
            pos=pos, # font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 500,
            edgecolors='white',  # This adds the white border
            linewidths=0.5)
    
    title = 'Cluster: ' + str(cluster_id) + ' sample_i: ' + str(i) + '\n' + samples[i] + '_ recurring community'
    ax.set_title(title)
    plt.tight_layout()
    name_to_save = 'network_cluster' + str(cluster_id) + '_' + str(i) + '_' + samples[i] + '_community'
    plt.savefig(os.path.join(path_to_save + '\\all_988_networks', name_to_save))
    plt.show()

    
    # plot community only
    # edges_with_gene_set = [(u, v) for u, v in G.edges() if u in gene_set or v in gene_set]
    # H = G.edge_subgraph(edges_with_gene_set).copy()
    
    edges_with_gene_set = [(u, v) for u, v in G.edges() if u in gene_set and v in gene_set]
    H = G.edge_subgraph(edges_with_gene_set).copy()

    
    # Calculate degree centrality for the subgraph
    degrees = np.array(list(nx.degree_centrality(H).values()))
    degrees_norm = degrees / np.max(degrees)
    
    # Get the 'LRP_norm' attribute for the edges in the subgraph
    widths = [G[u][v]['LRP_norm']*2 for u, v in H.edges()]
    
    # Determine the colors for the edges based on the LRP_norm attribute
    edge_colors = plt.cm.Greys(widths) if widths else 'black'
    
    # Update color_map for nodes in the subgraph
    color_map = ['blue' if node in gene_set else 'gray' for node in H.nodes()]
    
    # Get the positions for the nodes in the subgraph
    pos_subgraph = {node: pos for node, pos in pos_988[i].items() if node in H.nodes()}
    pos_subgraph = nx.spring_layout(H, weight='LRP_norm')
    
    # Draw the subgraph
    fig, ax = plt.subplots(figsize=(10,10))
    nx.draw(H, with_labels=False,
            node_color=color_map,
            width=[width * 1 for width in widths],
            pos=pos_subgraph,
            edge_color=edge_colors,
            ax=ax,
            node_size=degrees_norm * 500,
            edgecolors='white',  # Adds the white border to nodes
            linewidths=0.5)
    labels = {node: node.split('_')[0] for node in list(H.nodes)}
    nx.draw_networkx_labels(H, pos_subgraph, labels, font_size=10, font_color = 'red', ax=ax)
    
    
    title = 'Cluster: ' + str(cluster_id) + ' sample_i: ' + str(i) + '\n' + samples[i] + '_ recurring community only'
    ax.set_title(title)
    plt.tight_layout()
    name_to_save = 'network_cluster' + str(cluster_id) + '_' + str(i) + '_' + samples[i] + '_community_only'
    plt.savefig(os.path.join(path_to_save + '\\all_988_networks', name_to_save))
    plt.savefig(os.path.join(path_to_save + '\\all_988_networks', name_to_save+'.pdf'), format='pdf')
    plt.show()




















































# %%%% community similarities 2
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

top_communities['size'] = len(partitions_list)
top_communities['relative_frequency'] = top_communities['frequency'] / top_communities['size']
top_communities['intersection_within_cluster']  = ['']*top_communities.shape[0]
top_communities['union_within_cluster']  =  ['']*top_communities.shape[0]
top_communities['intersection_within_cluster_size']  = 0
top_communities['union_within_cluster_size']  =  0

 
top_communities['threshold'] = threshold

data_to_dendrogram = similarity_overlap
Z = linkage(data_to_dendrogram, method='ward')
dendrogram_cutoff = 40
cutoff = dendrogram_cutoff
fig, ax = plt.subplots(figsize=(10, 3))
dn = dendrogram(
    Z, 
    color_threshold=cutoff, 
    above_threshold_color='gray', 
    #truncate_mode='lastp',  # show only the last p merged clusters
    #p=30,  # show only the last 12 merged clusters
    #show_leaf_counts=True,  # show the number of samples in each cluster
    ax=ax
)

plt.axhline(y=cutoff, color='r', linestyle='--')
ax.set_ylabel('Distance')
ax.set_title('988 dataset\ncommunities_overlapsimilarity') 
plt.tight_layout()
plt.savefig(os.path.join(path_to_save, 'dendrogram_communities_jaccard_score_full_dataset_'.format(threshold)))
plt.show()

#sns.heatmap(similarity_jaccard, cmap = 'jet', vmin = 0, vmax = 1)
sns.clustermap(similarity_jaccard, cmap = 'jet', vmin = 0, vmax = 1, method = 'ward')
plt.title('Full dataset 988 ')
plt.show()
# sns.heatmap(similarity_dice, cmap = 'jet', vmin = 0, vmax = 1)
sns.clustermap(similarity_dice, cmap = 'jet', vmin = 0, vmax = 1, method = 'ward')

# sns.heatmap(similarity_overlap, cmap = 'jet', vmin = 0, vmax = 1)
sns.clustermap(similarity_overlap, cmap = 'jet', vmin = 0.5, vmax = 1, method = 'ward')

similarity_overlap_temp =similarity_overlap.copy()
similarity_overlap_temp[similarity_overlap<.9]= 0
sns.clustermap(similarity_overlap_temp, cmap = 'jet', vmin = 0.5, vmax = 1, method = 'ward')#, mask = similarity_overlap_temp>.8)
 
   
cluster_labels = fcluster(Z, t=cutoff, criterion='distance')
unique_labels, counts = np.unique(cluster_labels, return_counts=True)

top_communities['cluster_label'] = cluster_labels

for community_cluster in np.sort(unique_labels):
    
    communities_temp = top_communities[top_communities['cluster_label'] == community_cluster].reset_index(drop=True)
    genes_temp = communities_temp['genes'].to_list()
 
    genes_intersection = set(genes_temp[0]).intersection(*genes_temp[1:])
    genes_union = set().union(*genes_temp)

    top_communities.loc[top_communities['cluster_label'] == community_cluster, 'intersection_within_cluster']  = str(list(np.sort(list(genes_intersection))))
    top_communities.loc[top_communities['cluster_label'] == community_cluster, 'union_within_cluster']  = str(list(np.sort(list(genes_union))))
    
    top_communities.loc[top_communities['cluster_label'] == community_cluster, 'intersection_within_cluster_size']  = len(genes_intersection)
    top_communities.loc[top_communities['cluster_label'] == community_cluster, 'union_within_cluster_size']  = len(genes_union)

    
top_communities = top_communities.reset_index(drop=True)

    
top_communities.to_excel(os.path.join(path_to_save, 'top_communities_full_dataset_all_{}.xlsx'.format(threshold)))
    
 
# %%%% overalap

top_communities_temp = top_communities[top_communities['cluster_label']==1].reset_index(drop=True)
top_comm = top_communities_temp.loc[0,'genes']


top_communities_temp['overlap'] = [list(np.sort(list(set(x).intersection(set(top_comm))))) for x in top_communities_temp['genes']]
top_communities_temp['overlap_n']  = [len(x) for x in top_communities_temp['overlap']]
top_communities_temp['overlap_ratio']  = top_communities_temp['overlap_n'] / top_communities_temp['community_size']
# %%%%
 
    
top_communities_all_gr = top_communities.groupby(by = ['cluster_label']).first().reset_index()
    
top_communities_all_gr['intersection_within_cluster']
    
 
    
def count_communities_containing_genes(community_maps, genes_set):
   """
   Count how many communities the entire set of genes is contained within.
   
   Parameters:
   community_maps (list of dicts): A list where each element is a dictionary mapping community IDs to gene lists for each network.
   genes_set (set): The set of genes to check for across communities.

   Returns:
   int: The count of communities containing the entire set of genes.
   """
   count = 0
   for network_communities in community_maps:
       for community_genes in network_communities.values():
           # Check if all genes are in the current community
           if genes_set.issubset(set(community_genes)):
               count += 1
   return count

# Example usage:
#genes_set = set(['JAK2_exp', 'TNFAIP3_exp', 'JAK3_exp', 'PRF1_exp', 'CD79B_exp', 'CARD11_exp', 'CD74_exp', 'FCRL4_exp', 'SOCS1_exp', 'CYLD_exp', 'SYK_exp', 'IKZF1_exp', 'PTPRC_exp', 'CD28_exp', 'NFATC2_exp', 'CXCR4_exp', 'CD79A_exp', 'POU2AF1_exp', 'IL7R_exp', 'LCK_exp', 'B2M_exp', 'CCR4_exp', 'P2RY8_exp', 'CIITA_exp', 'BIRC3_exp', 'TNFRSF17_exp', 'BCL11B_exp', 'IRF4_exp', 'TCF7_exp', 'BTK_exp', 'CSF1R_exp', 'CCR7_exp', 'WAS_exp', 'STK4_exp', 'PDCD1LG2_exp'])
#gene_occurrences = count_communities_containing_genes(community_maps, genes_set)
import ast
gene_occurrences_list = []
gene_occurrences_outside_list = []

top_communities_all_gr_temp = top_communities_all_gr.copy()

community_maps = []
for partition in np.array(partitions_list):
    community_map = {}
    for node, comm in partition.items():
        community_map.setdefault(comm, []).append(node)
    community_maps.append(community_map)
    
for i,row, in top_communities_all_gr_temp.iterrows():
    print(i, row['intersection_within_cluster'])
    genes_set = set(ast.literal_eval(row['intersection_within_cluster']))
   #['JAK2_exp', 'TNFAIP3_exp', 'JAK3_exp', 'PRF1_exp', 'CD79B_exp', 'CARD11_exp', 'CD74_exp', 'FCRL4_exp', 'SOCS1_exp', 'CYLD_exp', 'SYK_exp', 'IKZF1_exp', 'PTPRC_exp', 'CD28_exp', 'NFATC2_exp', 'CXCR4_exp', 'CD79A_exp', 'POU2AF1_exp', 'IL7R_exp', 'LCK_exp', 'B2M_exp', 'CCR4_exp', 'P2RY8_exp', 'CIITA_exp', 'BIRC3_exp', 'TNFRSF17_exp', 'BCL11B_exp', 'IRF4_exp', 'TCF7_exp', 'BTK_exp', 'CSF1R_exp', 'CCR7_exp', 'WAS_exp', 'STK4_exp', 'PDCD1LG2_exp'])
    gene_occurrences = count_communities_containing_genes(community_maps, genes_set)
    
    print(gene_occurrences)
    
    gene_occurrences_list.append(gene_occurrences)
    print(row['cluster_label'])
    
community_maps = []

for partition in np.array(partitions_list):
    community_map = {}
    for node, comm in partition.items():
        community_map.setdefault(comm, []).append(node)
    community_maps.append(community_map)
    
for i,row, in top_communities_all_gr_temp.iterrows():
    print(i, row['intersection_within_cluster'])
    genes_set = set(ast.literal_eval(row['intersection_within_cluster']))
   #['JAK2_exp', 'TNFAIP3_exp', 'JAK3_exp', 'PRF1_exp', 'CD79B_exp', 'CARD11_exp', 'CD74_exp', 'FCRL4_exp', 'SOCS1_exp', 'CYLD_exp', 'SYK_exp', 'IKZF1_exp', 'PTPRC_exp', 'CD28_exp', 'NFATC2_exp', 'CXCR4_exp', 'CD79A_exp', 'POU2AF1_exp', 'IL7R_exp', 'LCK_exp', 'B2M_exp', 'CCR4_exp', 'P2RY8_exp', 'CIITA_exp', 'BIRC3_exp', 'TNFRSF17_exp', 'BCL11B_exp', 'IRF4_exp', 'TCF7_exp', 'BTK_exp', 'CSF1R_exp', 'CCR7_exp', 'WAS_exp', 'STK4_exp', 'PDCD1LG2_exp'])
    gene_occurrences = count_communities_containing_genes(community_maps, genes_set)
    
    print(gene_occurrences)
    
    gene_occurrences_outside_list.append(gene_occurrences)
    print(row['cluster_label'])    
    
    
        
        
        
top_communities_all_gr['gene_set_occurrences_within_cluster']     = gene_occurrences_list   
top_communities_all_gr['relative_gene_set_occurrences_within_cluster']     = top_communities_all_gr['gene_set_occurrences_within_cluster'] / top_communities_all_gr['size']   
top_communities_all_gr['gene_set_occurrences_outside_cluster']     = gene_occurrences_list   
top_communities_all_gr['relative_gene_set_occurrences_outside_cluster']     = top_communities_all_gr['gene_set_occurrences_outside_cluster'] / (988 - top_communities_all_gr['cluster_id_size']   )

top_communities_all_gr['cluster_id_comm_id'] = 'Cluster ' + top_communities_all_gr['cluster_id'].astype('str') + ' - community ' + top_communities_all_gr['cluster_label'].astype('str') 
top_communities_all_gr.to_excel(os.path.join(path_to_save, 'top_communities_all_gr.xlsx'))

to_plot = top_communities_all_gr.melt(id_vars='cluster_id_comm_id', value_vars=['relative_gene_set_occurrences_within_cluster','relative_gene_set_occurrences_outside_cluster'])#, var_name=None, value_name='value', col_level=None, ignore_index=True)
    
fig,ax=plt.subplots(figsize = (4,6))
sns.barplot(data=to_plot, y = 'cluster_id_comm_id', x = 'value', hue = 'variable',ax=ax, orient='h')
h, l = ax.get_legend_handles_labels()
labels = ['Inside cluster','Outside cluster']
ax.legend(h, labels, title="Relative occurances of community", bbox_to_anchor=(0.3, 1.1), loc = 'center')
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.set_xlim([0,1])
plt.tight_layout() 




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
