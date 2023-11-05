# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:55:05 2023

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


# %%
path_to_save = '/home/d07321ow/scratch/results_LRP_BRCA/networks'

topn = 1000

LRP_pd = pd.read_csv(os.path.join(path_to_save, 'LRP_individual_top{}.csv'.format(topn)), index_col = 0)


# %%
import community as community_louvain
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
from kneefinder import KneeFinder
from kneed import KneeLocator

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




edges = LRP_pd.iloc[:,0].reset_index()
edges['source_gene']  = edges['index'].str.split(' - ', expand = True)[0]
edges['target_gene']= edges['index'].str.split(' - ', expand = True)[1]
G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene')

nodes_all = list(G.nodes)
nodes_labaled_all = pd.DataFrame(nodes_all, columns=['node'])

G_list = []
partitions_list = []

for i in range(988):
    print(i)
    edges = LRP_pd.iloc[:,i].reset_index()
    
    edges['source_gene']  = edges['index'].str.split(' - ', expand = True)[0]
    edges['target_gene']= edges['index'].str.split(' - ', expand = True)[1]
    edges  = edges.rename(columns = {edges.columns[1]:'LRP'})
    edges['LRP_norm'] = edges['LRP'] / edges['LRP'].max()
    edges = edges.sort_values('LRP', ascending = False).reset_index(drop=True)
    #kf = KneeFinder(edges.index, edges['LRP'])
    #knee_x, knee_y = kf.find_knee()
    kl = KneeLocator(edges.index, edges['LRP'], curve="convex", direction = 'decreasing', S=1, interp_method='polynomial')
    knee_x = int(kl.knee)
    #kf.plot()
    #knee_x=2000
    edges = edges.iloc[:int(knee_x), :]
      
    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP_norm')
    partition = detect_communities(G)
    
    G_list.append(G)
    partitions_list.append(partition)
    
    labels = communities_to_labels(partition)
    edges_ = edges['index'].to_list()
    nodes_ = list(G.nodes)
    
    nodes_labaled = pd.DataFrame(np.array([nodes_, labels]).T, columns = ['node', 'community_{}'.format(i)])
    
    nodes_labaled_all = nodes_labaled_all.merge(nodes_labaled, on = 'node', how = 'left')
    
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



from itertools import combinations
# Number of networks
n_networks = len(G_list)
# Create an n x n matrix filled with zeros (or NaN if you prefer)
ars_matrix = np.zeros((n_networks, n_networks))
amis_matrix = np.zeros((n_networks, n_networks))
shared_nodes_matrix = np.zeros((n_networks, n_networks))
shared_edges_matrix = np.zeros((n_networks, n_networks))

# Iterate over all unique pairs of networks
for (i, G1), (j, G2) in combinations(enumerate(G_list), 2):
    print(i,j)
    # Find the shared nodes between the two graphs
    shared_nodes = set(G1.nodes) & set(G2.nodes)
    # Fill the symmetric matrix positions
    shared_nodes_matrix[i, j] = len(shared_nodes)
    shared_nodes_matrix[j, i] = len(shared_nodes)
    
    
    shared_edges = set(G1.edges) & set(G2.edges)
    # Fill the symmetric matrix positions
    shared_edges_matrix[i, j] = len(shared_edges)
    shared_edges_matrix[j, i] = len(shared_edges)
    
    
    # Get the community labels for the shared nodes in both networks
    labels1 = [partitions_list[i][node] for node in shared_nodes if node in partitions_list[i]]
    labels2 = [partitions_list[j][node] for node in shared_nodes if node in partitions_list[j]]

    # Calculate the ARS between the two sets of labels
    ars = adjusted_rand_score(labels1, labels2)
    amis = adjusted_mutual_info_score(labels1, labels2)
    
    # Fill the symmetric matrix positions
    ars_matrix[i, j] = ars
    ars_matrix[j, i] = ars
    
    amis_matrix[i, j] = amis
    amis_matrix[j, i] = amis


pd.DataFrame(shared_nodes_matrix).to_csv(os.path.join(path_to_save, 'shared_nodes_matrix_{}.csv'.format(topn)))
pd.DataFrame(shared_edges_matrix).to_csv(os.path.join(path_to_save, 'shared_edges_matrix_{}.csv'.format(topn)))
pd.DataFrame(ars_matrix).to_csv(os.path.join(path_to_save, 'ars_matrix_{}.csv'.format(topn)))
pd.DataFrame(amis_matrix).to_csv(os.path.join(path_to_save, 'amis_matrix_{}.csv'.format(topn)))


