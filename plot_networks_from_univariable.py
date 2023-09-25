# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:20:49 2023

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
# %% LOAD

path_to_read = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\networks'
group = 'her2'

pathway = 'PI3K'
pathway = 'TP53'

network_all = pd.read_excel(os.path.join(path_to_read , 'univariable_res_{}_all_BRCA.xlsx'.format(pathway)), index_col = 0)

network_pos = pd.read_excel(os.path.join(path_to_read , 'univariable_res_{}_{}_pos.xlsx'.format(pathway, group)), index_col = 0)
network_neg = pd.read_excel(os.path.join(path_to_read , 'univariable_res_{}_{}_neg.xlsx'.format(pathway, group)), index_col = 0)



# %% set thresholds

p_max = 0.01

network_all_th = network_all[network_all['p-val'] < p_max].reset_index(drop=True)
network_pos_th = network_pos[network_pos['p-val'] < p_max].reset_index(drop=True)
network_neg_th = network_neg[network_neg['p-val'] < p_max].reset_index(drop=True)


# %% functions

#cairocffi.install_as_pycairo()
from igraph import Graph, plot
import cairocffi
import igraph as ig
import matplotlib.cm as cm


def convert_networkx_to_igraph(G_nx):
    G_ig = Graph.TupleList(G_nx.edges(data=True), directed=False, weights=True)
    return G_ig
def plot_uni_netowrk(df_temp):
        
    G_nx = nx.from_pandas_edgelist(df_temp, source='source_gene', target='target_gene', edge_attr='width')
    
    g = convert_networkx_to_igraph(G_nx)
    g.es["weight"] = df_temp['width']
    
    
    # Setting the edge width based on the LRP column
    #g.es["width"] = df["LRP"] / df["LRP"].max() * 5   # Multiplied by 10 for better visibility, adjust as necessary
    
    # Setting the vertex size based on node degree
    #g.vs["size"] = [deg*0.2  for deg in g.degree()]  # Multiplied by 10 for better visibility, adjust as necessary
    
    # Setting the vertex label
    #g.vs["label"] = g.vs["name"]
    
    types_ = [name.split('_')[1] for name in g.vs["name"]]
    mapper = {'exp':'lightblue', 'mut':'red', 'amp':'green', 'del':'orange'}
    
    # Plotting the graph
    random.seed(42)
    layout = g.layout("fr", weights=g.es["weight"])
    #layout = g.layout("kk")  # Using the Kamada-Kaway layout which generally produces non-overlapping nodes
    #layout = g.layout("circle")  # Using the Kamada-Kaway layout which generally produces non-overlapping nodes
    
    
    
    # # Find cliques
    # clique_sizes = range(10,22)  # For example, find cliques of size 3 and 4
    # cliques = g.cliques(min=clique_sizes[0], max=clique_sizes[-1])
    
    # # Highlight cliques
    # for clique in cliques:
    #     for v in clique:
    #         g.vs[v]["color"] = "magenta"  # Highlighting clique vertices with yellow color
    
    
    
    visual_style = {}
    visual_style["vertex_size"] = [(3+deg/10 )/10   for deg in g.degree()] 
    visual_style["vertex_color"] = [mapper[type_] for type_ in types_]
    visual_style["vertex_label"] = [name.split('_')[0] for name in g.vs["name"]]
    visual_style["edge_width"] = df_temp["width"] / df_temp["width"].max() * 10 
    visual_style["vertex_frame_color"] = 'gray'
    
    
    normalized_LRP = np.array(df_temp["width"]) / df_temp["width"].max()
    colors = [cm.jet(val) for val in normalized_LRP]
    visual_style["edge_color"] = colors
    
    #visual_style["vertex_label_dist"] = 2
    visual_style["layout"] = layout
    visual_style["bbox"] = (10, 10)
    #visual_style["margin"] = 20
    
    
    fig, ax = plt.subplots(figsize = (10,10))
    plot(g,  target = ax, autocurve=True,  **visual_style) 
# %% ALL samples
import random

df_temp = network_all_th.copy()
plot_uni_netowrk(network_all_th)

plot_uni_netowrk(network_pos_th)

plot_uni_netowrk(network_neg_th)
