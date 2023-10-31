# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 23:19:08 2023

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

# %% get samples
path_to_networks = '/home/d07321ow/scratch/results_LRP_BRCA'
path_to_networks = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\networks'


path_to_lrp_results = '/home/d07321ow/scratch/results_LRP_BRCA'
path_to_lrp_results = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\results'

samples = f.get_samples_with_lrp(path_to_lrp_results)

path_to_data = '/home/d07321ow/scratch/scGeneRAI/data/data_BRCA'
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'


data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(path_to_data)


samples = df_clinical_features['bcr_patient_barcode'].to_list()
# %%% get sample goups

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

samples_groups = f.get_samples_by_group(df_clinical_features)  


# %% load networks

networks = {}

for group in samples_groups.keys():
    print(group)
    
    for subgroup in samples_groups[group].keys():
        print(subgroup)
        
        network_temp = pd.read_csv(os.path.join(path_to_networks, 'network_{}_allgenes.csv'.format(subgroup)))
        
        networks[subgroup] = network_temp


# %% LRP distirbutions


for index, (subgroup, network_temp) in enumerate(networks.items()):
    print(subgroup)
    
    
    # Create a histogram
    sns.histplot(network_temp['LRP'], bins=100, kde=False, color='blue')
    
    # Customize the plot
    plt.title(subgroup)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    
data = []
for key, df in networks.items():
    df['Network'] = key  # Add a column for the network name
    data.append(df)
all_data = pd.concat(data)

# Set up a FacetGrid
g = sns.displot(all_data, x="LRP", col="Network", kind="ecdf",  col_wrap = 4, height=4, aspect=.7,)

   
    
    
# %% plot networks
# %%% display
#cairocffi.install_as_pycairo()
from igraph import Graph, plot
import cairocffi
import igraph as ig
import matplotlib.cm as cm

LRP_threshold = .1 # % of highest LRP


for index, (subgroup, network_temp) in enumerate(networks.items()):
    print(subgroup)

    df_temp = network_temp.sort_values('LRP', ascending=False).reset_index(drop=True).iloc[:int(network_temp.shape[0] * LRP_threshold/100), : ]
    
    
    def convert_networkx_to_igraph(G_nx):
        G_ig = Graph.TupleList(G_nx.edges(data=True), directed=False, weights=True)
        return G_ig
    
    
    df_temp ['LRP'] = (df_temp ['LRP'] - df_temp ['LRP'].min() )/ (df_temp ['LRP'].max() - df_temp ['LRP'].min())+ 0.1
    
    G_nx = nx.from_pandas_edgelist(df_temp, source='source_gene', target='target_gene', edge_attr='LRP')
    
    g = convert_networkx_to_igraph(G_nx)
    g.es["weight"] = df_temp['LRP']
    
    
    types_ = [name.split('_')[1] for name in g.vs["name"]]
    mapper = {'exp':'lightblue', 'mut':'red', 'amp':'green', 'del':'orange','fus':'blue'}
    
    # Plotting the graph
    layout = g.layout("fr", weights=g.es["weight"])
    
    
    visual_style = {}
    visual_style["vertex_size"] = [(3+deg/10 )/10   for deg in g.degree()] 
    visual_style["vertex_color"] = [mapper[type_] for type_ in types_]
    visual_style["vertex_label"] = [name.split('_')[0] for name in g.vs["name"]]
    visual_style["vertex_label_size"] = 3
    visual_style["edge_width"] = df_temp["LRP"] / df_temp["LRP"].max() * 5 
    visual_style["vertex_frame_color"] = 'gray'
    
    normalized_LRP = np.array(df_temp["LRP"]) / df_temp["LRP"].max()
    colors = [cm.Reds(val) for val in normalized_LRP]
    visual_style["edge_color"] = colors
    
    #visual_style["vertex_label_dist"] = 2
    visual_style["layout"] = layout
    visual_style["bbox"] = (10, 10)
    #visual_style["margin"] = 20
    
    
    fig, ax = plt.subplots(figsize = (15,15))
    plot(g,  target = ax, autocurve=True,  **visual_style) 
    plt.title(subgroup)
    plt.tight_layout()
        
    plt.savefig(os.path.join(path_to_save, 'network_all_{}_th{}.png'.format(subgroup, str(LRP_threshold))), dpi = 400)        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        