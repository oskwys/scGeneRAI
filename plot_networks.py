# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:11:31 2023

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

# %%

files = os.listdir('./results')



df_clinical_features = pd.read_csv( os.path.join(path_to_data, 'CCE_clinical_features.csv') )


subtypes = ['BRCA', 'LGG', 'UCEC', 'LUAD', 'HNSC', 'PRAD', 'LUSC', 'STAD', 'COAD',
       'SKCM', 'CESC', 'SARC', 'OV', 'PAAD', 'ESCA', 'GBM', 'READ', 'UVM',
       'UCS', 'CHOL']
for subtype in subtypes[::-1]:
    print('\n ___________ NEW MODEL ___________ \n')
    print(subtype)
    
    
subtype = 'LUAD'

ids_ = df_clinical_features.loc[df_clinical_features['acronym'] == subtype, 'bcr_patient_barcode'].values
matching_files = [s for s in files if any(xs in s for xs in ids_)]

network_data = pd.DataFrame()

# %%

def remove_same_source_target(data):
    
    data = data[data['source_gene'] != data['target_gene']]

    return data

def get_node_colors(network):
    colors = pd.Series(list(network.node))
    colors [  colors.str.contains('exp') ] = 'lightblue'
    colors [  colors.str.contains('mut') ] = 'red'
    colors [  colors.str.contains('del') ] = 'green'
    colors [  colors.str.contains('amp') ] = 'orange'
    colors [  colors.str.contains('fus') ] = 'magenta'
    
    return list(colors.values)


# %% individual samples

for file in matching_files:
        
    df_temp = pd.read_csv('./results/' + file)
    df_temp['sample'] = file
    print(df_temp.sort_values(by='LRP', ascending=False).iloc[:20,1:4])
    network_data = pd.concat((network_data, df_temp))



network_data = network_data[network_data['source_gene'] != network_data['target_gene']]
average_network = network_data[['LRP', 'source_gene', 'target_gene']].groupby(['source_gene', 'target_gene']).mean().reset_index()


edges = average_network.sort_values(by='LRP', ascending=False).iloc[:20,:]


to_cluster = average_network.pivot_table(values = 'LRP', index = 'source_gene', columns = 'target_gene').fillna(0)

sns.clustermap(to_cluster, method = 'ward', cmap = 'Reds')



# %% comparing 2 samples
matching_files
df_1 = pd.read_csv('./results/' + matching_files[0])
df_1 = remove_same_source_target(df_1).sort_values(by='LRP', ascending=False).iloc[:100,:]
df_2 = pd.read_csv('./results/' + matching_files[1])
df_2 = remove_same_source_target(df_2).sort_values(by='LRP', ascending=False).iloc[:100,:]




network = nx.from_pandas_edgelist(df_1, source='source_gene', target='target_gene', edge_attr='LRP')
nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*100)

network = nx.from_pandas_edgelist(df_2, source='source_gene', target='target_gene', edge_attr='LRP')
pos = nx.spring_layout(network)

fig,ax = plt.subplots(figsize=  (15,15))
nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*100, pos = pos, font_size = 5,ax=ax)
plt.savefig('aaa.svg', format = 'svg')
   

# %% plot all netowrks
path_to_save = r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI\results_UCS'

top_n = 50
matching_files = os.listdir('./results_UCS/', )
for file in matching_files:
        
    #df_temp = pd.read_csv('./results_UCS/' + file)
    
    file_path = os.path.join(r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI\results_UCS' , file)
    print(file_path)
    with open(file_path, 'rb') as file_:
        df_temp = pickle.load(file_)
    
    df_temp = remove_same_source_target(df_temp).sort_values(by='LRP', ascending=False).iloc[:top_n,:]


    network = nx.from_pandas_edgelist(df_temp, source='source_gene', target='target_gene', edge_attr='LRP')
    degrees = np.array(list(nx.degree_centrality(network).values())) 
    degrees = degrees / np.max(degrees) * 2000
    #nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*100, node_size = network.degree)
    pos = nx.spring_layout(network)
    
    colors = get_node_colors(network)
    widths = df_temp['LRP'] / df_temp['LRP'].max() * 3
    edge_colors =cm.Reds((widths  - np.min(widths) )/ (np.max(widths) - np.min(widths)))
    
    fig,ax = plt.subplots(figsize=  (15,15))
    nx.draw(network, with_labels=True, 
            node_color=colors,
            width = widths,
            pos = pos, font_size = 6,
            #cmap = colors, 
            edge_color = edge_colors , 
            ax=ax,  
            node_size = degrees)
    plt.savefig(os.path.join(path_to_save , 'network_{}_{}.svg'.format(file, top_n)), format = 'svg')
    plt.savefig(os.path.join(path_to_save , 'network_{}_{}.png'.format(file, top_n)), dpi = 300)








file1 = matching_files[2]
file2 = matching_files[3]

df_temp = pd.read_csv('./results/' + file1)
df_temp = remove_same_source_target(df_temp).sort_values(by='LRP', ascending=False).iloc[:top_n,:]
network1 = nx.from_pandas_edgelist(df_temp, source='source_gene', target='target_gene', edge_attr='LRP')

df_temp = pd.read_csv('./results/' + file2)
df_temp = remove_same_source_target(df_temp).sort_values(by='LRP', ascending=False).iloc[:top_n,:]
network2 = nx.from_pandas_edgelist(df_temp, source='source_gene', target='target_gene', edge_attr='LRP')

paths, cost = nx.optimize_graph_edit_distance(network1, network2)
for v in nx.optimize_graph_edit_distance(network1, network2):
    minv = v
    print(v)
    
    
file = os.path.join(r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI\results_UCS' , file)
with open(file, 'rb') as file_:
    df_temp = pickle.load(file_)
    print(f'Object successfully loaded from "{file_name}"')







