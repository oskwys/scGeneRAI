# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:28:21 2023

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

pathway = 'TP53'
pathway = 'PI3K'
pathway = 'allgenes'


network_pos = pd.read_excel(os.path.join(path_to_read , 'network_{}_{}_pos.xlsx'.format(group, pathway)), index_col = 0)
network_neg = pd.read_excel(os.path.join(path_to_read , 'network_{}_{}_neg.xlsx'.format(group, pathway)), index_col = 0)

# %% investigate

network_pos = network_pos.sort_values('LRP', ascending=False).reset_index(drop=True).reset_index()
network_neg = network_neg.sort_values('LRP', ascending=False).reset_index(drop=True).reset_index()
# %% join networks pos + neg
network_neg['type'] = 'neg'
network_pos['type'] = 'pos'
network = pd.concat((network_pos, network_neg))
# %%
network_pos['LRP'].plot(label = 'pos')
network_neg['LRP'].plot(label = 'neg')
plt.legend()
plt.show()

sns.scatterplot(data = network_pos, x = 'index', y = 'LRP', hue = 'edge_type')
plt.show()
sns.scatterplot(data = network_neg, x = 'index', y = 'LRP', hue = 'edge_type')
plt.show()
# %%

g = sns.FacetGrid(network_pos, col="edge_type", col_wrap = 4)
g.map(sns.scatterplot, 'index', 'LRP')


g = sns.FacetGrid(network_neg, col="edge_type", col_wrap = 4)
g.map(sns.scatterplot, 'index', 'LRP')



# %%
g = sns.FacetGrid(network, col="edge_type", col_wrap = 4, hue= 'type', sharey=False, sharex=False)
g.map(sns.scatterplot, 'index', 'LRP')
plt.legend()

# %% threshold

threshold = 0.006


network_th = network[network['LRP'] > threshold]

g = sns.FacetGrid(network_th, col="edge_type", col_wrap = 4, hue= 'type', sharey=False, sharex=False)
g.map(sns.scatterplot, 'index', 'LRP')
plt.legend()

network_th_pos = network_th[network_th['type'] == 'pos'].reset_index(drop=True)
network_th_neg = network_th[network_th['type'] == 'neg'].reset_index(drop=True)

# %%
topn = 50
network_topn = network[network['index']<topn]

g = sns.FacetGrid(network_topn, col="edge_type", col_wrap = 4, hue= 'type', sharey=False, sharex=False)
g.map(sns.scatterplot, 'index', 'LRP')
plt.legend()

network_topn_pos = network_topn[network_topn['type'] == 'pos'].reset_index(drop=True)
network_topn_neg = network_topn[network_topn['type'] == 'neg'].reset_index(drop=True)

# %%


# %% PLOT LRP network

# %%% define df_temp


df_temp = network_th_neg.copy()
df_temp = network_th_pos.copy()
df_temp = network_topn_pos.copy()
df_temp = network_topn_neg.copy()

# %%% display
#cairocffi.install_as_pycairo()
from igraph import Graph, plot
import cairocffi
import igraph as ig
import matplotlib.cm as cm


def convert_networkx_to_igraph(G_nx):
    G_ig = Graph.TupleList(G_nx.edges(data=True), directed=False, weights=True)
    return G_ig


df_temp ['LRP'] = df_temp ['LRP'] / df_temp ['LRP'].max()

G_nx = nx.from_pandas_edgelist(df_temp, source='source_gene', target='target_gene', edge_attr='LRP')

g = convert_networkx_to_igraph(G_nx)
g.es["weight"] = df_temp['LRP']


# Setting the edge width based on the LRP column
#g.es["width"] = df["LRP"] / df["LRP"].max() * 5   # Multiplied by 10 for better visibility, adjust as necessary

# Setting the vertex size based on node degree
#g.vs["size"] = [deg*0.2  for deg in g.degree()]  # Multiplied by 10 for better visibility, adjust as necessary

# Setting the vertex label
#g.vs["label"] = g.vs["name"]

types_ = [name.split('_')[1] for name in g.vs["name"]]
mapper = {'exp':'lightblue', 'mut':'red', 'amp':'green', 'del':'orange','fus':'blue'}

# Plotting the graph
layout = g.layout("fr", weights=g.es["weight"])

  # Using the Kamada-Kaway layout which generally produces non-overlapping nodes
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
visual_style["edge_width"] = df_temp["LRP"] / df_temp["LRP"].max() * 10 
visual_style["vertex_frame_color"] = 'gray'

normalized_LRP = np.array(df_temp["LRP"]) / df_temp["LRP"].max()
colors = [cm.jet(val) for val in normalized_LRP]
visual_style["edge_color"] = colors

#visual_style["vertex_label_dist"] = 2
visual_style["layout"] = layout
visual_style["bbox"] = (10, 10)
#visual_style["margin"] = 20


fig, ax = plt.subplots(figsize = (15,15))
plot(g,  target = ax, autocurve=True,  **visual_style) 


# %%
import matplotlib.patches as mpatches



# Your provided color mapper
mapper = {'exp':'lightblue', 'mut':'red', 'amp':'green', 'del':'orange','fus':'blue'}

# Create patches (handles) for each type-color pair
patches = [mpatches.Patch(color=color, label=label) for label, color in mapper.items()]

# Create the legend
plt.figure(figsize=(10,5))
plt.legend(handles=patches, loc='upper left')
plt.axis('off')  # Turn off the axis
plt.show()


# %% INVESTIGATE in DETAILS

data_temp = data_to_model[data_to_model.index.isin(samples_her2_neg)]

gene1 = 'TP53_mut'
gene2 = 'CHEK2_exp'
fig,ax =plt.subplots(figsize = (3,5))
sns.swarmplot(x = data_temp[gene1], y = data_temp[gene2],ax=ax)



data_temp = data_to_model[data_to_model.index.isin(samples_her2_pos)]
gene1 = 'MDM4_amp'
gene2 = 'MDM2_exp'
fig,ax =plt.subplots(figsize = (3,5))
sns.swarmplot(x = data_temp[gene1], y = data_temp[gene2],ax=ax)

# %%% pairplot
genes_pairplot = ['NOTCH4_exp','SFRP2_exp','DKK3_exp','SFRP4_exp','PDGFRA_exp','LATS2_exp','TGFBR2_exp']

data_temp1 = data_to_model[data_to_model.index.isin(samples_her2_pos)]
data_temp1['label'] = 'her+'

data_temp2 = data_to_model[data_to_model.index.isin(samples_her2_neg)]
data_temp2['label'] = 'her-'

data_temp = pd.concat((data_temp1, data_temp2))
sns.pairplot(data_temp[genes_pairplot+['label']], hue='label',  markers=['s', 'o'], palette={'her+': 'red', 'her-': 'blue'})

sns.pairplot(data_temp2[genes_pairplot],  markers=[ 'o'])


data_temp1 = data_to_model[data_to_model.index.isin(samples_her2_neg)]
data_temp1
genes_pairplot
suffixes = ['_mut', '_fus', '_del', '_amp']

modified_genes = [gene.split('_exp')[0] for gene in genes_pairplot]
selected_columns = [col for col in data_temp2.columns if any(gene in col for gene in genes_pairplot) and not col.endswith('_exp')]

matching_strings = [s for s in data_temp2.columns for gene in [gene.replace('_exp','') for gene in  genes_pairplot] if gene in s]
filtered_lst = [s for s in matching_strings if "_exp" not in s]

    
# Set up the grid of plots
n_rows = len(genes_pairplot)
n_cols = len(filtered_lst)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 15))

for i, gene in enumerate(genes_pairplot):
    for j, col in enumerate(filtered_lst):
        print(gene,col)
        sns.swarmplot(x=data_temp2[col], y=data_temp2[gene], ax=axes[i, j], size = 2)
        axes[i, j].set_title(f"{col} vs {gene}")
        
plt.tight_layout()
plt.show()


# #################
genes_pairplot = ['MDM4_exp','TSC1_exp','MTOR_exp','SPEN_exp','EP300_exp','PIK3R2_exp','AKT1_exp','SFRP2_exp']

data_temp1 = data_to_model[data_to_model.index.isin(samples_her2_pos)]
data_temp1['label'] = 'her+'

data_temp2 = data_to_model[data_to_model.index.isin(samples_her2_neg)]
data_temp2['label'] = 'her-'

data_temp = pd.concat((data_temp1, data_temp2))[genes_pairplot+['label']]
sns.pairplot(data_temp, hue='label',  markers=['s', 'o'], palette={'her+': 'red', 'her-': 'blue'})

sns.pairplot(data_temp2[genes_pairplot])#,  markers=['o'], palette={'her-': 'blue'})


# %%


path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\networks_brca'

edges = network_topn_pos

G_average = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
H = nx.Graph()
H.add_nodes_from(sorted(G_average.nodes(data=True)))
H.add_edges_from(G_average.edges(data=True))

pos = nx.shell_layout(H)

def plot_network_(edges, color_values, top_n, subtype, i, file, path_to_save, node_size=100, layout=None, pos = None, sample_id=''):
    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
    #degrees = np.array(list(nx.degree_centrality(G).values())) 
    #degrees = degrees / np.max(degrees) * 500
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
    edge_colors =cm.Greys((widths  - np.min(widths) )/ (np.max(widths) - np.min(widths)))
    
    fig,ax = plt.subplots(figsize=  (15,15))
    nx.draw(G, with_labels=True, 
            #node_color=node_colors,
            width = widths,
            pos = pos, font_size = 25,
            #cmap = colors, 
            edge_color = edge_colors , 
            ax=ax,  
            #node_size = degrees
            node_size = node_size)
    ax.set_title(sample_id)
    plt.tight_layout()
    #plt.savefig(os.path.join(path_to_save , 'network_{}_{}_{}_{}.svg'.format(file, top_n, subtype, i)), format = 'svg')
    #plt.savefig(os.path.join(path_to_save , 'network_{}_{}_{}_{}.png'.format(file, top_n, subtype, i)), dpi = 300)
pos = nx.nx_agraph.graphviz_layout(G_average)
    
color_values = network_topn_pos['LRP'].values
plot_network_(network_topn_pos, color_values, 500, 'BRCA', 'average', 'brca_shell', path_to_save, layout='kamada_kawai_layout', sample_id='Average')
plot_network_(network_topn_neg, color_values, 500, 'BRCA', 'average', 'brca_shell', path_to_save, layout='kamada_kawai_layout', sample_id='Average')























