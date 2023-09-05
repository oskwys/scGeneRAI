# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:25:24 2023

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

import itertools
# %%

def remove_same_source_target(data):
    
    data = data[data['source_gene'] != data['target_gene']]

    return data

def get_node_colors(network):
    colors = pd.Series(list(network.nodes))
    colors  [  -colors.str.contains('mut') ]= 'lightblue'
    colors [  colors.str.contains('mut') ] = 'red'
    colors [  colors.str.contains('del') ] = 'green'
    colors [  colors.str.contains('amp') ] = 'orange'
    colors [  colors.str.contains('fus') ] = 'magenta'
    
    return list(colors.values)

def get_genes_in_all_i(lrp_dict, top_n, n = 5):
    
    genes = []
    for i in range(n):
        
        print(i)
        
        data_temp = lrp_dict[str(i)].iloc[:top_n,:]
        
        
        a = data_temp['source_gene'].to_list()
        b = data_temp['target_gene'].to_list()
        genes.append(a)
        genes.append(b)
        
    genes = [item for sublist in genes for item in sublist]
    genes = list(set(genes))
    genes.sort()
    
    return genes




def get_genes_related_to_gene_in_all_i(lrp_dict, gene, top_n, n = 5):
    
    genes = []
    for i in range(n):
        
        
        
        df_temp = lrp_dict[str(i)]
        df_temp = df_temp[df_temp['source_gene'].str.contains(gene) | df_temp['target_gene'].str.contains(gene) ].iloc[:top_n,:]
        
        a = df_temp['source_gene'].to_list()
        b = df_temp['target_gene'].to_list()
        genes.append(a)
        genes.append(b)
        
    genes = [item for sublist in genes for item in sublist]
    genes = list(set(genes))
    genes.sort()
    
    return genes


def get_all_unique_genes(df_temp):
    genes =[]
    a = df_temp['source_gene'].to_list()
    b = df_temp['target_gene'].to_list()
    genes.append(a)
    genes.append(b)
    
    gene_names = [item for sublist in genes for item in sublist]
    gene_names = list(set(gene_names))
    gene_names.sort()
    
    genes = [gene.split('_')[0] for gene in gene_names]
    genes = list(set(genes))
    genes.sort()

    return genes, gene_names


def remove_exp_string(edges_df):
    
    edges_df.loc[edges_df['source_gene'].str.contains('_exp'), 'source_gene'] = edges_df.loc[edges_df['source_gene'].str.contains('_exp'), 'source_gene'] .str.replace('_exp','')
    edges_df.loc[edges_df['target_gene'].str.contains('_exp'), 'target_gene'] = edges_df.loc[edges_df['target_gene'].str.contains('_exp'), 'target_gene'] .str.replace('_exp','')
    
    return edges_df

def plot_network_(edges, node_colors, top_n, subtype, i, file, path_to_save, layout=None, pos = None):
    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
    degrees = np.array(list(nx.degree_centrality(G).values())) 
    degrees = degrees / np.max(degrees) * 500
    #nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*100, node_size = network.degree)
    
        
    if pos is not None:
        print('using POS')
    else:
        if layout == None:
            pos = nx.spring_layout(G)
            
        elif layout== 'spectral':
            pos = nx.spectral_layout(G)
        elif layout== 'spectral':
            pos = nx.spectral_layout(G)
            
    #colors = get_node_colors(G)
    colors = node_colors
    widths = edges['LRP'] / edges['LRP'].max() * 10
    edge_colors =cm.Greys((widths  - np.min(widths) )/ (np.max(widths) - np.min(widths)))
    
    fig,ax = plt.subplots(figsize=  (15,15))
    nx.draw(G, with_labels=True, 
            node_color=node_colors,
            width = widths,
            pos = pos, font_size = 6,
            #cmap = colors, 
            edge_color = edge_colors , 
            ax=ax,  
            #node_size = degrees
            node_size = 100)
    plt.savefig(os.path.join(path_to_save , 'network_{}_{}_{}_{}.svg'.format(file, top_n, subtype, i)), format = 'svg')
    plt.savefig(os.path.join(path_to_save , 'network_{}_{}_{}_{}.png'.format(file, top_n, subtype, i)), dpi = 300)

    
def get_pivoted_heatmap(edges,genes):
    
    # filter edges
    edges = edges[edges['source_gene'].isin(genes) | edges['target_gene'].isin(genes) ]

    
    template_ = pd.DataFrame(columns = genes, index= genes).fillna(0)
    for row in edges.iterrows():
        
        row = row[1]
        template_.loc[row['source_gene'], row['target_gene']] = row['LRP']
        
        
    return template_


def add_suffixes_to_genes(genes):
    gene_names = []
    suffixes = ['_amp','_mut','_del','_fus','_exp']
    for gene in genes:
        
        for suffix in suffixes:
            gene_names.append(gene+suffix)
            
    return gene_names


def add_edge_colmn(edges):
    edges['edge'] = edges['source_gene'] + ' - '  + edges['target_gene']
    
    edges['edge_type'] = (edges['source_gene'].str.split('_',expand=True).iloc[:,1] + ' - '  + edges['target_gene'].str.split('_',expand=True).iloc[:,1]).str.split(' - ').apply(np.sort).str.join('-')
    
    
    '''edges['edge_type'] = 'exp-exp'
    edges.loc[ edges['edge'].str.contains('mut') & edges['edge'].str.contains('exp') , 'edge_type'] = 'mut-exp'
    edges.loc[ edges['edge'].str.contains('mut') & edges['edge'].str.contains('amp') , 'edge_type'] = 'mut-amp'
    edges.loc[ edges['edge'].str.contains('mut') & edges['edge'].str.contains('del') , 'edge_type'] = 'mut-del'
    edges.loc[ edges['edge'].str.contains('mut') & edges['edge'].str.contains('fus') , 'edge_type'] = 'mut-fus'
    
    edges.loc[ edges['edge'].str.contains('amp') & edges['edge'].str.contains('exp') , 'edge_type'] = 'amp-exp'
    edges.loc[ edges['edge'].str.contains('amp') & edges['edge'].str.contains('del') , 'edge_type'] = 'amp-del'
    edges.loc[ edges['edge'].str.contains('amp') & edges['edge'].str.contains('fus') , 'edge_type'] = 'amp-fus'
    
    
    edges.loc[ edges['edge'].str.contains('del') & edges['edge'].str.contains('exp') , 'edge_type'] = 'del-exp'
    edges.loc[ edges['edge'].str.contains('del') & edges['edge'].str.contains('del') , 'edge_type'] = 'del-fus'
    
    edges.loc[ edges['source_gene'].str.contains('mut') & edges['target_gene'].str.contains('mut') , 'edge_type'] = 'mut-mut'
    edges.loc[ edges['source_gene'].str.contains('amp') & edges['target_gene'].str.contains('amp') , 'edge_type'] = 'amp-amp'
    edges.loc[ edges['source_gene'].str.contains('del') & edges['target_gene'].str.contains('del') , 'edge_type'] = 'del-del'
    edges.loc[ edges['source_gene'].str.contains('fus') & edges['target_gene'].str.contains('fus') , 'edge_type'] = 'fus-fus'
    '''
    
    return edges

# %% get input data
# %% load data

#path_to_data = '/home/owysocki/Documents/KI_dataset/data_to_model'
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_model'
#path_to_data = 'KI_dataset/data_to_model'

#df_fus = feather.read_feather(os.path.join(path_to_data, 'CCE_fusions_to_model') )
#df_mut = feather.read_feather( os.path.join(path_to_data, 'CCE_mutations_to_model') ,)
#df_amp = feather.read_feather( os.path.join(path_to_data, 'CCE_amplifications_to_model') )
#df_del = feather.read_feather( os.path.join(path_to_data, 'CCE_deletions_to_model') , )
#df_exp = feather.read_feather( os.path.join(path_to_data, 'CCE_expressions_to_model') , )



df_fus = pd.read_csv(os.path.join(path_to_data, 'CCE_fusions_to_model.csv') ,index_col = 0)
df_mut = pd.read_csv( os.path.join(path_to_data, 'CCE_mutations_to_model.csv') ,index_col = 0)
df_amp = pd.read_csv( os.path.join(path_to_data, 'CCE_amplifications_to_model.csv'),index_col = 0 )
df_del = pd.read_csv( os.path.join(path_to_data, 'CCE_deletions_to_model.csv') ,index_col = 0 )
df_exp = pd.read_csv( os.path.join(path_to_data, 'CCE_expressions_to_model.csv') ,index_col = 0 )



df_clinical_features = pd.read_csv( os.path.join(path_to_data, 'CCE_clinical_features.csv') )


df_exp = df_exp.apply(lambda x: np.log(x + 1))

df_exp_stand = (df_exp-df_exp.mean(axis=0))/df_exp.std(axis=0)#.apply(lambda x: np.log(x +1))
df_mut_scale = (df_mut-df_mut.min(axis=0))/df_mut.max(axis=0)
df_fus_scale = (df_fus-df_fus.min(axis=0))/df_fus.max(axis=0)


df_exp_stand.columns = [col + '_exp' for col in df_exp_stand.columns]
df_mut_scale.columns = [col + '_mut' for col in df_mut.columns]
df_amp.columns = [col + '_amp' for col in df_amp.columns]
df_del.columns = [col + '_del' for col in df_del.columns]
df_fus_scale.columns = [col + '_fus' for col in df_fus.columns]


data = pd.concat((df_exp_stand, df_mut_scale, df_amp, df_del, df_fus_scale), axis = 1)

# %% analyse mutations





# %%%

########################################################################################################################################################################





# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SAME MODEL MULTPLE BRCA SAMPLES
# %% load data for same sybtypes from one model

lrp_dict = {}

subtype = 'BRCA'

path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\results_BRCA_0\results'
lrp_files = []

for file in os.listdir(path)[:6]:
    if file.startswith("LRP"):
        lrp_files.append(file)

samples = [i.split('_')[2] for i in lrp_files]
data = data.reset_index()
data = data[data['Tumor_Sample_Barcode'].isin(samples)].reset_index(drop=True)

print(samples)

# %% load lrp data
n = len(lrp_files)  

network_data = pd.DataFrame()

for i in range(n):
    
            
    file_name = lrp_files[i]
    print(file_name)
    data_temp = pd.read_pickle(os.path.join(path , file_name), compression='infer', storage_options=None)

    data_temp = remove_same_source_target(data_temp)
        
    data_temp = data_temp.sort_values('LRP', ascending= False).reset_index(drop=True)
    data_temp = add_edge_colmn(data_temp)
    
    network_data = pd.concat((network_data , data_temp))
        
    lrp_dict[str(i)] = data_temp
    

# %% average LPRs


for i in range(n):
    print(i)
    edges = lrp_dict[str(i)]
    
    network_data = pd.concat((network_data , edges))
    
#network_data['LRP'] = np.abs(network_data['LRP'])
#network_data = network_data[network_data['source_gene'] != network_data['target_gene']]

average_network = network_data[['LRP', 'source_gene', 'target_gene']].groupby(['source_gene', 'target_gene']).mean().reset_index().sort_values('LRP', ascending= False).reset_index(drop=True)

#average_network = remove_exp_string(average_network)
average_network = add_edge_colmn(average_network)

average_network = average_network.sort_values('edge')

#average_network[average_network['edge'] == 'STAT1_exp - B2M_exp']
#average_network[average_network['edge'] == 'B2M_exp - STAT1_exp']

# %% plot LRPs
import kneed
from kneed import DataGenerator, KneeLocator
for i in range(n):
    
    lrp_dict[str(i)]['LRP'].reset_index(drop=True).plot()
    plt.show()
    x = lrp_dict[str(i)].reset_index()['index'].values
    y = 1- lrp_dict[str(i)]['LRP'].values
    kneedle = KneeLocator(x, y, S=10.0, curve="concave", direction="increasing")
    kneedle.plot_knee_normalized()
    plt.show()
    kneedle.plot_knee()
    plt.show()
    print(round(kneedle.elbow, 3))



# %% define TOP N pairs
# input: average network

top_n = 200
top_n_pairs = average_network.loc[:top_n, 'edge'].values

# Lets select topn muts, amps, dels, expr

def get_top_ns(edges, top_n_mut = 50, top_n_amp = 50, top_n_del = 50, top_n_exp = 50, top_n_fus = None):

    if top_n_mut is not None:
        top_n_mut_pairs = list(edges[edges['edge'].str.contains('mut')].reset_index(drop=True).loc[:top_n_mut-1, 'edge'].values)
    else:
        top_n_mut_pairs = []
    
    
    if top_n_amp is not None:
        top_n_amp_pairs = list(edges[edges['edge'].str.contains('amp')].reset_index(drop=True).loc[:top_n_amp-1, 'edge'].values)
    else:
        top_n_amp_pairs = []
    
    
    if top_n_del is not None:
        top_n_del_pairs = list(edges[edges['edge'].str.contains('del')].reset_index(drop=True).loc[:top_n_del-1, 'edge'].values)
    else:
        top_n_del_pairs = []
    
    if top_n_exp is not None:
        top_n_exp_pairs = list(edges[edges['edge'].str.contains('exp')].reset_index(drop=True).loc[:top_n_exp-1, 'edge'].values)
    else:
        top_n_exp_pairs = []
    
    if top_n_fus is not None:
        top_n_fus_pairs = list(edges[edges['edge'].str.contains('fus')].reset_index(drop=True).loc[:top_n_fus-1, 'edge'].values)
    else:
        top_n_fus_pairs = []
    
    
    top_n_pairs = list(set(top_n_mut_pairs + top_n_amp_pairs + top_n_del_pairs + top_n_exp_pairs + top_n_fus_pairs ))
    top_n_pairs.sort()
    
    print('Number of pairs: ', len(top_n_pairs))
    return top_n_pairs

top_n_pairs = get_top_ns(average_network, top_n_mut = 50, top_n_amp = 50, top_n_del = 50, top_n_exp = 50, top_n_fus = None)

top_n_mut_alt = 50
index_ = (average_network['edge'].str.contains('amp').values + average_network['edge'].str.contains('del').values) and (average_network['edge'].str.contains('mut'))
top_n_mut_alt_pairs = average_network[ ].reset_index(drop=True).loc[:top_n_alterations, 'edge'].values



top_n = 100
top_n_pairs = average_network[index_].reset_index(drop=True).loc[:top_n, 'edge'].values



gene = 'TP53'
top_n = 200
index_ = average_network['edge'].str.contains(gene)
top_n_pairs = average_network.loc[index_, 'edge'].values[:top_n]


# %% LRP selected

lrp_dict_selected ={}
for i in range(n):
    print(i)
    edges = lrp_dict[str(i)]
    edges = edges[edges['edge'].isin(top_n_pairs) ].sort_values('edge')
    lrp_dict_selected[str(i)] = edges


# %% 
cl_faet = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)][['bcr_patient_barcode','HER2', 'Estrogen_receptor', 'Progesterone_receptor']]
cl_faet =cl_faet.set_index('bcr_patient_barcode')

labels = cl_faet['HER2'].values
lut = {'Positive':'blue', 'Equivocal':'gray', 'Negative':'red', 'Indeterminate':'white'}
her2_colors = pd.DataFrame(labels)[0].map(lut)

#Create additional row_colors here
labels2 = cl_faet['Estrogen_receptor'].values
#lut2 = dict(zip(set(labels2), sns.hls_palette(len(set(labels2)), l=0.5, s=0.8)))
estr_colors = pd.DataFrame(labels2)[0].map(lut)

labels3 = cl_faet['Progesterone_receptor'].values
#lut2 = dict(zip(set(labels2), sns.hls_palette(len(set(labels2)), l=0.5, s=0.8)))
proge_colors = pd.DataFrame(labels3)[0].map(lut)

col_colors = [her2_colors, estr_colors, proge_colors]





# %%% clustermaps for top n LRPs
edges_matrix = []

for i in range(n):
    print(i)
    edges = lrp_dict_selected[str(i)]
    edges_matrix.append(edges['LRP'].values)


edges_matrix = pd.DataFrame(edges_matrix).T
edges_matrix.columns = lrp_files
edges_matrix['index'] = top_n_pairs
edges_matrix = edges_matrix.set_index('index')


sns.clustermap(edges_matrix, cmap ='jet', row_cluster = True,
               col_colors = col_colors, 
               yticklabels=False,
               method = 'ward',
               dendrogram_ratio=0.2,
               figsize=(10, 10))
plt.savefig(os.path.join(path_to_save , 'clustermap_LRP.png'), dpi = 300)


# %%% compare to Pearson correlations

# all features


to_corrs_all = data.loc[data['Tumor_Sample_Barcode'].isin(samples), : ]
to_corrs_all =  to_corrs_all.iloc[:, 2:]
select_no_zero = (to_corrs_all.sum() != 0).values

to_corrs_all = to_corrs_all.loc[:, select_no_zero]
corrs_all = to_corrs_all.corr()
r_min = 0.5

def get_colors_for_corrs(to_corrs):
    
    types_ = [x.split('_')[1] for x in to_corrs.columns]
    color_map = {'exp' : 'blue', 'mut':'red','amp':'yellow', 'del':'orange', 'fus':'green' }
    colors = pd.Series(types_).map(color_map).values

    return colors

colors = get_colors_for_corrs(to_corrs_all)
# heatmaps
#sns.heatmap(corrs_all)

# pearson clustermap
sns.clustermap(corrs_all, method = 'ward',row_colors=colors, col_colors=colors, yticklabels=False, xticklabels=False, cmap = 'coolwarm', vmin = -1, vmax = 1)


# spearman clustermap
corrs_all = to_corrs_all.corr(method = 'spearman')
sns.clustermap(corrs_all, method = 'ward', row_colors=colors, col_colors=colors, yticklabels=False, xticklabels=False, cmap = 'coolwarm', vmin = -1, vmax = 1)




# only top n features
features = list(set(pd.Series(top_n_pairs).str.split(' - ', expand = True).melt()['value'].to_list()))
features.sort()


samples
to_corrs = data.loc[data['Tumor_Sample_Barcode'].isin(samples),features]
select_no_zero = (to_corrs.sum() != 0).values

to_corrs = to_corrs.loc[:, select_no_zero]

corrs = to_corrs.corr()
r_min = 0.5

# heatmaps
sns.heatmap(corrs)
colors = get_colors_for_corrs(to_corrs)
sns.clustermap(corrs, row_colors=colors, col_colors=colors, cmap ='coolwarm', method = 'ward')

corrs = to_corrs.corr(method = 'spearman')
sns.heatmap(corrs, mask = corrs < r_min)
sns.clustermap(corrs, row_colors=colors, col_colors=colors, yticklabels=False, xticklabels=False, cmap ='coolwarm', method = 'ward')


# graph from correlations
corrs.melt()

def edges(matr):
    edge = {}
    for m in matr.columns:
        for n in matr.index:
            a,b = m,n 
            if a > b: #only add edge once
                x = matr.at[m, n]
                edge[m,n] = float("{0:.4f}".format(x))
    return edge

r_edges = pd.DataFrame.from_dict(edges(corrs), orient='index').reset_index()
r_edges['index'].str.split(', ')


r_edges[['source_gene','target_gene']] = pd.DataFrame(r_edges['index'].to_list(), columns=['source_gene','target_gene'])
r_edges = r_edges.rename(columns = {0:'r'})
r_edges = r_edges[r_edges['r'].abs() > r_min]




G_r = nx.from_pandas_edgelist(r_edges, source='source_gene', target='target_gene', edge_attr='r')
pos = nx.spring_layout(G_r)
widths = r_edges['r'].abs() / r_edges['r'].abs().max() 
edge_colors =cm.Reds((widths  - np.min(widths) )/ (np.max(widths) - np.min(widths)))
degrees = np.array(list(nx.degree_centrality(G_r).values())) 
degrees = degrees / np.max(degrees) * 500

nx.draw(G_r, with_labels=True, 
        #node_color=node_colors,
        width = widths,
        pos = pos, font_size = 6,
        #cmap = colors, 
        edge_color = edge_colors , 
        #ax=ax,  
        node_size = degrees)


# %% clustermaps for DATA

# data all
data_to_clustermap = data.set_index('Tumor_Sample_Barcode').iloc[:, 1:]


sns.clustermap(data_to_clustermap.T, cmap ='jet', row_cluster = True,
               col_colors = col_colors, 
               yticklabels=False,
               xticklabels=True,
               method = 'ward',
               dendrogram_ratio=0.2,
               figsize=(10, 10))

plt.savefig(os.path.join(path_to_save , 'clustermap_all_data.png'), dpi = 300)

# data selected top n
features = list(set(pd.Series(top_n_pairs).str.split(' - ', expand = True).melt()['value'].to_list()))
features.sort()
data_to_clustermap = data.loc[data['Tumor_Sample_Barcode'].isin(samples),:].set_index('Tumor_Sample_Barcode')
data_to_clustermap = data_to_clustermap.loc[:, features]
data_to_clustermap

sns.clustermap(data_to_clustermap.T, cmap ='jet', row_cluster = True,
               col_colors = col_colors, 
               yticklabels=False,
               xticklabels=True,
               method = 'ward',
               dendrogram_ratio=0.2,
               figsize=(10, 10))

plt.savefig(os.path.join(path_to_save , 'clustermap_topn_data.png'), dpi = 300)


# %% distribution by edge type in average network
for type_ in average_network['edge_type'].unique():
    temp  = average_network[average_network['edge_type'] == type_]
    
    fig, ax = plt.subplots()
    ax.hist(temp['LRP'])
    ax.set_title(type_)

sns.histplot(data=average_network, x="LRP",
             element="poly",
             hue="edge_type")



sns.boxplot(data=average_network, x="LRP", y ='edge_type', showfliers=False)
sns.violinplot(data=average_network, x="LRP", y ='edge_type', showfliers=False)

# show top 100 lprs for type of edge
temp = average_network.groupby('edge_type').head(20).reset_index()
g = sns.FacetGrid(temp, col="edge_type", col_wrap=3, sharey=False, sharex=False)
g.map(sns.barplot, "LRP", "edge" )


for i in range(n):
    print(i)
    edges = lrp_dict[str(i)]
    
    temp = edges[edges['edge'].isin(top_n_pairs)].groupby('edge_type').head(20).reset_index()
    
    g = sns.FacetGrid(temp, col="edge_type", col_wrap=3, sharey=False, sharex=True)
    g.map(sns.barplot, "LRP", "edge" )
    plt.show()

# %% Anomaly networks
lrp_diff_dict = {}
for i in range(n):
    print(i)
    edges = lrp_dict[str(i)]

    temp = edges.copy()
    temp['LRP'] = temp['LRP']  - average_network['LRP']

    lrp_diff_dict[str(i)] =  temp.sort_values('LRP')


# %% plot individual networks
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\networks_brca'

edges = average_network[average_network['edge'].isin(top_n_pairs)].reset_index(drop=True)

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
        elif layout== 'spectral':
            pos = nx.spectral_layout(G)
            
    #colors = get_node_colors(G)
    
    
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
            node_color=node_colors,
            width = widths,
            pos = pos, font_size = 6,
            #cmap = colors, 
            edge_color = edge_colors , 
            ax=ax,  
            #node_size = degrees
            node_size = node_size)
    ax.set_title(sample_id)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save , 'network_{}_{}_{}_{}.svg'.format(file, top_n, subtype, i)), format = 'svg')
    plt.savefig(os.path.join(path_to_save , 'network_{}_{}_{}_{}.png'.format(file, top_n, subtype, i)), dpi = 300)

    



color_values = data.loc[:, nodes].mean().values
plot_network_(edges, color_values, 50, 'BRCA', 'average', 'brca_shell', path_to_save, pos=pos, sample_id='Average')

for i in range(n):
    print(i)
    edges = lrp_dict_selected[str(i)]
    color_values =  data.loc[data['Tumor_Sample_Barcode'] == samples[i], nodes].values[0]
    her2 = df_clinical_features.loc[df_clinical_features['bcr_patient_barcode'] == samples[i], 'HER2'].values[0]
    plot_network_(edges, color_values, 50, 'BRCA', i, 'brca_shell_her2_'+her2, path_to_save, pos=pos, sample_id = samples[i]+  '\nHER2: ' +her2)

# %% plot network comunitites
for i in range(n):
    print(i)
    edges = lrp_dict[str(i)]
    top_n_pairs_i = get_top_ns(edges, top_n_mut = 500, top_n_amp = 500, top_n_del = 500, top_n_exp = 250, top_n_fus = 250)
    edges_temp = edges[edges['edge'].isin(top_n_pairs_i) ].sort_values('edge')
    
    
    
    G = nx.from_pandas_edgelist(edges_temp, source='source_gene', target='target_gene', edge_attr='LRP')
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(components[0]).copy()
    
    her2 = df_clinical_features.loc[df_clinical_features['bcr_patient_barcode'] == samples[i], 'HER2'].values[0]
    
    plot_igraph(G, i, samples[i], her2, len(top_n_pairs_i))
    
    '''degrees = np.array(list(nx.degree_centrality(G).values())) 
    degrees = degrees / np.max(degrees) * 1000
    widths = edges['LRP'] / edges['LRP'].max() * 10
    edge_colors =cm.Reds((widths  - np.min(widths) )/ (np.max(widths) - np.min(widths)))
    
    #node_colors = get_node_colors(G)
    
    communities = list(nx.community.louvain_communities(G))
    node_colors = create_community_node_colors(G, communities)

    
    pos = nx.spring_layout(G)
    fig,ax = plt.subplots(figsize=  (15,15))
    nx.draw(G, with_labels=False, 
            node_color=node_colors,
            width = 1,
            pos = pos, font_size = 6,
            #cmap = colors, 
            edge_color = edge_colors , 
            ax=ax,  
            node_size = degrees,
            #node_size = node_size
            )
    ax.set_title(samples[i])
    plt.tight_layout()'''
    

from igraph import *
import igraph as ig

from mpl_toolkits.mplot3d import Axes3D

pos = nx.get_node_attributes(G, 'pos')

# Create 3D positions
pos_3d = {i: (pos[i][0], pos[i][1], i) for i in pos.keys()}


# Create a figure and plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw the nodes
nx.draw_networkx_nodes(G, pos_3d, ax=ax)


for edge in G.edges:
    x = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
    y = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
    z = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
    ax.plot(x, y, z, color='k')

plt.show()


def plot_igraph(G, i, sample_id, her2, size):
        
    g = Graph.from_networkx(G)
    
    
    
    layout= g.layout("fr")
    comms = g.community_multilevel()
    cmap =  plt.cm.hsv
    visual_style = {}
    visual_style["vertex_size"] = [(np.log(x)+1)*5 for x in g.degree()]
    visual_style["vertex_color"] =   [list(cmap(np.linspace(0, 1, np.size(np.unique(comms.membership))))[x]) for x in comms.membership]
    visual_style["vertex_label"] = G.nodes
    widths = (edges['LRP'] / edges['LRP'].max() * 10).values
    visual_style["edge_width"] = 1
    #visual_style["edge_color"] = width_labels
    #visual_style["labels"] = None
    visual_style["layout"] = layout
    visual_style["bbox"] = (1000, 1000)
    visual_style["margin"] = 40
    visual_style["vertex_label_size"] = 1  # Increase the distance of labels from the vertices
    #visual_style["vertex_label_angle"] = 30  # Rotate the labels by a specific angle
    visual_style["vertex_label"] = [""] * len(g.vs)
    
    #pal = ig.drawing.colors.ClusterColoringPalette(len(g.community_infomap()))
    #g.vs['color'] = pal.get_many(g.community_infomap().membership)
    
    plot1 = ig.plot(g, **visual_style, mark_groups = True)
    plot1.save(os.path.join(path_to_save , 'network_brca_{}_igraph_graph_{}_{}_{}.png'.format(i,sample_id,her2,size)))
    
    #comms.layout("fr")
    
    #plot2 = ig.plot(comms, **visual_style, mark_groups = True)
    #plot2.save(os.path.join(path_to_save , 'network_brca_{}_igraph_comm_{}_{}.png'.format(i,sample_id,her2)))



%matplotlib qt

# function to create node colour list
def create_community_node_colors(graph, communities):
    number_of_colors = len(communities[0])
    cmap = plt.cm.hsv
    colors = cmap(np.linspace(0, 1, number_of_colors))
    node_colors = []
    for node in graph:
        current_community_index = 0
        for community in communities:
            if node in community:
                node_colors.append(colors[current_community_index])
                break
            current_community_index += 1
    return node_colors


modularity = round(nx.community.modularity(graph, communities), 6)
    
    

plot_network_com(edges, color_values, top_n, subtype, i, file, path_to_save, node_size=100, layout=None, pos = None, sample_id=''):
    
    
def plot_network_com(edges, color_values, top_n, subtype, i, file, path_to_save, node_size=100, layout=None, pos = None, sample_id=''):
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
        elif layout== 'spectral':
            pos = nx.spectral_layout(G)
            
    #colors = get_node_colors(G)
    
    
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
            node_color=node_colors,
            width = widths,
            pos = pos, font_size = 6,
            #cmap = colors, 
            edge_color = edge_colors , 
            ax=ax,  
            #node_size = degrees
            node_size = node_size)
    ax.set_title(sample_id)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save , 'network_{}_{}_{}_{}.svg'.format(file, top_n, subtype, i)), format = 'svg')
    plt.savefig(os.path.join(path_to_save , 'network_{}_{}_{}_{}.png'.format(file, top_n, subtype, i)), dpi = 300)

    




# %% similarity of vectors heatmaps
from scipy.spatial.distance import pdist, jaccard

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial import distance


cos_distances = pd.DataFrame(np.ones((n,n))*0)
euc_distances = pd.DataFrame(np.ones((n,n))*0)


    
for i,j in list(itertools.combinations(range(size), 2)):
    
    
    #jaccard_ = distance.jaccard(heatmaps[i].values.ravel()*0, heatmaps[j].values.ravel()*0)  
    #distances.loc[i,j] = jaccard_
    
    cos_distances.loc[i,j] = cosine_similarity(lrp_dict_selected[str(i)]['LRP'].values.reshape(1, -1), lrp_dict_selected[str(j)]['LRP'].values.reshape(1, -1))  
    euc_distances.loc[i,j] = euclidean_distances(lrp_dict_selected[str(i)]['LRP'].values.reshape(1, -1), lrp_dict_selected[str(j)]['LRP'].values.reshape(1, -1))  

    
sns.heatmap(cos_distances + cos_distances.T + np.eye(n), cmap ='Greens', annot=False, vmax = 1, vmin = 0.6)
plt.title('Cosinus distance')
plt.show()
sns.heatmap(euc_distances, cmap ='Reds', annot=False,mask = euc_distances==0, vmin = 0.0)
plt.title('Euclidean distance')


sns.clustermap(cos_distances + cos_distances.T + np.eye(n), cmap ='Greens', annot=False, vmax = 1, method = 'ward')

    
# %% igraph

from igraph import Graph
import igraph as ig
G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
g = Graph.from_networkx(G)


layout = g.layout(layout='auto')
layout= g.layout("kk_3d")

visual_style = {}
visual_style["vertex_size"] = 20
visual_style["vertex_color"] =   get_node_colors(G)
visual_style["vertex_label"] = G.nodes
widths = (edges['LRP'] / edges['LRP'].max() * 10).values
visual_style["edge_width"] = widths
visual_style["edge_color"] = width_labels
visual_style["layout"] = layout
visual_style["bbox"] = (1000, 1000)
visual_style["margin"] = 20
visual_style["vertex_label_dist"] = 1  # Increase the distance of labels from the vertices
visual_style["vertex_label_angle"] = 30  # Rotate the labels by a specific angle

ig.plot(g, **visual_style)

import numpy as np

# Here, widths is your data
widths = widths #/ np.max(widths)
normalized_widths = (widths  - np.min(widths)) / (np.max(widths) - np.min(widths))

# Define color labels
color_labels = ['navy', 'blue', 'dodgerblue', 'deepskyblue', 'turquoise', 'limegreen', 'yellow', 'gold', 'tomato', 'red']
cmap = plt.cm.get_cmap('jet', 10)
# Convert normalized widths to labels
width_labels = [color_labels[int(width * 10)] if width < 1.0 else color_labels[9] for width in normalized_widths]

























# %% get samples

path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_model'

df_clinical_features = pd.read_csv( os.path.join(path_to_data, 'CCE_clinical_features.csv') )


subtypes = ['BRCA', 'LGG', 'UCEC', 'LUAD', 'HNSC', 'PRAD', 'LUSC']
subtypes = ['BRCA']


samples = df_clinical_features.groupby('acronym').head(1)

n_samples = 20
samples = df_clinical_features.groupby('acronym').head(n_samples)

samples = samples[samples['acronym'].isin(subtypes)].reset_index(drop = True)


# %% load data for many models
paths = []
lrp_dict = {}

subtype = 'BRCA'

for i in range(5):
    path = (r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\results_{}\results'.format(i))

    
    lrp_files = []
    
    for file in os.listdir(path):
        if file.startswith("LRP"):
            lrp_files.append(file)
            
    paths.append(path)
    
    sample_name = samples.loc[samples['acronym'] == subtype, 'bcr_patient_barcode'].values[0]
    file_name = [s for s in lrp_files if sample_name in s][0]

    data_temp = pd.read_pickle(os.path.join(path , file_name), compression='infer', storage_options=None)

    data_temp = remove_same_source_target(data_temp)
        
    data_temp = data_temp.sort_values('LRP', ascending= False).reset_index(drop=True)
    
    lrp_dict[str(i)] = data_temp
    



# %% clustermaps
top_n = 200

for i in range(5):
    
    data_temp = lrp_dict[str(i)]
        
    edges = data_temp.sort_values(by='LRP', ascending=False).iloc[:top_n,:]
    
    to_cluster = edges.pivot_table(values = 'LRP', index = 'source_gene', columns = 'target_gene').fillna(0)
    
    sns.clustermap(to_cluster, method = 'ward', cmap = 'Reds')
    plt.show()

# %% similarity heatmaps and scores
top_n = 100



genes = get_genes_in_all_i(lrp_dict, top_n)


heatmaps = {}

for i in range(5):

    edges = lrp_dict[str(i)].iloc[:top_n,:]

    template_ = get_pivoted_heatmap(edges,genes)
    heatmaps[i] = template_ 
    
    mask = template_==0
    fig,ax=plt.subplots(figsize = (12,12))
    sns.heatmap(template_, mask = mask, cmap = 'jet',  yticklabels=True, xticklabels=True,ax=ax)
    plt.show()

import itertools

    
from scipy.spatial.distance import pdist, jaccard

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial import distance
cos_distances = pd.DataFrame(np.ones((5,5))*0)
euc_distances = pd.DataFrame(np.ones((5,5))*0)
for i,j in list(itertools.combinations(range(5), 2)):
    
    
    #jaccard_ = distance.jaccard(heatmaps[i].values.ravel()*0, heatmaps[j].values.ravel()*0)  
    #distances.loc[i,j] = jaccard_
    
    cos_distances.loc[i,j] = cosine_similarity(heatmaps[i].values.reshape(1, -1), heatmaps[j].values.reshape(1, -1))  
    euc_distances.loc[i,j] = euclidean_distances(heatmaps[i].values.reshape(1, -1), heatmaps[j].values.reshape(1, -1))  

    
sns.heatmap(cos_distances, cmap ='Greens', annot=True, mask = cos_distances==0, vmax = 1, vmin = 0.6)
plt.title('Cosinus distance')
plt.show()
sns.heatmap(euc_distances, cmap ='Reds', annot=True,mask = euc_distances==0, vmin = 0.0)
plt.title('Euclidean distance')
        
# %% plot individual networks
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\networks_20230620'
top_n = 50
for i in range(5):
    
    df_temp = lrp_dict[str(i)]
    df_temp = df_temp.iloc[:top_n,:]
        
    edges  = remove_exp_string(df_temp )
    
    plot_network_(edges, top_n, subtype, i, file, path_to_save)
    
  
# %% plot average network

heatmap_mean = pd.DataFrame()

for i in range(5):
    
    temp= heatmaps[i]
    #temp['i'] = i
    heatmap_mean = pd.concat((heatmap_mean , temp.reset_index()))
    
heatmap_mean = heatmap_mean.groupby('index').sum()
heatmap_mean[heatmap_mean==0] = np.nan
edges_mean = heatmap_mean.reset_index().melt(id_vars = 'index', var_name = 'target_gene', value_name = 'LRP').dropna().reset_index(drop=True)
edges_mean = edges_mean.rename(columns = {'index':'source_gene'})


plot_network_(edges_mean, top_n, subtype, 'mean', file, path_to_save,)



# %% plot individual pathway
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\networks_brca'


all_unique_genes, all_unique_genes_names = get_all_unique_genes(lrp_dict[str(0)])
genes = list(pd.read_csv('genes_ESTROGEN_RESPONSE_EARLY.txt', sep = '\t',header=None).T[0].values)  # defined set of genes


# which genes do not match
diff = set.difference(set(genes), set(all_unique_genes))
genes = list(set.intersection(set(genes), set(all_unique_genes)))
genes.sort()

top_n = 1000
genes = add_suffixes_to_genes(genes)

heatmaps = {}
for i in range(len(lrp_dict.keys())):
    
    edges = lrp_dict[str(i)]
    edges = edges[edges['source_gene'].isin(genes) & edges['target_gene'].isin(genes) ]#.iloc[:top_n, :]
   
    #edges = remove_exp_string(edges)
    
    
    if i == 0:
        G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
        G1 = G.copy()
        pos = nx.spring_layout(G1)
        #pos = {(x,y):(y,-x) for x,y in G.nodes()}
    else:
        G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
        
    #plot_network_(G, edges, top_n, subtype, i, file, path_to_save, pos)

    
    template_ = get_pivoted_heatmap(edges, genes)
    #template_ = template_[template_.sum(axis=0)>0]
    template_ = template_.iloc[(template_.sum(axis=1)>0).values, (template_.sum(axis=0)>0).values]
    heatmaps[i] = template_ 

    mask = template_< 0.005
    mask = template_ == 0
    fig,ax=plt.subplots(figsize = (12,12))
    sns.heatmap(template_, mask = mask, cmap = 'Reds',  yticklabels=True, xticklabels=True,ax=ax)
    plt.show()





# %%%%%%%%%%%%%%%%%%%%%%%%%%%%









# %% 3D plotly graph
import igraph as ig
import plotly.graph_objects as go
import plotly.io as io

io.renderers.default='browser'

# Create a sample graph

# Create a 3D layout for the graph
layout = g.layout("kk3d")
edge_colors =cm.Reds((widths  - np.min(widths) )/ (np.max(widths) - np.min(widths)))

# Get the node coordinates from the layout
node_x = [coord[0] for coord in layout]
node_y = [coord[1] for coord in layout]
node_z = [coord[2] for coord in layout]

# Create a Plotly scatter3d trace for nodes
node_trace = go.Scatter3d(
    x=node_x,
    y=node_y,
    z=node_z,
    mode="markers",
    marker=dict(
        size=10,
        color='blue',
        line=dict(color="black", width=0.5),
        opacity=1,
    ),
)

# Create a Plotly scatter3d trace for edges
edge_trace = go.Scatter3d(
    x=node_x + [None] * len(g.es),
    y=node_y + [None] * len(g.es),
    z=node_z + [None] * len(g.es),
    mode="lines",
    line=dict(color="black", width=1),
)

# Create the Plotly data list
data = [node_trace, edge_trace]

# Create the Plotly layout
layout = go.Layout(
    showlegend=False,
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    ),
)

# Create the Plotly figure
fig = go.Figure(data=data, layout=layout)

# Show the figure
fig.show()


















# %% plot individual pathway
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\networks_brca'


all_unique_genes, all_unique_genes_names = get_all_unique_genes(lrp_dict[str(0)])
genes = list(pd.read_csv('genes_ESTROGEN_RESPONSE_EARLY.txt', sep = '\t',header=None).T[0].values)  # defined set of genes


# which genes do not match
diff = set.difference(set(genes), set(all_unique_genes))
genes = list(set.intersection(set(genes), set(all_unique_genes)))
genes.sort()

top_n = 1000
genes = add_suffixes_to_genes(genes)

heatmaps = {}
edges_matrix = []
for i in range(n):
    
    edges = lrp_dict[str(i)]
    edges = edges[edges['source_gene'].isin(genes) & edges['target_gene'].isin(genes) ]#.iloc[:top_n, :]
    edges_matrix.append(edges['LRP'].values)
    #edges = remove_exp_string(edges)
    
    
    #if i == 0:
      #  G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
     #   G1 = G.copy()
      #  pos = nx.spring_layout(G1)
        #pos = {(x,y):(y,-x) for x,y in G.nodes()}
    #else:
     #   G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
        
    #plot_network_(G, edges, top_n, subtype, i, file, path_to_save, pos)

    
    template_ = get_pivoted_heatmap(edges, genes)
    #template_ = template_[template_.sum(axis=0)>0]
    template_ = template_.iloc[(template_.sum(axis=1)>0).values, (template_.sum(axis=0)>0).values]
    heatmaps[i] = template_ 

    mask = template_< 0.005
    mask = template_ == 0
    fig,ax=plt.subplots(figsize = (12,12))
    sns.heatmap(template_, mask = mask, cmap = 'Reds',  yticklabels=True, xticklabels=True,ax=ax)
    plt.show()


    
from scipy.spatial.distance import pdist, jaccard

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial import distance
cos_distances = pd.DataFrame(np.ones((n,n))*0)
euc_distances = pd.DataFrame(np.ones((n,n))*0)
for i,j in list(itertools.combinations(range(n), 2)):
    
    
    #jaccard_ = distance.jaccard(heatmaps[i].values.ravel()*0, heatmaps[j].values.ravel()*0)  
    #distances.loc[i,j] = jaccard_
    
    cos_distances.loc[i,j] = cosine_similarity(heatmaps[i].values.reshape(1, -1), heatmaps[j].values.reshape(1, -1))  
    euc_distances.loc[i,j] = euclidean_distances(heatmaps[i].values.reshape(1, -1), heatmaps[j].values.reshape(1, -1))  

fig, ax = plt.subplots(figsize=(12,10)) 
sns.heatmap(cos_distances, cmap ='Greens', annot=True, mask = cos_distances==0, ax=ax,
            annot_kws={"fontsize":8})#, vmax = 1, vmin = 0.6)
plt.title('Cosinus distance')
plt.show()

fig, ax = plt.subplots(figsize=(12,10)) 
sns.heatmap(euc_distances, cmap ='Reds', annot=True, mask = cos_distances==0, ax=ax,
            annot_kws={"fontsize":8})#, vmax = 1, vmin = 0.6)
plt.title('Euclidean distance')





fig, ax = plt.subplots(figsize=(12,10)) 
sns.clustermap(cos_distances + cos_distances.T, cmap ='Greens', annot=True, #mask = cos_distances==0,# ax=ax,
            annot_kws={"fontsize":8})#, vmax = 1, vmin = 0.6)
plt.title('Cosinus distance')
plt.show()

edges_matrix = pd.DataFrame(edges_matrix).T
edges_matrix.columns = lrp_files



sns.clustermap(edges_matrix.T, cmap ='jet', method = 'ward')

















# %% hallmark genes


hall = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\PNET\pathways\MsigDB\h.all.v6.1.symbols.gmt', sep = '\t', header=None)
hall = hall.iloc[:, 2:].reset_index().melt( id_vars = 'index').dropna()

hall_genes = list(hall['value'].unique())
hall_genes .sort()

# %% 
top_n = 200
for i in range(5):
    
    df_temp = lrp_dict[str(i)]
    df_temp = df_temp.iloc[:top_n,:]
    
    df_temp['edge'] = df_temp['source_gene'] + ' - ' + df_temp['target_gene']








import igraph as ig
import cairo
random.seed(0)
g = ig.Graph.GRG(50, 0.15)
components = g.connected_components(mode='weak')

fig, ax = plt.subplots()
ig.plot(
    components,
    target=ax,
    palette=ig.RainbowPalette(),
    vertex_size=0.07,
    vertex_color=list(map(int, ig.rescale(components.membership, (0, 200), clamp=True))),
    edge_width=0.7
)
plt.show()


from igraph import Graph
G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')

g = Graph.DataFrame(edges[['source_gene','target_gene']], directed=False)
g = Graph.from_networkx(G)


layout = g.layout(layout='large')
fig, ax = plt.subplots()
ig.plot(g, target=ax, layout=layout)


g = ig.Graph(91, list(G.edges))




