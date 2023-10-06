# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:31:27 2023

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
# %% functions 

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

# %% load data
# %%%  get input data

#path_to_data = '/home/owysocki/Documents/KI_dataset/data_to_model'
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'
#path_to_data = 'KI_dataset/data_to_model'


data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(path_to_data)


# %%% clinical data

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

samples_estrogen_receptor_pos = df_clinical_features.loc[ df_clinical_features['Estrogen_receptor']=='Positive', 'bcr_patient_barcode'].to_list()
samples_estrogen_receptor_neg = df_clinical_features.loc[ df_clinical_features['Estrogen_receptor']=='Negative', 'bcr_patient_barcode'].to_list()

samples_her2_pos = df_clinical_features.loc[ df_clinical_features['HER2']=='Positive', 'bcr_patient_barcode'].to_list()
samples_her2_neg = df_clinical_features.loc[ df_clinical_features['HER2']=='Negative', 'bcr_patient_barcode'].to_list()

samples_progesterone_receptor_pos = df_clinical_features.loc[ df_clinical_features['Progesterone_receptor']=='Positive', 'bcr_patient_barcode'].to_list()
samples_progesterone_receptor_neg = df_clinical_features.loc[ df_clinical_features['Progesterone_receptor']=='Negative', 'bcr_patient_barcode'].to_list()

samples_groups = {}
samples_groups['her2'] = {'her2_pos':samples_her2_pos, 'her2_neg':samples_her2_neg}
samples_groups['progesterone_receptor'] = {'progesterone_receptor_pos':samples_progesterone_receptor_pos, 'progesterone_receptor_neg':samples_progesterone_receptor_neg}
samples_groups['estrogen_receptor'] = {'estrogen_receptor_pos':samples_estrogen_receptor_pos, 'estrogen_receptor_neg':samples_estrogen_receptor_neg}


    

# %%% define LRP files

lrp_dict = {}

path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\results'
lrp_files = []

for file in os.listdir(path):
    if file.startswith("LRP"):
        lrp_files.append(file)

samples = [i.split('_')[2] for i in lrp_files]
data_to_model = data_to_model.reset_index()
data_to_model = data_to_model[data_to_model['Tumor_Sample_Barcode'].isin(samples)].reset_index(drop=True)

print(samples)


# %%% load LRP data
n = len(lrp_files)  

#network_data = pd.DataFrame()

for i in range(n):
    
            
    file_name = lrp_files[i]
    sample_name = file_name.split('_')[2]
    print(i, file_name)
    data_temp = pd.read_pickle(os.path.join(path , file_name), compression='infer', storage_options=None)

    data_temp = remove_same_source_target(data_temp)
        
    #data_temp = data_temp.sort_values('LRP', ascending= False).reset_index(drop=True)
    #data_temp = add_edge_colmn(data_temp)
    
    #network_data = pd.concat((network_data , data_temp))
        
    lrp_dict[sample_name] = data_temp


# %% gene from pathways


genes_pathways = pd.read_csv(os.path.join(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\PATHWAYS', 'genes_pathways_pancanatlas_matched_cce.csv'))
genes_pathways_set = set(genes_pathways['cce_match'])

genes_pathways_dict = {}

for pathway in genes_pathways['Pathway'].unique():
    
    genes_pathways_dict[pathway] = genes_pathways.loc[genes_pathways['Pathway'] == pathway, 'Gene'].to_list()





# %% networks for patient groups and pathway genes

path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\networks'
group = 'progesterone_receptor'
group = 'estrogen_receptor'
group = 'her2'
#pathway = 'PI3K'

for pathway in genes_pathways['Pathway'].unique():
    print(pathway)
    genes = genes_pathways_dict[pathway]
    samples_pos = samples_groups[group][group + '_pos']
    samples_neg =  samples_groups[group][group + '_neg']
    
    network_pos = pd.DataFrame()
    i=0
    for sample_name in samples_pos:
        print(i, pathway, sample_name, 'pos')
        temp = lrp_dict[sample_name].copy()
        temp = temp[temp['source_gene'].str.split('_',expand=True)[0].isin(genes) & temp['target_gene'].str.split('_',expand=True)[0].isin(genes)]
        
        network_pos = pd.concat((network_pos, temp))
        i+=1
    network_pos = add_edge_colmn(network_pos)
    network_pos = network_pos.groupby(by = ['edge','source_gene', 'target_gene','edge_type'],as_index=False).mean()
    
    j=0
    network_neg = pd.DataFrame()
    for sample_name in samples_neg:
        print(j, pathway, sample_name, 'neg')
        temp = lrp_dict[sample_name].copy()
        temp = temp[temp['source_gene'].str.split('_',expand=True)[0].isin(genes) & temp['target_gene'].str.split('_',expand=True)[0].isin(genes)]
        network_neg = pd.concat((network_neg, temp))
        j+=1
    network_neg = add_edge_colmn(network_neg)
    network_neg = network_neg.groupby(by = ['edge','source_gene', 'target_gene','edge_type'],as_index=False).mean()
    
    
    network_pos.to_excel(os.path.join(path_to_save, 'network_{}_{}_pos.xlsx'.format(group, pathway)))
    network_neg.to_excel(os.path.join(path_to_save, 'network_{}_{}_neg.xlsx'.format(group, pathway)))



# %% networks for patient groups all genes (from pathways)

path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\networks'
group = 'progesterone_receptor'
group = 'estrogen_receptor'
group = 'her2'
#pathway = 'PI3K'


genes = list(genes_pathways_set)
samples_pos = samples_groups[group][group + '_pos']
samples_neg =  samples_groups[group][group + '_neg']

network_pos = pd.DataFrame()
i=0
for sample_name in samples_pos:
    print(i, sample_name, 'pos')
    temp = lrp_dict[sample_name].copy()
    temp = temp[temp['source_gene'].str.split('_',expand=True)[0].isin(genes) & temp['target_gene'].str.split('_',expand=True)[0].isin(genes)]
    
    network_pos = pd.concat((network_pos, temp))
    i+=1
network_pos = add_edge_colmn(network_pos)
network_pos = network_pos.groupby(by = ['edge','source_gene', 'target_gene','edge_type'],as_index=False).mean()

j=0
network_neg = pd.DataFrame()
for sample_name in samples_neg:
    print(j, sample_name, 'neg')
    temp = lrp_dict[sample_name].copy()
    temp = temp[temp['source_gene'].str.split('_',expand=True)[0].isin(genes) & temp['target_gene'].str.split('_',expand=True)[0].isin(genes)]
    network_neg = pd.concat((network_neg, temp))
    j+=1
network_neg = add_edge_colmn(network_neg)
network_neg = network_neg.groupby(by = ['edge','source_gene', 'target_gene','edge_type'],as_index=False).mean()


network_pos.to_excel(os.path.join(path_to_save, 'network_{}_allgenes_pos.xlsx'.format(group)))
network_neg.to_excel(os.path.join(path_to_save, 'network_{}_allgenes_neg.xlsx'.format(group)))

# %% BASELINE CLUSTERMAP
%matplotlib inline


# define how much % LRP remove
fraction = 0.75

data_temp = network_pos

# Compute the histogram values
values, bin_edges = np.histogram(data_temp['LRP'], bins=100, density=True)
# Calculate the cumulative density
cumulative_density = np.cumsum(values) * (bin_edges[1] - bin_edges[0])
# Find the index of the bin where the cumulative density is closest to 0.8
index_closest = np.abs(cumulative_density - fraction).argmin()
# The LRP value where the cumulative density is closest to 0.8 would be the right edge of the bin
lrp_value_closest = bin_edges[index_closest + 1]
fig,ax = plt.subplots()
sns.histplot(data=data_temp, x="LRP", element="step", fill=False,
    cumulative=True, stat="density", common_norm=False,ax=ax)
ax.axvline(lrp_value_closest)
ax.grid()

edges_above_threshold = data_temp.loc[data_temp['LRP'] > lrp_value_closest, 'edge' ].to_list()
data_temp

# %%% get lrp_dict_filtered all genes
genes = list(genes_pathways_set)

lrp_dict_filtered = {}
for index, (sample_name, data) in enumerate(lrp_dict.items()):
    print(index, sample_name)
    temp = data.copy()
    
    temp = temp[temp['LRP'] > lrp_value_closest]
    print(temp.shape[0])
    temp = temp[temp['source_gene'].str.split('_',expand=True)[0].isin(genes) & temp['target_gene'].str.split('_',expand=True)[0].isin(genes)]

    lrp_dict_filtered[sample_name] = temp.reset_index(drop=True)
    
    

# %%% get lrp_dict_filtered single pathway
def optimized_filtering(temp, genes):
    temp['source_gene_prefix'] = temp['source_gene'].str.split('_').str[0]
    temp['target_gene_prefix'] = temp['target_gene'].str.split('_').str[0]
    return temp[temp['source_gene_prefix'].isin(genes) & temp['target_gene_prefix'].isin(genes)]

pathway = 'PI3K'
pathway = 'TP53'
print(pathway)
genes = genes_pathways_dict[pathway]

lrp_dict_filtered = {}
lrp_dict_filtered[pathway] = {}
for index, (sample_name, data) in enumerate(lrp_dict.items()):
    print(index, sample_name)
    temp = data.copy()

    #temp = temp[temp['LRP'] > lrp_value_closest]
    #temp = temp[temp['source_gene_prefix'].isin(genes) & temp['target_gene_prefix'].isin(genes)]
    temp = optimized_filtering(temp, genes)
    #temp = temp[temp['source_gene'].str.split('_',expand=True)[0].isin(genes) & temp['target_gene'].str.split('_',expand=True)[0].isin(genes)]
    print(temp.shape[0])
    lrp_dict_filtered[pathway][sample_name] = temp.reset_index(drop=True)
    
    
    

# %%%% clustermap functions

def get_lrp_dict_filtered_pd(lrp_dict_filtered, pathway = 'PI3K'):
    
    lrp_dict_filtered_pd = pd.DataFrame()#lrp_dict_filtered)
    n = len(lrp_dict_filtered[pathway].keys())
    
    for index, (sample_name, data) in enumerate(lrp_dict_filtered[pathway].items()):
        print(index+1,'/',n, sample_name)
        
        data_temp = add_edge_colmn(data[['LRP', 'source_gene', 'target_gene']].copy())
        data_temp['sample'] = sample_name
    
        lrp_dict_filtered_pd = pd.concat((lrp_dict_filtered_pd, data_temp))
        
    
    lrp_dict_filtered_pd = lrp_dict_filtered_pd.reset_index(drop=True)
    
    return lrp_dict_filtered_pd

# lrp_dict_filtered_pd_sort_index = lrp_dict_filtered_pd.groupby('edge').mean().sort_values(['LRP'], ascending = False).index

# sns.pointplot(lrp_dict_filtered_pd)

# sns.pointplot(data  = lrp_dict_filtered_pd, x='edge', y='LRP', order = lrp_dict_filtered_pd_sort_index, estimator='mean', join = False, errorbar = 'ci')


def get_column_colors_from_clinical_df(df_clinical_features, df_to_clustermap):
    
    df_clinical_features_ = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(lrp_dict_filtered_pd_pivot.columns)]
    df_clinical_features_ = df_clinical_features_.set_index(df_to_clustermap.columns)
    
    
    return df_clinical_features_
    
def map_subtypes_to_col_color(df_clinical_features_):
        
    color_map = {
        "Negative": "red",
        "Positive": "blue",
        "Equivocal": "yellow",
        'Indeterminate':'gray',
        '[Not Evaluated]':'white',
        '[Not Available]':'white',}    
        
    column_colors_her2 = df_clinical_features_['HER2'].map(color_map)
    column_colors_er = df_clinical_features_['Estrogen_receptor'].map(color_map)
    column_colors_pro = df_clinical_features_['Progesterone_receptor'].map(color_map)

    column_colors = [column_colors_her2, column_colors_er, column_colors_pro]
    return column_colors


import matplotlib.patches as mpatches

def create_legend(color_map):
    patches = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
    plt.legend(handles=patches)

# Create a dummy figure just to show the legend
plt.figure(figsize=(10, 6))
create_legend(color_map)
plt.axis('off')
plt.show()

# %%%% Clustermap

pathway = 'PI3K'
pathway = 'TP53'
# ['Cell Cycle', 'HIPPO', 'MYC', 'NOTCH', 'NRF2', 'PI3K', 'RTK-RAS', 'TGF-Beta', 'TP53', 'WNT']

lrp_dict_filtered_pd = get_lrp_dict_filtered_pd(lrp_dict_filtered, pathway = pathway)
lrp_dict_filtered_pd_pivot = pd.pivot_table(lrp_dict_filtered_pd, index = 'edge', columns = 'sample', values = 'LRP')
lrp_dict_filtered_pd_pivot[lrp_dict_filtered_pd_pivot < 0.002] = 0
df_clinical_features_ = get_column_colors_from_clinical_df(df_clinical_features, lrp_dict_filtered_pd_pivot)
column_colors = map_subtypes_to_col_color(df_clinical_features_)

# %%%% plot clustermap
sns.clustermap(lrp_dict_filtered_pd_pivot, mask  = lrp_dict_filtered_pd_pivot < 0.001, cmap = 'jet', col_colors = column_colors,
               method = 'ward', yticklabels = False, xticklabels = False, vmax = 0.008, vmin = 0)



# %%% Correlation network
samples_selection = 'all_BRCA'
samples_selection = 'her2_pos'
samples_selection = 'her2_neg'
samples_to_univariable = samples_her2_pos
samples_to_univariable = samples_her2_neg
samples_to_univariable = samples


# %%%% functions


def bootstrap_iteration(group0, group1, M):
    boot_group0 = resample(group0, replace=True, n_samples=M)
    stat, p = mannwhitneyu(boot_group0, group1)
    cles = stat / (M * M)
    return p, cles

def get_mannwhitney_bootstrap(df_cat, df_num, cat_col, num_col, iters=100):
    M = (df_cat[cat_col] > 0).sum()  # Sample size for each bootstrap
    p_values = []
    cles_list = []

    x = df_num[num_col]
    y = df_cat[cat_col].astype('category').cat.codes

    group0 = x[y == 0]
    group1 = x[y > 0]
    
    for i in range(iters):
        try:
            p, cles = bootstrap_iteration(group0, group1, M)
        except:
            p, cles = 1, 0.5
        p_values.append(p)
        cles_list.append(cles)
    
    p_value_mean = np.mean(p_values)
    cles_mean = np.mean(cles_list)
    
    return p_value_mean, cles_mean

from scipy.stats import mannwhitneyu
from sklearn.utils import resample



def get_mannwhitneyu_matrix(df_cat, df_num, iters=10):
    
    cat_cols = df_cat.columns
    num_cols = df_num.columns
    # Initialize an empty DataFrame to store correlation values
    cles_matrix = pd.DataFrame(index=cat_cols, columns=num_cols)
    pval_matrix = pd.DataFrame(index=cat_cols, columns=num_cols)
    n = len(num_cols)
    # Compute biserial correlation for each pair of categorical and numerical columns
    for cat_col in cat_cols:
        
        for i, num_col in enumerate(num_cols):
            print(cat_col, num_col, num_col,i, '/',n)
            #u, pval = pointbiserialr(df_cat[cat_col].astype('category').cat.codes, df_num[num_col])
            pval, cles = get_mannwhitney_bootstrap(df_cat, df_num, cat_col , num_col, iters = iters)
            pval_matrix.loc[cat_col, num_col] = pval
            cles_matrix.loc[cat_col, num_col] = cles
            
    
    # Convert to float for plotting
    pval_matrix = pval_matrix.astype(float)
    cles_matrix = cles_matrix.astype(float)
    
    pval_matrix = pval_matrix.reset_index().melt(id_vars = 'index')
    pval_matrix=pval_matrix.rename(columns = {'index':'source_gene','variable':'target_gene','value':'p-val'})
    pval_matrix = add_edge_colmn(pval_matrix)

    cles_matrix = cles_matrix.reset_index().melt(id_vars = 'index')
    cles_matrix=cles_matrix.rename(columns = {'index':'source_gene','variable':'target_gene','value':'CLES'})
    cles_matrix = add_edge_colmn(cles_matrix)

    mwu_stats = pval_matrix.merge(cles_matrix)
    
    
    return pval_matrix, cles_matrix , mwu_stats


# %%%% Spearman

data_temp = data_to_model[data_to_model.index.isin(samples_to_univariable)]

#num_cols = [item for item in data_temp.columns if '_exp' in item or '_mut' in item]
num_cols = [item for item in data_temp.columns if '_exp' in item]
num_cols = [item for item in num_cols if any(gene in item for gene in genes)]


import pingouin as pg
corrs_spearman_exp = pg.pairwise_corr(data_temp[exp_cols], columns=exp_cols, method='spearman')
corrs_spearman_exp = corrs_spearman_exp.sort_values('p-unc').reset_index(drop=True)
corrs_spearman_exp = corrs_spearman_exp.rename(columns = {'X':'source_gene', 'Y':'target_gene', 'p-unc':'p-val'})

corrs_spearman_exp = corrs_spearman_exp[['source_gene', 'target_gene',  'r',  'p-val', 'power']]
corrs_spearman_exp = add_edge_colmn(corrs_spearman_exp)
corrs_spearman_exp['test'] = 'spearman'
# %%%% Mann Whitney


cat_cols = [item for item in data_temp.columns if '_exp' not in item and '_mut' not in item]
cat_cols = [item for item in cat_cols if any(gene in item for gene in genes)]


pval_matrix , cles_matrix, mwu_stats  = get_mannwhitneyu_matrix(data_temp[cat_cols], data_temp[num_cols], iters=100)

mwu_stats['test'] = 'mwu'
# %%%% Chi2

cat_cols = [item for item in data_temp.columns if '_exp' not in item and '_mut' not in item]
cat_cols = [item for item in cat_cols if any(gene in item for gene in genes)]

from itertools import combinations

chi2_res = pd.DataFrame()

# Loop over all possible pairs of genes using itertools and compute the chi2 test
for gene1, gene2 in combinations(cat_cols, 2):
    chi2_res_temp = pd.DataFrame()

    expected, observed, stats = pg.chi2_independence(data_temp, x=gene1,y=gene2)
    warning = ''
    if observed.min().min() < 5:
        warning = '<5'
    # Perform chi2 test using pingouin
    chi2_res_temp = stats.iloc[0,:].T
    chi2_res_temp['source_gene'] = gene1
    chi2_res_temp['target_gene'] = gene2
    chi2_res_temp['warning'] = warning
    chi2_res_temp = chi2_res_temp.rename({'pval':'p-val'})
    # Store the result
    chi2_res = pd.concat((chi2_res, pd.DataFrame(chi2_res_temp).T))
    
chi2_res  = chi2_res[chi2_res ['warning'] == ''].reset_index(drop=True)    
chi2_res = add_edge_colmn(chi2_res).drop(columns = ['test','lambda','dof','warning'])
chi2_res['test'] = 'chi2'
# %%%% merge

univariable_res = pd.concat((corrs_spearman_exp, mwu_stats, chi2_res))

univariable_res['width'] = 0
univariable_res.loc[univariable_res['test'] == 'spearman', 'width'] = univariable_res.loc[univariable_res['test'] == 'spearman', 'r'].abs() / .7
univariable_res.loc[univariable_res['test'] == 'chi2', 'width'] = univariable_res.loc[univariable_res['test'] == 'chi2', 'cramer'] / .7
univariable_res.loc[univariable_res['test'] == 'mwu', 'width'] = (univariable_res.loc[univariable_res['test'] == 'mwu', 'CLES'] - 0.5).abs() * 2

univariable_res.to_excel(os.path.join(path_to_save, 'univariable_res_{}_{}.xlsx'.format(pathway, samples_selection)))

# %%




































