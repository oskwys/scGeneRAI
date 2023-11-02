# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:31:35 2023

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
# %%

networks_for_patient_groups_and_ALL_genes = False
networks_for_patient_groups_and_pathway_genes = False
plot_clustermaps_pathway = False
plot_umaps = False
get_top1000 = True

# %% load data

path_to_data = '/home/d07321ow/scratch/scGeneRAI/data/data_BRCA'
#path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'

path_to_save = '/home/d07321ow/scratch/results_LRP_BRCA/networks'
#path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\networks'


#path_to_pathways = r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI\PATHWAYS'
path_to_pathways ='/home/d07321ow/scratch/scGeneRAI/PATHWAYS'

#path_to_lrp_results = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\results'
path_to_lrp_results = '/home/d07321ow/scratch/results_LRP_BRCA/results'

data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(path_to_data)

# %% get samples
samples = f.get_samples_with_lrp(path_to_lrp_results)
print('Samples: ', len(samples))
print('Samples: ', len(set(samples)))
# %%% get sample goups

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)
df_clinical_features = f.add_cluster0(df_clinical_features)



samples_groups = f.get_samples_by_group(df_clinical_features)


# %% gene from pathways

genes_pathways = pd.read_csv(os.path.join(path_to_pathways, 'genes_pathways_pancanatlas_matched_cce.csv'))
genes_pathways_set = set(genes_pathways['cce_match'])

genes_pathways_dict = {}

for pathway in genes_pathways['Pathway'].unique():
    
    genes_pathways_dict[pathway] = genes_pathways.loc[genes_pathways['Pathway'] == pathway, 'Gene'].to_list()

# %%% load LRP data
lrp_dict = {}
lrp_files = []

for file in os.listdir(path_to_lrp_results):
    if file.startswith("LRP"):
        lrp_files.append(file)
        
        
n = len(lrp_files)  

#network_data = pd.DataFrame()
start_time = datetime.now()

for i in range(n):
    
            
    file_name = lrp_files[i]
    sample_name = file_name.split('_')[2]
    print(i, file_name)
    data_temp = pd.read_pickle(os.path.join(path_to_lrp_results , file_name), compression='infer', storage_options=None)

    data_temp = f.remove_same_source_target(data_temp)
        
    #data_temp = data_temp.sort_values('LRP', ascending= False).reset_index(drop=True)
    #data_temp = add_edge_colmn(data_temp)
    
    #network_data = pd.concat((network_data , data_temp))
        
    lrp_dict[sample_name] = data_temp
end_time = datetime.now()

print(end_time - start_time)
    

    
# %% networks for patient groups and pathway genes




if networks_for_patient_groups_and_pathway_genes == True:
    for group in list(samples_groups.keys())[-1:]:
        print(group)
        
        for subgroup in samples_groups[group].keys():
            print(subgroup)
            
            samples_subgroup = samples_groups[group][subgroup]
                
            for pathway in genes_pathways['Pathway'].unique():
                
                genes = genes_pathways_dict[pathway]
                print(pathway, genes)
                network_all = pd.DataFrame()
                i=0
                index_ = []
                for sample_name in samples_subgroup:
                    print(i, pathway, sample_name, subgroup, ' ------  networks for patient groups and pathway genes')
                    temp = lrp_dict[sample_name].copy()
                    #print(temp.head())
                    #temp = temp.sort_values(['source_gene','target_gene'])
                    if i==0:
                        index_ = temp['source_gene'].str.split('_',expand=True)[0].isin(genes) & temp['target_gene'].str.split('_',expand=True)[0].isin(genes)
                    #print(temp.shape)
                    
                    temp = temp[index_]
                    
                    network_all = pd.concat((network_all, temp))
                    i+=1
                    
                network_all = f.add_edge_colmn(network_all)
                
                network_all_stats = network_all.groupby(by = ['edge','source_gene', 'target_gene','edge_type'],as_index=False).describe().reset_index()
                network_all_stats .columns = network_all_stats.columns.droplevel()
                network_all_stats = network_all_stats.drop(columns = ['min','max', 'count']).rename(columns = {'50%':'median'})
                
                
                
                #network_all = network_all.groupby(by = ['edge','source_gene', 'target_gene','edge_type'],as_index=False).mean()
                
                network_all_stats.to_excel(os.path.join(path_to_save, 'network_{}_{}.xlsx'.format(subgroup, pathway)))
                
        

# %% networks for patient groups and ALL genes

if networks_for_patient_groups_and_ALL_genes == True:
    for group in list(samples_groups.keys())[-1:]:
        print(group)
        
        for subgroup in samples_groups[group].keys():
            print(subgroup)
            
            samples_subgroup = samples_groups[group][subgroup]
            
            lrp_array = np.zeros((lrp_dict[sample_name].shape[0] , len(samples_subgroup)))
            
            
            for i, sample_name in enumerate(samples_subgroup):
                print(i, sample_name, subgroup, ' ------ networks for patient groups and ALL genes')
                temp = lrp_dict[sample_name].copy()
                lrp_array[:, i] = temp['LRP'].values
                
    
            lrp_array_mean = np.round(  np.mean(lrp_array,axis=1), 5)
            lrp_array_std = np.round( np.std(lrp_array,axis=1), 5)
            lrp_array_median = np.round( np.median(lrp_array,axis=1), 5)
            lrp_array_q1 = np.round( np.quantile(lrp_array, .25, axis=1), 5)
            lrp_array_q3 = np.round( np.quantile(lrp_array, .75, axis=1), 5)
            
            network_all = temp.copy()
            network_all['LRP'] = lrp_array_mean
            network_all['LRP_std'] = lrp_array_std
            network_all['LRP_median'] = lrp_array_median
            network_all['LRP_q1'] = lrp_array_q1
            network_all['LRP_q3'] = lrp_array_q3
            
            network_all = f.add_edge_colmn(network_all)
    
            network_all.to_csv(os.path.join(path_to_save, 'network_{}_allgenes.csv'.format(subgroup)))
        
# %% CLUSTERMAPS and UMAPs
# %%% get lrp_dict_filtered for each pathway
if plot_clustermaps_pathway or plot_umaps:
    lrp_dict_filtered = {}
     
    for pathway in genes_pathways['Pathway'].unique():
        print(pathway, ' ---- get lrp_dict_filtered for each pathway')
        lrp_dict_filtered[pathway] = {}
        genes = genes_pathways_dict[pathway]     
        for index, (sample_name, data) in enumerate(lrp_dict.items()):
            #print(index, sample_name)
            if index ==0:
                 index_ = data['source_gene'].str.split('_',expand=True)[0].isin(genes) & data['target_gene'].str.split('_',expand=True)[0].isin(genes)
             
            
            temp = data[index_].copy()
            #print(temp.shape[0])
            lrp_dict_filtered[pathway][sample_name] = temp.reset_index(drop=True)
            
        
    lrp_dict_filtered_pd = {}
    for pathway in genes_pathways['Pathway'].unique():
        print(pathway, ' ---- get lrp_dict_filtered for each pathway PD')
        lrp_dict_filtered_pd[pathway] = f.get_lrp_dict_filtered_pd(lrp_dict_filtered, pathway = pathway)    
    


# %%% plot clustermap

thresholds = [0, 0.0001, 0.0002]

if plot_clustermaps_pathway:
    for threshold in thresholds:
        for pathway in genes_pathways['Pathway'].unique():
            print('Clustermap for ', pathway, str(threshold))
            data_to_clustermap = lrp_dict_filtered_pd[pathway].copy()
            
            data_to_clustermap[data_to_clustermap < threshold] = 0
            
            df_clinical_features_ = f.get_column_colors_from_clinical_df(df_clinical_features,data_to_clustermap )
            column_colors = f.map_subtypes_to_col_color(df_clinical_features_)    
            
            sns.clustermap(data_to_clustermap, mask  = data_to_clustermap < threshold, cmap = 'jet', col_colors = column_colors,
                           method = 'ward', yticklabels = False, xticklabels = False, vmax = 0.008, vmin = 0)       
            plt.suptitle(pathway, fontsize=20)       
                   
            plt.savefig(os.path.join(path_to_save, 'clustermap_{}_th{}.png'.format(pathway, str(threshold)[2:])), dpi = 400)        
        
# %%% plot UMAP
import umap
titles = ['her2', 'estrogen_receptor','progesterone_receptor',  'TNBC']


if plot_umaps:
    for threshold in thresholds:
        for pathway in genes_pathways['Pathway'].unique():
            print('UMAP for ', pathway, str(threshold))
            data_to_umap = lrp_dict_filtered_pd[pathway].copy().T
            
            data_to_umap[data_to_umap < threshold] = 0
            
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(data_to_umap)
            
            df_clinical_features_ = f.get_column_colors_from_clinical_df(df_clinical_features, lrp_dict_filtered_pd[pathway].copy() )
            column_colors = f.map_subtypes_to_col_color(df_clinical_features_)    
             
            
            # Create a 2x2 grid of subplots
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
            axes = axes.flatten()  # Flatten axes to easily iterate over
            
            # Iterate over subplots and color lists to create UMAP plots
            i=0
            for ax, colors in zip(axes, column_colors):
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors.values)
                ax.set_title(titles[i])
                i+=1
                ax.set_xticks([])
                ax.set_yticks([])
            plt.suptitle(pathway, fontsize=20)    
            plt.tight_layout()
                 
            plt.savefig(os.path.join(path_to_save, 'UMAP_{}_th{}.png'.format(pathway, str(threshold)[2:])), dpi = 400)        
    
    
# %% get top 1000 interactions for each sample

if get_top1000:
    topn = 250
    
    df_topn = pd.DataFrame(np.zeros((topn,len(samples)), dtype = 'str'), columns = samples)
    
    for i, sample_name in enumerate(samples):
        print(i)
        data_temp = lrp_dict[sample_name]
    
        data_temp = data_temp.sort_values('LRP', ascending = False).iloc[:topn, :]
        data_temp = f.add_edge_colmn(data_temp)
        df_topn.iloc[:, i] = data_temp['edge'].values
    
    df_topn.to_csv(os.path.join(path_to_save, 'df_topn_for_individuals.csv'))
    
    
    unique_edges = list(df_topn.melt()['value'].unique())
    unique_edges_count = []
    for i, unique_edge in enumerate(unique_edges):
        print(i,'/', len(unique_edges), unique_edge)
        a = np.sum(np.sum(df_topn == unique_edge, axis=0))
        unique_edges_count.append(a)
        
    unique_edges_df = pd.DataFrame([unique_edges, unique_edges_count]).T#, columns = )
    unique_edges_df.columns = ['edge','count']
    
    unique_edges_df.to_csv(os.path.join(path_to_save, 'unique_edges_count_in_top1000.csv'))

    #unique_edges_df['count'].plot(kind = 'hist')

    #unique_edges_df.sort_values('count', ascending = False).reset_index(drop=True)['count'].plot()
# %%% clustermap legend

# import matplotlib.patches as mpatches

# def create_legend(color_map):
#     patches = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
#     plt.legend(handles=patches)

# # Create a dummy figure just to show the legend
# plt.figure(figsize=(10, 6))
# create_legend(color_map)
# plt.axis('off')
# plt.show()
    











         