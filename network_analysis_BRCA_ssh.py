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
get_top1000 = False

get_top1000_noexpexp = False
get_pathway_LRP = False
get_LRP_median_matrix = False
get_top_lrp = False
get_top_lrp_groups = False
get_stat_diff_groups = False
get_mean_lrp_groups = False
get_sum_lrp_groups = False
get_sum_lrp_matrix =True 
# %% load data

path_to_data = r'G:\My Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'
path_to_data = '/home/d07321ow/scratch/scGeneRAI/data/data_BRCA'
#path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'

path_to_save = '/home/d07321ow/scratch/results_LRP_BRCA/networks'
#path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\networks'


#path_to_pathways = r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI\PATHWAYS'
path_to_pathways = r'C:\Users\owysocky\Documents\GitHub\scGeneRAI\PATHWAYS'
path_to_pathways ='/home/d07321ow/scratch/scGeneRAI/PATHWAYS'

#path_to_lrp_results = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\results'
path_to_lrp_results = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\results'
path_to_lrp_results = '/home/d07321ow/scratch/results_LRP_BRCA/results'

data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(path_to_data)

# %% get samples
samples = f.get_samples_with_lrp(path_to_lrp_results)
#samples = samples[:10]
print('Samples: ', len(samples))
print('Samples: ', len(set(samples)))
# %%% get sample goups

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)
df_clinical_features = f.add_cluster2(df_clinical_features)



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
    topn = 100
    
    df_topn = pd.DataFrame(np.zeros((topn,len(samples)), dtype = 'str'), columns = samples)
    
    for i, sample_name in enumerate(samples):
        print(i)
        data_temp = lrp_dict[sample_name]
    
        data_temp = data_temp.sort_values('LRP', ascending = False).iloc[:topn, :]
        data_temp = f.add_edge_colmn(data_temp)
        df_topn.iloc[:, i] = data_temp['edge'].values
    
    df_topn.to_csv(os.path.join(path_to_save, 'df_topn_for_individuals_top{}.csv'.format(topn)))
    
    unique_edges, unique_edges_count = np.unique(df_topn.values.ravel(), return_counts=True)
    
    # unique_edges = list(df_topn.melt()['value'].unique())
    # unique_edges_count = []
    # for i, unique_edge in enumerate(unique_edges):
    #     print(i,'/', len(unique_edges), unique_edge)
    #     a = np.sum(np.sum(df_topn == unique_edge, axis=0))
    #     unique_edges_count.append(a)
        
    unique_edges_df = pd.DataFrame([unique_edges, unique_edges_count]).T#, columns = )
    unique_edges_df.columns = ['edge','count']
    
    unique_edges_df.to_csv(os.path.join(path_to_save, 'unique_edges_count_in_top_{}.csv'.format(topn)))

    #unique_edges_df['count'].plot(kind = 'hist')

    #unique_edges_df.sort_values('count', ascending = False).reset_index(drop=True)['count'].plot()
    

    
# %% get top 1000 interactions (exclude exp-exp) for each sample

if get_top1000_noexpexp:
    topn = 1000
    
    df_topn = pd.DataFrame(np.zeros((topn,len(samples)), dtype = 'str'), columns = samples)
    
    for i, sample_name in enumerate(samples):
        print(i)
        data_temp = lrp_dict[sample_name]
        data_temp = data_temp[ - (data_temp['source_gene'] .str.contains('_exp') & data_temp['target_gene'] .str.contains('_exp'))]
        data_temp = data_temp.sort_values('LRP', ascending = False)
        data_temp = data_temp.iloc[:topn, :]
        data_temp = f.add_edge_colmn(data_temp)
        df_topn.iloc[:, i] = data_temp['edge'].values
    
    df_topn.to_csv(os.path.join(path_to_save, 'df_topn_for_individuals_top{}_noexpexp.csv'.format(topn)))
    
    unique_edges, unique_edges_count = np.unique(df_topn.values.ravel(), return_counts=True)
    
    # unique_edges = list(df_topn.melt()['value'].unique())
    # unique_edges_count = []
    # for i, unique_edge in enumerate(unique_edges):
    #     print(i,'/', len(unique_edges), unique_edge)
    #     a = np.sum(np.sum(df_topn == unique_edge, axis=0))
    #     unique_edges_count.append(a)
        
    unique_edges_df = pd.DataFrame([unique_edges, unique_edges_count]).T#, columns = )
    unique_edges_df.columns = ['edge','count']
    
    unique_edges_df.to_csv(os.path.join(path_to_save, 'unique_edges_noexpexp_count_in_top_{}_noexpexp.csv'.format(topn)))
    
    
    
# %% get LRP only for interaction including PATHWAY genes

if get_pathway_LRP:
    temp = lrp_dict['TCGA-3C-AAAU']


    for pathway in genes_pathways['Pathway'].unique():
        genes = genes_pathways.loc[genes_pathways['Pathway'] == pathway, 'cce_match'].to_list()
        index_ = (temp['source_gene'].str.split('_', expand = True)[0].isin(genes)) + (temp['target_gene'].str.split('_', expand = True)[0].isin(genes))

        df_topn = pd.DataFrame(np.zeros((topn,len(samples)), dtype = 'str'), columns = samples)

        np_lrp_temp = np.zeros((index_.sum(),len(samples)) )

        for i, sample_name in enumerate(samples):
            print(i)
            data_temp = lrp_dict[sample_name]
            data_temp = data_temp[index_]
            
            np_lrp_temp[:, i] = data_temp['LRP'].values
            
            data_temp = data_temp.sort_values('LRP', ascending = False).reset_index(drop=True)
            data_temp = data_temp.iloc[:topn, :]
            data_temp = f.add_edge_colmn(data_temp)
            df_topn.iloc[:, i] = data_temp['edge'].values
        
        df_topn.to_csv(os.path.join(path_to_save, 'df_topn_for_individuals_top{}_{}.csv'.format(topn, pathway)))
        
        unique_edges, unique_edges_count = np.unique(df_topn.values.ravel(), return_counts=True)
        
        # unique_edges = list(df_topn.melt()['value'].unique())
        # unique_edges_count = []
        # for i, unique_edge in enumerate(unique_edges):
        #     print(i,'/', len(unique_edges), unique_edge)
        #     a = np.sum(np.sum(df_topn == unique_edge, axis=0))
        #     unique_edges_count.append(a)
            
        unique_edges_df = pd.DataFrame([unique_edges, unique_edges_count]).T#, columns = )
        unique_edges_df.columns = ['edge','count']
        
        unique_edges_df.to_csv(os.path.join(path_to_save, 'unique_edges_noexpexp_count_in_top_{}_{}.csv'.format(topn, pathway)))
        
        # median LRP matrix
        # lrp_median = np.nanmedian(np_lrp_temp, axis = 1)
        # lrp_median_df = pd.DataFrame()
        # lrp_median_df ['LRP_median'] = lrp_median
        # lrp_median_df[['source_gene', 'target_gene']] =   temp.loc[index_, ['source_gene', 'target_gene']].reset_index(drop=True)
        
        # lrp_median_df_doubled = pd.DataFrame()
        # lrp_median_df_doubled ['source_gene'] = lrp_median_df['target_gene']
        # lrp_median_df_doubled ['target_gene'] = lrp_median_df['source_gene']
        # lrp_median_df_doubled['LRP_median'] = lrp_median_df['LRP_median']
        
        # lrp_median_df = pd.concat((lrp_median_df, lrp_median_df_doubled)).reset_index(drop=True)
        # lrp_median_df_pivot = lrp_median_df.pivot_table(columns = 'target_gene', values = 'LRP_median', index = 'source_gene')
        #sns.heatmap(lrp_median_df_pivot)


# %% get LRP median clustermap matix


if get_LRP_median_matrix:
    temp = lrp_dict['TCGA-3C-AAAU']

    np_lrp_temp = np.zeros((temp.shape[0],len(samples)) )

    for i, sample_name in enumerate(samples):
        print(i)
        data_temp = lrp_dict[sample_name]
        np_lrp_temp[:, i] = data_temp['LRP'].values
        
    
    # median LRP matrix
    print(np_lrp_temp)
    lrp_median = np.nanmedian(np_lrp_temp, axis = 1)
    lrp_median_df = pd.DataFrame()
    print(lrp_median)
    lrp_median_df ['LRP_median'] = lrp_median
    lrp_median_df['source_gene'] = temp['source_gene'].to_list()
    lrp_median_df['target_gene'] = temp['target_gene'].to_list()

    lrp_median_df.to_csv(os.path.join(path_to_save, 'lrp_median_list.csv'))

    lrp_median_df_doubled = pd.DataFrame()
    lrp_median_df_doubled ['source_gene'] = lrp_median_df['target_gene'].copy()
    lrp_median_df_doubled ['target_gene'] = lrp_median_df['source_gene'].copy()
    lrp_median_df_doubled['LRP_median'] = lrp_median_df['LRP_median'].copy()
    
    lrp_median_df = pd.concat((lrp_median_df, lrp_median_df_doubled)).reset_index(drop=True)
    lrp_median_df_pivot = lrp_median_df.pivot_table(columns = 'target_gene', values = 'LRP_median', index = 'source_gene')

    lrp_median_pivot = lrp_median_df_pivot.fillna(0).values + lrp_median_df_pivot.T.fillna(0).values - np.diag(np.diag(lrp_median_df_pivot.values))
    np.fill_diagonal(lrp_median_pivot, 0)
    lrp_median_df_pivot = pd.DataFrame(lrp_median_pivot, index = lrp_median_df_pivot.index, columns = lrp_median_df_pivot.columns)
    
    lrp_median_df_pivot.to_csv(os.path.join(path_to_save, 'lrp_median_matrix.csv'))
    
    
    
    
    # mean LRP matrix
    print(np_lrp_temp)
    lrp_mean = np.nanmean(np_lrp_temp, axis = 1)
    lrp_mean_df = pd.DataFrame()
    print(lrp_mean)
    lrp_mean_df ['LRP_mean'] = lrp_mean
    lrp_mean_df['source_gene'] = temp['source_gene'].to_list()
    lrp_mean_df['target_gene'] = temp['target_gene'].to_list()
    lrp_mean_df.to_csv(os.path.join(path_to_save, 'lrp_mean_list.csv'))

    lrp_mean_df_doubled = pd.DataFrame()
    lrp_mean_df_doubled ['source_gene'] = lrp_mean_df['target_gene'].copy()
    lrp_mean_df_doubled ['target_gene'] = lrp_mean_df['source_gene'].copy()
    lrp_mean_df_doubled['LRP_mean'] = lrp_mean_df['LRP_mean'].copy()
    
    lrp_mean_df = pd.concat((lrp_mean_df, lrp_mean_df_doubled)).reset_index(drop=True)
    lrp_mean_df_pivot = lrp_mean_df.pivot_table(columns = 'target_gene', values = 'LRP_mean', index = 'source_gene')

    lrp_mean_pivot = lrp_mean_df_pivot.fillna(0).values + lrp_mean_df_pivot.T.fillna(0).values - np.diag(np.diag(lrp_mean_df_pivot.values))
    np.fill_diagonal(lrp_mean_pivot, 0)
    lrp_mean_df_pivot = pd.DataFrame(lrp_mean_pivot, index = lrp_mean_df_pivot.index, columns = lrp_mean_df_pivot.columns)

    lrp_mean_df_pivot.to_csv(os.path.join(path_to_save, 'lrp_mean_matrix.csv'))







# %% get LRP top values
def get_median_or_mean_from_all_LRP(lrp_np, temp, index_, type_ = 'mean'):
    
    if type_ == 'median':
        lrp_temp = np.nanmedian(lrp_np, axis = 1)
    elif type_ == 'mean':
        lrp_temp = np.nanmean(lrp_np, axis = 1)
    lrp_df = pd.DataFrame()
    print(lrp_temp)
    lrp_df ['LRP'] = lrp_temp
    lrp_df['source_gene'] = temp.loc[index_,'source_gene'].to_list()
    lrp_df['target_gene'] = temp.loc[index_,'target_gene'].to_list()
    lrp_df['edge'] =  temp.loc[index_,'edge'].to_list()
    lrp_df['edge_type'] =  temp.loc[index_,'edge_type'].to_list()
    lrp_df = lrp_df.reset_index()
    lrp_df = lrp_df.sort_values(by = 'LRP', ascending = False).reset_index(drop = True)
    return lrp_df
    



def calculate_lrp_measures(lrp_dict, samples, temp, edge_types, add_name = ''):
    """
    Calculates the mean and median of LRP values for different edge types.
    
    :param lrp_dict: Dictionary containing LRP values.
    :param samples: List of sample names.
    :param temp: DataFrame with 'edge_type' and 'LRP' columns.
    :param edge_types: List of edge types to process.
    :return: Dictionary of DataFrames for mean and median values of each edge type.
        """
    results = {}
    
    for edge_type in edge_types:
        # Filter the DataFrame based on the edge type
        index_ = temp['edge_type'].str.contains(edge_type)
        np_lrp_temp = np.zeros((index_.sum(), len(samples)))
        print(edge_type)
        # Populate the numpy array with LRP values
        for i, sample_name in enumerate(samples):
            #print(i)
            data_temp = lrp_dict[sample_name]
            data_temp = data_temp[index_]
            np_lrp_temp[:, i] = data_temp['LRP'].values
    
        # Calculate mean and median and store in results
        lrp_mean = get_median_or_mean_from_all_LRP(np_lrp_temp, temp, index_, type_='mean')
        lrp_median = get_median_or_mean_from_all_LRP(np_lrp_temp, temp, index_, type_='median')
        results[f'lrp_top_{edge_type}_{add_name}_mean'] = lrp_mean
        results[f'lrp_top_{edge_type}_{add_name}_median'] = lrp_median

    return results

def save_lrp_results(results, path_to_save):
    """
    Saves the LRP results to CSV files.

    :param results: Dictionary containing the LRP results.
    :param path_to_save: Path where the files will be saved.
    """
    for key, df in results.items():
        filename = f'{key}.csv'
        df.to_csv(os.path.join(path_to_save, filename))

if get_top_lrp:
    temp = lrp_dict['TCGA-3C-AAAU'].copy()
    temp = f.add_edge_colmn(temp)
    edge_types = ['exp', 'mut', 'del', 'amp', 'fus', 'exp-exp', 'mut-mut', 'del-del', 'amp-amp', 'fus-fus']
    results = calculate_lrp_measures(lrp_dict, samples, temp, edge_types, add_name = 'all_')
    save_lrp_results(results, path_to_save)



if get_top_lrp_groups:
    edge_types = ['exp', 'mut', 'del', 'amp', 'fus', 'exp-exp', 'mut-mut', 'del-del', 'amp-amp', 'fus-fus']
    temp = lrp_dict['TCGA-3C-AAAU'].copy()
    temp = f.add_edge_colmn(temp)
    for group in list(samples_groups.keys()):
        print(group)
        
        for subgroup in samples_groups[group].keys():
            print(subgroup)
            
            samples_subgroup = samples_groups[group][subgroup]
    
            results = calculate_lrp_measures(lrp_dict, samples_subgroup, temp, edge_types, add_name = subgroup)
            save_lrp_results(results, path_to_save)
# %% get median and mean lrp for each group

def get_mean_or_median_lrp_from_np(np_lrp, temp, type_ = 'mean'):
        
    if type_ == 'median':
        lrp_temp = np.nanmedian(np_lrp, axis = 1)
    elif type_ == 'mean':
        lrp_temp = np.nanmean(np_lrp, axis = 1)
    
    lrp_df = pd.DataFrame()
    print(lrp_temp)
    lrp_df ['LRP'] = lrp_temp
    lrp_df['source_gene'] = temp.loc[:,'source_gene'].to_list()
    lrp_df['target_gene'] = temp.loc[:,'target_gene'].to_list()
    lrp_df['edge'] =  temp.loc[:,'edge'].to_list()
    lrp_df['edge_type'] =  temp.loc[:,'edge_type'].to_list()
    lrp_df = lrp_df.reset_index()
    
    return lrp_df


    

if get_mean_lrp_groups:
#    temp = lrp_dict['TCGA-3C-AAAU']
    temp = lrp_dict['TCGA-EW-A1P0']
    temp = f.add_edge_colmn(temp)
    for group in list(samples_groups.keys()):
        print(group)

        subgroup_keys = list(samples_groups[group].keys())
        subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
        print(group, subgroup1, subgroup2)
        
        samples_subgroup1 = samples_groups[group][subgroup1]
        samples_subgroup2 = samples_groups[group][subgroup2]
        
        np_lrp_temp1 = np.zeros((temp.shape[0], len(samples_subgroup1)))
        np_lrp_temp2 = np.zeros((temp.shape[0], len(samples_subgroup2)))
        
        for i, sample_name in enumerate(samples_subgroup1):
            np_lrp_temp1[:, i] = lrp_dict[sample_name]['LRP'].values
        
        for i, sample_name in enumerate(samples_subgroup2):
            np_lrp_temp2[:, i] = lrp_dict[sample_name]['LRP'].values
        
        df_1 = get_mean_or_median_lrp_from_np(np_lrp_temp1, temp, type_ = 'mean')
        df_2 = get_mean_or_median_lrp_from_np(np_lrp_temp2, temp, type_ = 'mean')
            
        df_1.to_csv(os.path.join(path_to_save, 'lrp_mean_{}.csv'.format(subgroup1)))
        df_2.to_csv(os.path.join(path_to_save, 'lrp_mean_{}.csv'.format(subgroup2)))
        
        df_1 = get_mean_or_median_lrp_from_np(np_lrp_temp1, temp, type_ = 'median')
        df_2 = get_mean_or_median_lrp_from_np(np_lrp_temp2, temp, type_ = 'median')
            
        df_1.to_csv(os.path.join(path_to_save, 'lrp_median_{}.csv'.format(subgroup1)))
        df_2.to_csv(os.path.join(path_to_save, 'lrp_median_{}.csv'.format(subgroup2)))

# %% get_sum_lrp_matrix

if get_sum_lrp_matrix:
#    temp = lrp_dict['TCGA-3C-AAAU']
    temp = lrp_dict['TCGA-EW-A1P0']
    temp = f.add_edge_colmn(temp)
    
    gene_set = set(temp['source_gene']).union(temp['target_gene'])
    n_genes = len(gene_set)
    gene_list = sorted(list(gene_set))  # Create a consistent ordered list of genes
    

    np_lrp_temp = np.zeros((n_genes, len(samples)))
    

    for i, sample_name in enumerate(samples):
        data_temp = lrp_dict[sample_name]
        sum_ = data_temp.groupby('source_gene')['LRP'].sum().add(
            data_temp.groupby('target_gene')['LRP'].sum(), fill_value=0
        ).reindex(gene_list, fill_value=0)

        np_lrp_temp[:, i] = sum_.values
        
    df = pd.DataFrame()
    df ['LRP_sum_mean'] = np.nanmean(np_lrp_temp, axis = 1)
    df ['gene'] = gene_list            
    df.to_csv(os.path.join(path_to_save, 'lrp_sum_median.csv'))
    print(df.head())
    
    
    
# %% get_sum_lrp_groups       
     
if get_sum_lrp_groups:
#    temp = lrp_dict['TCGA-3C-AAAU']
    temp = lrp_dict['TCGA-EW-A1P0']
    temp = f.add_edge_colmn(temp)
    
    gene_set = set(temp['source_gene']).union(temp['target_gene'])
    n_genes = len(gene_set)
    gene_list = sorted(list(gene_set))  # Create a consistent ordered list of genes
    
    
    for group in list(samples_groups.keys()):
        print(group)

        subgroup_keys = list(samples_groups[group].keys())
        subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
        print(group, subgroup1, subgroup2)
    
        samples_subgroup1 = samples_groups[group][subgroup1]
        samples_subgroup2 = samples_groups[group][subgroup2]
    
        np_lrp_temp1 = np.zeros((n_genes, len(samples_subgroup1)))
        np_lrp_temp2 = np.zeros((n_genes, len(samples_subgroup2)))
    
        for i, sample_name in enumerate(samples_subgroup1):
            data_temp = lrp_dict[sample_name]
            sum_ = data_temp.groupby('source_gene')['LRP'].sum().add(
                data_temp.groupby('target_gene')['LRP'].sum(), fill_value=0
            ).reindex(gene_list, fill_value=0)
    
            np_lrp_temp1[:, i] = sum_.values
    
        for i, sample_name in enumerate(samples_subgroup2):
            data_temp = lrp_dict[sample_name]
            sum_ = data_temp.groupby('source_gene')['LRP'].sum().add(
                data_temp.groupby('target_gene')['LRP'].sum(), fill_value=0
            ).reindex(gene_list, fill_value=0)
    
            np_lrp_temp2[:, i] = sum_.values
        
        print(np_lrp_temp1.shape ,np_lrp_temp2.shape)
        # mean
        df_1 = pd.DataFrame()
        df_1 ['LRP_sum_mean'] = np.nanmean(np_lrp_temp1, axis = 1)
        df_1 ['gene'] = gene_list
        
        df_2 = pd.DataFrame()
        df_2 ['LRP_sum_mean'] = np.nanmean(np_lrp_temp2, axis = 1)
        df_2 ['gene'] = gene_list
        
        df = df_1.merge(df_2, on = 'gene',suffixes=('_'+subgroup1, '_'+subgroup2))
        df.to_csv(os.path.join(path_to_save, 'lrp_sum_mean_{}.csv'.format(group)))

        print(df.head())

        # median
        df_1 = pd.DataFrame()
        df_1 ['LRP_sum_median'] = np.nanmedian(np_lrp_temp1, axis = 1)
        df_1 ['gene'] = gene_list
        
        df_2 = pd.DataFrame()
        df_2 ['LRP_sum_median'] = np.nanmedian(np_lrp_temp2, axis = 1)
        df_2 ['gene'] = gene_list
        
        df = df_1.merge(df_2, on = 'gene',suffixes=('_'+subgroup1, '_'+subgroup2))
        df.to_csv(os.path.join(path_to_save, 'lrp_sum_median_{}.csv'.format(group)))
        print(df.head())

# %% compute statistcal difference between groups in LRP


if get_stat_diff_groups:
    import pingouin as pg
#    temp = lrp_dict['TCGA-3C-AAAU']
    temp = lrp_dict['TCGA-EW-A1P0']
    temp = f.add_edge_colmn(temp)
    for group in list(samples_groups.keys()):
        print(group)

        subgroup_keys = list(samples_groups[group].keys())
        subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
        print(group, subgroup1, subgroup2)
        
        samples_subgroup1 = samples_groups[group][subgroup1]
        samples_subgroup2 = samples_groups[group][subgroup2]
        
        np_lrp_temp1 = np.zeros((temp.shape[0], len(samples_subgroup1)))
        np_lrp_temp2 = np.zeros((temp.shape[0], len(samples_subgroup2)))
        
        for i, sample_name in enumerate(samples_subgroup1):
            np_lrp_temp1[:, i] = lrp_dict[sample_name]['LRP'].values
        
        for i, sample_name in enumerate(samples_subgroup2):
            np_lrp_temp2[:, i] = lrp_dict[sample_name]['LRP'].values
    
        # Compare rows
        mwu_res = []
        for pair_i in range(temp.shape[0]):
            x1 = np_lrp_temp1[pair_i, :]
            x2 = np_lrp_temp2[pair_i, :]
    
            mwu = pg.mwu(x1, x2).values[0]
            mwu_res.append(mwu)
        mwu_df = pd.DataFrame(mwu_res, columns=['U-val', 'alternative', 'p-val', 'RBC', 'CLES'])
        mwu_df['edge'] = temp['edge']
        #mwu_df['edge_type'] = temp['edge_type']
        mwu_df.to_csv(os.path.join(path_to_save, 'mwu_edges_LRP_{}.csv'.format(group)))




if get_stat_diff_groups:
    import pingouin as pg
    gene_set = set(temp['source_gene']).union(temp['target_gene'])
    n_genes = len(gene_set)
    gene_list = sorted(list(gene_set))  # Create a consistent ordered list of genes
    
    for group, subgroups in samples_groups.items():
        print(group)
        
        subgroup_keys = list(samples_groups[group].keys())
        subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
        print(group, subgroup1, subgroup2)
    
        samples_subgroup1 = subgroups[subgroup1]
        samples_subgroup2 = subgroups[subgroup2]
    
        np_lrp_temp1 = np.zeros((n_genes, len(samples_subgroup1)))
        np_lrp_temp2 = np.zeros((n_genes, len(samples_subgroup2)))
    
        for i, sample_name in enumerate(samples_subgroup1):
            data_temp = lrp_dict[sample_name]
            sum_ = data_temp.groupby('source_gene')['LRP'].sum().add(
                data_temp.groupby('target_gene')['LRP'].sum(), fill_value=0
            ).reindex(gene_list, fill_value=0)
    
            np_lrp_temp1[:, i] = sum_.values
    
        for i, sample_name in enumerate(samples_subgroup2):
            data_temp = lrp_dict[sample_name]
            sum_ = data_temp.groupby('source_gene')['LRP'].sum().add(
                data_temp.groupby('target_gene')['LRP'].sum(), fill_value=0
            ).reindex(gene_list, fill_value=0)
    
            np_lrp_temp2[:, i] = sum_.values
    
        # Compare genes
        mwu_res = []
        for gene_i in range(n_genes):
            x1 = np_lrp_temp1[gene_i, :]
            x2 = np_lrp_temp2[gene_i, :]
    
            mwu = pg.mwu(x1, x2).values[0]
            mwu_res.append(mwu)
        
        mwu_df = pd.DataFrame(mwu_res, columns=['U-val', 'alternative', 'p-val', 'RBC', 'CLES'])
        mwu_df['genes'] = n_genes
        mwu_df.to_csv(os.path.join(path_to_save, 'mwu_sum_LRP_genes_{}.csv'.format(group)))



















    
