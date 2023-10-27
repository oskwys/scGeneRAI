# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:27:47 2023

@author: d07321ow
"""


from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error , r2_score

import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import pingouin as pg
import numpy as np
import sys
import os

import pyarrow.feather as feather
from datetime import datetime

def get_samples_with_lrp(path_to_lrp_results, starts_with = 'LRP_'):
    files = [f for f in os.listdir(path_to_lrp_results) if f.startswith(starts_with)]
    print(files)
    samples = [file.split('_')[2] for file in files]
    return samples


def get_input_data(path_to_data):
        
        
    df_fus = pd.read_csv(os.path.join(path_to_data, 'CCE_fusions_to_model.csv') ,index_col = 0)
    df_mut = pd.read_csv( os.path.join(path_to_data, 'CCE_mutations_to_model.csv') ,index_col = 0)
    df_amp = pd.read_csv( os.path.join(path_to_data, 'CCE_amplifications_to_model.csv'),index_col = 0 )
    df_del = pd.read_csv( os.path.join(path_to_data, 'CCE_deletions_to_model.csv') ,index_col = 0 )
    
    df_exp = feather.read_feather( os.path.join(path_to_data, 'CCE_expressions_to_model') , )
    
    df_clinical_features = pd.read_csv( os.path.join(path_to_data, 'CCE_clinical_features.csv') )
    
    df_exp = df_exp.apply(lambda x: np.log(x + 1))
    df_exp_stand = (df_exp-df_exp.mean(axis=0))/df_exp.std(axis=0)
    
    df_mut_scale = (df_mut-df_mut.min(axis=0))/df_mut.max(axis=0)
    df_fus_scale = (df_fus-df_fus.min(axis=0))/df_fus.max(axis=0)
    
    df_amp[df_amp==2] =1

    data_to_model = pd.concat((df_exp_stand, df_mut_scale, df_amp, df_del, df_fus_scale), axis = 1)
    
    return data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features





def get_samples_by_group(df_clinical_features):
        
    #df_clinical_features = df_clinical_features_all[df_clinical_features_all['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)
    
    
    # Estrogen_receptor
    samples_estrogen_receptor_pos = df_clinical_features.loc[ df_clinical_features['Estrogen_receptor']=='Positive', 'bcr_patient_barcode'].to_list()
    samples_estrogen_receptor_neg = df_clinical_features.loc[ df_clinical_features['Estrogen_receptor']=='Negative', 'bcr_patient_barcode'].to_list()
    
    # HER2
    samples_her2_pos = df_clinical_features.loc[ df_clinical_features['HER2']=='Positive', 'bcr_patient_barcode'].to_list()
    samples_her2_neg = df_clinical_features.loc[ df_clinical_features['HER2']=='Negative', 'bcr_patient_barcode'].to_list()
    
    # Progesterone_receptor
    samples_progesterone_receptor_pos = df_clinical_features.loc[ df_clinical_features['Progesterone_receptor']=='Positive', 'bcr_patient_barcode'].to_list()
    samples_progesterone_receptor_neg = df_clinical_features.loc[ df_clinical_features['Progesterone_receptor']=='Negative', 'bcr_patient_barcode'].to_list()
    
    # triple negative - TNBC
    triple_neg_index = (df_clinical_features['HER2']=='Negative') & (df_clinical_features['Progesterone_receptor']=='Negative') & (df_clinical_features['Estrogen_receptor']=='Negative')
    samples_triple_neg = df_clinical_features.loc[triple_neg_index , 'bcr_patient_barcode'].to_list()
    samples_no_triple_neg = df_clinical_features.loc[ -triple_neg_index, 'bcr_patient_barcode'].to_list()
    
    # triple negative - TNBC
    cluster0_neg_index = (df_clinical_features['cluster0']==0)
    samples_cluster0_neg = df_clinical_features.loc[cluster0_neg_index  , 'bcr_patient_barcode'].to_list()
    samples_cluster0_pos = df_clinical_features.loc[ -cluster0_neg_index , 'bcr_patient_barcode'].to_list()
    
    samples_groups = {}
    samples_groups['her2'] = {'her2_pos':samples_her2_pos, 'her2_neg':samples_her2_neg}
    samples_groups['progesterone_receptor'] = {'progesterone_receptor_pos':samples_progesterone_receptor_pos, 'progesterone_receptor_neg':samples_progesterone_receptor_neg}
    samples_groups['estrogen_receptor'] = {'estrogen_receptor_pos':samples_estrogen_receptor_pos, 'estrogen_receptor_neg':samples_estrogen_receptor_neg}
    samples_groups['TNBC'] = {'TNBC':samples_triple_neg, 'no_TNBC':samples_no_triple_neg}
    samples_groups['cluster0'] = {'cluster0_pos':samples_cluster0_pos, 'cluster0_neg':samples_cluster0_neg}
    
    
    
    return samples_groups



from sklearn.metrics import roc_curve, auc


def get_auc_artificial_homogeneous(corrs, plot = True, title = ''):

    matrix2 = corrs
    
    # Groubnd truth homo
    matrix1 = np.zeros((32,32))
    matrix1 [:8,:8] = 1
    matrix1 [8:16,8:16] = 1
    matrix1 [16:24,16:24] = 1
    matrix1 [24:, 24:] = 1
    sns.heatmap(matrix1, cmap = 'bwr')
    
    np.fill_diagonal(matrix1, 0)
    np.fill_diagonal(matrix2, 0)

        
    # Flatten matrices
    flat_matrix1 = matrix1.ravel().astype(int)
    flat_matrix2 = matrix2.ravel()
    
    
    # Define custom thresholds for increased granularity
    thresholds = np.linspace(0, 1, 100)
    
    # Manually compute TPRs and FPRs based on custom thresholds
    fpr = []
    tpr = []
    for threshold in thresholds:
        # Predictions based on threshold
        predicted = (flat_matrix2 >= threshold).astype(int)
        
        # True positives, false positives, etc.
        TP = np.sum((predicted == 1) & (flat_matrix1 == 1))
        TN = np.sum((predicted == 0) & (flat_matrix1 == 0))
        FP = np.sum((predicted == 1) & (flat_matrix1 == 0))
        FN = np.sum((predicted == 0) & (flat_matrix1 == 1))
        
        # Append TPR and FPR values
        tpr.append(TP / (TP + FN))
        fpr.append(FP / (FP + TN))
    
    
    # Sort FPR and TPR based on FPR values for correct plotting
    sort_idx = np.argsort(fpr)
    fpr = np.array(fpr)[sort_idx]
    tpr = np.array(tpr)[sort_idx]

    roc_auc = auc(fpr, tpr)

    
    print("AUC:", roc_auc)
    
    if plot:
        
        # Plotting the ROC curve using fig, ax
        fig, ax = plt.subplots(figsize=(4,4))
        ax.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True)
        plt.show()
    
        sns.heatmap(matrix1, cmap = 'bwr')
        plt.show()
        sns.heatmap(matrix1, cmap = 'bwr')
        plt.show()
        
    return roc_auc



def ensure_get_score_size(dict_input, m_features):
    if len(dict_input.keys()) < m_features:
        print('ensured_get_score_size')
        lst = [0] * m_features
        dict_output = {f'f{i}': 0 for i in range(m_features+1)}

        for i, key in enumerate(list(dict_input.keys())):
            lst[int(key[1:])] = dict_input[key]
    else:
        dict_output = dict_input
                                
    return dict_output



import os
from datetime import datetime


def create_folder_with_datetime(absolute_path):
    # Ensure the provided path exists; if not, create it
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    # Get the current date and time
    now = datetime.now()
    folder_name = "results_" + now.strftime("%Y-%m-%d %H-%M-%S")

    # Combine the absolute path with the new folder name
    full_path = os.path.join(absolute_path, folder_name)

    # Create the new folder at the specified location
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Folder '{folder_name}' created at {full_path}!")
    else:
        print(f"Folder '{full_path}' already exists!")
        
    return full_path


def get_shap_martix(shap_values_comb_mean_dict, super_cols, s, n_features, shap_values_type = 'abs_mean_global'):
    shap_matrix = np.zeros((s, n_features))
    for column in super_cols:
        print(column)
    
        shap_temp = shap_values_comb_mean_dict[column][shap_values_type]
        
        shap_matrix[int(column.split('_')[0]), int(column.split('_')[1])] = shap_temp
        
    return shap_matrix



def get_metrics_all(xgboost_eval_dict, s, iterations_per_feature):

    r2_pd_all = pd.DataFrame(np.zeros(iterations_per_feature))
    mape_pd_all = pd.DataFrame(np.zeros(iterations_per_feature))
    mse_pd_all = pd.DataFrame(np.zeros(iterations_per_feature))
    
    for i in range(s):
        print('Feature: ', i)
        r2_i = []
        mse_i = []
        mape_i = []
        for iter_ in range(iterations_per_feature):
            # performance
            r2_i.append(xgboost_eval_dict[str(i)][iter_]['r2'])
            mse_i.append(xgboost_eval_dict[str(i)][iter_]['mse'])
            mape_i.append(xgboost_eval_dict[str(i)][iter_]['mape'])
            
            
        r2_pd_all[i] = r2_i
        mse_pd_all[i] = mse_i
        mape_pd_all[i] = mape_i
        
        
    
    r2_pd_all = r2_pd_all.melt(var_name = 'target_feature', value_name = 'r2')
    
    mse_pd_all = mse_pd_all.melt(var_name = 'target_feature', value_name = 'mse')
    
    mape_pd_all = mape_pd_all.melt(var_name = 'target_feature', value_name = 'mape')
    
    return r2_pd_all, mse_pd_all, mape_pd_all




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



def get_lrp_dict_filtered_pd(lrp_dict_filtered, pathway = 'PI3K'):
    
    lrp_dict_filtered_pd = pd.DataFrame()#lrp_dict_filtered)
    n = len(lrp_dict_filtered[pathway].keys())
    
    for index, (sample_name, data) in enumerate(lrp_dict_filtered[pathway].items()):
        #print(index+1,'/',n, sample_name)
        
        data_temp = add_edge_colmn(data[['LRP', 'source_gene', 'target_gene']].copy())
        data_temp['sample'] = sample_name
    
        lrp_dict_filtered_pd = pd.concat((lrp_dict_filtered_pd, data_temp))
        
    
    lrp_dict_filtered_pd = lrp_dict_filtered_pd.reset_index(drop=True)
    
    lrp_dict_filtered_pd_pivot = pd.pivot_table(lrp_dict_filtered_pd , index = 'edge', columns = 'sample', values = 'LRP')

    
    return lrp_dict_filtered_pd_pivot

    
def get_column_colors_from_clinical_df(df_clinical_features, df_to_clustermap):
    
    df_clinical_features_ = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(df_to_clustermap.columns)]
    df_clinical_features_ = df_clinical_features_.set_index(df_to_clustermap.columns)
    
    
    return df_clinical_features_
    
def map_subtypes_to_col_color(df_clinical_features_):
        
    triple_neg_index = (df_clinical_features_['HER2']=='Negative') & (df_clinical_features_['Progesterone_receptor']=='Negative') & (df_clinical_features_['Estrogen_receptor']=='Negative')
    df_clinical_features_['TNBC'] = "Positive"
    df_clinical_features_.loc[triple_neg_index, 'TNBC'] = "Negative" # it means that triple negative is RED
    
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
    column_colors_tnbc = df_clinical_features_['TNBC'].map(color_map)


    column_colors = [column_colors_her2, column_colors_er, column_colors_pro, column_colors_tnbc]
    return column_colors



    
def get_correlation_r(data, num_cols, method = 'pearson'):
             
    corrs = pg.pairwise_corr(data[num_cols], columns=num_cols, method=method)
    corrs = corrs.sort_values('p-unc').reset_index(drop=True)
    corrs = corrs.rename(columns = {'X':'source_gene', 'Y':'target_gene', 'p-unc':'p-val'})
    
    corrs = corrs[['source_gene', 'target_gene',  'r',  'p-val', 'power']]
    corrs = add_edge_colmn(corrs)
    corrs['test'] = method
    return corrs




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
            #print(cat_col, num_col, num_col,i, '/',n)
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














    
 
    