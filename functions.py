# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:27:47 2023

@author: d07321ow
"""

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

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


from typing import List

def get_lrp_files(path_to_lrp_results: str) -> List[str]:
    """
    Retrieves a list of files starting with "LRP" from the specified directory.
    Args:
        path_to_lrp_results (str): The path to the directory containing LRP result files.
    Returns:
        list: A list of filenames that start with "LRP".
    Raises:
        FileNotFoundError: If the specified directory does not exist or if no LRP files are found in the directory.
    """
    if not os.path.exists(path_to_lrp_results):
        raise FileNotFoundError(f"The directory {path_to_lrp_results} does not exist.")

    lrp_files = []
    for file in os.listdir(path_to_lrp_results):
        if file.startswith("LRP"):
            lrp_files.append(file)

    if not lrp_files:
        raise FileNotFoundError(
            f"No LRP files found in the directory {path_to_lrp_results}."
            )
    
    return lrp_files


def load_lrp_file(file_name: str, path_to_lrp_results: str, lrp_dict: dict) -> None:
    """
    Loads an LRP (Layer-wise Relevance Propagation) result file, processes it, and stores it in a dictionary.

    Args:
        file_name (str): The name of the LRP result file. The filename is expected to have at least three parts separated by underscores.
        path_to_lrp_results (str): The directory path where the LRP result files are stored.
        lrp_dict (dict): A dictionary to store the processed LRP data, with the sample name as the key.

    Raises:
        ValueError: If the filename does not have the expected format (at least three parts separated by underscores).

    Returns:
        None: The function updates the lrp_dict in place.
    """
    file_parts = file_name.split("_")
    if len(file_parts) < 3:
        raise ValueError(f"Filename {file_name} does not have the expected format.")
    sample_name = file_parts[2]
    print(file_name)
    data_temp = pd.read_pickle(
        os.path.join(path_to_lrp_results, file_name),
        compression="infer",
        storage_options=None,
    )
    data_temp = remove_same_source_target(data_temp)
    lrp_dict[sample_name] = data_temp


def load_lrp_data(lrp_files: list[str], path_to_lrp_results: str) -> dict:
    """
    Loads LRP (Layer-wise Relevance Propagation) data from a list of files and stores it in a dictionary.
    Args:
        lrp_files (list[str]): A list of filenames containing LRP data.
        path_to_lrp_results (str): The path to the directory where the LRP result files are located.
    Returns:
        dict: A dictionary containing the loaded LRP data.
    """
    lrp_dict = {}
    n = len(lrp_files)

    for i in range(n):
        load_lrp_file(lrp_files[i], path_to_lrp_results, lrp_dict)

    return lrp_dict


def filter_and_sort_data(data_temp: pd.DataFrame, node_type: str = None, topn: int = 100) -> pd.DataFrame:
    """
    Filters and sorts a DataFrame based on the presence of a specified node type in the 
    'source_gene' or 'target_gene' columns, and returns the top N rows sorted by the 
    'LRP' column in descending order.

    Args:
        data_temp (pd.DataFrame): The input DataFrame containing the data to be filtered 
                                  and sorted.
        node_type (str, optional): The node type to filter the 'source_gene' and 'target_gene' 
                                   columns by. Defaults to None.
        topn (int): The number of top rows to return after sorting.

    Returns:
        pd.DataFrame: The filtered and sorted DataFrame with an additional edge column.
    """
    if node_type is not None:
        data_temp = data_temp[
            (data_temp["source_gene"].str.contains(node_type)) |
            (data_temp["target_gene"].str.contains(node_type))
        ]
    data_temp = data_temp.sort_values("LRP", ascending=False)
    data_temp = data_temp.iloc[:topn, :]
    data_temp = add_edge_column(data_temp)
    return data_temp



def get_samples_with_lrp(path_to_lrp_results, starts_with="LRP_"):
    """
    Retrieves sample identifiers from filenames in a specified directory that start with a given prefix.

    Args:
        path_to_lrp_results (str): The path to the directory containing LRP result files.
        starts_with (str, optional): The prefix that filenames should start with. Defaults to "LRP_".

    Returns:
        list: A list of sample identifiers extracted from the filenames.
    """
    files = [f for f in os.listdir(path_to_lrp_results) if f.startswith(starts_with)]
    print(files)
    samples = [file.split("_")[2] for file in files]
    return samples

def remove_same_source_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where 'source_gene' is the same as 'target_gene'.

    Args:
        data (pd.DataFrame): The input DataFrame containing 'source_gene' and 'target_gene' columns.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if 'source_gene' not in data.columns or 'target_gene' not in data.columns:
        raise ValueError("DataFrame must contain 'source_gene' and 'target_gene' columns.")

    data = data[data['source_gene'] != data['target_gene']]
    return data


def add_edge_column(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'edge' and 'edge_type' columns to the given DataFrame.
    The 'edge' column is created by concatenating the 'source_gene' and 'target_gene' columns with ' - '.
    The 'edge_type' column is created by splitting the 'source_gene' and 'target_gene' columns on '_',
    taking the second part, sorting them, and then joining them with '-'.
    Parameters:
    edges (pd.DataFrame): A DataFrame containing 'source_gene' and 'target_gene' columns.
    Returns:
    pd.DataFrame: The input DataFrame with added 'edge' and 'edge_type' columns.
    """
    edges['edge'] = edges['source_gene'] + ' - ' + edges['target_gene']

    edges['edge_type'] = (
                edges['source_gene'].str.split('_', expand=True).iloc[:, 1] + ' - ' + edges['target_gene'].str.split(
            '_', expand=True).iloc[:, 1]).str.split(' - ').apply(np.sort).str.join('-')

    return edges


def add_to_main_df_topn(df_topn: pd.DataFrame, sample_name: str, data_temp: pd.DataFrame) -> None:
    """
    Adds the 'edge' values from the data_temp DataFrame to the df_topn DataFrame under the column named sample_name.

    Parameters:
    df_topn (pd.DataFrame): The main DataFrame to which the 'edge' values will be added.
    sample_name (str): The name of the column in df_topn where the 'edge' values will be stored.
    data_temp (pd.DataFrame): The DataFrame containing the 'edge' values to be added to df_topn.

    Returns:
    None
    """
    df_topn[sample_name] = data_temp['edge'].values

def count_unique_edges_in_df_topn(df_topn: pd.DataFrame):
    """
    Count unique edges in the given DataFrame and return a DataFrame with the counts.
    Parameters:
    df_topn (pd.DataFrame): A DataFrame containing edges.
    Returns:
    pd.DataFrame: A DataFrame with two columns: 'edge' and 'count', where 'edge' 
                  represents the unique edges and 'count' represents their respective counts.
    """
    unique_edges, unique_edges_count = np.unique(
        df_topn.values.ravel(), return_counts=True
    )

    unique_edges_df = pd.DataFrame(
        [unique_edges, unique_edges_count]
    ).T  # , columns = )
    unique_edges_df.columns = ["edge", "count"]
    return unique_edges_df














def get_input_data(path_to_data):
    df_fus = pd.read_csv(os.path.join(path_to_data, 'CCE_fusions_to_model.csv'), index_col=0)
    df_mut = pd.read_csv(os.path.join(path_to_data, 'CCE_mutations_to_model.csv'), index_col=0)
    df_amp = pd.read_csv(os.path.join(path_to_data, 'CCE_amplifications_to_model.csv'), index_col=0)
    df_del = pd.read_csv(os.path.join(path_to_data, 'CCE_deletions_to_model.csv'), index_col=0)

    df_exp = feather.read_feather(os.path.join(path_to_data, 'CCE_expressions_to_model'), )

    df_clinical_features = pd.read_csv(os.path.join(path_to_data, 'CCE_clinical_features.csv'))

    df_exp = df_exp.apply(lambda x: np.log(x + 1))
    df_exp_stand = (df_exp - df_exp.mean(axis=0)) / df_exp.std(axis=0)

    df_mut_scale = (df_mut - df_mut.min(axis=0)) / df_mut.max(axis=0)
    df_fus_scale = (df_fus - df_fus.min(axis=0)) / df_fus.max(axis=0)

    df_amp[df_amp == 2] = 1

    data_to_model = pd.concat((df_exp_stand, df_mut_scale, df_amp, df_del, df_fus_scale), axis=1)

    return data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features


def get_samples_by_group(df_clinical_features):
    # df_clinical_features = df_clinical_features_all[df_clinical_features_all['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)

    # Estrogen_receptor
    samples_estrogen_receptor_pos = df_clinical_features.loc[
        df_clinical_features['Estrogen_receptor'] == 'Positive', 'bcr_patient_barcode'].to_list()
    samples_estrogen_receptor_neg = df_clinical_features.loc[
        df_clinical_features['Estrogen_receptor'] == 'Negative', 'bcr_patient_barcode'].to_list()

    # HER2
    samples_her2_pos = df_clinical_features.loc[
        df_clinical_features['HER2'] == 'Positive', 'bcr_patient_barcode'].to_list()
    samples_her2_neg = df_clinical_features.loc[
        df_clinical_features['HER2'] == 'Negative', 'bcr_patient_barcode'].to_list()

    # Progesterone_receptor
    samples_progesterone_receptor_pos = df_clinical_features.loc[
        df_clinical_features['Progesterone_receptor'] == 'Positive', 'bcr_patient_barcode'].to_list()
    samples_progesterone_receptor_neg = df_clinical_features.loc[
        df_clinical_features['Progesterone_receptor'] == 'Negative', 'bcr_patient_barcode'].to_list()

    # triple negative - TNBC
    triple_neg_index = (df_clinical_features['HER2'] == 'Negative') & (
                df_clinical_features['Progesterone_receptor'] == 'Negative') & (
                                   df_clinical_features['Estrogen_receptor'] == 'Negative')
    samples_triple_neg = df_clinical_features.loc[triple_neg_index, 'bcr_patient_barcode'].to_list()
    samples_no_triple_neg = df_clinical_features.loc[-triple_neg_index, 'bcr_patient_barcode'].to_list()

    # triple negative - TNBC
    cluster2_neg_index = (df_clinical_features['cluster2'] == 0)
    samples_cluster2_neg = df_clinical_features.loc[cluster2_neg_index, 'bcr_patient_barcode'].to_list()
    samples_cluster2_pos = df_clinical_features.loc[-cluster2_neg_index, 'bcr_patient_barcode'].to_list()

    samples_groups = {}
    samples_groups['her2'] = {'her2_pos': samples_her2_pos, 'her2_neg': samples_her2_neg}
    samples_groups['progesterone_receptor'] = {'progesterone_receptor_pos': samples_progesterone_receptor_pos,
                                               'progesterone_receptor_neg': samples_progesterone_receptor_neg}
    samples_groups['estrogen_receptor'] = {'estrogen_receptor_pos': samples_estrogen_receptor_pos,
                                           'estrogen_receptor_neg': samples_estrogen_receptor_neg}
    samples_groups['TNBC'] = {'TNBC': samples_triple_neg, 'no_TNBC': samples_no_triple_neg}
    samples_groups['cluster2'] = {'cluster2_pos': samples_cluster2_pos, 'cluster2_neg': samples_cluster2_neg}

    return samples_groups


from sklearn.metrics import roc_curve, auc


def get_auc_artificial_homogeneous(corrs, plot=True, title=''):
    matrix2 = corrs

    # Groubnd truth homo
    matrix1 = np.zeros((32, 32))
    matrix1[:8, :8] = 1
    matrix1[8:16, 8:16] = 1
    matrix1[16:24, 16:24] = 1
    matrix1[24:, 24:] = 1
    sns.heatmap(matrix1, cmap='bwr')

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
        fig, ax = plt.subplots(figsize=(4, 4))
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

        sns.heatmap(matrix1, cmap='bwr')
        plt.show()
        sns.heatmap(matrix1, cmap='bwr')
        plt.show()

    return roc_auc


def ensure_get_score_size(dict_input, m_features):
    if len(dict_input.keys()) < m_features:
        print('ensured_get_score_size')
        lst = [0] * m_features
        dict_output = {f'f{i}': 0 for i in range(m_features + 1)}

        for i, key in enumerate(list(dict_input.keys())):
            lst[int(key[1:])] = dict_input[key]
    else:
        dict_output = dict_input

    return dict_output


import os
from datetime import datetime




def get_node_colors(network):
    colors = pd.Series(list(network.nodes))
    colors[-colors.str.contains('mut')] = 'lightblue'
    colors[colors.str.contains('mut')] = 'red'
    colors[colors.str.contains('del')] = 'green'
    colors[colors.str.contains('amp')] = 'orange'
    colors[colors.str.contains('fus')] = 'magenta'
    colors[colors.str.contains('prot')] = 'black'

    return list(colors.values)


def get_genes_in_all_i(lrp_dict, top_n, n=5):
    genes = []
    for i in range(n):
        print(i)

        data_temp = lrp_dict[str(i)].iloc[:top_n, :]

        a = data_temp['source_gene'].to_list()
        b = data_temp['target_gene'].to_list()
        genes.append(a)
        genes.append(b)

    genes = [item for sublist in genes for item in sublist]
    genes = list(set(genes))
    genes.sort()

    return genes


def get_genes_related_to_gene_in_all_i(lrp_dict, gene, top_n, n=5):
    genes = []
    for i in range(n):
        df_temp = lrp_dict[str(i)]
        df_temp = df_temp[df_temp['source_gene'].str.contains(gene) | df_temp['target_gene'].str.contains(gene)].iloc[
                  :top_n, :]

        a = df_temp['source_gene'].to_list()
        b = df_temp['target_gene'].to_list()
        genes.append(a)
        genes.append(b)

    genes = [item for sublist in genes for item in sublist]
    genes = list(set(genes))
    genes.sort()

    return genes


def get_all_unique_genes(df_temp):
    genes = []
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
    edges_df.loc[edges_df['source_gene'].str.contains('_exp'), 'source_gene'] = edges_df.loc[
        edges_df['source_gene'].str.contains('_exp'), 'source_gene'].str.replace('_exp', '')
    edges_df.loc[edges_df['target_gene'].str.contains('_exp'), 'target_gene'] = edges_df.loc[
        edges_df['target_gene'].str.contains('_exp'), 'target_gene'].str.replace('_exp', '')

    return edges_df


# def plot_network_(edges, node_colors, top_n, subtype, i, file, path_to_save, layout=None, pos = None):
#     G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
#     degrees = np.array(list(nx.degree_centrality(G).values()))
#     degrees = degrees / np.max(degrees) * 500
#     #nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*100, node_size = network.degree)


#     if pos is not None:
#         print('using POS')
#     else:
#         if layout == None:
#             pos = nx.spring_layout(G)

#         elif layout== 'spectral':
#             pos = nx.spectral_layout(G)
#         elif layout== 'spectral':
#             pos = nx.spectral_layout(G)

#     #colors = get_node_colors(G)
#     colors = node_colors
#     widths = edges['LRP'] / edges['LRP'].max() * 10
#     edge_colors =cm.Greys((widths  - np.min(widths) )/ (np.max(widths) - np.min(widths)))

#     fig,ax = plt.subplots(figsize=  (15,15))
#     nx.draw(G, with_labels=True,
#             node_color=node_colors,
#             width = widths,
#             pos = pos, font_size = 6,
#             #cmap = colors,
#             edge_color = edge_colors ,
#             ax=ax,
#             #node_size = degrees
#             node_size = 100)
#     plt.savefig(os.path.join(path_to_save , 'network_{}_{}_{}_{}.svg'.format(file, top_n, subtype, i)), format = 'svg')
#     plt.savefig(os.path.join(path_to_save , 'network_{}_{}_{}_{}.png'.format(file, top_n, subtype, i)), dpi = 300)


def get_pivoted_heatmap(edges, genes):
    # filter edges
    edges = edges[edges['source_gene'].isin(genes) | edges['target_gene'].isin(genes)]

    template_ = pd.DataFrame(columns=genes, index=genes).fillna(0)
    for row in edges.iterrows():
        row = row[1]
        template_.loc[row['source_gene'], row['target_gene']] = row['LRP']

    return template_


def add_suffixes_to_genes(genes):
    gene_names = []
    suffixes = ['_amp', '_mut', '_del', '_fus', '_exp']
    for gene in genes:

        for suffix in suffixes:
            gene_names.append(gene + suffix)

    return gene_names




def get_lrp_dict_filtered_pd(lrp_dict_filtered, pathway='PI3K'):
    lrp_dict_filtered_pd = pd.DataFrame()  # lrp_dict_filtered)
    n = len(lrp_dict_filtered[pathway].keys())

    for index, (sample_name, data) in enumerate(lrp_dict_filtered[pathway].items()):
        # print(index+1,'/',n, sample_name)

        data_temp = add_edge_colmn(data[['LRP', 'source_gene', 'target_gene']].copy())
        data_temp['sample'] = sample_name

        lrp_dict_filtered_pd = pd.concat((lrp_dict_filtered_pd, data_temp))

    lrp_dict_filtered_pd = lrp_dict_filtered_pd.reset_index(drop=True)

    lrp_dict_filtered_pd_pivot = pd.pivot_table(lrp_dict_filtered_pd, index='edge', columns='sample', values='LRP')

    return lrp_dict_filtered_pd_pivot


def get_column_colors_from_clinical_df(df_clinical_features, df_to_clustermap):
    df_clinical_features_ = df_clinical_features[
        df_clinical_features['bcr_patient_barcode'].isin(df_to_clustermap.columns)]
    df_clinical_features_ = df_clinical_features_.set_index(df_to_clustermap.columns)

    return df_clinical_features_


def map_subtypes_to_col_color(df_clinical_features_):
    triple_neg_index = (df_clinical_features_['HER2'] == 'Negative') & (
                df_clinical_features_['Progesterone_receptor'] == 'Negative') & (
                                   df_clinical_features_['Estrogen_receptor'] == 'Negative')
    df_clinical_features_['TNBC'] = "Positive"
    df_clinical_features_.loc[triple_neg_index, 'TNBC'] = "Negative"  # it means that triple negative is RED

    color_map = {
        "Negative": "red",
        "Positive": "blue",
        "Equivocal": "yellow",
        'Indeterminate': 'gray',
        '[Not Evaluated]': 'white',
        '[Not Available]': 'white', }

    column_colors_her2 = df_clinical_features_['HER2'].map(color_map)
    column_colors_er = df_clinical_features_['Estrogen_receptor'].map(color_map)
    column_colors_pro = df_clinical_features_['Progesterone_receptor'].map(color_map)
    column_colors_tnbc = df_clinical_features_['TNBC'].map(color_map)

    column_colors = [column_colors_her2, column_colors_er, column_colors_pro, column_colors_tnbc]
    return column_colors


def get_correlation_r(data, num_cols, method='pearson'):
    corrs = pg.pairwise_corr(data[num_cols], columns=num_cols, method=method)
    corrs = corrs.sort_values('p-unc').reset_index(drop=True)
    corrs = corrs.rename(columns={'X': 'source_gene', 'Y': 'target_gene', 'p-unc': 'p-val'})

    corrs = corrs[['source_gene', 'target_gene', 'r', 'p-val', 'power']]
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
            # print(cat_col, num_col, num_col,i, '/',n)
            # u, pval = pointbiserialr(df_cat[cat_col].astype('category').cat.codes, df_num[num_col])
            pval, cles = get_mannwhitney_bootstrap(df_cat, df_num, cat_col, num_col, iters=iters)
            pval_matrix.loc[cat_col, num_col] = pval
            cles_matrix.loc[cat_col, num_col] = cles

    # Convert to float for plotting
    pval_matrix = pval_matrix.astype(float)
    cles_matrix = cles_matrix.astype(float)

    pval_matrix = pval_matrix.reset_index().melt(id_vars='index')
    pval_matrix = pval_matrix.rename(columns={'index': 'source_gene', 'variable': 'target_gene', 'value': 'p-val'})
    pval_matrix = add_edge_colmn(pval_matrix)

    cles_matrix = cles_matrix.reset_index().melt(id_vars='index')
    cles_matrix = cles_matrix.rename(columns={'index': 'source_gene', 'variable': 'target_gene', 'value': 'CLES'})
    cles_matrix = add_edge_colmn(cles_matrix)

    mwu_stats = pval_matrix.merge(cles_matrix)

    return pval_matrix, cles_matrix, mwu_stats


def color_mapper(input_list):
    # Define the color for each specific keyword
    keyword_to_color = {

        'del': 'green',  # assuming 'green' is the color for 'del'
        'amp': 'orange',  # assuming 'blue' is the color for 'amp'
        'fus': 'blue',  # assuming 'orange' is the color for 'fus'
        'mut': 'red',  # assuming 'red' is the color for 'mut'
    }

    # This dictionary will store the items with their corresponding color
    color_map = {}

    # Iterate through each item in the input list
    for item in input_list:
        # Default color if no keyword is matched, it's set to 'black' here
        color = 'gray'

        # Check each keyword to see if it exists in the item
        for keyword, assigned_color in keyword_to_color.items():
            if '_' + keyword in item:  # the underscore ensures we are checking, e.g., '_mut' and not just 'mut'
                color = assigned_color
                break  # if we found a keyword, we don't need to check the others for this item

        # Add the item and its color to the dictionary
        color_map[item] = color

    return color_map


import networkx as nx


def plot_network_(edges, path_to_save, layout=None, pos=None, title='', name_to_save='network', ax=None):
    G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    # nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*100, node_size = network.degree)

    if pos is not None:
        print('using POS')
    else:
        if layout == None:
            pos = nx.spring_layout(G)

        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        elif layout == 'kamada_kawai_layout':
            pos = nx.kamada_kawai_layout(G)

    widths = edges['LRP'] / edges['LRP'].max() * 10

    edges_from_G = [i[0] + ' - ' + i[1] for i in list(G.edges)]
    edge_colors = list(color_mapper(edges_from_G).values())

    node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}
    # Plot legend
    ls = list(node_color_mapper.keys())
    cl = list(node_color_mapper.values())
    for label, color in zip(ls, cl):
        ax.plot([], [], 'o', color=color, label=label)
    ax.legend(title='Nodes', loc='best')

    node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper)

    if ax == None:
        fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(G, with_labels=False,
            node_color=node_colors,
            width=widths,
            pos=pos,  # font_size=0,
            # cmap = colors,
            edge_color=edge_colors,
            ax=ax,
            # node_size = degrees,
            node_size=degrees_norm * 2000)

    labels = {node: node.split('_')[0] for node in list(G.nodes)}
    nx.draw_networkx_labels(G, pos, labels, font_size=25, ax=ax)

    ax.set_title(title)
    plt.tight_layout()

    # plt.savefig(os.path.join(path_to_save , name_to_save + '.svg'), format = 'svg')
    # plt.savefig(os.path.join(path_to_save , name_to_save + '.png'), dpi = 300)

