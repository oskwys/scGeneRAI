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

import numpy as np
import sys
import os

import pyarrow.feather as feather
from datetime import datetime





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
    
    samples_groups = {}
    samples_groups['her2'] = {'her2_pos':samples_her2_pos, 'her2_neg':samples_her2_neg}
    samples_groups['progesterone_receptor'] = {'progesterone_receptor_pos':samples_progesterone_receptor_pos, 'progesterone_receptor_neg':samples_progesterone_receptor_neg}
    samples_groups['estrogen_receptor'] = {'estrogen_receptor_pos':samples_estrogen_receptor_pos, 'estrogen_receptor_neg':samples_estrogen_receptor_neg}
    samples_groups['TNBC'] = {'TNBC':samples_triple_neg, 'no_TNBC':samples_no_triple_neg}
    
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


