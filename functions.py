# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:27:47 2023

@author: d07321ow
"""


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