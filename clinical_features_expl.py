# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 20:13:20 2023

@author: d07321ow
"""


import pyarrow.feather as feather

import pandas as pd

import numpy as np
import sys
import os
import seaborn as sns

# %% laod data

#path_to_data = '/home/owysocki/Documents/KI_dataset/data_to_model'
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset'
#path_to_data = 'KI_dataset/data_to_model'

df = pd.read_csv(os.path.join(path_to_data, 'pancantlas_clinical_PANCAN_patient_with_followup.tsv'), sep = '\t' , encoding='windows-1252')
df = df.rename(columns = {'lab_proc_her2_neu_immunohistochemistry_receptor_status':'HER2',
                          'breast_carcinoma_estrogen_receptor_status':'Estrogen_receptor',
                              'breast_carcinoma_progesterone_receptor_status':'Progesterone_receptor'})


df = df[['bcr_patient_barcode', 'acronym','tumor_tissue_site', 'HER2', 'Estrogen_receptor' ,'Progesterone_receptor']]
df_exp = feather.read_feather( os.path.join(path_to_data, 'data_to_model/CCE_expressions_to_model') , )
df_res = pd.read_csv( os.path.join(path_to_data, 'CCE_response_paper.csv')  )


df = df.merge(df_exp.reset_index().iloc[:, :1], right_on = 'Tumor_Sample_Barcode', left_on = 'bcr_patient_barcode' , how = 'right')
df = df.merge(df_res, right_on = 'id', left_on = 'bcr_patient_barcode' , how = 'right')

#df.pop('age_at_initial_pathologic_diagnosis')
#df.pop('days_to_initial_pathologic_diagnosis')
#df.pop('days_to_initial_pathologic_diagnosis')


# %% sweetviz
import sweetviz as sv

my_report = sv.analyze(df)
my_report.show_html()


#%% safe 

df_clinical_features = df.set_index('bcr_patient_barcode')[['acronym','tumor_tissue_site','response', 'HER2', 'Estrogen_receptor' ,'Progesterone_receptor']]
df_clinical_features.to_csv(os.path.join(path_to_data, 'data_to_model/CCE_clinical_features.csv'))



# %% other
df.groupby('acronym').nunique()['tumor_tissue_site']
df.groupby('tumor_tissue_site').nunique()['acronym']
a  = df.iloc[:30, :].T










a = df_clinical_features[df_clinical_features['acronym']=='BRCA']

my_report = sv.analyze(a)
my_report.show_html()



samples = a[(a['HER2'] == 'Negative') & (a['Estrogen_receptor'] == 'Negative')].reset_index()['bcr_patient_barcode']
samples = a[(a['HER2'] == 'Positive') & (a['Estrogen_receptor'] == 'Negative')].reset_index()['bcr_patient_barcode']

samples = a[(a['HER2'] == 'Negative') & (a['Estrogen_receptor'] == 'Positive')].reset_index()['bcr_patient_barcode']





a = df_clinical_features[df_clinical_features['acronym']=='BRCA']

b = a.groupby(['HER2', 'Estrogen_receptor','Progesterone_receptor']).tail()
c = b[b['HER2'] == 'Positive']
samples = c.reset_index()['bcr_patient_barcode']







