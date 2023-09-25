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
df = df.iloc[:, :20]
df_exp = feather.read_feather( os.path.join(path_to_data, 'data_to_model/CCE_expressions_to_model') , )


df = df.merge(df_exp.reset_index().iloc[:, :1], right_on = 'Tumor_Sample_Barcode', left_on = 'bcr_patient_barcode' , how = 'right')
df.pop('age_at_initial_pathologic_diagnosis')
df.pop('days_to_initial_pathologic_diagnosis')
#df.pop('days_to_initial_pathologic_diagnosis')

# %% sweetviz
import sweetviz as sv

my_report = sv.analyze(df)
my_report.show_html()


#%% safe 

df_clinical_features = df.set_index('bcr_patient_barcode')[['acronym','tumor_tissue_site']]
df_clinical_features.to_csv(os.path.join(path_to_data, 'data_to_model/CCE_clinical_features.csv'))



# %% other
df.groupby('acronym').nunique()['tumor_tissue_site']
df.groupby('tumor_tissue_site').nunique()['acronym']
a  = df.iloc[:30, :].T






























