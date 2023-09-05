# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 22:59:47 2023

@author: d07321ow
"""


import pickle
import pandas as pd

import numpy as np
import sys
import os
import seaborn as sns


# %% load data

#path_to_data = '/home/owysocki/Documents/KI_dataset/data_to_model'
#path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_model'
path_to_data = 'KI_dataset/data_to_model'

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


# %% data to model
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

device = 'cuda'
# %% define samples


#subtype = 'LUAD'
subtypes = ['BRCA', 'LGG', 'UCEC', 'LUAD', 'HNSC', 'PRAD', 'LUSC']#, 'PRAD', 'LUSC', 'STAD', 'COAD',

      # 'SKCM', 'CESC', 'SARC', 'OV', 'PAAD', 'ESCA', 'GBM', 'READ', 'UVM',
      # 'UCS', 'CHOL']
      
temp = df_clinical_features.groupby('acronym').head(100)
temp = temp[temp['acronym'].isin(subtypes)]

samples = temp['bcr_patient_barcode'].values

# %% models
path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\results_model_all_batch5_nepochs300_depth2_lr02'
files = []#os.listdir(path)
for file in os.listdir(path):
    if file.endswith(".pkl"):
        files.append(file)


data_temp = data[data.index.isin(samples)]

for i, file_name in enumerate(files):
        print('\n ___________ NEW MODEL ___________ \n')
        print(i, file)
                
        try:
                    
            with open(os.path.join(path , file_name), 'rb') as file:
                model = pickle.load(file)
                print(f'Object successfully loaded from "{file_name}"')
                
            preds = model.predict_networks(data_temp, descriptors = None, LRPau = True, remove_descriptors = True, device_name = device, PATH = 'results_{}'.format(i))
        except:
            print('Failed')
            
            
            
            
            
            
            