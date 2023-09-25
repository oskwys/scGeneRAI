# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 07:47:38 2023

@author: d07321ow
"""


import pickle
import pandas as pd

import numpy as np
import sys
import os
import seaborn as sns

from scGeneRAI import scGeneRAI
import pyarrow.feather as feather

# %% load data


#path_to_data = '/home/owysocki/Documents/KI_dataset/data_to_model'
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'
#path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_model'
#path_to_data = 'KI_dataset/data_to_model'

#df_fus = feather.read_feather(os.path.join(path_to_data, 'CCE_fusions_to_model') )
#df_mut = feather.read_feather( os.path.join(path_to_data, 'CCE_mutations_to_model') ,)
#df_amp = feather.read_feather( os.path.join(path_to_data, 'CCE_amplifications_to_model') )
#df_del = feather.read_feather( os.path.join(path_to_data, 'CCE_deletions_to_model') , )
df_exp = feather.read_feather( os.path.join(path_to_data, 'CCE_expressions_to_model') , )



df_fus = pd.read_csv(os.path.join(path_to_data, 'CCE_fusions_to_model.csv') ,index_col = 0)
df_mut = pd.read_csv( os.path.join(path_to_data, 'CCE_mutations_to_model.csv') ,index_col = 0)
df_amp = pd.read_csv( os.path.join(path_to_data, 'CCE_amplifications_to_model.csv'),index_col = 0 )
df_del = pd.read_csv( os.path.join(path_to_data, 'CCE_deletions_to_model.csv') ,index_col = 0 )
#df_exp = pd.read_csv( os.path.join(path_to_data, 'CCE_expressions_to_model.csv') ,index_col = 0 )

#df_clinical_features = pd.read_csv( os.path.join(path_to_data, 'CCE_clinical_features.csv') )

# %% preprocess data
df_exp = df_exp.apply(lambda x: np.log(x + 1))
df_exp_stand = (df_exp-df_exp.mean(axis=0))/df_exp.std(axis=0)

df_mut_scale = (df_mut-df_mut.min(axis=0))/df_mut.max(axis=0)
df_fus_scale = (df_fus-df_fus.min(axis=0))/df_fus.max(axis=0)

df_amp[df_amp==2] =1


# %% data to model

data = pd.concat((df_exp_stand, df_mut_scale, df_amp, df_del, df_fus_scale), axis = 1)

# %%

device = 'cuda'


file_name = 'model_BRCA_batch5_nepochs500_depth2_lr02.pkl'
path_to_model = r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI'


data_temp = data.copy()

path = os.path.join(path_to_model, file_name)#.replace('.pkl','')


with open(path, 'rb') as file:
    model = pickle.load(file)
    print(f'Object successfully loaded from "{file_name}"')
    
    
    
path_to_save_lrp = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP'

files = os.listdir(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP\results')

samples = [file.split('_')[2] for file in files[1:]]

# %%
#preds = model.predict_networks(data_temp.iloc[30:900,:], descriptors = None, LRPau = True, remove_descriptors = True, device_name = device, PATH = path_to_save_lrp)
preds = model.predict_networks(data_temp.iloc[931:,:], descriptors = None, LRPau = True, remove_descriptors = True, device_name = device, PATH = path_to_save_lrp)


preds = model.predict_networks(data_temp[~data_temp.index.isin(samples)].iloc[700:720,:], descriptors = None, LRPau = True, remove_descriptors = True, device_name = device, PATH = path_to_save_lrp)




# %%






data_temp[~data_temp.index.isin(samples)]









    