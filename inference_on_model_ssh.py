# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:59:17 2023

@author: d07321ow
"""

import pickle
import pandas as pd

import numpy as np
import sys
import os
import seaborn as sns
import functions as f

from scGeneRAI import scGeneRAI
import pyarrow.feather as feather

# %% load data

#path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'
path_to_data = '/home/d07321ow/scratch/scGeneRAI/data/data_BRCA'
data, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(path_to_data)

print('DATA shape ', data.shape)
# %%

device = 'cuda'

file_name = 'model_BRCA_batch5_nepochs500_depth2_lr02.pkl'
#path_to_model = r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI'
path_to_model = '/home/d07321ow/scratch/results_LRP_BRCA'

data_temp = data.copy()

path = os.path.join(path_to_model, file_name)#.replace('.pkl','')


with open(path, 'rb') as file:
    model = pickle.load(file)
    print(f'Object successfully loaded from "{file_name}"')
    
    
    
#path_to_save_lrp = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\LRP'
path_to_save_lrp = '/home/d07321ow/scratch/results_LRP_BRCA'

files = os.listdir(path_to_save_lrp)

samples = [file.split('_')[2] for file in files]

# %%
#preds = model.predict_networks(data_temp.iloc[30:900,:], descriptors = None, LRPau = True, remove_descriptors = True, device_name = device, PATH = path_to_save_lrp)
#preds = model.predict_networks(data_temp.iloc[0:10,:], descriptors = None, LRPau = True, remove_descriptors = True, device_name = device, PATH = path_to_save_lrp)


preds = model.predict_networks(data_temp[~data_temp.index.isin(samples)], descriptors = None, LRPau = True, remove_descriptors = True, device_name = device, PATH = path_to_save_lrp)



    