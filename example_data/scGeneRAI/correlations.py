# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:16:25 2023

@author: d07321ow
"""

import pyarrow.feather as feather
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

df_fus = feather.read_feather(os.path.join(path_to_data, 'CCE_fusions_to_model') )
df_mut = feather.read_feather( os.path.join(path_to_data, 'CCE_mutations_to_model') ,)
df_amp = feather.read_feather( os.path.join(path_to_data, 'CCE_amplifications_to_model') )
df_del = feather.read_feather( os.path.join(path_to_data, 'CCE_deletions_to_model') , )
df_exp = feather.read_feather( os.path.join(path_to_data, 'CCE_expressions_to_model') , )

df_clinical_features = pd.read_csv( os.path.join(path_to_data, 'CCE_clinical_features.csv') )


# %% corrletions


