# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:49:24 2023

@author: d07321ow
"""


# %% READ DATA


import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import numpy as np
import sys
import os
import json
import pickle
# %%


path_to_res = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\Synthetic_data\results_2023-10-03 14-02-35'
    
with open(os.path.join(path_to_res, 'shaps.pkl'), 'rb') as file:
    xgboost_shaps_dict = pickle.load(file) 
# MODELS
#with open(os.path.join(path_to_save, 'models.pkl'), 'wb') as file:
#    pickle.dump(xgboost_models_dict, file)
    

with open(os.path.join(path_to_res, 'evals.json'), 'r') as file:
    xgboost_eval_dict = json.load(file)    
    
shap_matrix_abs = pd.read_csv(os.path.join(path_to_res , 'shap_matrix_abs.csv'), index_col = 0).values   
shap_matrix = pd.read_csv(os.path.join(path_to_res , 'shap_matrix.csv'), index_col = 0)   .values
gain_matrix = pd.read_csv(os.path.join(path_to_res , 'gain_matrix.csv'), index_col = 0)   .values

info = pd.read_csv(os.path.join(path_to_res , 'info.csv'), index_col = 0)

m_features, iterations_per_feature = info['m_features'][0], info['iterations_per_feature: '][0]

# %% Evaluate AUC
f.get_auc_artificial_homogeneous(shap_matrix_abs, plot = True, title= 'XGBOOST SHAP')
f.get_auc_artificial_homogeneous(gain_matrix, plot = True, title= 'XGBOOST total gain')


# %% investigate r2

r2_pd_all, mse_pd_all, mape_pd_all = f.get_metrics_all(xgboost_eval_dict, s, iterations_per_feature)


# %% plot r2
fig,ax = plt.subplots(figsize = (7,5))
sns.boxplot(data = r2_pd_all, x='target_feature', y='r2', color = 'white', showfliers=False,ax=ax)
sns.stripplot(data = r2_pd_all, x='target_feature', y='r2',ax=ax, color = 'k')
ax.axhline(1, linestyle = '--')
ax.axhline(.7, linestyle = '--')
