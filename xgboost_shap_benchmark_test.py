# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 21:58:43 2023

@author: d07321ow
"""



import xgboost as xgb
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error , r2_score
from xgboost import plot_importance

import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import numpy as np
import sys
import os

#from scGeneRAI import scGeneRAI
import functions as f
import pyarrow.feather as feather
from datetime import datetime


import importlib, sys
importlib.reload(f)
#%matplotlib inline 
import random
# %%


#path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\Synthetic_data\keylPatientlevelProteomicNetwork2022'

path = '/home/d07321ow/scratch/scGeneRAI/data/keylPatientlevelProteomicNetwork2022'
data = pd.read_csv( os.path.join(path, 'artificial_homogeneous.csv'), index_col = 0 )

#data = data.sample(100, random_state =1).reset_index(drop=True)

# %% define random sets of features
import random
import itertools

n_features = len(data.columns)
m_times = 3
m_features = 5

s = n_features

features_sampled_dict = {}
for i in range(s):
    print(i)
    feature_list = list(np.arange(0,n_features))
    feature_list.pop(i)
    
    feature_list_sampled_i = []
    for m_time in range(m_times):
        random.seed((m_time*10+1)*(i*100+5)*2)
        
        random.shuffle(feature_list)
        
        for iter_ in range(0, int(np.floor(n_features / m_features))):
            feature_list_sampled  = feature_list[m_features * iter_ : m_features * (iter_+1)]
            feature_list_sampled.sort()
            feature_list_sampled_i.append(feature_list_sampled)
            
    features_sampled_dict[str(i)] = feature_list_sampled_i

iterations_per_feature = int(np.floor(n_features / m_features) * m_times)
print('iterations_per_feature: ', iterations_per_feature)
# %% train and get shap
print('train and get shap')
import warnings
warnings.filterwarnings("ignore")
start_time = datetime.now()

data_to_xg = data.copy().values

xgboost_models_dict = {}
xgboost_shaps_dict = {}
xgboost_eval_dict = {}
for i in range(s):
    print('\n\n Feature: ', i, '/', s)
    xgboost_models_dict[str(i)] = {}
    xgboost_shaps_dict[str(i)] = {}
    xgboost_eval_dict[str(i)] = {}
    
    for iter_ in range(iterations_per_feature):
        xgboost_models_dict[str(i)][iter_] = {}
        xgboost_shaps_dict[str(i)][iter_] = {}
        xgboost_eval_dict[str(i)][iter_] = {}
        
        print('Iteration: ', iter_, '/', iterations_per_feature)
        feature_list = features_sampled_dict[str(i)][iter_]

        X = data_to_xg[:,feature_list]
        y = data_to_xg[:, i]

        
        # Convert data to DMatrix format
        dtrain = xgb.DMatrix(X, label=y)
        
        # Set parameters for GPU training
        param = {
            'max_depth': 10,
            'eta': 0.3,
            'objective': 'reg:squarederror',  # regression task
            'tree_method' : "gpu_hist", 'device' : "cuda"
        }
        num_round = 10  # Number of boosting rounds
        
        # Train the model
        model = xgb.train(param, dtrain, num_round)
        
        # evaluate
        preds = model.predict(dtrain)
        mse = mean_squared_error(y, preds)
        mape = mean_absolute_percentage_error(y, preds)
        r2 = r2_score(y, preds)
        print('r2:', np.round(r2,3))
        
        # save to dict
        xgboost_models_dict[str(i)][iter_]['model'] = model
        
        xgboost_eval_dict[str(i)][iter_]['mse'] = mse
        xgboost_eval_dict[str(i)][iter_]['mape'] = mape
        xgboost_eval_dict[str(i)][iter_]['r2'] = r2
        
        xgboost_eval_dict[str(i)][iter_]['total_gain'] = f.ensure_get_score_size(model.get_score(importance_type='total_gain'), m_features)
        xgboost_eval_dict[str(i)][iter_]['gain'] = f.ensure_get_score_size(model.get_score(importance_type='gain'), m_features)
        xgboost_eval_dict[str(i)][iter_]['weight'] = f.ensure_get_score_size(model.get_score(importance_type='weight'), m_features)
        xgboost_eval_dict[str(i)][iter_]['total_cover'] = f.ensure_get_score_size(model.get_score(importance_type='total_cover'), m_features)
        xgboost_eval_dict[str(i)][iter_]['cover'] = f.ensure_get_score_size(model.get_score(importance_type='cover'), m_features)
            
        
        
        model.set_param({"device": "cuda:0"})
        shap_values = model.predict(dtrain, pred_contribs=True)[:,:-1]
        
        
        # Get SHAP
        #explainer = shap.TreeExplainer(model, X)
        #shap_values = explainer(X).values
        
        # save feature importance
        feature_list_str = [str(i) + '_' + str(feature) + '_' + str(iter_) for feature in feature_list]
        
        xgboost_shaps_dict[str(i)][iter_]['shap_values'] = shap_values
        xgboost_shaps_dict[str(i)][iter_]['feature_list_str'] = feature_list_str
end_time = datetime.now() 

print(end_time - start_time)
    
    
# %% shap_values_dict
print('shap_values_dict')
shap_values_dict = {}
cols_all = []

r2_threshold = 0.5

for i in range(s):
    print('Feature: ', i)
    shap_values_dict[str(i)] = {}
    shaps_temp_i = np.zeros((data.shape[0], iterations_per_feature*m_features))
    cols = []
    for iter_ in range(iterations_per_feature):
        
        r2 = xgboost_eval_dict[str(i)][iter_]['r2'] 
        
        shaps_temp = xgboost_shaps_dict[str(i)][iter_]['shap_values']
        
        if r2 < r2_threshold:
            shaps_temp = shaps_temp * np.nan
            print(r2)
            print(shaps_temp.shape)
        shaps_temp_i[:, iter_*m_features: (iter_+1)*m_features] = shaps_temp
        
        cols += xgboost_shaps_dict[str(i)][iter_]['feature_list_str']
        cols_all += xgboost_shaps_dict[str(i)][iter_]['feature_list_str']
        
    shap_values_dict[str(i)]['shap_values'] = shaps_temp_i
    shap_values_dict[str(i)]['feature_list_str'] = cols
    shap_values_dict[str(i)]['shap_values_pd'] = pd.DataFrame(data = shaps_temp_i, columns = cols)
    #print(pd.DataFrame(data = shaps_temp_i, columns = cols).shape)
    #print(pd.DataFrame(data = shaps_temp_i, columns = cols).dropna(axis=1).shape)


# %% shap_values_comb_dict
print('shap_values_comb_dict')
shap_values_comb_dict = {}

super_cols = []
for i in range(s):
    cols = shap_values_dict[str(i)]['feature_list_str'] 
    #cols = ['_'.join(column.split('_')[:-1]) for column in cols]
    cols.sort()
    print('Feature: ', i)
    for j in range(0,iterations_per_feature*m_features, m_times):
        
        cols_to_select = cols[j:j+m_times]
        super_col = '_'.join(cols_to_select[0].split('_')[:-1])
        shap_values_comb_dict[super_col] = shap_values_dict[str(i)]['shap_values_pd'].loc[:, cols_to_select].values
        super_cols.append(super_col)
        
# %% shap_values_comb_mean_dict
print('shap_values_comb_mean_dict')
shap_values_comb_mean_dict = {}

for column in super_cols:
    print(column)

    shap_values_comb_mean_dict[column]={}
    shap_values_comb_mean_dict[column]['mean_local'] = np.round(np.nanmean(shap_values_comb_dict[column],axis = 1), 5)
    shap_values_comb_mean_dict[column]['abs_mean_local'] = np.round(np.nanmean(np.abs(shap_values_comb_dict[column]),axis = 1), 5)
    shap_values_comb_mean_dict[column]['mean_global'] = np.round(np.nanmean(shap_values_comb_dict[column]), 5)
    shap_values_comb_mean_dict[column]['abs_mean_global'] = np.round(np.nanmean(np.abs(shap_values_comb_dict[column] / np.nanmax(np.abs(shap_values_comb_dict[column])))), 5)




# %% edges SHAP
shap_matrix_abs = f.get_shap_martix(shap_values_comb_mean_dict, super_cols, s, n_features,shap_values_type = 'abs_mean_global')
shap_matrix = f.get_shap_martix(shap_values_comb_mean_dict, super_cols, s, n_features,shap_values_type = 'mean_global')




sns.heatmap(shap_matrix, cmap = 'jet')
plt.show()

sns.heatmap(shap_matrix, cmap = 'jet', vmin = -1, vmax = 1)
plt.show()

sns.heatmap(shap_matrix_abs, cmap = 'jet', vmin = 0)
plt.show()

sns.heatmap(shap_matrix_abs, cmap = 'jet', vmin = 0)
plt.show()

shap_matrix_pd = pd.DataFrame(shap_matrix, columns = data.columns, index = data.columns[:s]).T

shap_edges = shap_matrix_pd.reset_index().melt(id_vars='index', value_vars=shap_matrix_pd.columns)
shap_edges = shap_edges.rename(columns = {'index':'source','variable':'target', 'value':'weight'})



# %% SAVE

print('SAVE !')

import json
import pickle

path_to_save = '/home/d07321ow/scratch/results/'

path_to_save = f.create_folder_with_datetime(path_to_save)


# Save dictionary
with open(os.path.join(path_to_save, 'shaps.pkl'), 'wb') as file:
    pickle.dump(xgboost_shaps_dict, file)
    

# Save dictionary
with open(os.path.join(path_to_save, 'models.pkl'), 'wb') as file:
    pickle.dump(xgboost_models_dict, file)
    
# Save dictionary
with open(os.path.join(path_to_save, 'evals.json'), 'w') as file:
    json.dump(xgboost_eval_dict, file)
    
pd.DataFrame(shap_matrix_abs).to_csv(os.path.join(path_to_save , 'shap_matrix_abs.csv'))
pd.DataFrame(shap_matrix).to_csv(os.path.join(path_to_save , 'shap_matrix.csv'))

print('done')