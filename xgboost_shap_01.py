# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 21:25:56 2023

@author: d07321ow
"""
# %%

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

df_clinical_features = pd.read_csv( os.path.join(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_model', 'CCE_clinical_features.csv') )
df_clinical_features= df_clinical_features [df_clinical_features ['acronym']=='BRCA'].reset_index(drop=True)
# %% preprocess data
df_exp = df_exp.apply(lambda x: np.log(x + 1))
df_exp_stand = (df_exp-df_exp.mean(axis=0))/df_exp.std(axis=0)

df_mut_scale = (df_mut-df_mut.min(axis=0))/df_mut.max(axis=0)
df_fus_scale = (df_fus-df_fus.min(axis=0))/df_fus.max(axis=0)

df_amp[df_amp==2] =1


# %% data to model

data = pd.concat((df_exp_stand, df_mut_scale, df_amp, df_del, df_fus_scale), axis = 1)


#########################################################################################################################################

# %%

samples_group_dict = f.get_samples_by_group(df_clinical_features)
print(samples_group_dict.keys())


# %% define random sets of features
import random
import itertools

n_features = len(data.columns)
m_times = 2
m_features = 50

s = 250

features_sampled_dict = {}
for i in range(s):
    print(i)
    feature_list = list(np.arange(0,n_features))
    feature_list.pop(i)
    
    feature_list_sampled_i = []
    for m_time in range(m_times):
        random.seed((m_time*10+1)*(i*100+5))
        
        random.shuffle(feature_list)
        
        for iter_ in range(0, int(np.floor(n_features / m_features))):
            feature_list_sampled  = feature_list[m_features * iter_ : m_features * (iter_+1)]
            feature_list_sampled.sort()
            feature_list_sampled_i.append(feature_list_sampled)
            
    features_sampled_dict[str(i)] = feature_list_sampled_i

iterations_per_feature = int(np.floor(n_features / m_features) * m_times)
# %% train and get shap
import warnings
warnings.filterwarnings("ignore")
start_time = datetime.now()

data_to_xg = data.copy().values

xgboost_models_dict = {}
xgboost_shaps_dict = {}
xgboost_eval_dict = {}
for i in range(s):
    print('Feature: ', i)
    xgboost_models_dict[str(i)] = {}
    xgboost_shaps_dict[str(i)] = {}
    xgboost_eval_dict[str(i)] = {}
    
    for iter_ in range(iterations_per_feature):
        xgboost_models_dict[str(i)][iter_] = {}
        xgboost_shaps_dict[str(i)][iter_] = {}
        xgboost_eval_dict[str(i)][iter_] = {}
        
        print('Iteration: ', iter_)
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
            'tree_method' : "hist", 'device' : "cuda"
        }
        num_round = 10  # Number of boosting rounds
        
        # Train the model
        model = xgb.train(param, dtrain, num_round)
        
        # evaluate
        preds = model.predict(dtrain)
        mse = mean_squared_error(y, preds)
        mape = mean_absolute_percentage_error(y, preds)
        r2 = r2_score(y, preds)
        print(mse, mape, r2)
        
        # save to dict
        xgboost_models_dict[str(i)][iter_]['model'] = model
        
        xgboost_eval_dict[str(i)][iter_]['mse'] = mse
        xgboost_eval_dict[str(i)][iter_]['mape'] = mape
        xgboost_eval_dict[str(i)][iter_]['r2'] = r2
        
        xgboost_eval_dict[str(i)][iter_]['total_gain'] = model.get_score(importance_type='total_gain')
        xgboost_eval_dict[str(i)][iter_]['gain'] = model.get_score(importance_type='gain')
        xgboost_eval_dict[str(i)][iter_]['weight'] = model.get_score(importance_type='weight')
        xgboost_eval_dict[str(i)][iter_]['total_cover'] = model.get_score(importance_type='total_cover')
        xgboost_eval_dict[str(i)][iter_]['cover'] = model.get_score(importance_type='cover')
            
        
        
        
        
        
        # Get SHAP
        explainer = shap.TreeExplainer(model, X)
        shap_values = explainer(X).values
        
        # save feature importance
        feature_list_str = [str(i) + '_' + str(feature) + '_' + str(iter_) for feature in feature_list]
        
        xgboost_shaps_dict[str(i)][iter_]['shap_values'] = shap_values
        xgboost_shaps_dict[str(i)][iter_]['feature_list_str'] = feature_list_str
end_time = datetime.now()   
        
# %% save
import json
import pickle

path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\XGBOOST_SHAP_results'

# Save dictionary
with open(os.path.join(path_to_save, 'shaps.pkl'), 'wb') as file:
    pickle.dump(xgboost_shaps_dict, file)
    

# Save dictionary
with open(os.path.join(path_to_save, 'models.pkl'), 'wb') as file:
    pickle.dump(xgboost_models_dict, file)
    
# Save dictionary
with open(os.path.join(path_to_save, 'evals.json'), 'w') as file:
    json.dump(xgboost_eval_dict, file)

       
# %% shap_values_dict
shap_values_dict = {}
cols_all = []
for i in range(s):
    print('Feature: ', i)
    shap_values_dict[str(i)] = {}
    shaps_temp_i = np.zeros((data.shape[0], iterations_per_feature*m_features))
    cols = []
    for iter_ in range(iterations_per_feature):
        
        
        shaps_temp = xgboost_shaps_dict[str(i)][iter_]['shap_values']
        
        shaps_temp_i[:, iter_*m_features: (iter_+1)*m_features] = shaps_temp
        
        cols += xgboost_shaps_dict[str(i)][iter_]['feature_list_str']
        cols_all += xgboost_shaps_dict[str(i)][iter_]['feature_list_str']
    shap_values_dict[str(i)]['shap_values'] = shaps_temp_i
    shap_values_dict[str(i)]['feature_list_str'] = cols
    shap_values_dict[str(i)]['shap_values_pd'] = pd.DataFrame(data = shaps_temp_i, columns = cols)
        
        


# %% shap_values_comb_dict
shap_values_comb_dict = {}
# for column in cols_all:

#     a = column.split('_')[0]
#     b = column.split('_')[1]
#     ab = a + '_' + b
#     print(column, str(ab))
    
#     shap_values_comb_dict[ab] = np.zeros((data.shape[0], m_times ))
    
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
shap_values_comb_mean_dict = {}

for column in super_cols:
    print(column)

    shap_values_comb_mean_dict[column]={}
    shap_values_comb_mean_dict[column]['mean_local'] = np.round(np.mean(shap_values_comb_dict[column],axis = 1), 5)
    shap_values_comb_mean_dict[column]['abs_mean_local'] = np.round(np.mean(np.abs(shap_values_comb_dict[column]),axis = 1), 5)
    shap_values_comb_mean_dict[column]['mean_global'] = np.round(np.mean(shap_values_comb_dict[column]), 5)
    shap_values_comb_mean_dict[column]['abs_mean_global'] = np.round(np.mean(np.abs(shap_values_comb_dict[column])), 5)
    



# %% shap_values_comb_mean_groups_dict
shap_values_comb_mean_groups_dict = {}

for group in list(samples_group_dict.keys()):
    print(group)
    subgroups = list(samples_group_dict[group].keys())
    samples1 = samples_group_dict[group][subgroups[0]]
    samples2 = samples_group_dict[group][subgroups[1]]

    indices1 = df_clinical_features['bcr_patient_barcode'].isin(samples1).values
    indices2 = df_clinical_features['bcr_patient_barcode'].isin(samples2).values

    shap_values_comb_mean_groups_dict[group] = {subgroups[0]:{}, subgroups[1]:{}}
    
    for column in super_cols:
        print(column)
    
        
        shap_values_comb_mean_groups_dict[group][subgroups[0]][column]={}        
        shap_values_comb_mean_groups_dict[group][subgroups[0]][column]['mean_global'] = np.round(np.mean(shap_values_comb_dict[column][indices1]), 5)
        shap_values_comb_mean_groups_dict[group][subgroups[0]][column]['abs_mean_global'] = np.round(np.mean(np.abs(shap_values_comb_dict[column][indices1])), 5)    
        
        
        shap_values_comb_mean_groups_dict[group][subgroups[1]][column]={}        
        shap_values_comb_mean_groups_dict[group][subgroups[1]][column]['mean_global'] = np.round(np.mean(shap_values_comb_dict[column][indices2]), 5)
        shap_values_comb_mean_groups_dict[group][subgroups[1]][column]['abs_mean_global'] = np.round(np.mean(np.abs(shap_values_comb_dict[column][indices2])), 5)    
        
# %% save shap_values_comb_mean_dict
import json
import pickle

path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\XGBOOST_SHAP_results'

# Save dictionary
with open(os.path.join(path_to_save, 'shaps_mean.pkl'), 'wb') as file:
    pickle.dump(shap_values_comb_mean_dict, file)
    

# Save dictionary
with open(os.path.join(path_to_save, 'shaps_groups_mean.pkl'), 'wb') as file:
    pickle.dump(shap_values_comb_mean_groups_dict, file)
# %% investigate r2
r2_pd_all = pd.DataFrame(np.zeros(iterations_per_feature))
mape_pd_all = pd.DataFrame(np.zeros(iterations_per_feature))
mse_pd_all = pd.DataFrame(np.zeros(iterations_per_feature))

for i in range(s):
    print('Feature: ', i)
    r2_i = []
    mse_i = []
    mape_i = []
    for iter_ in range(iterations_per_feature):
        # performance
        r2_i.append(xgboost_eval_dict[str(i)][iter_]['r2'])
        mse_i.append(xgboost_eval_dict[str(i)][iter_]['mse'])
        mape_i.append(xgboost_eval_dict[str(i)][iter_]['mape'])
        
        
    r2_pd_all[i] = r2_i
    mse_pd_all[i] = mse_i
    mape_pd_all[i] = mape_i
    
    

r2_pd_all = r2_pd_all.melt(var_name = 'target_feature', value_name = 'r2')

mse_pd_all = mse_pd_all.melt(var_name = 'target_feature', value_name = 'mse')

mape_pd_all = mape_pd_all.melt(var_name = 'target_feature', value_name = 'mape')

# %% plot r2
fig,ax = plt.subplots(figsize = (15,5))
sns.boxplot(data = r2_pd_all, x='target_feature', y='r2', color = 'white', showfliers=False,ax=ax)
sns.stripplot(data = r2_pd_all, x='target_feature', y='r2',ax=ax, color = 'k')
ax.axhline(1, linestyle = '--')
ax.axhline(.7, linestyle = '--')

fig,ax = plt.subplots(figsize = (15,5))
sns.boxplot(data = mse_pd_all, x='target_feature', y='mse', color = 'white', showfliers=False,ax=ax)
sns.stripplot(data = mse_pd_all, x='target_feature', y='mse',ax=ax, color = 'k')
#ax.axhline(1, linestyle = '--')

fig,ax = plt.subplots(figsize = (15,5))
sns.boxplot(data = mape_pd_all, x='target_feature', y='mape', color = 'white', showfliers=False,ax=ax)
sns.stripplot(data = mape_pd_all, x='target_feature', y='mape',ax=ax, color = 'k')
#ax.axhline(1, linestyle = '--')

# %% edges

shap_matrix = np.zeros((s, n_features))

for column in super_cols:
    print(column)

    shap_temp = shap_values_comb_mean_dict[column]['abs_mean_global']
    
    shap_matrix[int(column.split('_')[0]), int(column.split('_')[1])] = shap_temp

sns.heatmap(shap_matrix, mask = shap_matrix<0.1, cmap = 'jet')


shap_matrix_pd = pd.DataFrame(shap_matrix, columns = data.columns, index = data.columns[:s]).T

shap_edges = shap_matrix_pd.reset_index().melt(id_vars='index', value_vars=shap_matrix_pd.columns)
shap_edges = shap_edges.rename(columns = {'index':'source','variable':'target', 'value':'weight'})


