# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:47:19 2023

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

# %% artificial_homogeneous

path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\Synthetic_data\keylPatientlevelProteomicNetwork2022'
data = pd.read_csv( os.path.join(path, 'artificial_homogeneous.csv'), index_col = 0 )

data.describe().T
# %%% correlations
%matplotlib inline 

corr_p = data.corr()

sns.heatmap(corr_p.abs(), cmap ='cool', vmin=0, vmax = 1, yticklabels=False, xticklabels=False)
plt.title('Abs. Pearson correlation - artificial_homogeneous')
plt.show()

corr_sper= data.corr(method = 'spearman')
sns.heatmap(corr_sper.abs(), cmap ='cool', vmin=0, vmax = 1, yticklabels=False, xticklabels=False,)
plt.title('Abs. Spearman correlation - artificial_homogeneous')

# %%% PPS

import ppscore as pps

pps_matrix = pps.matrix(data)

pps_matrix_pivot = pps_matrix[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

sns.heatmap(pps_matrix_pivot, cmap ='cool', vmin=0, vmax = 1, yticklabels=False, xticklabels=False,)
plt.title('PPS - artificial_homogeneous')

# %%% pairplot

sns.pairplot(data)





# %% artificial_heterogeneous

path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\Synthetic_data\keylPatientlevelProteomicNetwork2022'
data = pd.read_csv( os.path.join(path, 'artificial_heterogeneous.csv'), index_col = 0 )
data.describe().T


# %%% correlations
%matplotlib inline 

corr_p = data.corr()

sns.heatmap(corr_p.abs(), cmap ='cool', vmin=0, vmax = 1, yticklabels=False, xticklabels=False)
plt.title('Abs. Pearson correlation - artificial_heterogeneous')
plt.show()

corr_sper= data.corr(method = 'spearman')
sns.heatmap(corr_sper.abs(), cmap ='cool', vmin=0, vmax = 1, yticklabels=False, xticklabels=False,)
plt.title('Abs. Spearman correlation - artificial_heterogeneous')


for i in range(4):
    
    data_temp = data.iloc[i*1000 : (i+1)*1000, :]

    
    corr_p = data_temp.corr()
    
    sns.heatmap(corr_p.abs(), cmap ='cool', vmin=0, vmax = 1, yticklabels=False, xticklabels=False)
    plt.title('{} Abs. Pearson correlation - artificial_heterogeneous'.format(i+1))
    plt.show()
    
 
for i in range(4):
    
    data_temp = data.iloc[i*1000 : (i+1)*1000, :]   
    
    corr_sper= data_temp.corr(method = 'spearman')
    sns.heatmap(corr_sper.abs(), cmap ='cool', vmin=0, vmax = 1, yticklabels=False, xticklabels=False,)
    plt.title('{} Abs. Spearman correlation - artificial_heterogeneous'.format(i+1))
    plt.show()
    
    
# %%% PPS

import ppscore as pps

pps_matrix = pps.matrix(data)

pps_matrix_pivot = pps_matrix[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

sns.heatmap(pps_matrix_pivot, cmap ='cool', vmin=0, vmax = 1, yticklabels=False, xticklabels=False,)
plt.title('PPS - artificial_heterogeneous')



for i in range(4):
    
    data_temp = data.iloc[i*1000 : (i+1)*1000, :]   

    pps_matrix = pps.matrix(data_temp)
    
    pps_matrix_pivot = pps_matrix[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
    
    sns.heatmap(pps_matrix_pivot, cmap ='cool', vmin=0, vmax = 1, yticklabels=False, xticklabels=False,)
    plt.title('{} PPS - artificial_heterogeneous'.format(i))
    plt.show()


























