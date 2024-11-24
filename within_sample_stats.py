# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:28:10 2024

@author: owysocky
"""


%matplotlib inline 
import pickle
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\LRP_stats_within_sample.csv', index_col = 0)
df = df[['sum_LRP','mean_LRP','std_LRP','median_LRP','q1_LRP','q3_LRP']]
# %%
fig, axs = plt.subplots(1,6,figsize= (10,3))
for i, col in enumerate(df.columns):

    ax = axs[i]
    
    sns.violinplot(df[col],ax=ax)

    ax.set_ylim([0,None])
    
plt.tight_layout()

# %% lrp_sum_mean
df_lrp_sum = pd.read_csv(r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\lrp_sum_mean.csv', index_col = 0)


sum_ = df_lrp_sum.sum()
mean = df_lrp_sum.mean()
median = df_lrp_sum.median()
std = df_lrp_sum.std()


fig, axs = plt.subplots(1,4,figsize= (8,3))

sns.violinplot(sum_,ax=axs[0], label = 'LRP sum')
sns.violinplot(mean,ax=axs[1])
sns.violinplot(median,ax=axs[3])
sns.violinplot(std,ax=axs[2])
titles = ['$LRP_{sum}$ sample sum','$LRP_{sum}$ sample mean','$LRP_{sum}$ sample median','$LRP_{sum}$ sample std']
for i in range(4):
    axs[i].set_title(titles[i])
    axs[i].set_ylim([0,None])
    if i >0:
        axs[i].set_ylim([0,5])
plt.tight_layout()
# %%
