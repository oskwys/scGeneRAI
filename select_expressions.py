# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 21:46:15 2023

@author: d07321ow
"""


import pandas as pd

import numpy as np
import sys
import os
import seaborn as sns
#import scanpy as sc
#import torch
import scipy

import seaborn as sns
import matplotlib.pyplot as plt

# %%

path_to_corrs = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection'

file = 'brca_spearman_all.csv'

df = pd.read_csv(os.path.join(path_to_corrs , file))

#df = df.set_index('Unnamed: 0')

r_min = 0.3



# Melt the DataFrame to long format
df_melted = pd.melt(df, id_vars='Unnamed: 0', var_name='Variable_2', value_name='Correlation')

index_r = df_melted['Correlation'].abs() > r_min



# %%

path_to_univariable_results = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selectioncols_to_keep_fus_pval{}_cles{}