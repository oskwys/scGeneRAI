# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 20:58:50 2023

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

# %% load pancanatlas pathways

path_to_pancanatlas = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\PATHWAYS'

file = 'pathways_Pancanatlas.xlsx'

df = pd.read_excel(os.path.join(path_to_pancanatlas, file), )

df = df[df['TYPE'] == 'GENE'].reset_index(drop=True).drop(columns = ['ID'])
df['cce_match'] = ''

genes_pathways_all = set(df['Gene'])

# %% load CCE gene names

path_to_cce_genes = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\PATHWAYS'

file = 'cce_genes_new_symbols.csv'

df_cce = pd.read_csv(os.path.join(path_to_pancanatlas, file), )


# %% LOAD Expressions
#file = 'CCE_gene_expression.csv'
#df= pd.read_csv(os.path.join(path_to_data, file) , index_col = 0)
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset'


import pyarrow.feather as feather
#feather.write_feather(df, os.path.join(path_to_data, 'CCE_gene_expression') )
df_exp = feather.read_feather(os.path.join(path_to_data, 'CCE_gene_expression'))
df_exp = df_exp.fillna(0)

genes_exp = pd.DataFrame(df_exp.columns, columns = ['CCE_symbol'])

all_symbols_cce = df_cce['all_symbols'].dropna()

genes_intersection = set.intersection(set(df_cce['CCE_symbol']) , set(df_exp.columns))

# %% find genes from pathways not in CCE genes

genes_missing = set.difference(set(df['Gene']), set(genes_exp['CCE_symbol']))

            
for i, row_pathways in df.iterrows():

    gene = row_pathways['Gene']          
    index_ = df_cce['CCE_symbol'] == gene
    if index_.sum() == 1:
        print(gene, )
        df.loc[i, 'cce_match'] = df_cce.loc[index_, 'CCE_symbol'].values[0]
    else:
        for j, row_cce in df_cce.dropna(subset = 'all_symbols').iterrows():
            if gene in row_cce['all_symbols'] :
                print(gene,'found in : ', row_cce['all_symbols'])
                df.loc[i, 'cce_match'] = row_cce['CCE_symbol']
    

# %% SAVE    
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\PATHWAYS'
df.to_csv(os.path.join(path_to_save, 'genes_pathways_pancanatlas_matched_cce.csv'))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
