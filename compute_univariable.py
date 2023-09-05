# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:34:46 2023

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

path_to_data = '/home/owysocki/Documents/KI_dataset'
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset'

df_clinical_features = pd.read_csv( os.path.join( r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_model', 'CCE_clinical_features.csv') )

# %%  get only BRCA patients
subtype = 'BRCA'
index_ = (df_clinical_features['acronym'] == subtype).values
index_ = (df_clinical_features['acronym'] != 0).values


# %% FUSION
file = 'CCE_fusion_genes.csv'
df= pd.read_csv(os.path.join(path_to_data, file) , index_col = 0)
df = df[index_]

# Prepare fusions to model
# condition
min_n_with_condition = 2

df_fus = df.iloc[:, ((df != 0).sum() > min_n_with_condition).values]
(df_fus>0).sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (10,5),  title = 'No. samples with Fusions')


# %% MUTATIONS
file = 'CCE_final_analysis_mutations.csv'
df= pd.read_csv(os.path.join(path_to_data, file) , index_col = 0)
df = df[index_]

# Prepare mutations to model
# condition
min_n_with_condition = 2

df_mut = df.iloc[:, ((df != 0).sum() > min_n_with_condition).values]

(df_mut>0).sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (20,5), title = 'No. samples with Mutations')
plt.show()

(df_mut>0).sum().plot(kind = 'hist', bins = 20)

# %% CNA
file = 'CCE_data_CNA_paper.csv'
df= pd.read_csv(os.path.join(path_to_data, file) , index_col = 0)
df = df[index_]


sns.heatmap(df, cmap = 'Reds')
sns.clustermap(df, cmap = 'RdBu', method = 'ward')
plt.show()


# Prepare CNA amplification to model
# condition
min_n_with_condition = 2
df_amp = df[df>0].fillna(0)
# any amplificaiton (1 or 2) is a binary == 1
df_amp[df_amp >0] = 2
df_amp = df_amp.iloc[:, ((df_amp > 0).sum() > min_n_with_condition).values]

# Prepare CNA deletion to model
# condition
min_n_with_condition = 2
df_del = df[df<0].fillna(0)
# any amplificaiton (1 or 2) is a binary == 1
df_del[df_del <0] = 1
df_del = df_del.iloc[:, ((df_del > 0).sum() > min_n_with_condition).values]


(df_amp>0).sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (20,5), title = 'No. samples with Amplifications')
plt.show()
(df_del>0).sum().sort_values().reset_index().plot(x = 'index', y =0, kind='bar', figsize = (20,5), title = 'No. samples with Deletions')
plt.show()
# %% Expressions
#file = 'CCE_gene_expression.csv'
#df= pd.read_csv(os.path.join(path_to_data, file) , index_col = 0)

import pyarrow.feather as feather
#feather.write_feather(df, os.path.join(path_to_data, 'CCE_gene_expression') )
df = feather.read_feather(os.path.join(path_to_data, 'CCE_gene_expression'))
df = df[index_]
df_exp = df.fillna(0)

# adjust per each patient
df_exp = df_exp.div(df.sum(axis=1), axis=0) * 10000

# %% save cols with alteration, mutation or fusion

cols_fus = pd.DataFrame()
cols_fus['cols'] = df_fus.columns
cols_fus['type'] = 'fusion'

cols_mut = pd.DataFrame()
cols_mut['cols'] = df_mut.columns
cols_mut['type'] = 'mutation'


cols_del = pd.DataFrame()
cols_del['cols'] = df_del.columns
cols_del['type'] = 'deletion'

cols_amp = pd.DataFrame()
cols_amp['cols'] = df_amp.columns
cols_amp['type'] = 'amplification'

cols = pd.concat((cols_fus, cols_mut, cols_del, cols_amp))

cols.nunique()
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection'

cols.to_csv(os.path.join(path_to_save, 'all_genes_selected_alterations.csv'))

# %% add names to columns
df_exp.columns = [col + '_exp' for col in df_exp.columns]
df_mut.columns = [col + '_mut' for col in df_mut.columns]
df_amp.columns = [col + '_amp' for col in df_amp.columns]
df_del.columns = [col + '_del' for col in df_del.columns]
df_fus.columns = [col + '_fus' for col in df_fus.columns]

print(np.sum(index_))


# copmute log x + 1

df_exp = df_exp.apply(lambda x: np.log(x + 1))


# comput z score for each column EXPR
df_exp_z_score = (df_exp - df_exp.mean()) / df_exp.std()

# remove nans
cols_to_keep = df_exp_z_score.columns[df_exp_z_score.isna().sum() == 0]
df_exp_z_score =df_exp_z_score[cols_to_keep]
df_exp = df_exp[cols_to_keep]




# %% plot z_score threshold

z_score_thresholds = range(2,35)
n_cols = []
for z_score_threshold in z_score_thresholds:
    
    a=((df_exp_z_score.abs() > z_score_threshold).sum() > 0).sum()
    
    n_cols.append(a)

fig,ax = plt.subplots(figsize = (10,4))
ax.bar(z_score_thresholds, n_cols)
ax.set_xlabel('z_score')
ax.set_ylabel('N genes selected')



# %% MANN WHITNEY
from scipy.stats import mannwhitneyu
from sklearn.utils import resample



def get_mannwhitneyu_matrix(df_cat, df_num, iters=10):
    
    cat_cols = df_cat.columns
    num_cols = df_num.columns
    # Initialize an empty DataFrame to store correlation values
    cles_matrix = pd.DataFrame(index=cat_cols, columns=num_cols)
    pval_matrix = pd.DataFrame(index=cat_cols, columns=num_cols)
    n = len(num_cols)
    # Compute biserial correlation for each pair of categorical and numerical columns
    for cat_col in cat_cols:
        
        for i, num_col in enumerate(num_cols):
            print(cat_col, num_col, num_col,i, '/',n)
            #u, pval = pointbiserialr(df_cat[cat_col].astype('category').cat.codes, df_num[num_col])
            pval, cles = get_mannwhitney_bootstrap(df_cat, df_num, cat_col , num_col, iters = iters)
            pval_matrix.loc[cat_col, num_col] = pval
            cles_matrix.loc[cat_col, num_col] = cles
            
    
    # Convert to float for plotting
    pval_matrix = pval_matrix.astype(float)
    cles_matrix = cles_matrix.astype(float)
    
    return pval_matrix, cles_matrix 



# %%% compute mannwhitneyu PARALLEL

from concurrent.futures import ThreadPoolExecutor
from sklearn.utils import resample
from scipy.stats import mannwhitneyu
import numpy as np

def bootstrap_iteration(group0, group1, M, index):
    boot_group0 = resample(group0, replace=True, n_samples=M)
    stat, p = mannwhitneyu(boot_group0, group1)
    cles = stat / (M * M)
    return p, cles

def get_mannwhitney_bootstrap(df_cat, df_num, cat_col, num_col, iters=100):
    M = (df_cat[cat_col]>0).sum()  # Sample size for each bootstrap
    p_values = []
    cles_list = []

    x = df_num[num_col]
    y = df_cat[cat_col].astype('category').cat.codes

    group0 = x[y == 0]
    group1 = x[y > 0]

    with ThreadPoolExecutor() as executor:
        results = [executor.submit(bootstrap_iteration, group0, group1, M, i) for i in range(iters)]
        
        for f in results:
            try:
                p, cles = f.result()
            except:
                p, cles = 1, 0.5
            p_values.append(p)
            cles_list.append(cles)
    
    p_value_mean = np.mean(p_values)
    cles_mean = np.mean(cles_list)
    
    return p_value_mean, cles_mean


# %%% compute mannwhitneyu 


def bootstrap_iteration(group0, group1, M):
    boot_group0 = resample(group0, replace=True, n_samples=M)
    stat, p = mannwhitneyu(boot_group0, group1)
    cles = stat / (M * M)
    return p, cles

def get_mannwhitney_bootstrap(df_cat, df_num, cat_col, num_col, iters=100):
    M = (df_cat[cat_col] > 0).sum()  # Sample size for each bootstrap
    p_values = []
    cles_list = []

    x = df_num[num_col]
    y = df_cat[cat_col].astype('category').cat.codes

    group0 = x[y == 0]
    group1 = x[y > 0]
    
    for i in range(iters):
        try:
            p, cles = bootstrap_iteration(group0, group1, M)
        except:
            p, cles = 1, 0.5
        p_values.append(p)
        cles_list.append(cles)
    
    p_value_mean = np.mean(p_values)
    cles_mean = np.mean(cles_list)
    
    return p_value_mean, cles_mean



# %%% FUS vs EXP

df_cat = df_fus.copy()
df_cat[df_cat >0] =1
df_num = df_exp.copy()

pval_matrix , cles_matrix  = get_mannwhitneyu_matrix(df_cat, df_num, iters=1)


cles_matrix.to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_CLES_fus_vs_expr.csv')
pval_matrix.to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_pval_fus_vs_expr.csv')

# select only expr where abs corr > X
pvalmax = 0.1
cles_abs_max = 0.6

index_pval  = pval_matrix < pvalmax
index_cles  = (cles_matrix - 0.5).abs() > (cles_abs_max - 0.5)
cols_to_keep_pval = pval_matrix.columns[pval_matrix [index_cles & index_pval].sum() > 0]

cles_matrix_filtered = cles_matrix[cols_to_keep_pval]
pval_matrix_filtered = pval_matrix[cols_to_keep_pval]
cles_matrix_filtered .to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_manwhit_CLES_fus_vs_expr_filtered.csv')
pval_matrix_filtered .to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_manwhit_pvals_fus_vs_expr_filtered.csv')

pd.DataFrame(cols_to_keep_pval, columns = ['cols_to_keep_fus']).to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\cols_to_keep_fus_pval{}_cles{}.csv'.format(pvalmax, cles_abs_max))


pval_matrix_filtered_melted = pval_matrix_filtered.melt(value_name ='pval')
cles_matrix_filtered_melted = cles_matrix_filtered.melt(value_name ='CLES')

merged_matrix_filtered_melted = cles_matrix_filtered_melted .merge(pval_matrix_filtered_melted ) 

sns.scatterplot(merged_matrix_filtered_melted[merged_matrix_filtered_melted['pval'] < 0.05] , x='CLES', y='pval', alpha = 0.2)



# %%% DEL vs EXP

df_cat = df_del.copy()
df_cat[df_cat >0] =1
df_num = df_exp.copy()

pval_matrix , cles_matrix  = get_mannwhitneyu_matrix(df_cat, df_num, iters=2)


cles_matrix.to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_CLES_del_vs_expr.csv')
pval_matrix.to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_pval_del_vs_expr.csv')

# select only expr where abs corr > X
pvalmax = 0.05
cles_abs_max = 0.6

index_pval  = pval_matrix < pvalmax
index_cles  = (cles_matrix - 0.5).abs() > (cles_abs_max - 0.5)
cols_to_keep_pval = pval_matrix.columns[pval_matrix [index_cles & index_pval].sum() > 0]

cles_matrix_filtered = cles_matrix[cols_to_keep_pval]
pval_matrix_filtered = pval_matrix[cols_to_keep_pval]
cles_matrix_filtered .to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_manwhit_CLES_del_vs_expr_filtered.csv')
pval_matrix_filtered .to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_manwhit_pvals_del_vs_expr_filtered.csv')

pd.DataFrame(cols_to_keep_pval, columns = ['cols_to_keep_del']).to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\cols_to_keep_del_pval{}_cles{}.csv'.format(pvalmax, cles_abs_max))


pval_matrix_filtered_melted = pval_matrix_filtered.melt(value_name ='pval')
cles_matrix_filtered_melted = cles_matrix_filtered.melt(value_name ='CLES')

merged_matrix_filtered_melted = cles_matrix_filtered_melted .merge(pval_matrix_filtered_melted ) 

sns.scatterplot(merged_matrix_filtered_melted[merged_matrix_filtered_melted['pval'] < 0.05] , x='CLES', y='pval', alpha = 0.2)




# %%% AMP vs EXP

df_cat = df_amp.copy()
df_cat[df_cat >0] =1
df_num = df_exp.copy()

pval_matrix , cles_matrix  = get_mannwhitneyu_matrix(df_cat, df_num, iters=2)


cles_matrix.to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_CLES_amp_vs_expr.csv')
pval_matrix.to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_pval_amp_vs_expr.csv')

# select only expr where abs corr > X
pvalmax = 0.05
cles_abs_max = 0.6

index_pval  = pval_matrix < pvalmax
index_cles  = (cles_matrix - 0.5).abs() > (cles_abs_max - 0.5)
cols_to_keep_pval = pval_matrix.columns[pval_matrix [index_cles & index_pval].sum() > 0]

cles_matrix_filtered = cles_matrix[cols_to_keep_pval]
pval_matrix_filtered = pval_matrix[cols_to_keep_pval]
cles_matrix_filtered .to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_manwhit_CLES_amp_vs_expr_filtered.csv')
pval_matrix_filtered .to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_manwhit_pvals_amp_vs_expr_filtered.csv')

pd.DataFrame(cols_to_keep_pval, columns = ['cols_to_keep_amp']).to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\cols_to_keep_amp_pval{}_cles{}.csv'.format(pvalmax, cles_abs_max))


# %%% MUT vs EXP

df_cat = df_mut.copy()
df_cat[df_cat >0] =1
df_num = df_exp.copy().iloc[:,10000:]

pval_matrix , cles_matrix  = get_mannwhitneyu_matrix(df_cat, df_num, iters=1)


cles_matrix.to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_CLES_mut_vs_expr_2.csv')
pval_matrix.to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_pval_mut_vs_expr_2.csv')

pval_matrix0 = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_pval_mut_vs_expr_0.csv')
pval_matrix1 = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_pval_mut_vs_expr_1.csv')
pval_matrix = pd.concat((pval_matrix0, pval_matrix1),axis=1)

# select only expr where abs corr > X
pvalmax = 0.05
cles_abs_max = 0.6

index_pval  = pval_matrix < pvalmax
index_cles  = (cles_matrix - 0.5).abs() > (cles_abs_max - 0.5)
cols_to_keep_pval = pval_matrix.columns[pval_matrix [index_cles & index_pval].sum() > 0]

cles_matrix_filtered = cles_matrix[cols_to_keep_pval]
pval_matrix_filtered = pval_matrix[cols_to_keep_pval]
cles_matrix_filtered .to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_manwhit_CLES_mut_vs_expr_filtered.csv')
pval_matrix_filtered .to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_manwhit_pvals_mut_vs_expr_filtered.csv')

pd.DataFrame(cols_to_keep_pval, columns = ['cols_to_keep_mut']).to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\cols_to_keep_mut_pval{}_cles{}.csv'.format(pvalmax, cles_abs_max))


# %%% plot MannWhitney example

index_0 = df_fus['ESR1_fus'] ==0
index_1= df_fus['ESR1_fus'] ==1

df_exp['OCIAD2_exp']

cols = ['OCIAD2_exp','C12orf5_exp','PSMB7_exp']
for col in cols:
    fig,ax = plt.subplots(figsize = (2,10))
    sns.stripplot( y=df_exp[col], hue = df_fus['ESR1_fus'] , dodge = True, alpha = 0.3,ax=ax)
    



cols = ['NOTCH2_exp','BRD4_exp','MYC_exp']
for col in cols:
    fig,ax = plt.subplots(figsize = (2,10))
    sns.stripplot( y=df_exp[col], hue = df_mut['PIK3CA_mut']>0 , dodge = True, alpha = 0.3,ax=ax)
    
    

# %%% plot examples

n=10
fig,axs = plt.subplots(n,1,figsize = (10,n))
axs = axs.flatten()
for i, col in enumerate(df_exp.columns[500:500+n]):
    
    
    ax = axs[i]
    sns.stripplot(x=df_exp_z_score.loc[:, col], ax=ax, alpha= 0.5)
    ax.axvline(-2, linestyle = '--', c='k')
    ax.axvline(2, linestyle = '--', c='k')

plt.tight_layout()




# %%% PLOT n_cols vs pval threshold



def plot_ncols_by_pval_threshold(pval_matrix, cles_matrix, title, cles_abs_max = 0.6):
        
    n_cols = []
    pvals_list = [0.05, 0.02, 0.01, 0.005, 0.001, 0.0001]
    cles_list = [0.6, 0.65, .7, .75, .8, .85, .9, .95]
    
    cles_res = []
    pval_res = []
    
    for pvalmax in pvals_list:
        for cles_abs_max in cles_list:
            print(pvalmax, cles_abs_max)
            index_pval  = pval_matrix < pvalmax
            index_cles  = (cles_matrix - 0.5).abs() > (cles_abs_max - 0.5)
        
            cols_to_keep_pval = pval_matrix.columns[pval_matrix [index_cles & index_pval].sum() > 0]
            n_col = len(cols_to_keep_pval )
            n_cols.append(n_col)
            cles_res.append(cles_abs_max)
            pval_res.append(pvalmax)
        
        
    res = pd.DataFrame(np.array([pval_res, cles_res, n_cols]).T, columns = ['pval','CLES','n_cols'])
    
    res = res.pivot_table(index='pval', columns = 'CLES')
    res.columns = res.columns.droplevel()
    
    
    fig,ax = plt.subplots(figsize = (8,4))
    sns.heatmap(res, annot=True,ax=ax, fmt=".0f" , vmax = 1000, cmap = 'Reds', cbar=False)
  

    #sns.barplot(x=pvals_list, y=n_cols,ax=ax)
    #ax.set_xlabel('pval')
    #ax.set_ylabel('genes selected')
    ax.set_title(title)
    
    
pval_matrix = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_pval_fus_vs_expr.csv', index_col = 0)
cles_matrix = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_cles_fus_vs_expr.csv', index_col = 0)
plot_ncols_by_pval_threshold(pval_matrix, cles_matrix, title='selected by Fusions')
    

pval_matrix = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_pval_amp_vs_expr.csv', index_col = 0)
cles_matrix = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_cles_amp_vs_expr.csv', index_col = 0)
plot_ncols_by_pval_threshold(pval_matrix, cles_matrix, title='selected by Amplification')


pval_matrix = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_pval_del_vs_expr.csv', index_col = 0)
cles_matrix = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_cles_del_vs_expr.csv', index_col = 0)
plot_ncols_by_pval_threshold(pval_matrix, cles_matrix, title='selected by Deletion')

pval_matrix = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_pval_mut_vs_expr.csv', index_col = 0)
cles_matrix = pd.read_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_mannwhit_cles_mut_vs_expr.csv', index_col = 0)
plot_ncols_by_pval_threshold(pval_matrix, cles_matrix, title='selected by Mutation')




# %% Spearman correlation


# use pandas first
spearman_matrix  = df_exp_z_score.corr('spearman')
spearman_matrix.to_csv(r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection\brca_spearman_all.csv')
# now select only high correlations
r_min = 0.2

index_r = spearman_matrix.abs() > r_min



import pingouin as pg

spearman_matrix = pg.pairwise_corr(df_exp_z_score.iloc[:,:], method='spearman', padjust='sidak').round(3)



import numpy as np
from numba import jit

@jit(nopython=True)
def rank_array(arr):
    sorter = np.argsort(arr)
    ranks = np.empty_like(sorter, dtype=np.float64)
    ranks[sorter] = np.arange(arr.shape[0]) + 1  # +1 because Spearman's rank starts at 1
    return ranks

@jit(nopython=True)
def spearman_rank_correlation(rank_x, rank_y):
    n = len(rank_x)
    sum_of_square_differences = np.sum((rank_x - rank_y) ** 2)
    return 1 - (6 * sum_of_square_differences) / (n * (n ** 2 - 1))

@jit(nopython=True)
def spearman_correlation_matrix(data):
    n_rows, n_cols = data.shape
    result = np.zeros((n_cols, n_cols))
    
    for i in range(n_cols):
        for j in range(i, n_cols):
            print(i,j)
            rank_x = rank_array(data[:, i])
            rank_y = rank_array(data[:, j])
            corr = spearman_rank_correlation(rank_x, rank_y)
            result[i, j] = corr
            result[j, i] = corr  # Symmetry
            
    return result

# Test the functions

correlation_matrix = spearman_correlation_matrix(df_exp_z_score)

print("Spearman Correlation Matrix:")
print(correlation_matrix)



# %% categorical vs categorical

import pingouin as pg

df_mut_binary = df_mut.copy()
df_mut_binary[df_mut_binary >0] = 1


df_categ = pd.concat((df_fus, df_mut_binary, df_amp, df_del), axis=1).astype('int')



expected, observed, stats =  pg.chi2_independence(df_categ, x=col1, y=col2)







from scipy.stats import chi2_contingency
from itertools import combinations

def chi2_test_all_columns(df):
    """
    Perform a chi2 test for all combinations of columns in the dataframe.
    
    Parameters:
        df (pd.DataFrame): The input dataframe with categorical data.
        
    Returns:
        dict: A dictionary with column pairs as keys and p-values as values.
    """
    columns = df.columns
    results = {}
    i = 0
    len_ = combinations(columns, 2)
    
    pairs = []
    pvals = []
    chi2 = []
    cramervs = []
    powers = []
    
    for col1, col2 in combinations(columns, 2):
        print(i,col1,col2,)
        
        # Create a contingency table
        #contingency = pd.crosstab(df[col1], df[col2])
        
        expected, observed, stats = pg.chi2_independence(df, col1, col2)
        
        # Perform the chi2 test
        #stat, p, _, _ = chi2_contingency(contingency)
        
        pairs.append([col1,col2])
        pvals.append(stats['pval'].values[0])
        chi2.append(stats['chi2'].values[0])
        cramervs.append(stats['cramer'].values[0])
        powers.append(stats['power'].values[0])
        i+=1

    return pairs, pvals, chi2, cramervs, powers


# Perform the tests
pairs, pvals, chi2, cramers, powers = chi2_test_all_columns(df_categ)
results = pd.DataFrame(np.array([ pvals, chi2, cramers, powers]).T, columns = ['p','chi2', 'cramerv' ,'power'])
results['pair'] = pairs


results_sig = results.copy()[results['p'] < 0.05].reset_index(drop=True)

path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\feature_selection'

results_sig.to_csv(os.path.join(path_to_save, 'braca_chi2_005.csv'))


def combinations_count(n, k):
    from math import factorial
    return factorial(n) // (factorial(k) * factorial(n - k))

# Example usage:
n = 770
k = 2
print(combinations_count(n, k))  # This should print 10








# %% PLOT EXP by Z_score

df_exp_z_score 

df_exp_z_score = df_exp_z_score


n=20
fig,axs = plt.subplots(n,1,figsize = (10,n))
axs = axs.flatten()
for i, col in enumerate(df.columns[:n]):
    
    
    ax = axs[i]
    sns.stripplot(x=df_exp_z_score.loc[:, col], ax=ax, alpha= 0.5)
    ax.axvline(-2, linestyle = '--', c='k')
    ax.axvline(2, linestyle = '--', c='k')

plt.tight_layout()

df_z_score


cols_z_score_higher2 = df_z_score.columns[(df_z_score.abs() > 25).sum() > 0]
plt.hist(cols_z_score_higher2)


corrs = df_z_score[cols_z_score_higher2].corr()

sns.clustermap(corrs, mask  = (corrs.abs() < 0.3), method= 'ward', cmap = 'Reds')



std_ = df.std()
std_.sort_values(ascending=False)[:1000].plot(kind='bar')

mean_ = np.log(df+1).mean()
mean_.sort_values(ascending=False).plot()

np.log(df+1).mean().sort_values(ascending=False).plot()
np.log(df+1).max().sort_values(ascending=False).plot()
np.log(df+1).min().sort_values(ascending=False).plot()

# condition
# highest average expression values
#std_min = 5000

mean_min = df.mean().sort_values(ascending=False)[1000]

df_expr = df.loc[:, list(df.mean()[df.mean() > mean_min].index)]


sns.clustermap(np.log(df_expr.values+1), cmap = 'Reds', method = 'ward')

# %% SAVE DATA 
path_to_save = '/home/owysocki/Documents/KI_dataset/data_to_model'
path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_model'


feather.write_feather(df_fusions, os.path.join(path_to_save, 'CCE_fusions_to_model') )
feather.write_feather(df_mutations, os.path.join(path_to_save, 'CCE_mutations_to_model') )
feather.write_feather(df_amp, os.path.join(path_to_save, 'CCE_amplifications_to_model') )
feather.write_feather(df_del, os.path.join(path_to_save, 'CCE_deletions_to_model') )
feather.write_feather(df_expr, os.path.join(path_to_save, 'CCE_expressions_to_model') )

df_fusions.to_csv(os.path.join(path_to_save, 'CCE_fusions_to_model.csv') )
df_mutations.to_csv(os.path.join(path_to_save, 'CCE_mutations_to_model.csv') )
df_amp.to_csv( os.path.join(path_to_save, 'CCE_amplifications_to_model.csv') )
df_del.to_csv(os.path.join(path_to_save, 'CCE_deletions_to_model.csv') )
df_expr.to_csv(os.path.join(path_to_save, 'CCE_expressions_to_model.csv') )




