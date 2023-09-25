# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:04:56 2023

@author: d07321ow
"""
import seaborn as sns
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html

# %% load data
df = pd.read_csv(r'C:\Users\d07321ow\Google Drive\Abroad\GME\data_to_plot_true_CL_pairs_GENE_DRUG.csv', index_col = 0)
df_out = pd.read_csv(r'C:\Users\d07321ow\Google Drive\Abroad\GME\out_true_CL_pairs_GENE_DRUG.csv', index_col = 0).reset_index()

df = df.merge(df_out, on = 'reference_doi').sort_values('index').reset_index(drop=True)
dataset = 'CL_pairs_GENE_DRUG'
# %%
df = pd.read_csv(r'C:\Users\d07321ow\Google Drive\Abroad\GME\data_to_plot_true_DrugBank.csv', index_col = 0)
df_out = pd.read_csv(r'C:\Users\d07321ow\Google Drive\Abroad\GME\out_DrugBank.csv', index_col = 0).reset_index()

df = df[['pmid', 'variable', 'drug_x', 'target_x', 'interaction_x', 'drug_y',   'target_y', 'interaction_y', ]]

df = df.merge(df_out[['pmid','index']], on='pmid').sort_values('index').reset_index(drop=True)
dataset = 'drugbank'
# %% compute occurance
def analyze_occurance(df , column, step =100 , plot_hist = False, plot = False):
    
    n = df.shape[0]
 
    occurance_results = pd.DataFrame()
    sizes = []
    nunique = []

    stds = []
    means = []
    mins = []
    maxs = []
    for size in range(20,n,step):
    
        #x_data = np.sort(np.unique(df.loc[:size, column], return_counts = True)[1])[::-1]
        
        x_data = df[:size].groupby(column).count().sort_values('index', ascending= False)['index'].values
        #sns.ecdfplot(data=pd.DataFrame(x_data), x= 0),ax=ax)
        #plt.show()
        #sns.displot(data=pd.DataFrame(x_data), kind= 'ecdf',ax=ax)
        #plt.show()
        #x_data = x_data/np.max(x_data)
        print('Size: ', size, 'Data shape: ', x_data.shape)
        
        nunique.append(x_data.shape[0])
        sizes.append(size)
               
        stds.append(np.std(x_data))
        means.append(np.mean(x_data))
        mins.append(np.min(x_data))
        maxs.append(np.max(x_data))
        
        if plot_hist:
            plt.bar(x = range(len(x_data)),height=x_data)
            plt.title('Column: '+ column+ ' Size: {}'.format(size))
            plt.show()
        
        
    occurance_results['size'] = sizes
    occurance_results['nunique'] = nunique
    occurance_results['std'] = stds
    occurance_results['mean'] = means
    occurance_results['min'] = mins
    occurance_results['max'] = maxs
        
    if plot:
        
        fig,ax= plt.subplots()
        #ax.plot(occurance_results['size'], occurance_results['std'], '-o',label = 'std')
        ax.plot(occurance_results['size'], occurance_results['min'], '--', label  ='min')
        ax.plot(occurance_results['size'], occurance_results['mean'], label  ='mean', c='k')
        ax.plot(occurance_results['size'], occurance_results['max'], '--', label  ='max')
        ax.set_title('Occurance of entity '+ column+ ' in the dataset')
        ax.legend()
        ax.set_xlabel('Size')
        plt.tight_layout()
        
        fig,ax= plt.subplots()
        #ax.plot(occurance_results['size'], occurance_results['std'], '-o',label = 'std')
        ax.plot(occurance_results['size'], occurance_results['min'], '--', label  ='min')
        ax.plot(occurance_results['size'], occurance_results['mean'], label  ='mean', c='k')
        ax.plot(occurance_results['size'], occurance_results['max'], '--', label  ='max')
        ax.set_title('Occurance of entity '+ column+ ' in the dataset')
        ax.set_yscale('log')
        y_ticks = [0,1, 10, 100]
        ax.set_yticklabels(['','0','1', '10', '100'])
        ax.legend()
        ax.set_xlabel('Size')
        plt.tight_layout()
        
        
    return occurance_results


def analyze_occuranvce_random(df , column, step =100 , iters = 20, plot = False):
    n = df.shape[0]
 
    occurance_results = pd.DataFrame()
    sizes = []
    nunique = []
    
    stds = []
    means = []
    mins = []
    maxs = []
    iters_list = []
    for size in range(20,n,step):
    
        for iter_ in range(iters):
            
            x_data = df.sample(size).groupby(column).count().sort_values('train', ascending= False)['train'].values
            
            print(iter_, 'Size: ', size, 'Data shape: ', x_data.shape)
            
            nunique.append(x_data.shape[0])
            sizes.append(size)
                                    
            stds.append(np.std(x_data))
            means.append(np.mean(x_data))
            mins.append(np.min(x_data))
            maxs.append(np.max(x_data))
            iters_list.append(iter_)
   
        
        
    occurance_results['size'] = sizes
    occurance_results['nunique'] = nunique
    occurance_results['std'] = stds
    occurance_results['mean'] = means
    occurance_results['min'] = mins
    occurance_results['max'] = maxs
    occurance_results['random_iter'] = iters_list
        
    if plot:
        
        fig,ax= plt.subplots()
        #ax.plot(occurance_results['size'], occurance_results['std'], '-o',label = 'std')
        ax.plot(occurance_results['size'], occurance_results['min'], '--', label  ='min')
        ax.plot(occurance_results['size'], occurance_results['mean'], label  ='mean', c='k')
        ax.plot(occurance_results['size'], occurance_results['max'], '--', label  ='max')
        ax.set_title('Occurance of entity '+ column+ ' in the dataset')
        ax.legend()
        ax.set_xlabel('Size')
        plt.tight_layout()
        
        fig,ax= plt.subplots()
        #ax.plot(occurance_results['size'], occurance_results['std'], '-o',label = 'std')
        ax.plot(occurance_results['size'], occurance_results['min'], '--', label  ='min')
        ax.plot(occurance_results['size'], occurance_results['mean'], label  ='mean', c='k')
        ax.plot(occurance_results['size'], occurance_results['max'], '--', label  ='max')
        ax.set_title('Occurance of entity '+ column+ ' in the dataset')
        ax.set_yscale('log')
        y_ticks = [0,1, 10, 100]
        ax.set_yticklabels(['','0','1', '10', '100'])
        ax.legend()
        ax.set_xlabel('Size')
        plt.tight_layout()
        
        
    return occurance_results



# %% GET OCCURANCE

column = 'drug_x'
column = 'gene_x'

occurance_results = analyze_occurance(df , column = column, step = 10, plot_hist = False, plot = False)
#occurance_results = analyze_occurance(df , column = 'gene_x', step = 20, plot_hist = False, plot = True)

# PLOT GME OCCURANCE

fig,ax= plt.subplots()
#ax.plot(occurance_results['size'], occurance_results['std'], '-o',label = 'std')
ax.plot(occurance_results['size'], occurance_results['min'], '--', label  ='min')
ax.plot(occurance_results['size'], occurance_results['mean'], label  ='mean', c='k')
ax.plot(occurance_results['size'], occurance_results['max'], '--', label  ='max')
ax.set_title('Occurance of entity '+ column+ ' in the dataset')
ax.legend()
ax.set_xlabel('Dataset size')
plt.tight_layout()



fig,ax= plt.subplots()
#ax.plot(occurance_results['size'], occurance_results['std'], '-o',label = 'std')
ax.plot(occurance_results['size'], occurance_results['min'], '--', label  ='min')
ax.plot(occurance_results['size'], occurance_results['mean'], label  ='mean', c='k')
ax.plot(occurance_results['size'], occurance_results['max'], '--', label  ='max')
ax.set_title('Occurance of entity '+ column+ ' in the dataset')
ax.set_yscale('log')
y_ticks = [0,1, 10, 100]
ax.set_yticklabels(['','0','1', '10', '100'])
ax.legend()
ax.set_xlabel('Dataset size')
plt.tight_layout()



fig,ax= plt.subplots()
#ax.plot(occurance_results['size'], occurance_results['std'], '-o',label = 'std')
ax.plot(occurance_results['nunique'], occurance_results['min'], '--', label  ='min')
ax.plot(occurance_results['nunique'], occurance_results['mean'], label  ='mean', c='k')
ax.plot(occurance_results['nunique'], occurance_results['max'], '--', label  ='max')
ax.set_title('Occurance of entity '+ column+ ' in the dataset')
ax.legend()
ax.set_xlabel('n unique')
plt.tight_layout()




# GET OCCURANCE RANDOM SELECTION from dataset

occurance_results_random = analyze_occuranvce_random(df , column, step =10 , iters = 20, plot = False)
occurance_results_random_gr_min = occurance_results_random.groupby('size').min()['min'].reset_index()
occurance_results_random_gr_mean = occurance_results_random.groupby('size').mean()['mean'].reset_index()
occurance_results_random_gr_max = occurance_results_random.groupby('size').max()['max'].reset_index()


# PLOTS


fig,ax= plt.subplots()
#ax.plot(occurance_results['size'], occurance_results['std'], '-o',label = 'std')
ax.plot(occurance_results_random_gr_min['size'], occurance_results_random_gr_min['min'], '--', label  ='min')
ax.plot(occurance_results_random_gr_mean['size'], occurance_results_random_gr_mean['mean'], label  ='mean', c='k')
ax.plot(occurance_results_random_gr_max['size'], occurance_results_random_gr_max['max'], '--', label  ='max')
ax.set_title('Occurance of entity '+ column+ ' in the dataset after random selection')
ax.legend()
ax.set_xlabel('Dataset size')
plt.tight_layout()



fig,ax= plt.subplots()
#ax.plot(occurance_results['size'], occurance_results['std'], '-o',label = 'std')
ax.plot(occurance_results_random_gr_min['size'], occurance_results_random_gr_min['min'], '--', label  ='min')
ax.plot(occurance_results_random_gr_mean['size'], occurance_results_random_gr_mean['mean'], label  ='mean', c='k')
ax.plot(occurance_results_random_gr_max['size'], occurance_results_random_gr_max['max'], '--', label  ='max')
ax.set_title('Occurance of entity '+ column+ ' in the dataset after random selection')
ax.set_yscale('log')
y_ticks = [0,1, 10, 100]
ax.set_yticklabels(['','0','1', '10', '100'])
ax.legend()
ax.set_xlabel('Dataset size')
plt.tight_layout()


fig,ax= plt.subplots()
#ax.plot(occurance_results['size'], occurance_results['std'], '-o',label = 'std')
ax.plot(occurance_results_random['size'], occurance_results_random['min'], '--', label  ='min')
ax.plot(occurance_results_random['size'], occurance_results_random['mean'], label  ='mean', c='k')
ax.plot(occurance_results_random['size'], occurance_results_random['max'], '--', label  ='max')
ax.set_title('Occurance of entity '+ column+ ' in the dataset')
ax.legend()
ax.set_xlabel('Size')
plt.tight_layout()



fig,ax= plt.subplots()
#ax.plot(occurance_results['size'], occurance_results['std'], '-o',label = 'std')
sns.lineplot(x=occurance_results_random['size'], y=occurance_results_random['min'], linestyle='--', label ='min',ax=ax)
sns.lineplot(x=occurance_results_random['size'], y=occurance_results_random['mean'], linestyle='--', label ='mean',ax=ax)
sns.lineplot(x=occurance_results_random['size'], y=occurance_results_random['max'], linestyle='--', label ='max',ax=ax)
ax.set_title('Occurance of entity '+ column+ ' in the dataset')
ax.legend()
ax.set_xlabel('Size')
plt.tight_layout()





occurance_results['max_deviation'] = occurance_results['max'] - occurance_results['mean']
occurance_results_random['max_deviation'] = (occurance_results_random['max'] - occurance_results_random['mean'])

# JOINT PLOT FILL BETWEEN
fig,ax= plt.subplots(figsize = (8,6))
ax.plot(occurance_results_random_gr_mean['size'], occurance_results_random_gr_mean['mean'], label  ='Random', c='k')
ax.fill_between(occurance_results_random_gr_min['size'], occurance_results_random_gr_min['min'], occurance_results_random_gr_max['max'], alpha = 0.3, color = 'gray')
ax.plot(occurance_results['size'], occurance_results['mean'], label  ='GME', c='green')
ax.fill_between(occurance_results['size'], occurance_results['min'], occurance_results['max'], alpha = 0.3, color = 'green')
#sns.lineplot(x=occurance_results_random['size'], y=occurance_results_random['max_deviation'], linestyle='--', label ='max_deviation',ax=ax)
#sns.lineplot(x=occurance_results['size'], y=occurance_results['max_deviation'], linestyle='--', label ='max_deviation',ax=ax)
ax.set_title('Occurance of entity '+ column+ ' in the dataset')
ax.legend()
ax.set_ylim([1,None])
ax.set_xlim([0,None])
ax.set_xlabel('Size')
plt.tight_layout()


# max occurance - mean accurance

fig,ax= plt.subplots(figsize = (8,6))
sns.lineplot(x=occurance_results_random['size'], y=occurance_results_random['max_deviation'], linestyle='--', label ='max_deviation',ax=ax)
sns.lineplot(x=occurance_results['size'], y=occurance_results['max_deviation'], linestyle='--', label ='max_deviation',ax=ax)
ax.set_title('Maximal difference in occurance mean-max'+ column+ ' in the dataset')
ax.legend()
ax.set_ylim([1,None])
ax.set_xlim([0,None])
ax.set_xlabel('Size')
plt.tight_layout()




# %% compute n_unique vs size


def calculate_unique_entities(df_y, columns, increment):
    num_rows = len(df_y)
    rows_counts = []
    unique_entities_counts = {col: [] for col in columns}

    for n in range(1, num_rows + increment, increment):
        print(n)
        df_x = df_y.iloc[:n]
        for column in columns:
            unique_entities = df_x[column].nunique()
            unique_entities_counts[column].append(unique_entities)
        rows_counts.append(n)

    result_df = pd.DataFrame({'size': rows_counts})
    for column in columns:
        result_df['n_unique_{}'.format(column)] = unique_entities_counts[column]

    return result_df

def calculate_unique_entities_by_pmid(df_y, columns, column_pmid, increment):
    rows_counts = []
    unique_entities_counts = {col: [] for col in columns}
    num_rows = df_y[column_pmid].nunique()

    for n in range(1, num_rows + increment, increment):
        print(n)
        
        unique_entities_Z = df_y[column_pmid].drop_duplicates().head(n)
        
        df_x = df_y[df_y[column_pmid].isin(unique_entities_Z)]
                
        for column in columns:
            unique_entities = df_x[column].nunique()
            unique_entities_counts[column].append(unique_entities)
        rows_counts.append(n)

    result_df = pd.DataFrame({'PMID_size': rows_counts})
    for column in columns:
        result_df['n_unique_{}'.format(column)] = unique_entities_counts[column]

    return result_df



def random_unique(df, n, columns, random_state):
    # First, randomly select n rows from the dataframe
    random_df = df.sample(n, random_state=random_state)
    
    # Initialize an empty dictionary to store the unique counts per column
    unique_counts = {}

    # Loop through the columns specified
    for col in columns:
        if col in random_df.columns:
            # Calculate the number of unique entities in the column
            unique_counts[col] = random_df[col].nunique()
        else:
            print(f"Column {col} does not exist in the dataframe.")
    
    # Convert the dictionary to a DataFrame and return it
    unique_counts_df = pd.DataFrame.from_dict(unique_counts, orient='index', columns=['Unique Count'])
    return unique_counts_df


def random_unique_pmid(df, n, columns, column_pmid, random_state):
    # First, randomly select n rows from the dataframe
    unique_entities_Z = df[column_pmid].drop_duplicates().sample(n, random_state=random_state)
    
    random_df = df[df[column_pmid].isin(unique_entities_Z)]
        
    # Initialize an empty dictionary to store the unique counts per column
    unique_counts = {}

    # Loop through the columns specified
    for col in columns:
        if col in random_df.columns:
            # Calculate the number of unique entities in the column
            unique_counts[col] = random_df[col].nunique()
        else:
            print(f"Column {col} does not exist in the dataframe.")
    
    # Convert the dictionary to a DataFrame and return it
    unique_counts_df = pd.DataFrame.from_dict(unique_counts, orient='index', columns=['Unique Count'])
    return unique_counts_df

# %% Size VS nunique


df['drug_x + interaction + target_x'] = df['drug_x'] + ' - ' + df['interaction_x'] + ' - ' + df['target_x'] 
df['drug_x + target_x'] = df['drug_x'] + ' - ' + df['target_x'] 


iters = 10
step = 100
#columns = ['drug_x', 'gene_x']
columns = ['drug_x', 'target_x', 'interaction_x', 'drug_x + interaction + target_x', 'drug_x + target_x']


# %%% BY RELATION
n_unique_entities_df = calculate_unique_entities(df, columns, step)
n_unique_entities_random = pd.DataFrame()
for size in range(1, df.shape[0], step):
    for iter_ in range(iters):
        print(size, iter_)
        unique_counts_df = random_unique(df, size, columns, iter_)

        res_temp = unique_counts_df.T.reset_index()
        res_temp['size'] = size
        res_temp['iter'] = iter_
        n_unique_entities_random = pd.concat((n_unique_entities_random  , res_temp))


# %%%% Nunique

fig, ax = plt.subplots(figsize = (9,4))
for column in columns:
    ax.plot(n_unique_entities_df['size'], n_unique_entities_df['n_unique_'+column], label=column)
    ax.plot(n_unique_entities_random.groupby('size').mean().index, n_unique_entities_random.groupby('size').mean()[column], linestyle = '--', label=column+'_random')
    
    #sns.lineplot(data = n_unique_entities_random , x= 'size', y = column+'_ratio', label = 'random '+column+'_ratio', linestyle = '--',ax=ax)
    #ax.plot(n_unique_entities_random['size'],n_unique_entities_random[column], label = 'random '+column, linestyle = '--')
ax.legend( bbox_to_anchor=(1.05, 0.5), loc = 'center left',  fancybox=True)
ax.set_xlabel('Dataset size')    
ax.set_ylabel('n unique / dataset size')    
ax.set_xlim([0,None])
ax.set_ylim([0,None])
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.png', dpi = 200)
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.svg', format= 'svg')

# PLOTS
fig, axs = plt.subplots(1, len(columns), figsize = (4*len(columns), 3))
for i, column in enumerate(columns):
    ax = axs[i]
    ax.plot(n_unique_entities_df['size'], n_unique_entities_df['n_unique_'+column], label=column)
    ax.plot(n_unique_entities_random.groupby('size').mean().index, n_unique_entities_random.groupby('size').mean()[column], linestyle = '--', label=column+'_random')
    
    
    #sns.lineplot(data = n_unique_entities_random , x= 'size', y = column, label = 'random '+column, linestyle = '--',ax=ax)
    #ax.plot(n_unique_entities_random_pmid['pmid'],n_unique_entities_random_pmid[column], label = 'random '+column, linestyle = '--')
    ax.legend()
    ax.set_xlabel('Dataset size')    
    ax.set_ylabel('n unique')    
    ax.set_xlim([0,None])
    ax.set_ylim([0,None])
    
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.png', dpi = 200)
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.svg', format= 'svg')


# RATIO

for column in columns:
    n_unique_entities_df['n_unique_{}_ratio'.format(column)] = n_unique_entities_df['n_unique_{}'.format(column)] / n_unique_entities_df['size']
    n_unique_entities_random['{}_ratio'.format(column)] = n_unique_entities_random['{}'.format(column)] / n_unique_entities_random['size']



fig, ax = plt.subplots(figsize = (6,4))
for column in columns:
    ax.plot(n_unique_entities_df['size'], n_unique_entities_df['n_unique_'+column+'_ratio'], label=column)
    ax.plot(n_unique_entities_random.groupby('size').mean().index, n_unique_entities_random.groupby('size').mean()[column+'_ratio'], linestyle = '--', label=column+'_random')
    
    #sns.lineplot(data = n_unique_entities_random , x= 'size', y = column+'_ratio', label = 'random '+column+'_ratio', linestyle = '--',ax=ax)
    #ax.plot(n_unique_entities_random['size'],n_unique_entities_random[column], label = 'random '+column, linestyle = '--')
ax.set_xlabel('Dataset size')    
ax.set_ylabel('n unique / dataset size')    
ax.set_xlim([0,None])
ax.set_ylim([0,None])
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.png', dpi = 200)
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.svg', format= 'svg')

   



   
    
    
# %%% BY PMID
n_unique_entities_df_pmid = calculate_unique_entities_by_pmid(df, columns,'pmid', 10)
n_unique_entities_random_pmid = pd.DataFrame()
for size in range(1, df['pmid'].nunique(), step):
    for iter_ in range(iters):
        print(size, iter_)
        unique_counts_df = random_unique_pmid(df, size, columns,'pmid', np.random.randint(1000))

        res_temp = unique_counts_df.T.reset_index()
        res_temp['pmid'] = size
        res_temp['iter'] = iter_
        n_unique_entities_random_pmid = pd.concat((n_unique_entities_random_pmid  , res_temp))



# %%%% Nunique

fig, ax = plt.subplots(figsize = (9,4))
for column in columns:
    ax.plot(n_unique_entities_df_pmid['PMID_size'], n_unique_entities_df_pmid['n_unique_'+column], label=column)
    ax.plot(n_unique_entities_random_pmid.groupby('pmid').mean().index, n_unique_entities_random_pmid.groupby('pmid').mean()[column], linestyle = '--', label=column+'_random')
    
    #sns.lineplot(data = n_unique_entities_random_pmid , x= 'size', y = column+'_ratio', label = 'random '+column+'_ratio', linestyle = '--',ax=ax)
    #ax.plot(n_unique_entities_random_pmid['size'],n_unique_entities_random_pmid[column], label = 'random '+column, linestyle = '--')
ax.legend( bbox_to_anchor=(1.05, 0.5), loc = 'center left',  fancybox=True)
ax.set_xlabel('Dataset size (pmid)')    
ax.set_ylabel('n unique / dataset size')    
ax.set_xlim([0,None])
ax.set_ylim([0,None])
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.png', dpi = 200)
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.svg', format= 'svg')

# PLOTS
fig, axs = plt.subplots(1, len(columns), figsize = (4*len(columns), 3))
for i, column in enumerate(columns):
    ax = axs[i]
    ax.plot(n_unique_entities_df_pmid['PMID_size'], n_unique_entities_df_pmid['n_unique_'+column], label=column)
    ax.plot(n_unique_entities_random_pmid.groupby('pmid').mean().index, n_unique_entities_random_pmid.groupby('pmid').mean()[column], linestyle = '--', label=column+'_random')
    
    
    #sns.lineplot(data = n_unique_entities_random_pmid , x= 'size', y = column, label = 'random '+column, linestyle = '--',ax=ax)
    #ax.plot(n_unique_entities_random_pmid_pmid['pmid'],n_unique_entities_random_pmid_pmid[column], label = 'random '+column, linestyle = '--')
    ax.legend()
    ax.set_xlabel('Dataset size (pmid)')    
    ax.set_ylabel('n unique')    
    ax.set_xlim([0,None])
    ax.set_ylim([0,None])
    
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.png', dpi = 200)
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.svg', format= 'svg')


# RATIO

for column in columns:
    n_unique_entities_df_pmid['n_unique_{}_ratio'.format(column)] = n_unique_entities_df_pmid['n_unique_{}'.format(column)] / n_unique_entities_df_pmid['PMID_size']
    n_unique_entities_random_pmid['{}_ratio'.format(column)] = n_unique_entities_random_pmid['{}'.format(column)] / n_unique_entities_random_pmid['pmid']



fig, ax = plt.subplots(figsize = (6,4))
for column in columns:
    ax.plot(n_unique_entities_df_pmid['PMID_size'], n_unique_entities_df_pmid['n_unique_'+column+'_ratio'], label=column)
    ax.plot(n_unique_entities_random_pmid.groupby('pmid').mean().index, n_unique_entities_random_pmid.groupby('pmid').mean()[column+'_ratio'], linestyle = '--', label=column+'_random')
    
    #sns.lineplot(data = n_unique_entities_random_pmid , x= 'size', y = column+'_ratio', label = 'random '+column+'_ratio', linestyle = '--',ax=ax)
    #ax.plot(n_unique_entities_random_pmid['size'],n_unique_entities_random_pmid[column], label = 'random '+column, linestyle = '--')
ax.set_xlabel('Dataset size (pmid)')    
ax.set_ylabel('n unique / dataset size')    
ax.set_xlim([0,None])
ax.set_ylim([0,None])
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.png', dpi = 200)
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.svg', format= 'svg')

# %% Stratified sampling

import numpy as np
import pandas as pd

path_to_save = r'C:\Users\d07321ow\Google Drive\Abroad\GME'

def stratified_sample(df, strata, n_samples, iter_):
    """
    df: Pandas DataFrame
    strata: column on which to do stratification
    n_samples: number of samples to take from each stratum
    """
    # Group the data by the strata and sample n_samples from each group
    stratified_sample = df.groupby(strata, group_keys=False).apply(lambda x: x.sample(min(len(x), n_samples), random_state =iter_))

    return stratified_sample
   
columns = ['drug_x', 'target_x', 'interaction_x']

   
column1 = 'interaction_x' # define strata column
column2 = 'target_x' # dfine any other different to strata
n_points = 20
sizes = np.unique(np.geomspace(1, df.groupby(column1).count()[column2].max(), n_points, dtype='int'))
iters=5
unique_entities_counts = {col: [] for col in columns}
rows_counts = []
for size in sizes:
    
    
    for iter_ in range(iters):
        print(size, iter_)

        sample = stratified_sample(df, column1, size, iter_)
        
        
        
        for column in columns:
            unique_entities = sample[column].nunique()
            unique_entities_counts[column].append(unique_entities)
        rows_counts.append(sample.shape[0])
   
result_df = pd.DataFrame({'size': rows_counts})
for column in columns:
    result_df['n_unique_{}'.format(column)] = unique_entities_counts[column]  
    
    

fig, ax = plt.subplots(figsize = (9,4))
for column in columns:
    ax.plot(result_df['size'], result_df['n_unique_'+column], label=column)
    #ax.plot(n_unique_entities_random.groupby('size').mean().index, n_unique_entities_random.groupby('size').mean()[column], linestyle = '--', label=column+'_random')
    
    #sns.lineplot(data = n_unique_entities_random , x= 'size', y = column+'_ratio', label = 'random '+column+'_ratio', linestyle = '--',ax=ax)
    #ax.plot(n_unique_entities_random['size'],n_unique_entities_random[column], label = 'random '+column, linestyle = '--')
ax.legend( bbox_to_anchor=(1.05, 0.5), loc = 'center left',  fancybox=True)
ax.set_xlabel('Dataset size')    
ax.set_ylabel('n unique / dataset size')    
ax.set_xlim([0,None])
ax.set_ylim([0,None])
plt.tight_layout()


result_df.to_csv(os.path.join(path_to_save, '{}_stratified_by_{}.csv'.format(dataset, column1)))


   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   

# %%
# PLOTS
fig, ax = plt.subplots(figsize = (8,8))
for column in columns:
    ax.plot(n_unique_entities_df_pmid_pmid['PMID_size'], n_unique_entities_df_pmid_pmid['n_unique_'+column], label=column)
    sns.lineplot(data = n_unique_entities_random_pmid_pmid , x= 'pmid', y = column, label = 'random '+column, linestyle = '--',ax=ax)
    #ax.plot(n_unique_entities_random_pmid_pmid['pmid'],n_unique_entities_random_pmid_pmid[column], label = 'random '+column, linestyle = '--')
ax.legend( bbox_to_anchor=(1.05, 0.5), loc = 'center left',  fancybox=True)
ax.set_xlabel('PMID Size')    
ax.set_ylabel('n unique')    
ax.set_xlim([0,None])
ax.set_ylim([0,None])
plt.tight_layout()


# interaction for drug dataset
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(n_unique_entities_df_pmid_pmid['PMID_size'], n_unique_entities_df_pmid_pmid['n_unique_interaction_x'], label=column)
sns.lineplot(data = n_unique_entities_random_pmid_pmid , x= 'pmid', y = 'interaction_x', label = 'random interaction_x', linestyle = '--',ax=ax)

ax.legend()
ax.set_xlabel('PMID Size')    
ax.set_ylabel('n unique')    
ax.set_xlim([0,None])
ax.set_ylim([0,None])
plt.tight_layout()



n_unique_entities_df_pmid['n_unique_drug_x_ratio'] = n_unique_entities_df_pmid['n_unique_drug_x'] / n_unique_entities_df_pmid['size']
n_unique_entities_df_pmid['n_unique_gene_x_ratio'] = n_unique_entities_df_pmid['n_unique_gene_x'] / n_unique_entities_df_pmid['size']

n_unique_entities_random_pmid['drug_x_ratio'] = n_unique_entities_random_pmid['drug_x'] / n_unique_entities_random_pmid['size']
n_unique_entities_random_pmid['gene_x_ratio'] = n_unique_entities_random_pmid['gene_x'] / n_unique_entities_random_pmid['size']


fig, ax = plt.subplots(figsize = (6,4))
for column in columns:
    ax.plot(n_unique_entities_df['PMID_size'], n_unique_entities_df['n_unique_'+column+'_ratio'], label=column)
    sns.lineplot(data = n_unique_entities_random_pmid , x= 'pmid', y = column+'_ratio', label = 'random '+column+'_ratio', linestyle = '--',ax=ax)
    #ax.plot(n_unique_entities_random_pmid['size'],n_unique_entities_random_pmid[column], label = 'random '+column, linestyle = '--')
ax.set_xlabel('Dataset size')    
ax.set_ylabel('n unique / dataset size')    
ax.set_xlim([0,None])
ax.set_ylim([0,None])
plt.tight_layout()
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.png', dpi = 200)
plt.savefig(r'C:\Users\d07321ow\Google Drive\Abroad\GME\NAZWA.svg', format= 'svg')





# For n-ary relations







# %% unique pairs, triples

df['triple'] = df['drug_x'] + ' - ' + df['interaction_x'] + ' - ' + df['target_x'] 

df['triple'].nunique()
df.size 

calculate_unique_entities(df, ['triple'], 100)

# %%

n = df.shape[0]
step = 20
column = 'drug_x'

entities = df.loc[:, column].values
mapper = dict([(y,x+1) for x,y in enumerate(sorted(set(entities)))])
df['mapped_column'] = df[column].map(mapper)

pareto_shapes = []
pareto_locs = []
pareto_scales = []

pareto_results = pd.DataFrame()
bs = []
for size in range(2,n,step):
    #b =5

    x_data = np.sort(np.unique(df.loc[:size, 'mapped_column'], return_counts = True)[1])[::-1]
    print(x_data.shape)
    shape_param, loc_param, scale_param = stats.pareto.fit(x_data)
    
    pareto_shapes.append(shape_param)
    pareto_locs.append(loc_param)
    pareto_scales.append(scale_param)
    
    plt.hist(x_data)
    plt.show()
    bs.append(b)
pareto_results['size'] = bs
pareto_results['shape'] = pareto_shapes
pareto_results['loc'] = pareto_locs
pareto_results['scale'] = pareto_scales
pareto_results = pareto_results.round(3)


pareto_results['shape'].plot()
pareto_results['loc'].plot()
pareto_results['scale'].plot()



# %%
import os 
path = r'C:\Users\d07321ow\Google Drive\Abroad\sinfonia_data'
file = 'prompts_umls_relations.csv'
umls = pd.read_csv(os.path.join(path, file))


import sweetviz as sv

my_report = sv.analyze(umls)
my_report.show_html()



file = 'prompts_cdt_relations.csv'
cdt = pd.read_csv(os.path.join(path, file))


import sweetviz as sv

my_report = sv.analyze(cdt)
my_report.show_html()

cdt.groupby('property')['relation'].unique()





















