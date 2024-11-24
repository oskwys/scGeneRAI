import functions as f
"""
This script performs the following tasks:
1. Imports necessary libraries and modules.
2. Reloads the custom functions module.
3. Defines the path to save figures.
4. Loads input data for the model.
5. Filters clinical features based on sample list and adds cluster information.
6. Groups samples based on clinical features.
7. Reads LRP (Layer-wise Relevance Propagation) sum values and calculates their mean.
8. Generates and saves a violin plot for LRP sum values.
9. Reads LRP sum values for different sample groups.
10. Generates and saves violin plots for LRP sum values across different sample groups.
Functions:
- `f.get_input_data(path_to_data)`: Loads input data for the model.
- `f.add_cluster2(df_clinical_features)`: Adds cluster information to clinical features.
- `f.get_samples_by_group(df_clinical_features)`: Groups samples based on clinical features.
Variables:
- `path_to_save_figures`: Path to save the generated figures.
- `path_to_data`: Path to the input data for the model.
- `data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features`: DataFrames containing input data and clinical features.
- `samples`: List of sample identifiers.
- `samples_groups`: Dictionary containing grouped samples.
- `df_lrp_sum`: DataFrame containing LRP sum values.
- `df_lrp_sum_mean`: DataFrame containing mean LRP sum values.
- `path_to_lrp_mean`: Path to the LRP mean values.
- `lrp_dict_groups`: Dictionary containing LRP sum values for different sample groups.
"""
import importlib, sys
importlib.reload(f)
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# %%
path_to_save_figures = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\plots_to_paper'

# %% samples
path_to_data = r'G:\My Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'
data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = f.get_input_data(path_to_data)

samples = pd.read_csv(r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt', index_col = 0)['samples'].to_list()

df_clinical_features = df_clinical_features[df_clinical_features['bcr_patient_barcode'].isin(samples)].reset_index(drop=True)
df_clinical_features = f.add_cluster2(df_clinical_features)

# samples groups
samples_groups = f.get_samples_by_group(df_clinical_features)


# %% lrp_sum_mean for samples
df_lrp_sum = pd.read_csv(r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\lrp_sum_mean.csv', index_col = 0)

df_lrp_sum_mean = pd.DataFrame(df_lrp_sum.mean().values, columns = ['LRP_sum'])
# Create a figure and axes for the plots
fig, ax = plt.subplots(figsize=(3, 5), sharey=True)
sns.violinplot(ax=ax, y='LRP_sum', data=df_lrp_sum_mean)
handles, labels = ax.get_legend_handles_labels()
labels = [label.replace('LRP_sum_mean_', '').replace('pos', 'Positive').replace('_neg', ' Negative').replace('_',' ') for label in labels]  # Remove 'LRP' from the labels
ax.set_ylabel('$LRP_{sum}$')
ax.set_title('All samples')
plt.tight_layout()
plt.savefig(os.path.join(path_to_save_figures , 'LRP_sum_violinplot_samples'+ '.pdf'), format = 'pdf')


df_lrp_sum_mean = pd.DataFrame(df_lrp_sum.mean(axis=1).values, columns = ['LRP_sum'])
# Create a figure and axes for the plots
fig, ax = plt.subplots(figsize=(3, 5), sharey=True)
sns.violinplot(ax=ax, y='LRP_sum', data=df_lrp_sum_mean)
handles, labels = ax.get_legend_handles_labels()
labels = [label.replace('LRP_sum_mean_', '').replace('pos', 'Positive').replace('_neg', ' Negative').replace('_',' ') for label in labels]  # Remove 'LRP' from the labels
ax.set_ylabel('$LRP_{sum}$')
ax.set_title('Genes')
plt.tight_layout()
plt.savefig(os.path.join(path_to_save_figures , 'LRP_sum_violinplot_genes'+ '.pdf'), format = 'pdf')



# %% LRP mean violinplots 
# This cell reads a CSV file containing LRP (Layer-wise Relevance Propagation) sum values into a DataFrame.
# It then calculates the mean of these values and creates a new DataFrame with the mean values.
# A violin plot is generated to visualize the distribution of the LRP sum values.
# The plot is saved as a PDF file.

path_to_lrp_mean = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'
lrp_dict_groups = {}

for group in list(samples_groups.keys())[:-1]:
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)
    
    df_lrp = pd.read_csv(os.path.join(path_to_lrp_mean, 'LRP_sum_mean_{}.csv'.format(group)), index_col=0)
    lrp_dict_groups[group] = df_lrp


# Create a figure and axes for the plots
fig, axes = plt.subplots(1, len(lrp_dict_groups), figsize=(14, 5), sharey=True)

# Iterate over the dictionary and create violin plots for each key
for i, (key, data) in enumerate(lrp_dict_groups.items()):
    ax = axes[i]
    sns.violinplot(ax=ax, y='LRP_sum', hue='group', data=data.melt('gene', var_name='group', value_name='LRP_sum'), split=True)
    #ax.set_title(key)
    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace('LRP_sum_mean_', '').replace('pos', 'Positive').replace('_neg', ' Negative').replace('_',' ') for label in labels]  # Remove 'LRP' from the labels
    ax.legend_.set_title('Group')
    ax.legend(handles, labels)
    if i == 0:
        ax.set_ylabel('$LRP_{sum}$')
plt.suptitle('Mean $LRP_{sum}$ for a sample, stratified by sample group')
plt.tight_layout()
plt.savefig(os.path.join(path_to_save_figures , 'LRP_sum_groups_violinplot'.format(group) + '.pdf'), format = 'pdf')


# %%
