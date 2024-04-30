# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:13:05 2024

@author: owysocky
"""
import pandas as pd
import requests
import json
import os

gene_list = pd.read_csv(r'C:\Users\owysocky\Desktop\gene_set.csv')['gene_name'].to_list()
import requests
r = requests.post(
    url='https://biit.cs.ut.ee/gprofiler/api/gost/profile/',
    json={
        'organism':'hsapiens',
        'query':gene_list,
        'sources':["GO:MF","GO:CC","GO:BP"],#,"KEGG","REAC","WP","TF","MIRNA","HPA","CORUM","HP"],
        'user_threshold':0.05, 
        'significance_threshold_method':'bonferroni'
    }
    )
a = r.json()['result']
a = pd.DataFrame(a)


pvalue_threshold = 0.01

df = pd.DataFrame(a)

df_f = df.loc[df['p_value'] < pvalue_threshold]

df_f = df_f.to_dict()


sort_by = ['recall']
select_topn = 20

enrichment_data_filtered = df_f
df = pd.DataFrame(enrichment_data_filtered)
df = df.sort_values(by = sort_by, ascending = False)
df = df.iloc[:select_topn, :].reset_index(drop = True)

result = df.to_dict()

cols = ['name', 'description', 'intersection_size', 'precision','recall']

df_to_markdown = df[cols]

a = df_to_markdown.to_markdown(index=False)

prompt_instruction ='sa da aNAN NANANAN '
prompt = prompt_instruction + '\n\n\n' + a

import numpy as np
from scipy.stats import fisher_exact
def compute_fisher_test(list1, list2):
    """
    Compute Fisher's Exact Test of overlap between two lists of strings.

    :param list1: First list of unique strings
    :param list2: Second list of unique strings
    :param total_population: Total population size from which lists are drawn
    :return: p-value from the Fisher's Exact Test
    """
    # Calculate overlap and unique counts
    total_population = 17489
    overlap = len(set(list1) & set(list2))
    only_list1 = len(set(list1) - set(list2))
    only_list2 = len(set(list2) - set(list1))
    neither = total_population - (overlap + only_list1 + only_list2)
    
    # Create contingency table
    contingency_table = np.array([[overlap, only_list1], [only_list2, neither]])
    
    # Compute Fisher's Exact Test
    _, p_value = fisher_exact(contingency_table, alternative='two-sided')
    
    return p_value


list1 = ['gene1', 'gene2', 'gene3', 'gene4']
list2 = ['gene2', 'gene3', 'gene5', 'gene6']
  # Example total population size

p_value = compute_fisher_test(list1, list2)












