# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:44:45 2023

@author: d07321ow
"""


import pickle
import pandas as pd

import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.cm as cm

import matplotlib
import pyarrow.feather as feather
import itertools

#from scGeneRAI import scGeneRAI
import functions as f
from datetime import datetime

%matplotlib inline 
import importlib, sys
importlib.reload(f)


#path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'
path_to_data = '/home/d07321ow/scratch/results_LRP_BRCA/networks'
#path_to_save = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'
topn = 1000
edges_count = pd.read_csv(os.path.join(path_to_data, 'unique_edges_count_in_top_{}.csv'.format(topn)), index_col = 0)
edges_count = pd.read_csv(os.path.join(path_to_data, 'unique_edges_noexpexp_count_in_top_1000_noexpexp.csv'), index_col = 0)

# %% plot histograms

edges_count = edges_count.sort_values('count', ascending=False).reset_index(drop=True)

# from kneed import KneeLocator
# kneedle = KneeLocator(edges_count.index, edges_count['count'].max() - edges_count['count'], S=1.0, curve="concave", direction="increasing",interp_method="interp1d")
# kneedle.plot_knee_normalized()
# kneedle.plot_knee()
# kneedle.Ds_y


from kneefinder import KneeFinder
kf = KneeFinder(edges_count.index, edges_count['count'])
knee_x, knee_y = kf.find_knee()
# plotting to check the results
kf.plot()




threshold = 0.01 # 0.1%

edges_to_select = edges_count.iloc[:int(edges_count.shape[0] * threshold), :]
edges_to_select = edges_count.iloc[:knee_x, :]



fig, ax = plt.subplots(figsize = (15,5))
ax.plot(edges_to_select.index, edges_to_select['count'])

fig, ax = plt.subplots(figsize = (10,5))
ax.plot(edges_count.index, edges_count['count'])
ax.axvline(edges_to_select.shape[0], linestyle = '--')

fig, ax = plt.subplots(figsize = (10,5))
sns.displot(edges_count['count'],ax=ax,  kind="ecdf")

sns.displot(edges_to_select['count'],height=3, aspect =2)
plt.xlim([0,None])
plt.axvline(edges_to_select['count'].min(), linestyle = '--', color= 'red', label = str(edges_to_select['count'].min()))
plt.xlabel('Interaction occurance in the top 1000 LRP')
plt.ylabel('Count')
plt.legend()

#edges_to_select['edge'].to_csv(os.path.join(path_to_data, 'edges_to_select_{}.csv'.format(topn)))

edges_to_select['edge'].to_csv(os.path.join(path_to_data, 'edges_to_select_1000_noexpexp.csv'))





