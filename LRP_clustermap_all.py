# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:50:50 2024

@author: d07321ow
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

import networkx as nx
import matplotlib.cm as cm

import matplotlib
import pyarrow.feather as feather
import itertools

#from scGeneRAI import scGeneRAI
import functions as f
from datetime import datetime

import importlib, sys
importlib.reload(f)

 #%%
 
path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples'
df = pd.read_csv(os.path.join(path_to_data, 'lrp_median_matrix.csv'))

df = df.set_index('source_gene')



sns.heatmap(df)

df.shape


sns.clustermap(df, method = 'ward', mask = (df == 0) , cmap = 'jet')














