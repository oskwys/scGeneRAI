# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:03:51 2023

@author: d07321ow
"""


import pickle
import pandas as pd

import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt


# %% Load files

path = r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI\results_model_all_batch5_nepochs300_depth2_lr002'
files = os.listdir(path)*2
fig, axs = plt.subplots(1, len(files), figsize = (len(files)*7,5))

for i, file in enumerate(files):
    ax = axs[i]
    losses  = pd.read_csv(os.path.join(path, file), index_col = 0)
    
    
    losses.columns = ['test','train','epochs']
    
    
    losses.plot(x = 'epochs',ax = ax)
    ax.set_ylim([0,.5])
    ax.set_title(file)
    ax.grid()
plt.tight_layout()