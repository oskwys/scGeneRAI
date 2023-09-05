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
import glob
glob.glob('./*.txt')

# %% Load files

path = r'C:\Users\d07321ow\Documents\GitHub\scGeneRAI\results_model_all_batch5_nepochs300_depth2_lr02'
path = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\scGeneRAI_results\results_model_all_batch5_nepochs300_depth2_lr02'
files = []#os.listdir(path)
for file in os.listdir(path):
    if file.endswith("losses.csv"):
        files.append(file)
fig, axs = plt.subplots(len(files),1, figsize = (5,len(files)*4))

for i, file in enumerate(files):
    ax = axs[i]
    losses  = pd.read_csv(os.path.join(path, file), index_col = 0)
    
    
    losses.columns = ['test','train','epochs']
    
    
    losses.plot(x = 'epochs',ax = ax)
    ax.set_ylim([0,.5])
    ax.set_title(file)
    ax.grid()
plt.tight_layout()



# %%
fig, axs = plt.subplots(1,2, figsize = (10,5))

for i, file in enumerate(files):
    
    losses  = pd.read_csv(os.path.join(path, file), index_col = 0)
    
    
    losses.columns = ['test','train','epochs']
    
    ax = axs[0]
    losses.plot(x = 'epochs',y = 'train', ax = ax, label = i)
    ax.set_ylim([0,.25])
    #ax.set_title(file)
    ax.set_title('Training loss')
    ax.grid()
    
    
    ax = axs[1]
    losses.plot(x = 'epochs',y = 'test', ax = ax, label = i)
    ax.set_ylim([0,.25])
    #ax.set_title(file)
    ax.set_title('Testing loss')
    ax.grid()
    
    
plt.tight_layout()