# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:24:43 2023

@author: d07321ow
"""

#import pyarrow.feather as feather
import pickle
import pandas as pd

import numpy as np
import sys
import os
import seaborn as sns
from scGeneRAI import scGeneRAI
import pyarrow.feather as feather

# %% load data

#path_to_data = r'C:\Users\d07321ow\Google Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model'
path_to_data = '/home/d07321ow/scratch/scGeneRAI/data/data_BRCA'

data = pd.read_csv( os.path.join(path_to_data, 'data_to_model_proteomics.csv') ,index_col = 0 )

# %% TRAIN
model_depth = 2
nepochs = 500
device = 'cuda'
batch_size = 5
learning_rate = 2e-2 # default = 2e-2
#file_name = 'model_BRCA_batch{}_nepochs{}_depth{}_lr{}.pkl'.format(batch_size, nepochs, model_depth, str(learning_rate).replace('0.', ''))
file_name = 'model_BRCA_GenProt_batch{}_nepochs{}_depth{}_lr{}.pkl'.format(batch_size, nepochs, model_depth, str(learning_rate).replace('0.', ''))
print(file_name)

print('\n ___________ NEW MODEL ___________ \n')

data_temp = data.copy()

# %% fit

model = scGeneRAI()

testlosses, trainlosses, epoch_list = model.fit(data_temp, nepochs, model_depth, lr=learning_rate, batch_size=batch_size, lr_decay = 0.99, descriptors = None, early_stopping = False, device_name = device)
print('\n\n __________________ MODEL TRAINED ! __________________\n')

# %% saving the model


with open(file_name, 'wb') as file:
    pickle.dump(model, file)
    print(f'Object successfully saved to "{file_name}"')


path_to_save_models = '/home/d07321ow/scratch/results_LRP_BRCA_Proteomics'

pd.DataFrame(np.array([testlosses, trainlosses, epoch_list]).T, columns = ['test_loss','train_loss','epochs']).to_csv(os.path.join(path_to_save_models, file_name+'losses.csv'))

print('\n\n __________________ END ! __________________\n')




#preds = model.predict_networks(data.iloc[:30,:], descriptors = None, LRPau = True, remove_descriptors = True, device_name = device, PATH = '.')



#with open(file_name, 'rb') as file:
#    model_ = pickle.load(file)
#    print(f'Object successfully loaded from "{file_name}"')

