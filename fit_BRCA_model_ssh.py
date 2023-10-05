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

df_exp = feather.read_feather( os.path.join(path_to_data, 'CCE_expressions_to_model') , )

df_fus = pd.read_csv(os.path.join(path_to_data, 'CCE_fusions_to_model.csv') ,index_col = 0)
df_mut = pd.read_csv( os.path.join(path_to_data, 'CCE_mutations_to_model.csv') ,index_col = 0)
df_amp = pd.read_csv( os.path.join(path_to_data, 'CCE_amplifications_to_model.csv'),index_col = 0 )
df_del = pd.read_csv( os.path.join(path_to_data, 'CCE_deletions_to_model.csv') ,index_col = 0 )
#df_exp = pd.read_csv( os.path.join(path_to_data, 'CCE_expressions_to_model.csv') ,index_col = 0 )

#df_clinical_features = pd.read_csv( os.path.join(path_to_data, 'CCE_clinical_features.csv') )

# %% preprocess data
df_exp = df_exp.apply(lambda x: np.log(x + 1))
df_exp_stand = (df_exp-df_exp.mean(axis=0))/df_exp.std(axis=0)

df_mut_scale = (df_mut-df_mut.min(axis=0))/df_mut.max(axis=0)
df_fus_scale = (df_fus-df_fus.min(axis=0))/df_fus.max(axis=0)

df_amp[df_amp==2] =1


# %% data to model

data = pd.concat((df_exp_stand, df_mut_scale, df_amp, df_del, df_fus_scale), axis = 1)


# %% TRAIN
model_depth = 2
nepochs = 500
device = 'cuda'
batch_size = 5
learning_rate = 2e-2 # default = 2e-2
file_name = 'model_BRCA_batch{}_nepochs{}_depth{}_lr{}.pkl'.format(batch_size, nepochs, model_depth, str(learning_rate).replace('0.', ''))
file_name = 'model_TCGA_batch{}_nepochs{}_depth{}_lr{}.pkl'.format(batch_size, nepochs, model_depth, str(learning_rate).replace('0.', ''))
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


path_to_save_models = '/home/d07321ow/scratch/results_LRP_BRCA'

pd.DataFrame(np.array([testlosses, trainlosses, epoch_list]).T, columns = ['test_loss','train_loss','epochs']).to_csv(os.path.join(path_to_save_models, file_name+'losses.csv'))

print('\n\n __________________ END ! __________________\n')




#preds = model.predict_networks(data.iloc[:30,:], descriptors = None, LRPau = True, remove_descriptors = True, device_name = device, PATH = '.')



#with open(file_name, 'rb') as file:
#    model_ = pickle.load(file)
#    print(f'Object successfully loaded from "{file_name}"')

