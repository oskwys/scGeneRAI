#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 23:17:50 2023

@author: owysocki
"""
import pyarrow.feather as feather

import pandas as pd
import networkx as nx

import numpy as np
import sys
import os
import seaborn as sns

files = os.listdir('./results')

network_data = pd.concat([pd.read_csv('./results/' + file) for file in files])

network_data['LRP'] = np.abs(network_data['LRP'])
network_data = network_data[network_data['source_gene'] != network_data['target_gene']]

average_network = network_data[['LRP', 'source_gene', 'target_gene']].groupby(['source_gene', 'target_gene']).mean().reset_index()

top_exp = 100
top_mut = 50
top_fus = 50
top_amp = 50
top_del = 50

network_amp = average_network[average_network['source_gene'].str.contains('amp').values].sort_values(by='LRP', ascending=False).iloc[:top_amp,:]
network_mut = average_network[average_network['source_gene'].str.contains('mut').values].sort_values(by='LRP', ascending=False).iloc[:top_mut,:]
network_del = average_network[average_network['source_gene'].str.contains('del').values].sort_values(by='LRP', ascending=False).iloc[:top_del,:]
network_fus = average_network[average_network['source_gene'].str.contains('fus').values].sort_values(by='LRP', ascending=False).iloc[:top_fus,:]




network_exp = average_network.sort_values(by='LRP', ascending=False).iloc[:top_exp,:]

edges = pd.concat((network_exp, network_mut, network_fus, network_amp, network_del))


network = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP')
nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*10)


# %% to gephi

nodes = pd.DataFrame(network.nodes)
nodes = nodes.reset_index().rename(columns = {'index':'id', 0:'label'}).set_index('label')

edges_gephi = edges.reset_index(drop = True)
edges_gephi = edges_gephi.rename(columns = {'source_gene':'Source','target_gene':'Target','LRP':'Weights'})


edges_gephi['Source'] = edges_gephi['Source'].map(nodes.to_dict()['id'])
edges_gephi['Target'] = edges_gephi['Target'].map(nodes.to_dict()['id'])
edges_gephi .set_index('Source').to_csv('edges.csv')

nodes = nodes.reset_index().set_index('id')
nodes.to_csv('nodes.csv')
