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

# from scGeneRAI import scGeneRAI
import functions as f
from datetime import datetime

import importlib, sys
import argparse

importlib.reload(f)
# %%



# %% get samples
# Parse command line arguments
parser = argparse.ArgumentParser(description='Process LRP data.')
parser.add_argument('--path_to_save', type=str, default="/home/d07321ow/scratch/results_LRP_BRCA/results", help='Path to save results')
parser.add_argument('--path_to_lrp_results', type=str, default="/home/d07321ow/scratch/results_LRP_BRCA/results", help='Path to LRP results')
parser.add_argument('--get_topN', type=bool, default=True, help='Flag to get top N interactions')
parser.add_argument('--get_topN_topn', type=int, default=100, help='Number of top N interactions to get')
parser.add_argument('--get_topN_node_type', type=str, default=None, help='Node type to filter top N interactions')

args = parser.parse_args()

path_to_save = args.path_to_save
path_to_lrp_results = args.path_to_lrp_results
get_topN = args.get_topN
get_topN_topn = args.get_topN_topn
get_topN_node_type = args.get_topN_node_type



samples = f.get_samples_with_lrp(path_to_lrp_results)
# samples = samples[:10]
print("Samples: ", len(samples))
print("Samples: ", len(set(samples)))

# %%% load LRP data


lrp_files = f.get_lrp_files(path_to_lrp_results)

start_time = datetime.now()
lrp_dict = f.load_lrp_data(lrp_files[:20], path_to_lrp_results)
end_time = datetime.now()
print(end_time - start_time)
print(lpr_files[:20])
n = len(lrp_files)
print("Number of LRP files: ", n)


# %% get top N interactions for each sample that contain specified node_type (e.g. '_exp')


if get_topN:

    df_topn = pd.DataFrame(np.zeros((get_topN_topn, len(samples)), dtype="str"), columns=samples)
    print(samples[:20])
    for i, sample_name in enumerate(samples[:20]):
        print('get_topN: ', i, sample_name)
        data_temp = lrp_dict[sample_name]
        data_temp = f.filter_and_sort_data(data_temp, get_topN_node_type, get_topN_topn)
        data_temp = f.add_edge_column(data_temp)
        f.add_to_main_df_topn(df_topn, sample_name, data_temp)

        # save data_temp to csv
        data_temp.to_csv(
            os.path.join(
                path_to_save,
                "df_topn_for_individuals_top{}_{}_{}.csv".format(
                    get_topN_topn, get_topN_node_type, sample_name
                ),
            )
        )

    df_topn.to_csv(
        os.path.join(
            path_to_save, "df_topn_for_individuals_top{}_{}.csv".format(get_topN_topn, get_topN_node_type)
        )
    )

    unique_edges_df = f.count_unique_edges(df_topn)

    unique_edges_df.to_csv(
        os.path.join(
            path_to_save,
            "unique_edges_count_in_top_{}_{}.csv".format(get_topN_topn, str(get_topN_node_type)),
        )
    )
