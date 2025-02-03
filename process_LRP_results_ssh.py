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
parser = argparse.ArgumentParser(description="Process LRP data.")
parser.add_argument(
    "--path_to_save",
    type=str,
    default="/home/d07321ow/scratch/results_LRP_BRCA_Proteomics/results",
    help="Path to save results",
)
parser.add_argument(
    "--path_to_lrp_results",
    type=str,
    default="/home/d07321ow/scratch/results_LRP_BRCA_Proteomics/results",
    help="Path to LRP results",
)
parser.add_argument(
    "--get_topN", type=bool, default=True, help="Flag to get top N interactions"
)
parser.add_argument(
    "--get_topN_topn", type=int, default=100, help="Number of top N interactions to get"
)
parser.add_argument(
    "--get_topN_node_type",
    type=str,
    default=None,
    help="Node type to filter top N interactions",
)

args = parser.parse_args()

path_to_save = args.path_to_save
path_to_lrp_results = args.path_to_lrp_results
get_topN = args.get_topN
get_topN_topn = args.get_topN_topn
get_topN_node_type = args.get_topN_node_type


# %%% load LRP data
lrp_files = f.get_lrp_files(path_to_lrp_results)[:10]
samples = f.get_samples_with_lrp(lrp_files)
# samples = samples[:10]
print("Samples: ", len(samples))
print("Samples: ", len(set(samples)))

start_time = datetime.now()
lrp_dict = f.load_lrp_data(lrp_files, path_to_lrp_results)
end_time = datetime.now()
print(end_time - start_time)
print(lrp_files)
n = len(lrp_files)
print("Number of LRP files: ", n)


# %% get top N interactions for each sample that contain specified node_type (e.g. '_exp')


if get_topN:

    df_topn_edges = pd.DataFrame(
        np.zeros((get_topN_topn, len(samples)), dtype="str"), columns=samples
    )
    df_topn_lrps = pd.DataFrame(
        np.zeros((get_topN_topn, len(samples)), dtype="str"), columns=samples
    )

    print(samples)
    for i, sample_name in enumerate(samples):
        print("get_topN: ", i, sample_name)
        data_temp = lrp_dict[sample_name]
        data_temp = f.filter_and_sort_data(data_temp, get_topN_node_type, get_topN_topn)
        data_temp = f.add_edge_column(data_temp)
        f.add_edges_to_main_df_topn(df_topn_edges, sample_name, data_temp)
        f.add_lrps_to_main_df_topn(df_topn_lrps, sample_name, data_temp)

        # save data_temp to csv
        data_temp.to_csv(
            os.path.join(
                path_to_save,
                "df_topn_for_individuals_top{}_{}_{}.csv".format(
                    get_topN_topn, get_topN_node_type, sample_name
                ),
            )
        )

    df_topn_edges.to_csv(
        os.path.join(
            path_to_save,
            "df_topn_edges_for_individuals_top{}_{}.csv".format(
                get_topN_topn, get_topN_node_type
            ),
        )
    )
    # print that the data is saved
    print(
        "df_topn_edges saved to: ",
        os.path.join(
            path_to_save,
            "df_topn_for_individuals_top{}_{}.csv".format(
                get_topN_topn, get_topN_node_type
            ),
        ),
    )

    df_topn_lrps.to_csv(
        os.path.join(
            path_to_save,
            "df_topn_lrps_for_individuals_top{}_{}.csv".format(
                get_topN_topn, get_topN_node_type
            ),
        )
    )
    # print that the data is saved
    print(
        "df_topn_lrps saved to: ",
        os.path.join(
            path_to_save,
            "df_topn_for_individuals_top{}_{}.csv".format(
                get_topN_topn, get_topN_node_type
            ),
        ),
    )

    unique_edges_df = f.count_unique_edges_in_df_topn(df_topn_edges)

    unique_edges_df.to_csv(
        os.path.join(
            path_to_save,
            "unique_edges_count_in_top_{}_{}.csv".format(
                get_topN_topn, str(get_topN_node_type)
            ),
        )
    )
    # print that unique_edges_df is saved
    print(
        "unique_edges_df saved to: ",
        os.path.join(
            path_to_save,
            "unique_edges_count_in_top_{}_{}.csv".format(
                get_topN_topn, str(get_topN_node_type)
            ),
        ),
    )

    print(df_topn_lrps.head())
    print(df_topn_edges.head())
