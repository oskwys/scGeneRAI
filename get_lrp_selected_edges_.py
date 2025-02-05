#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:26:52 2023

This script loads LRP data for BRCA, selects specific network edges,
builds an LRP matrix, and saves the results as a CSV file.

Usage (from the terminal):
    python run_lrp.py \
        --path_to_lrp_results /home/d07321ow/scratch/scGeneRAI/data/data_BRCA \
        --path_and_file_name_edges selected_edges_prot.csv \
        --path_to_save /home/d07321ow/scratch/results_LRP_BRCA/networks \
        --file_name_to_save LRP_individual_selected_edges.csv
"""

import argparse
import os
from datetime import datetime

import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib
import pyarrow.feather as feather
import itertools

# Import your custom functions from functions.py
import functions as f

def main(args):
    # Retrieve command line arguments
    path_to_lrp_results = args.path_to_lrp_results
    path_and_file_name_edges = args.path_and_file_name_edges
    path_to_save = args.path_to_save
    file_name_to_save = args.file_name_to_save

    # Print paths and filenames to inform the user
    print(f"Path to LRP results: {path_to_lrp_results}")
    print(f"Path and filename for selected edges: {path_and_file_name_edges}")
    print(f"Path to save the output: {path_to_save}")
    print(f"Filename for the saved output CSV: {file_name_to_save}")

    # %% Load the edges to select from CSV
    edges_to_select = pd.read_csv(path_and_file_name_edges)['edge'].to_list()

    # %% Load LRP data and process each file
    lrp_files = f.get_lrp_files(path_to_lrp_results)
    n = len(lrp_files)
    samples = f.get_samples_with_lrp(lrp_files)

    start_time = datetime.now()

    # Create an empty LRP matrix with dimensions (#edges, #samples)
    LRP_matrix = np.zeros((len(edges_to_select), n))

    # Process each sample's LRP data
    for i in range(n):
        file_name = lrp_files[i]
        sample_name = samples[i]

        print(f"Processing file {i}: {file_name}")
        file_path = os.path.join(path_to_lrp_results, file_name)
        data_temp = pd.read_pickle(file_path, compression='infer', storage_options=None)

        # Remove rows where the source and target are the same
        data_temp = f.remove_same_source_target(data_temp)

        # For the first sample, add the edge column and extract the selected edges
        if i == 0:
            data_temp_0 = f.add_edge_column(data_temp)
            data_temp_0 = f.add_edge_column(data_temp_0)  # as in your original code
            index_ = data_temp_0['edge'].isin(edges_to_select)
            edges = data_temp_0.loc[index_, 'edge'].values

        # Use the same index (from the first iteration) to select the rows of interest
        data_temp = data_temp[index_].reset_index(drop=True)
        LRP_matrix[:, i] = data_temp['LRP'].values

    end_time = datetime.now()
    print("Total processing time:", end_time - start_time)

    # Create a DataFrame from the LRP matrix with appropriate row and column labels
    LRP_pd = pd.DataFrame(LRP_matrix, columns=samples, index=edges)

    # Ensure the output directory exists
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    # Save the resulting DataFrame to a CSV file
    output_file = os.path.join(path_to_save, file_name_to_save)
    LRP_pd.to_csv(output_file)
    print("CSV file saved at:", output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process LRP data and extract network edges.'
    )
    parser.add_argument(
        '--path_to_lrp_results',
        type=str,
        required=True,
        help='Path to the LRP results directory.'
    )
    parser.add_argument(
        '--path_and_file_name_edges',
        type=str,
        required=True,
        help='Path and filename for the CSV file containing the selected edges.'
    )
    parser.add_argument(
        '--path_to_save',
        type=str,
        required=True,
        help='Directory where the output CSV will be saved.'
    )
    parser.add_argument(
        '--file_name_to_save',
        type=str,
        required=True,
        help='Filename for the saved output CSV.'
    )

    args = parser.parse_args()
    main(args)
