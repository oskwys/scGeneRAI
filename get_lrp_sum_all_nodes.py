import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Import your custom functions from functions.py
import functions as f

def save_lrp_sum_to_csv(path_to_save, file_name_to_save, node_list, np_lrp_temp):
    df = pd.DataFrame(np_lrp_temp)
    df ['node'] = node_list        
    df = df.set_index('node')    
    df.to_csv(os.path.join(path_to_save, file_name_to_save), index=True)
    print('LRP sum dataframe 5 first rows:', df.head())
    print('LRP sum dataframe saved to CSV: ', os.path.join(path_to_save, file_name_to_save))

def get_LRP_sum(data_temp, node_list):
    """
    Calculate the sum of LRP values for each node in the node list.
    This function groups the input DataFrame by 'source_gene' and 'target_gene',
    sums the 'LRP' values for each group, and then combines these sums. The result
    is reindexed to match the provided node list, filling any missing values with 0.
    Parameters:
    data_temp (pd.DataFrame): A DataFrame containing at least 'source_gene', 'target_gene', and 'LRP' columns.
    node_list (list): A list of inputs (genes) for which the LRP sums are to be calculated.
    Returns:
    pd.Series: A Series with the sum of LRP values for each node in the node list.
    """
    lrp_sum = data_temp.groupby('source_gene')['LRP'].sum().add(
            data_temp.groupby('target_gene')['LRP'].sum(), fill_value=0
        ).reindex(node_list, fill_value=0)
    
    return lrp_sum


def main(args):
    # Retrieve command line arguments
    path_to_lrp_results = args.path_to_lrp_results
    path_to_save = args.path_to_save
    file_name_to_save = args.file_name_to_save
    
    # Print paths and filenames to inform the user
    print(f"Path to LRP results: {path_to_lrp_results}")
    print(f"Path to save the output: {path_to_save}")
    print(f"Filename for the saved output CSV: {file_name_to_save}")

    # %% Load LRP data and process each file
    lrp_files = f.get_lrp_files(path_to_lrp_results)
    n = len(lrp_files)
    samples = f.get_samples_with_lrp(lrp_files)

    start_time = datetime.now()

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
            node_set = set(data_temp['source_gene']).union(data_temp['target_gene'])
            n_nodes = len(node_set)
            node_list = sorted(list(node_set))  # Create a consistent ordered list of genes
            np_lrp_temp = np.zeros((n_nodes, len(samples)))

        sum_ = get_LRP_sum(data_temp, node_list)
        np_lrp_temp[:, i] = sum_.values

    end_time = datetime.now()
    print(f"Time taken to process {n} files: {end_time - start_time}")

    save_lrp_sum_to_csv(path_to_save, file_name_to_save, node_list, np_lrp_temp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute LRP_sum for each node and each sample')
    parser.add_argument('--path_to_lrp_results', type=str, help='Path to the LRP results folder')
    parser.add_argument('--path_to_save', type=str, help='Path to save the output')
    parser.add_argument('--file_name_to_save', type=str, help='Filename for the saved output CSV with LRP_sum for each node (row) and each sample (column)') 
    args = parser.parse_args()
    main(args)













    