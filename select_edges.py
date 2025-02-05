#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for processing edge counts:
  - Loads a CSV file containing edge counts.
  - Sorts the edges and either selects a top fraction (if a threshold is provided)
    or uses a knee‐finder algorithm to select the edges.
  - Creates several plots (e.g. the knee finder plot, the ECDF plots, etc.) and
    optionally saves and/or displays them.
  - Saves the selected edges (the "edge" column) to a CSV file.
  
Usage (from the command line):
    python process_edges.py --data_path "path/to/data" [options]

Options include:
    --file_name: Name of the input CSV file (default: unique_edges_count_in_top_1000_None.csv)
    --output_edges_file: Name for saving selected edges (default: edges_to_select_1000_None.csv)
    --save_plots_dir: Directory to save plots (if not provided, plots won’t be saved)
    --show_plots: Flag to display plots (by default, plots are not displayed)
    --threshold: A float between 0 and 1 (e.g., 0.01 for top 1%). If provided the top fraction
                 of edges will be selected. If not provided, the knee-finder method is used.
                 
Note: This script uses the external package "kneefinder" for the knee-finding algorithm.
      Make sure it is installed (e.g. via pip install kneefinder).
"""

import os
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def process_edges(edges_count, threshold=None):
    """
    Process the edges_count DataFrame by sorting and selecting edges either via
    a fraction threshold or a knee-finder method.
    
    Parameters:
        edges_count (pd.DataFrame): DataFrame with columns 'edge' and 'count'.
        threshold (float or None): If provided, use the top fraction of edges.
                                   Otherwise, use the knee-finder method.
        
    Returns:
        edges_count (pd.DataFrame): The sorted DataFrame.
        selected_edges (pd.DataFrame): The selected subset of edges.
        knee_x (int or None): The index at which the knee occurs (if computed).
        knee_y (float or None): The y-value at the knee (if computed).
        kf (KneeFinder or None): The KneeFinder object (if computed).
    """
    # Sort the data by the "count" column in descending order.
    edges_count = edges_count.sort_values('count', ascending=False).reset_index(drop=True)
    
    if threshold is not None:
        num_edges = int(edges_count.shape[0] * threshold)
        selected_edges = edges_count.iloc[:num_edges, :]
        return edges_count, selected_edges, None, None, None
    else:
        # Use the KneeFinder to determine a threshold index.
        from kneefinder import KneeFinder
        kf = KneeFinder(edges_count.index, edges_count['count'])
        knee_x, knee_y = kf.find_knee()
        selected_edges = edges_count.iloc[:knee_x, :]
        return edges_count, selected_edges, knee_x, knee_y, kf

def plot_results(edges_count, selected_edges, knee_x=None, knee_y=None, kf=None, 
                 save_plots_dir=None):
    """
    Create and optionally save/display plots.
    
    Plots include:
      - The knee finder plot (if using the knee-finder method).
      - A plot of the selected edges count vs. index.
      - A plot of the full edges count with a vertical line at the selection threshold.
      - ECDF plots for both the full and selected edges counts.
      
    Parameters:
        edges_count (pd.DataFrame): The DataFrame with all edges.
        selected_edges (pd.DataFrame): The DataFrame with selected edges.
        knee_x (int or None): The knee index (if computed).
        knee_y (float or None): The knee y-value (if computed).
        kf (KneeFinder or None): The KneeFinder object (if computed).
        save_plots_dir (str or None): If provided, the directory in which to save plots.
        show_plots (bool): Whether to display the plots.
    """
    
    # Plot the knee finder results, if available.
    if kf is not None:
        # Create a new figure and axis for the knee finder plot.
        fig_kf, ax_kf = plt.subplots(figsize=(8, 6))
        # Set the current axis to the one we just created.
        plt.sca(ax_kf)
        # Call kf.plot(), which will now plot on ax_kf.
        kf.plot()
        ax_kf.set_title("Knee Finder Plot")
        if save_plots_dir:
            fig_kf.savefig(os.path.join(save_plots_dir, "knee_finder_plot.png"))
        plt.close(fig_kf)
    
    
    # Plot: Selected edges count vs. index.
    fig1, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(selected_edges.index, selected_edges['count'])
    ax1.set_title("Selected Edges Count")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Count")
    if save_plots_dir:
        fig1.savefig(os.path.join(save_plots_dir, "selected_edges_plot.png"))
    plt.close(fig1)
    
    # Plot: Full edges count with a vertical line at the selection threshold.
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(edges_count.index, edges_count['count'], label="All Edges")
    ax2.axvline(selected_edges.shape[0], linestyle='--', color='red', label="Selection Threshold")
    ax2.set_title("All Edges Count with Selection Threshold")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Count")
    ax2.legend()
    if save_plots_dir:
        fig2.savefig(os.path.join(save_plots_dir, "edges_count_with_threshold.png"))
    plt.close(fig2)
    
    
def main(args):
    # Create the plots directory if needed.
    if args.save_plots_dir:
        os.makedirs(args.save_plots_dir, exist_ok=True)
    
    # Construct the full path to the CSV file.
    csv_file_path = os.path.join(args.data_path, args.file_name)
    print('csv_file_path: ', csv_file_path)
    if not os.path.isfile(csv_file_path):
        print("Error: File " + csv_file_path + " does not exist.")
        sys.exit(1)
    
    # Load the CSV data.
    edges_count = pd.read_csv(csv_file_path, index_col=0)
    
    # Process the edges (using either the provided threshold or the knee finder).
    edges_count, selected_edges, knee_x, knee_y, kf = process_edges(edges_count, threshold=args.threshold)
    
    # Plot the results.
    plot_results(edges_count, selected_edges, knee_x=knee_x, knee_y=knee_y, kf=kf,
                    save_plots_dir=args.save_plots_dir)
    
    # Save the selected edges (only the 'edge' column).
    output_csv_path = os.path.join(args.data_path, args.output_edges_file)
    selected_edges['edge'].to_csv(output_csv_path, index=False)
    print("Selected edges saved to " + output_csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process edge counts, detect a knee in the distribution, and save selected edges."
    )
    parser.add_argument(
        "--data_path", "-d", type=str, required=True,
        help="Path to the data directory containing the CSV file."
    )
    parser.add_argument(
        "--file_name", "-f", type=str, default="unique_edges_count_in_top_1000_None.csv",
        help="Name of the CSV file with edge counts."
    )
    parser.add_argument(
        "--output_edges_file", "-o", type=str, default="edges_to_select_1000_None.csv",
        help="Output CSV file name for selected edges."
    )
    parser.add_argument(
        "--save_plots_dir", type=str, default=None,
        help="Directory to save the plots. If not provided, plots will not be saved."
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help=("Fraction threshold to select top edges (e.g., 0.01 for top 1%%). "
                "If not provided, the knee-finder method is used.")
    )
    
    args = parser.parse_args()
    main(args)
