"""
This script performs a comprehensive analysis of gene expression and pathway data using Mann-Whitney U-tests and Chi-squared tests. 
It calculates group averages, assigns pathway memberships to genes, and generates various plots to visualize the results.
The script includes the following main steps:
1. Load gene and pathway data.
2. Calculate group averages for LRP (Layer-wise Relevance Propagation) sums.
3. Perform Mann-Whitney U-tests to compare LRP sums between different groups.
4. Generate volcano plots and bubble plots to visualize the results.
5. Perform traditional statistical comparisons (Mann-Whitney U-tests) on gene expression data.
6. Perform Chi-squared tests on categorical data (mutations, fusions, amplifications, deletions).
7. Save the results to CSV files and generate plots for publication.
Functions:
- calculate_group_averages(df, group_dict): Calculates the average values for each group.
- assign_pathway_memebership_to_gene(genes_pathways, df): Assigns pathway memberships to genes.
"""


import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

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

importlib.reload(f)

# %% Load genes and pathways
path_to_genes = r"G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\PATHWAYS"
genes_pathways = pd.read_csv(
    os.path.join(path_to_genes, "genes_pathways_pancanatlas_matched_cce.csv"),
    index_col=0,
)
pathways = list(genes_pathways["Pathway"].unique())
# Load LRP_sum data
path_to_lrp_sum_mean = r"G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples"
df_lrp = pd.read_csv(
    os.path.join(path_to_lrp_sum_mean, "LRP_sum_mean.csv"), index_col=0
)

# add sample_names as columns
samples = pd.read_csv(
    r"G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt",
    index_col=0,
)["samples"].to_list()
df_lrp.columns = samples


path_to_data = r"G:\My Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model"
data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = (
    f.get_input_data(path_to_data)
)

df_clinical_features = df_clinical_features[
    df_clinical_features["bcr_patient_barcode"].isin(samples)
].reset_index(drop=True)
df_clinical_features = f.add_cluster2(df_clinical_features)

# samples groups
samples_groups = f.get_samples_by_group(df_clinical_features)

#


def calculate_group_averages(df, group_dict):
    group_averages = {}

    for group, subgroups in group_dict.items():
        for subgroup, samples in subgroups.items():
            # Check if all samples exist in the dataframe columns
            valid_samples = [sample for sample in samples if sample in df.columns]

            if valid_samples:
                # Calculate the average for the valid samples
                group_averages[subgroup] = df[valid_samples].mean(axis=1)
            else:
                group_averages[subgroup] = None
                print(f"No valid samples found for subgroup: {subgroup}")

    return pd.DataFrame(group_averages)


# Calculate group averages
group_averages_df = calculate_group_averages(df_lrp, samples_groups)

# %%% paths
path_to_data = r"G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\mwu"
path_to_lrp_mean = r"G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples"
path_to_save = r"G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\mwu\plots"
path_to_save_csv = r"G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\mwu"
path_to_save_figures = r"G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\plots_to_paper"
import os


# %% analyse nodes MWU LRP
def assign_pathway_memebership_to_gene(genes_pathways, df):
    df["gene_name"] = df["gene"].apply(lambda x: x.split("_")[0])

    # add columns with pathway names
    for pathway in genes_pathways["Pathway"].unique():
        df[pathway] = df["gene_name"].apply(
            lambda x: (
                1
                if x
                in genes_pathways.loc[
                    genes_pathways["Pathway"] == pathway, "cce_match"
                ].values
                else 0
            )
        )
    return df


# define dataframe
mwu_results_all = pd.DataFrame()

for group in list(samples_groups.keys())[:-1]:
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)

    df_lrp = pd.read_csv(
        os.path.join(path_to_lrp_mean, "LRP_sum_median_{}.csv".format(group)),
        index_col=0,
    )
    df_lrp.columns
    df_lrp = df_lrp[[df_lrp.columns[1], df_lrp.columns[0], df_lrp.columns[2]]]

    mwu_results = pd.read_csv(
        os.path.join(path_to_data, "mwu_sum_LRP_genes_{}.csv".format(group)),
        index_col=0,
    )
    mwu_results["gene"] = df_lrp["gene"]

    mwu_results = mwu_results.merge(df_lrp, on="gene", how="inner")
    mwu_results = mwu_results.drop(columns="genes")
    # add column with the gene type, e.g. exp, amp, del etc
    mwu_results["type"] = mwu_results["gene"].str.split("_", expand=True)[1]

    # add column with the difference in LRP sum between the two subgroups
    mwu_results["LRP_sum_diff"] = mwu_results.iloc[:, 6] - mwu_results.iloc[:, 7]

    mwu_results = assign_pathway_memebership_to_gene(genes_pathways, mwu_results)

    # sort by gene name
    mwu_results = mwu_results.sort_values("gene")

    # add column with the group name
    mwu_results["group"] = group

    # concatenate to the dataframe with all pathways
    mwu_results_all = pd.concat([mwu_results_all, mwu_results])


mwu_results_all["p-val"] = mwu_results_all["p-val"].astype(float)

mwu_results_all["-log10(p-val)"] = -np.log(mwu_results_all["p-val"])

# add a string column that contains all the pathways for each gene
mwu_results_all["pathway"] = mwu_results_all[pathways].apply(
    lambda x: ", ".join(x.index[x == 1]), axis=1
)


# save to csv
mwu_results_all.to_csv(os.path.join(path_to_save_csv, "mwu_sum_LRP_genes_all.csv"))


# %% plot vulcano plot
mwu_results_all = pd.read_csv(
    os.path.join(path_to_data, "mwu_sum_LRP_genes_all.csv"),
    index_col=0,
)


for group in list(samples_groups.keys())[:-1]:
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)

    # filter mwu_results_all for the group
    mwu_results = mwu_results_all.loc[mwu_results_all["group"] == group, :]

    # select only genes from pathweays
    mwu_results = mwu_results[mwu_results[pathways].sum(axis=1) > 0]

    # plot scatterplot for p-val vs difference in LRP sum, color the points by type, add text annotation with gene name for p-val < 0.00001
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=mwu_results,
        y="-log10(p-val)",
        x="LRP_sum_diff",
        ax=ax,
        hue="type",
        style="type",
        alpha=0.8,
        s=10,
    )

    # Add text annotation with gene name for p-val < 0.00001
    annotation_threshold = 5
    for i, row in mwu_results.iterrows():
        if row["-log10(p-val)"] > annotation_threshold:
            ax.text(
                row["LRP_sum_diff"] + 0.01,
                row["-log10(p-val)"] + 1,
                row["gene_name"],
                fontsize=5,
                alpha=1,
            )

    # Make x axis symmetric, center 0
    ax.axhline(-np.log(0.001), linestyle="--", color="gray")
    ax.axvline(0, linestyle="--", color="gray")
    ax.set_title("Mann-Whitney U-test between two groups: {}".format(group))
    ax.set_xlabel("Difference in median $LRP_{sum}$ for each gene")
    ax.set_ylabel("-log10(p-val)")
    # ax.set_xlim([-3, 3])
    plt.tight_layout()
    # save figure to pdf
    plt.savefig(
        os.path.join(
            path_to_save_figures,
            "scatterplot_LRP_sum_diff_vs_pval_{}.pdf".format(group),
        ),
        format="pdf",
    )

    # plot facetGrid scatterplot for p-val vs difference in LRP sum, color the points by type, add text annotation with gene name for p-val < 0.00001
    # Create FacetGrid
    g = sns.FacetGrid(
        mwu_results,
        col="pathway",
        col_wrap=3,  # Adjust number of columns here according to the number of pathways and layout preference
        height=4,
        aspect=1.5,
    )

    # Mapping the scatterplot
    g.map_dataframe(
        sns.scatterplot,
        x="LRP_sum_diff",
        y="-log10(p-val)",
        hue="type",
        style="type",
        alpha=1,
        s=15,
    )
    annotation_threshold = 5

    # Add text annotation with gene name for p-val < 0.00001
    def add_annotations(data, **kwargs):
        ax = plt.gca()
        for i, row in data.iterrows():
            if -np.log(row["p-val"]) > annotation_threshold:
                ax.text(
                    row["LRP_sum_diff"] + 0.01,
                    row["-log10(p-val)"] + 1,
                    row["gene_name"],
                    fontsize=5,
                    alpha=1,
                )

    g.map_dataframe(add_annotations)
    # define x and y axis labels
    g.set_axis_labels("Difference in median $LRP_{sum}$ for each gene", "-log10(p-val)")
    # add axvline and axhline
    # Add vertical and horizontal lines for threshold and zero
    for ax in g.axes.flat:
        ax.axhline(annotation_threshold, linestyle="--", color="gray")
        ax.axvline(0, linestyle="--", color="gray")

    # Manually add legend to each axis
    for ax in g.axes.flat:
        unique_handles, unique_labels = ax.get_legend_handles_labels()
        ax.legend(
            unique_handles,
            unique_labels,
            title="Type",
            fontsize=8,
            title_fontsize=10,
            loc="upper right",  # or any preferred location
            frameon=True,
        )

    g.set_titles("Pathway: {col_name}")

    plt.savefig(
        os.path.join(
            path_to_save_figures,
            "scatterplot_LRP_sum_diff_vs_pval_{}_facetgrid.pdf".format(group),
        ),
        format="pdf",
    )


# %% prepate bubble plot with all scores
# define dataframe with all groups
'''df_bubbleplot = pd.DataFrame()
for group in list(samples_groups.keys())[:-1]:
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)

    mwu_results = pd.read_csv(
        os.path.join(
            path_to_data, "mwu_sum_LRP_genes_{}_pathwaygenesonly.csv".format(group)
        ),
        index_col=0,
    )

    mwu_results["-log10(p-val)"] = -np.log(mwu_results["p-val"])
    # select only column with -log10(p-val), LRP_sum_diff, gene_name, type, pathway
    mwu_results_filtered = mwu_results[
        ["-log10(p-val)", "LRP_sum_diff", "gene_name", "type", "pathway"]
    ]
    # add column with group name
    mwu_results_filtered["group"] = group

    # concatenate to the dataframe with all groups
    df_bubbleplot = pd.concat([df_bubbleplot, mwu_results_filtered])'''


# define dataframe with all groups
df_bubbleplot = pd.read_csv(
    os.path.join(path_to_data, "all_genes_with_stats_and_LRP_pvals.csv"), index_col=0
)

df_bubbleplot["type"] = df_bubbleplot["gene"].str.split("_", expand=True)[1]
# sort by gene name
df_bubbleplot = df_bubbleplot.sort_values("gene")

df_bubbleplot = df_bubbleplot[["-log10(p-val)_lrp", "gene_name", "type", "pathway", "group"]]

# drop rows with NaN pathway
df_bubbleplot = df_bubbleplot.dropna(subset=["pathway"]).reset_index(drop=True)

# melt the dataframe so the type is in the columns, index is gene_name, pathway and type, valeus are -log10(p-val)
df_bubbleplot_melted = df_bubbleplot.melt(
    id_vars=["gene_name", "pathway", "group", "type"],
    value_vars=["-log10(p-val)_lrp"],
    value_name="Value",
)
# pivot the dataframe so the type is in the columns, index is gene_name, pathway and type, valeus are -log10(p-val)
df_bubbleplot_pivot = df_bubbleplot_melted.pivot_table(
    index=["gene_name", "pathway", "group"], columns="type", values="Value"
).reset_index()

# Pivoting the DataFrame
pivoted_df = df_bubbleplot_pivot.melt(
    id_vars=["gene_name", "pathway", "group"],
    value_vars=["amp", "del", "exp", "fus", "mut"],
    var_name="variable",
    value_name="value",
)

# Create new column names combining group and variable
pivoted_df["new_column"] = pivoted_df["group"] + "_" + pivoted_df["variable"]

# Pivot the table to get desired format
result_df = pivoted_df.pivot_table(
    index=["gene_name", "pathway"], columns="new_column", values="value"
).reset_index()

# Due to pivoting, the column names become a MultiIndex, we can fix this:
result_df.columns.name = None
# sort column names after third column, but keep gene_name and pathway in the first two columns
result_df = result_df[["gene_name", "pathway"] + sorted(result_df.columns[2:])]

# melt the dataframe so the columns starting from the thirds are in the rows, index is gene_name, pathway
result_df_melted = result_df.melt(
    id_vars=["gene_name", "pathway"],
    value_vars=result_df.columns[2:],
    value_name="Value",
    var_name="Type",
)
# sort the dataframe by pathway, and then by gene_name, and then by Type
result_df_melted = result_df_melted.sort_values(
    ["pathway", "gene_name", "Type"], ascending=True
).reset_index(drop=True)

# add column that joins gene name and pathway
result_df_melted["gene_pathway"] = (
    result_df_melted["gene_name"] + " (" + result_df_melted["pathway"] + ")"
)

# Extract color categories from 'Type'
result_df_melted["color_category"] = result_df_melted["Type"].apply(
    lambda x: x.split("_")[-1]
)


# Define the color mapping function
def map_color(string):
    if "mut" in string:
        return "red"
    elif "amp" in string:
        return "blue"
    elif "exp" in string:
        return "gray"
    elif "del" in string:
        return "green"
    else:
        return "black"


# Apply the color mapping function to create a new column for colors
result_df_melted["color"] = result_df_melted["Type"].apply(map_color)


# Define a dictionary for the legend with descriptions
legend_labels = {
    "Amplification": "blue",
    "Deletion": "green",
    "Expression": "gray",
    "Fusion": "black",
    "Mutation": "red",
}
color_palette = {
    "red": "red",
    "blue": "blue",
    "gray": "gray",
    "green": "green",
    "black": "black",
}
from matplotlib.lines import Line2D  # Make sure to import this

# Create the scatter plot
fig, ax = plt.subplots(figsize=(8, 30))
sns.scatterplot(
    data=result_df_melted,
    y="gene_pathway",
    x="Type",
    ax=ax,
    hue="color",  # Use color column for the hue
    size="Value",
    alpha=1,
    sizes=(1, 200),
    palette=color_palette,
    legend=False,  # Disable default legend
    edgecolor="black",
    linewidth=0.1,
)

# Manually create legend entries
legend_elements = [
    Line2D(
        [0], [0], marker="o", color="w", label=key, markersize=10, markerfacecolor=value
    )
    for key, value in legend_labels.items()
]

# Add the custom legend to the plot, horizontal layout, at the bottom
ax.legend(
    handles=legend_elements,
    title=None,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.03),
    title_fontsize=10,
    fontsize=10,
    ncols=5,
)

# Add vertical lines every 5 x-ticks
xticks = ax.get_xticks()
for i in range(5, len(xticks), 5):
    ax.axvline(
        x=(xticks[i - 1] + xticks[i]) / 2, color="grey", linestyle="-", linewidth=1
    )

# Replace underscore and adjust names for x-ticks
x_labels = ax.get_xticklabels()
new_labels = []
# Modify each label based on conditions
for idx, label in enumerate(x_labels):
    text = label.get_text()
    modified_text = (
        text.replace("_", " ")  # Remove underscores
        .replace("her2", "HER2")  # Specific replacements
        .replace("progesterone receptor", "PR")
        .replace("estrogen receptor", "ER")
        .split(" ")[0]
    )  # Split by space
    # Show every 5th label starting from the third
    if (idx + 1 - 3) % 5 == 0:
        new_labels.append(modified_text)
    else:
        new_labels.append("")

ax.set_xticklabels(new_labels, rotation=0, ha="center")
ax.set_ylabel("Gene (Pathway)")
ax.set_xlabel("Group")
ax.set_title(
    "$LRP_{sum}$ difference between positive and negative groups\n$-log10(p_{val})$ values (Mann-Whitney U-test)"
)
ax.grid(axis="y", linestyle="--", alpha=0.5)

ax.set_ylim([result_df_melted.shape[0] / 4 / 5 + 1, -0.7])
# Enhance the layout for readability
plt.tight_layout()
plt.savefig(
    os.path.join(
        path_to_save_figures, "bubbleplot_LRP_sum_diff_vs_pval_all_groups.pdf"
    ),
    format="pdf",
)


# %% compare using traditional statistics - Expressions
import pingouin as pg

df_temp = df_exp.copy()
# df_exp, df_mut, df_amp, df_del, df_fus
# df_exp.max(axis=1).plot()

# total table for mwu results
mwu_results_total = pd.DataFrame()
# iterate over all groups
for group in list(samples_groups.keys())[:-1]:
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)

    # select only samples from the group1 from df_temp
    samples_group1 = samples_groups[group][subgroup1]
    df_group1 = df_temp.loc[samples_group1, :]
    # select only samples from the group2 from df_temp
    samples_group2 = samples_groups[group][subgroup2]
    df_group2 = df_temp.loc[samples_group2, :]

    # calculate MWU test for each gene (column) between group1 and group2, add column with the median, min, and max values for each group
    mwu_results = pd.DataFrame(
        index=df_group1.columns,
        columns=[
            "p-val",
            "CLES",
            "median_group1",
            "median_group2",
            "min_group1",
            "min_group2",
            "max_group1",
            "max_group2",
        ],
    )
    for gene in df_group1.columns:
        # use pinguing MWU test to calculate p-value
        mwu_temp = pg.mwu(df_group1[gene], df_group2[gene])
        mwu_results.loc[gene, "p-val"] = mwu_temp["p-val"].values[0]
        mwu_results.loc[gene, "CLES"] = mwu_temp["CLES"].values[0]
        mwu_results.loc[gene, "median_group1"] = df_group1[gene].median()
        mwu_results.loc[gene, "median_group2"] = df_group2[gene].median()
        mwu_results.loc[gene, "min_group1"] = df_group1[gene].min()
        mwu_results.loc[gene, "min_group2"] = df_group2[gene].min()
        mwu_results.loc[gene, "max_group1"] = df_group1[gene].max()
        mwu_results.loc[gene, "max_group2"] = df_group2[gene].max()

    # reset index to have gene as a column
    mwu_results = mwu_results.reset_index()
    mwu_results = mwu_results.rename(columns={"index": "gene"})

    # assign pathway membership to gene
    mwu_results = assign_pathway_memebership_to_gene(genes_pathways, mwu_results)

    # remove genes that are not in any columns from the pathways
    # mwu_results = mwu_results[mwu_results[pathways].sum(axis = 1) > 0]

    # add column with group label
    mwu_results["group"] = group

    # concatenate to the dataframe with all groups
    mwu_results_total = pd.concat([mwu_results_total, mwu_results])

mwu_results_total["p-val"] = mwu_results_total["p-val"].astype(float)

mwu_results_total["-log10(p-val)"] = -np.log(mwu_results_total["p-val"])
# add column with the difference in medians between the two groups
mwu_results_total["median_diff"] = (
    mwu_results_total["median_group1"] - mwu_results_total["median_group2"]
)

# add a string column that contains all the pathways for each gene
mwu_results_total["pathway"] = mwu_results_total[pathways].apply(
    lambda x: ", ".join(x.index[x == 1]), axis=1
)

# save to csv as a mwu results for expression data group comparison
mwu_results_total.to_csv(
    os.path.join(path_to_save_csv, "mwu_expression_data_group_comparison.csv")
)

# %% plot vulcano plot for traditional statistics
# define dataframe with all groups

mwu_results_exp = pd.read_csv(
    os.path.join(path_to_save_csv, "mwu_expression_data_group_comparison.csv"),
    index_col=0,
)


for group in list(samples_groups.keys())[:-1]:
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)

    # select only samples from the group from mwu_results_exp
    mwu_results_temp = mwu_results_exp[mwu_results_exp["group"] == group].reset_index(
        drop=True
    )

    # select only genes from pathweays
    mwu_results_temp = mwu_results_temp[mwu_results_temp[pathways].sum(axis=1) > 0]

    # plot scatterplot for p-val vs difference in LRP sum, color the points by type, add text annotation with gene name for p-val < 0.00001
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=mwu_results_temp,
        y="-log10(p-val)",
        x="median_diff",
        style="pathway",
        hue="pathway",
        ax=ax,
        alpha=0.9,
        s=10,
    )

    # Add text annotation with gene name for p-val < 0.00001
    annotation_threshold = 5
    for i, row in mwu_results_temp.iterrows():
        if row["-log10(p-val)"] > annotation_threshold:
            ax.text(
                row["median_diff"] + 0.01,
                row["-log10(p-val)"] + 1,
                row["gene_name"],
                fontsize=5,
                alpha=1,
            )

    # Make x axis symmetric, center 0
    ax.axhline(-np.log(0.001), linestyle="--", color="gray")
    ax.axvline(0, linestyle="--", color="gray")
    ax.set_title("Mann-Whitney U-test between two groups: {}".format(group))
    ax.set_xlabel("Difference in median expression for each gene")
    ax.set_ylabel("-log10(p-val)")
    # ax.set_ylim([None, 100])

    plt.tight_layout()
    # save figure to pdf
    plt.savefig(
        os.path.join(
            path_to_save_figures,
            "scatterplot_median_expression_vs_pval_{}.pdf".format(group),
        ),
        format="pdf",
    )


# %% compare using traditional statistics - categorical data
import pingouin as pg

# concatenate all df_mut, df_fus, df_amp, df_del
df_temp = pd.concat([df_mut, df_fus, df_amp, df_del], axis=1)

# binerize the table
df_temp = df_temp.map(lambda x: 1 if x > 0 else 0)

# total table for chi2 results
chi2_results_total = pd.DataFrame()
# iterate over all groups
for group in list(samples_groups.keys())[:-1]:
    print(group)

    subgroup_keys = list(samples_groups[group].keys())
    subgroup1, subgroup2 = subgroup_keys[0], subgroup_keys[1]
    print(group, subgroup1, subgroup2)

    # select only samples from the group1 from df_temp
    samples_group1 = samples_groups[group][subgroup1]
    df_group1 = df_temp.loc[samples_group1, :]
    # select only samples from the group2 from df_temp
    samples_group2 = samples_groups[group][subgroup2]
    df_group2 = df_temp.loc[samples_group2, :]

    # calculate MWU test for each gene (column) between group1 and group2, add column with the median, min, and max values for each group
    chi2_results = pd.DataFrame(
        index=df_group1.columns, columns=["p-val", "cramer", "power"]
    )
    for gene in df_group1.columns:

        # concatenate the two columns from df_group1[gene] and df_group2[gene], add label column with group1 and group2
        df_group1_temp = pd.DataFrame(df_group1[gene])
        df_group1_temp["group"] = subgroup1
        df_group2_temp = pd.DataFrame(df_group2[gene])
        df_group2_temp["group"] = subgroup2
        df_temp_gene = pd.concat([df_group1_temp, df_group2_temp])

        expected, observed, stats = pg.chi2_independence(
            data=df_temp_gene, x=gene, y="group"
        )
        chi2_results.loc[gene, "p-val"] = stats["pval"].values[0]
        chi2_results.loc[gene, "cramer"] = stats["cramer"].values[0]
        chi2_results.loc[gene, "power"] = stats["power"].values[0]
        # convert observed to four columns and add to the dataframe
        # add try except to avoid error when the observed is not 2x2
        try:
            chi2_results.loc[gene, "obs_pos_0"] = observed.iloc[0, 0]
            chi2_results.loc[gene, "obs_pos_1"] = observed.iloc[0, 1]
            chi2_results.loc[gene, "obs_neg_0"] = observed.iloc[1, 0]
            chi2_results.loc[gene, "obs_neg_1"] = observed.iloc[1, 1]
        except:
            print("error", gene)
            pass

        chi2_results["group"] = group

        # concatenate to the dataframe with all groups
    chi2_results_total = pd.concat([chi2_results_total, chi2_results.reset_index()])

# rename column index to gene
chi2_results_total = chi2_results_total.rename(columns={"index": "gene"})

# assign pathway membership to gene
chi2_results_total = assign_pathway_memebership_to_gene(
    genes_pathways, chi2_results_total
)

# add column with the pathway name
chi2_results_total["pathway"] = chi2_results_total[pathways].apply(
    lambda x: ", ".join(x.index[x == 1]), axis=1
)

# column p-val as float
chi2_results_total["p-val"] = chi2_results_total["p-val"].astype(float)

chi2_results_total = chi2_results_total.reset_index(drop=True)
chi2_results_total["-log10(p-val)"] = -np.log(chi2_results_total["p-val"].values)


# save to csv as a mwu results for expression data group comparison
chi2_results_total.to_csv(os.path.join(path_to_save_csv, "chi2_group_comparison.csv"))


# %% prepate bubble plot with all stats MWU and chi2

# define dataframe with all groups
df_bubbleplot = pd.read_csv(
    os.path.join(path_to_data, "all_genes_with_stats_and_LRP_pvals.csv"), index_col=0
)

df_bubbleplot["type"] = df_bubbleplot["gene"].str.split("_", expand=True)[1]
# sort by gene name
df_bubbleplot = df_bubbleplot.sort_values("gene")

df_bubbleplot = df_bubbleplot[["-log10(p-val)", "gene_name", "type", "pathway", "group"]]

# drop rows with NaN pathway
df_bubbleplot = df_bubbleplot.dropna(subset=["pathway"]).reset_index(drop=True)


# melt the dataframe so the type is in the columns, index is gene_name, pathway and type, valeus are -log10(p-val)
df_bubbleplot_melted = df_bubbleplot.melt(
    id_vars=["gene_name", "pathway", "group", "type"],
    value_vars=["-log10(p-val)"],
    value_name="Value",
)
# pivot the dataframe so the type is in the columns, index is gene_name, pathway and type, valeus are -log10(p-val)
df_bubbleplot_pivot = df_bubbleplot_melted.pivot_table(
    index=["gene_name", "pathway", "group"], columns="type", values="Value"
).reset_index()

# Pivoting the DataFrame
pivoted_df = df_bubbleplot_pivot.melt(
    id_vars=["gene_name", "pathway", "group"],
    value_vars=["amp", "del", "exp", "fus", "mut"],
    var_name="variable",
    value_name="value",
)

# Create new column names combining group and variable
pivoted_df["new_column"] = pivoted_df["group"] + "_" + pivoted_df["variable"]

# Pivot the table to get desired format
result_df = pivoted_df.pivot_table(
    index=["gene_name", "pathway"], columns="new_column", values="value"
).reset_index()

# Due to pivoting, the column names become a MultiIndex, we can fix this:
result_df.columns.name = None
# sort column names after third column, but keep gene_name and pathway in the first two columns
result_df = result_df[["gene_name", "pathway"] + sorted(result_df.columns[2:])]

# melt the dataframe so the columns starting from the thirds are in the rows, index is gene_name, pathway
result_df_melted = result_df.melt(
    id_vars=["gene_name", "pathway"],
    value_vars=result_df.columns[2:],
    value_name="Value",
    var_name="Type",
)
# sort the dataframe by pathway, and then by gene_name, and then by Type
result_df_melted = result_df_melted.sort_values(
    ["pathway", "gene_name", "Type"], ascending=True
).reset_index(drop=True)

# add column that joins gene name and pathway
result_df_melted["gene_pathway"] = (
    result_df_melted["gene_name"] + " (" + result_df_melted["pathway"] + ")"
)

# Extract color categories from 'Type'
result_df_melted["color_category"] = result_df_melted["Type"].apply(
    lambda x: x.split("_")[-1]
)


# Define the color mapping function
def map_color(string):
    if "mut" in string:
        return "red"
    elif "amp" in string:
        return "blue"
    elif "exp" in string:
        return "gray"
    elif "del" in string:
        return "green"
    else:
        return "black"


# Apply the color mapping function to create a new column for colors
result_df_melted["color"] = result_df_melted["Type"].apply(map_color)


# Define a dictionary for the legend with descriptions
legend_labels = {
    "Amplification": "blue",
    "Deletion": "green",
    "Expression": "gray",
    "Fusion": "black",
    "Mutation": "red",
}
color_palette = {
    "red": "red",
    "blue": "blue",
    "gray": "gray",
    "green": "green",
    "black": "black",
}
from matplotlib.lines import Line2D  # Make sure to import this

# Create the scatter plot
fig, ax = plt.subplots(figsize=(8, 30))
sns.scatterplot(
    data=result_df_melted,
    y="gene_pathway",
    x="Type",
    ax=ax,
    hue="color",  # Use color column for the hue
    size="Value",
    alpha=1,
    sizes=(1, 200),
    palette=color_palette,
    legend=False,  # Disable default legend
    edgecolor="black",
    linewidth=0.1,
)

# Manually create legend entries
legend_elements = [
    Line2D(
        [0], [0], marker="o", color="w", label=key, markersize=10, markerfacecolor=value
    )
    for key, value in legend_labels.items()
]

# Add the custom legend to the plot, horizontal layout, at the bottom
ax.legend(
    handles=legend_elements,
    title=None,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.03),
    title_fontsize=10,
    fontsize=10,
    ncols=5,
)

# Add vertical lines every 5 x-ticks
xticks = ax.get_xticks()
for i in range(5, len(xticks), 5):
    ax.axvline(
        x=(xticks[i - 1] + xticks[i]) / 2, color="grey", linestyle="-", linewidth=1
    )

# Replace underscore and adjust names for x-ticks
x_labels = ax.get_xticklabels()
new_labels = []
# Modify each label based on conditions
for idx, label in enumerate(x_labels):
    text = label.get_text()
    modified_text = (
        text.replace("_", " ")  # Remove underscores
        .replace("her2", "HER2")  # Specific replacements
        .replace("progesterone receptor", "PR")
        .replace("estrogen receptor", "ER")
        .split(" ")[0]
    )  # Split by space
    # Show every 5th label starting from the third
    if (idx + 1 - 3) % 5 == 0:
        new_labels.append(modified_text)
    else:
        new_labels.append("")

ax.set_xticklabels(new_labels, rotation=0, ha="center")
ax.set_ylabel("Gene (Pathway)")
ax.set_xlabel("Group")
ax.set_title(
    "Univariable tests for differences between positive and negative groups\n$-log10(p_{val})$ values (Mann-Whitney U-test or $\chi^2$ )"
)
ax.grid(axis="y", linestyle="--", alpha=0.5)

ax.set_ylim([result_df_melted.shape[0] / 4 / 5 + 1, -0.7])
# Enhance the layout for readability
plt.tight_layout()
plt.savefig(
    os.path.join(
        path_to_save_figures, "bubbleplot_MWU_chi2_vs_pval_all_groups.pdf"
    ),
    format="pdf",
)

# %%


# %% PLOT both bubble on one figure

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D  # For custom legend entries

# Read data
df_bubbleplot = pd.read_csv(
    os.path.join(path_to_data, "all_genes_with_stats_and_LRP_pvals.csv"), index_col=0
)

# Function to process data for plotting
def process_data(df, value_column):
    df = df.copy()
    df["type"] = df["gene"].str.split("_", expand=True)[1]
    df = df.sort_values("gene")
    df = df[[value_column, "gene_name", "type", "pathway", "group"]]
    df = df.dropna(subset=["pathway"]).reset_index(drop=True)
    df_melted = df.melt(
        id_vars=["gene_name", "pathway", "group", "type"],
        value_vars=[value_column],
        value_name="Value",
    )
    df_pivot = df_melted.pivot_table(
        index=["gene_name", "pathway", "group"], columns="type", values="Value"
    ).reset_index()
    pivoted_df = df_pivot.melt(
        id_vars=["gene_name", "pathway", "group"],
        value_vars=["amp", "del", "exp", "fus", "mut"],
        var_name="variable",
        value_name="value",
    )
    pivoted_df["new_column"] = pivoted_df["group"] + "_" + pivoted_df["variable"]
    result_df = pivoted_df.pivot_table(
        index=["gene_name", "pathway"], columns="new_column", values="value"
    ).reset_index()
    result_df.columns.name = None
    result_df = result_df[["gene_name", "pathway"] + sorted(result_df.columns[2:])]
    result_df_melted = result_df.melt(
        id_vars=["gene_name", "pathway"],
        value_vars=result_df.columns[2:],
        value_name="Value",
        var_name="Type",
    )
    result_df_melted = result_df_melted.sort_values(
        ["pathway", "gene_name", "Type"], ascending=True
    ).reset_index(drop=True)
    result_df_melted["gene_pathway"] = (
        result_df_melted["gene_name"] + " (" + result_df_melted["pathway"] + ")"
    )
    result_df_melted["color_category"] = result_df_melted["Type"].apply(
        lambda x: x.split("_")[-1]
    )
    # Define the color mapping function
    def map_color(string):
        if "mut" in string:
            return "red"
        elif "amp" in string:
            return "blue"
        elif "exp" in string:
            return "gray"
        elif "del" in string:
            return "green"
        else:
            return "black"
    result_df_melted["color"] = result_df_melted["Type"].apply(map_color)
    return result_df_melted

# Process data for both plots
result_df_melted1 = process_data(df_bubbleplot, "-log10(p-val)")
result_df_melted2 = process_data(df_bubbleplot, "-log10(p-val)_lrp")

# Create subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 30), sharey=True)

# Define color palette and legend labels
legend_labels = {
    "Amplification": "blue",
    "Deletion": "green",
    "Expression": "gray",
    "Fusion": "black",
    "Mutation": "red",
}
color_palette = {
    "red": "red",
    "blue": "blue",
    "gray": "gray",
    "green": "green",
    "black": "black",
}

# Plot the first subplot
sns.scatterplot(
    data=result_df_melted1,
    y="gene_pathway",
    x="Type",
    ax=ax1,
    hue="color",  # Use color column for the hue
    size="Value",
    alpha=1,
    sizes=(1, 200),
    palette=color_palette,
    legend=False,  # Disable default legend
    edgecolor="black",
    linewidth=0.1,
)

# Plot the second subplot
sns.scatterplot(
    data=result_df_melted2,
    y="gene_pathway",
    x="Type",
    ax=ax2,
    hue="color",  # Use color column for the hue
    size="Value",
    alpha=1,
    sizes=(1, 200),
    palette=color_palette,
    legend=False,  # Disable default legend
    edgecolor="black",
    linewidth=0.1,
)

# Manually create legend entries
legend_elements = [
    Line2D(
        [0], [0], marker="o", color="w", label=key, markersize=15, markerfacecolor=value
    )
    for key, value in legend_labels.items()
]


# Adjust x-ticks and labels for both subplots
for ax in [ax1, ax2]:
    xticks = ax.get_xticks()
    for i in range(5, len(xticks), 5):
        ax.axvline(
            x=(xticks[i - 1] + xticks[i]) / 2, color="grey", linestyle="-", linewidth=1
        )
    x_labels = ax.get_xticklabels()
    new_labels = []
    for idx, label in enumerate(x_labels):
        text = label.get_text()
        modified_text = (
            text.replace("_", " ")  # Remove underscores
            .replace("her2", "HER2")  # Specific replacements
            .replace("progesterone receptor", "PR")
            .replace("estrogen receptor", "ER")
            .split(" ")[0]
        )  # Split by space
        if (idx + 1 - 3) % 5 == 0:
            new_labels.append(modified_text)
        else:
            new_labels.append("")
    ax.set_xticklabels(new_labels, rotation=0, ha="center")

# Set labels and titles
ax1.set_ylabel("Gene (Pathway)")
ax1.set_xlabel("Group")
ax1.set_title(
    "Univariable tests for differences between positive and negative groups\n$-log10(p_{val})$ values (Mann-Whitney U-test or $\chi^2$ )"
)
ax1.grid(axis="y", linestyle="--", alpha=0.5)

ax2.set_ylabel("")  # No y-axis label on the second plot
ax2.set_xlabel("Group")
ax2.set_title(
    "$LRP_{sum}$ difference between positive and negative groups\n$-log10(p_{val})$ values (Mann-Whitney U-test)"
)
ax2.grid(axis="y", linestyle="--", alpha=0.5)

# Hide y-axis labels on the second plot
ax2.tick_params(labelleft=False)

# Set consistent y-axis limits
ylim_value = [result_df_melted1.shape[0] / 4 / 5 + 1, -0.7]
ax1.set_ylim(ylim_value)
ax2.set_ylim(ylim_value)

plt.tight_layout()
# Add the custom legend to the first subplot
ax1.legend(
    handles=legend_elements,
    title=None,
    loc="lower center",
    bbox_to_anchor=(1, -0.04),
    title_fontsize=12,
    fontsize=12,
    ncols=5,
)
# Enhance the layout for readability

plt.subplots_adjust(bottom=0.045)  # Adjust bottom to make room for the legend

# Save the combined figure
plt.savefig(
    os.path.join(
        path_to_save_figures, "combined_bubbleplot_all_groups.pdf"
    ),
    format="pdf",
)

# Display the plot
plt.show()

# %% plot scatterplot with annotations for -log10(p-val) > 5 only for genes from pathways
# define dataframe with all groups
df = pd.read_csv(
    os.path.join(path_to_data, "all_genes_with_stats_and_LRP_pvals.csv"), index_col=0
)


df["type"] = df["gene"].str.split("_", expand=True)[1]
df = df.sort_values("gene")
df = df.dropna(subset=["pathway"]).reset_index(drop=True)

df = df[[ "-log10(p-val)","-log10(p-val)_lrp", "gene_name", "type", "pathway", "group"]]

# select only group TNBC
df = df[df["group"] == "TNBC"]

# plot scatterplot -log10(p-val) vs -log10(p-val)_lrp, color the points by type, add text annotation with gene name for -log10(p-val) > 5
fig, ax = plt.subplots(figsize=(12, 7))
sns.scatterplot(
    data=df,
    x="-log10(p-val)",
    y="-log10(p-val)_lrp",
    style="type",
    hue="pathway",
    ax=ax,
    alpha=0.9,
    s=10,
)

# Add text annotation with gene name for -log10(p-val) > 5
annotation_threshold = 3
for i, row in df.iterrows():
    if row["-log10(p-val)_lrp"] > annotation_threshold:
        ax.text(
            row["-log10(p-val)"] + 0.01,
            row["-log10(p-val)_lrp"] + 1,
            row["gene_name"],
            fontsize=5,
            alpha=1,
        )

# Make x axis symmetric, center 0
ax.axhline(-np.log10(0.001), linestyle="--", color="gray")
ax.axvline(0, linestyle="--", color="gray")
ax.set_title("TNBC group")
ax.set_xlabel("$-log10(p_{val})$ (univariable tests)")
ax.set_ylabel("$-log10(p_{val})$ ($LRP_{sum}$ difference)")
# ax.set_ylim([None, 100])
# adjust legend positon outside the plot
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Pathway", markerscale=2.0)

# save to pdf
plt.tight_layout()
plt.savefig(
    os.path.join(
        path_to_save_figures,
        "scatterplot_LRP_sum_diff_vs_pval_TNBC.pdf",
    ),
    format="pdf",
)

# %%
