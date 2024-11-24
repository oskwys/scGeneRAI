import importlib, sys

"""
This script performs the following tasks:
1. Imports necessary libraries and modules.
2. Defines file paths for saving figures and loading data.
3. Loads input data and clinical features.
4. Filters clinical features based on sample list and adds cluster information.
5. Groups samples based on clinical features.
6. Loads LRP (Layer-wise Relevance Propagation) data and maps clinical features to column colors.
7. Maps gene types to row colors.
8. Performs hierarchical clustering on the LRP data.
9. Plots a clustermap of the LRP data with customized aesthetics.
10. Adds legends for row and column colors.
11. Saves the clustermap as a PNG file.
12. Plots and saves the distribution of LRP values in both log scale and original scale.
Functions:
- map_color(string): Maps gene types to specific colors.
"""
import functions as f

importlib.reload(f)
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


# %%
path_to_save_figures = r"G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\plots_to_paper"

# %% samples
path_to_data = r"G:\My Drive\SAFE_AI\CCE_DART\KI_dataset\data_to_BRCA_model"
data_to_model, df_exp, df_mut, df_amp, df_del, df_fus, df_clinical_features = (
    f.get_input_data(path_to_data)
)

samples = pd.read_csv(
    r"G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\BRCA_samples.txt",
    index_col=0,
)["samples"].to_list()

df_clinical_features = df_clinical_features[
    df_clinical_features["bcr_patient_barcode"].isin(samples)
].reset_index(drop=True)
df_clinical_features = f.add_cluster2(df_clinical_features)

# samples groups
samples_groups = f.get_samples_by_group(df_clinical_features)

# %% plot clustermap
path_to_lrp_mean = r"G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples"

df_lrp = pd.read_csv(os.path.join(path_to_lrp_mean, "LRP_sum_mean.csv"), index_col=0)
df_lrp.columns = samples

# col colors
df_clinical_features_ = (
    df_lrp.T.reset_index()
    .iloc[:, 0:1]
    .merge(df_clinical_features, left_on="index", right_on="bcr_patient_barcode")
    .set_index("index")
)
column_colors = pd.DataFrame(f.map_subtypes_to_col_color(df_clinical_features_)).T
column_colors = column_colors.rename(
    columns={"Estrogen_receptor": "ER", "Progesterone_receptor": "PR"}
)


# row colors
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


row_colors = df_lrp.reset_index().iloc[:, 0].apply(map_color)
row_colors.index = df_lrp.index
row_colors = row_colors.rename(index="Type")

data_to_dendrogram = df_lrp.T.values
Z_col = linkage(data_to_dendrogram, method="ward")
Z_row = linkage(data_to_dendrogram.T, method="ward")

# %%
vmax = 5
g = sns.clustermap(
    df_lrp,
    vmax=vmax,
    method="ward",
    cmap="jet",
    col_linkage=Z_col,
    row_linkage=Z_row,
    yticklabels=False,
    xticklabels=False,
    col_colors=column_colors,
    row_colors=row_colors,
    cbar_kws={"orientation": "horizontal", "fraction": 0.1, "pad": 0.1},
)
# Adjust the height of the column colors to make them less tall
col_colors_height = 0.07  # Set desired height for col_colors
g.ax_col_colors.set_position(
    [
        g.ax_col_colors.get_position().x0,
        g.ax_col_colors.get_position().y0,
        g.ax_col_colors.get_position().width,
        col_colors_height,
    ]
)

# Adjust the height of the column dendrogram to match the adjusted col_colors
g.ax_col_dendrogram.set_position(
    [
        g.ax_col_dendrogram.get_position().x0,
        g.ax_col_colors.get_position().y0 + col_colors_height + 0.01,
        g.ax_col_dendrogram.get_position().width,
        0.18,
    ]
)

# Move the colorbar up
g.cax.set_position([0.05, 0.9, 0.25, 0.03])
g.ax_heatmap.set_xlabel("Samples")
g.ax_heatmap.set_ylabel("Genes")


color_mapping = {
    "mut": "red",
    "amp": "blue",
    "exp": "gray",
    "del": "green",
    #'other': 'black'
}

# Create legend patches for row colors
legend_elements_row = [
    mpatches.Patch(color=color, label=label) for label, color in color_mapping.items()
]

# Create a legend for column colors
col_color_mapping = {
    "Negative": "red",
    "Positive": "blue",
    "Equivocal": "yellow",
    "Indeterminate": "gray",
    "NA": "white",
}

# Create legend patches for column colors
legend_elements_col = [
    mpatches.Patch(color=color, label=label)
    for label, color in col_color_mapping.items()
]

# Add the legend for row colors to the plot
g.ax_row_dendrogram.legend(
    handles=legend_elements_row,
    title="Type",
    bbox_to_anchor=(0.65, 0.01),
    loc="lower right",
)

# Add the legend for column colors to the plot
g.ax_col_dendrogram.legend(
    handles=legend_elements_col,
    title="Group",
    bbox_to_anchor=(-0.01, 0.4),
    loc="upper right",
)

# plt.savefig(os.path.join(path_to_save_figures , 'LRP_sum_clustermap'+'.pdf'), format = 'pdf')
plt.savefig(
    os.path.join(path_to_save_figures, "LRP_sum_clustermap" + ".png"),
    format="png",
    dpi=300,
)


# %%

# plot distribution of LRP values, with xaxis in log scale, xticklabels are the original values, like 1, 10, 100, 1000
df_lrp_log = np.log10(df_lrp)
fig, ax = plt.subplots(figsize=(10, 5))
ax = sns.histplot(df_lrp_log.values.flatten(), bins=100, color="blue", kde=True)
ax.set_xlabel("LRP Values (log10 scale)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of LRP Values (all samples x all genes)")
ax.vlines(x=np.log10(vmax), ymin=0, ymax=ax.get_ylim()[1], color="red", linestyle="--")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{10**x:.1f}"))
plt.tight_layout()
plt.savefig(
    os.path.join(path_to_save_figures, "LRP_sum_histogram" + ".pdf"), format="pdf"
)


fig, ax = plt.subplots(figsize=(10, 5))
ax = sns.histplot(df_lrp.values.flatten(), bins=100, color="blue", kde=True)
ax.set_xlabel("LRP Values")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of LRP Values")
