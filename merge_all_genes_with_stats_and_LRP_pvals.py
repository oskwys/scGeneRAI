"""
This script loads and merges statistical data from multiple CSV files, processes the data, and generates plots comparing p-values from different statistical tests.
The script performs the following steps:
1. Imports necessary libraries.
2. Defines paths for saving results and loading data.
3. Loads data from three CSV files: 'mwu_sum_LRP_genes_all.csv', 'chi2_group_comparison.csv', and 'mwu_expression_data_group_comparison.csv'.
4. Merges the data from 'chi2_group_comparison.csv' and 'mwu_expression_data_group_comparison.csv' into a single DataFrame.
5. Selects specific columns from the merged DataFrame.
6. Merges the resulting DataFrame with data from 'mwu_sum_LRP_genes_all.csv'.
7. Creates a simplified DataFrame with selected columns.
8. Saves the simplified DataFrame and the full merged DataFrame to CSV files.
9. Generates and saves joint plots comparing the -log10(p-values) from different statistical tests for each group in the data.


"""


import pandas as pd
import numpy as np


path_to_save = r"G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\plots_to_paper"

# import data csv from path
path = r"G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\mwu"

# %% laod data and merge
# import mwu_sum_LRP_genes_all.csv
df_lrp = pd.read_csv(path + r"\mwu_sum_LRP_genes_all.csv", index_col=0)
# import chi2_group_comparison.csv
df_chi2 = pd.read_csv(path + r"\chi2_group_comparison.csv", index_col=0)
# import mwu_expression_data_group_comparison.csv
df_mwu = pd.read_csv(path + r"\mwu_expression_data_group_comparison.csv", index_col=0)

# merge chi2 with mwu
df_stats = pd.concat((df_chi2, df_mwu), axis=0)

df_stats = df_stats[
    [
        "group",
        "gene",
        "p-val",
        "-log10(p-val)",
        "gene_name",
        "Cell Cycle",
        "HIPPO",
        "MYC",
        "NOTCH",
        "NRF2",
        "PI3K",
        "RTK-RAS",
        "TGF-Beta",
        "TP53",
        "WNT",
        "pathway",
        "CLES",
        "cramer",
        "power",
        "obs_pos_0",
        "obs_pos_1",
        "obs_neg_0",
        "obs_neg_1",
        "median_group1",
        "median_group2",
        "min_group1",
        "min_group2",
        "max_group1",
        "max_group2",
        "median_diff",
    ]
]


# merge df with df_lrp
df = pd.merge(
    df_stats,
    df_lrp,
    on=[
        "gene",
        "group",
        "gene_name",
        "Cell Cycle",
        "HIPPO",
        "MYC",
        "NOTCH",
        "NRF2",
        "PI3K",
        "RTK-RAS",
        "TGF-Beta",
        "TP53",
        "WNT",
    ],
    how="outer",
    suffixes=("", "_lrp"),
)


df_simple = df[
    [
        "group",
        "gene",
        "p-val",
        "p-val_lrp",
        "-log10(p-val)",
        "-log10(p-val)_lrp",
        "gene_name",
        "Cell Cycle",
        "HIPPO",
        "MYC",
        "NOTCH",
        "NRF2",
        "PI3K",
        "RTK-RAS",
        "TGF-Beta",
        "TP53",
        "WNT",
        "pathway",
    ]
].copy()


# save df_simple
df_simple.to_csv(path + r"\all_genes_with_stats_and_LRP_pvals.csv")
df.to_csv(path + r"\all_genes_with_stats_and_LRP_pvals_full_table.csv")


# %% plot p-val vs p-val_lrp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

# for each group in df_simple, plot p-val vs p-val_lrp

for group in df_simple["group"].unique():
    # filter df_simple by group
    df_group = df_simple[df_simple["group"] == group]
    # Create jointplot
    g = sns.jointplot(
        data=df_group, x="-log10(p-val)", y="-log10(p-val)_lrp", kind="hex", height=4
    )

    # Add labels
    g.ax_joint.set_xlabel("-log10(p-val) from MWU or $\chi^2$ test")
    g.ax_joint.set_ylabel("-log10(p-val) from $LRP_{sum}$ MWU test")

    # Add title
    g.fig.suptitle(group, y=1.02)

    plt.tight_layout()
    # save fig
    # plt.savefig(path_to_save + r'\statistics_vs_LRPsum_{}.png'.format(group), dpi=300)
    # save as pdf
    plt.savefig(
        path_to_save + r"\statistics_vs_LRPsum_{}.pdf".format(group), format="pdf"
    )


# %%
