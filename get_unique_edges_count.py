import pandas as pd
import os
from datetime import datetime
import argparse
import functions as f


def main(args):
    
    """
    Main function to process LRP data and get top N interactions.

    Parameters:
    args (argparse.Namespace): The arguments passed to the script.
    - path_to_save (str): Path to save the output files.
    - path_to_lrp_results (str): Path to the LRP results.
    - get_topN (bool): Flag to determine if top N interactions should be retrieved.
    - get_topN_topn (int): Number of top N interactions to retrieve.
    - get_topN_node_type (str): Type of node to consider for top N interactions.
    - save_individual_samples (bool): Flag to determine if individual sample results should be saved.
    
    Returns:
    None
    """
    path_to_save = args.path_to_save
    path_to_lrp_results = args.path_to_lrp_results
    get_topN = args.get_topN
    get_topN_topn = args.get_topN_topn
    get_topN_node_type = args.get_topN_node_type
    save_individual_samples = args.save_individual_samples

    # Load LRP data
    lrp_files = f.get_lrp_files(path_to_lrp_results)[:10]
    samples = f.get_samples_with_lrp(lrp_files)
    print("Samples count:", len(samples))
    print("Unique samples count:", len(set(samples)))

    start_time = datetime.now()
    lrp_dict = f.load_lrp_data(lrp_files, path_to_lrp_results)
    end_time = datetime.now()
    print("Time taken to load data:", end_time - start_time)
    print("LRP files loaded:", lrp_files)
    print("Number of LRP files:", len(lrp_files))

    # Get top N interactions
    if get_topN:
        df_topn_edges = pd.DataFrame(columns=samples)
        df_topn_lrps = pd.DataFrame(columns=samples)

        for sample_name in samples:
            data_temp = f.get_topN_interactions(
                lrp_dict[sample_name], get_topN_node_type, get_topN_topn
            )
            data_temp = f.add_edge_column(data_temp)
            f.add_edges_to_main_df_topn(df_topn_edges, sample_name, data_temp)
            f.add_lrps_to_main_df_topn(df_topn_lrps, sample_name, data_temp)

            if save_individual_samples:
                data_temp.to_csv(
                    os.path.join(
                        path_to_save,
                        f"df_topn_for_individuals_top{get_topN_topn}_{get_topN_node_type}_{sample_name}.csv",
                    ),
                    index=False,
                )

        df_topn_edges.to_csv(
            os.path.join(
                path_to_save,
                f"df_topn_edges_for_individuals_top{get_topN_topn}_{get_topN_node_type}.csv",
            ),
            index=False,
        )
        print("df_topn_edges saved.")

        df_topn_lrps.to_csv(
            os.path.join(
                path_to_save,
                f"df_topn_lrps_for_individuals_top{get_topN_topn}_{get_topN_node_type}.csv",
            ),
            index=False,
        )
        print("df_topn_lrps saved.")

        unique_edges_count_df = f.count_unique_edges_in_df_topn(df_topn_edges)
        unique_edges_count_df.to_csv(
            os.path.join(
                path_to_save,
                f"unique_edges_count_in_top_{get_topN_topn}_{get_topN_node_type}.csv",
            ),
            index=False,
        )
        print("Unique edges count saved.")


    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LRP data.")
    parser.add_argument(
        "--path_to_save", type=str, default="./results", help="Path to save results"
    )
    parser.add_argument(
        "--path_to_lrp_results",
        type=str,
        default="./results",
        help="Path to LRP results",
    )
    parser.add_argument(
        "--get_topN",
        type=bool,
        default=True,
        help="Flag to get top N interactions. If False then no processing is done.",
    )
    parser.add_argument(
        "--get_topN_topn",
        type=int,
        default=100,
        help="Number of top N interactions to get",
    )
    parser.add_argument(
        "--get_topN_node_type",
        type=str,
        default=None,
        help="Node type to filter top N interactions. Interactions must have at least one node of this type. If None then no filtering is done.",
    )
    parser.add_argument(
        "--save_individual_samples",
        type=bool,
        default=False,
        help="Flag to save individual samples. If True then LRP data files for individual samples are saved.",
    )
    args = parser.parse_args()
    main(args)
