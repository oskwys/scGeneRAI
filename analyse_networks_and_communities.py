# %% import
import pickle
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
from kneefinder import KneeFinder
from kneed import KneeLocator

import community as community_louvain
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score

from cdlib import algorithms
import random
random.seed(42)
np.random.seed(42)

# %% functions
def detect_communities(G):
    # Use the Louvain method for community detection
    #partition = community_louvain.best_partition(G)
    G_ig = convert_networkx_to_igraph(G)
    partition = algorithms.leiden(G_ig, weights=G_ig.es()['weight']).to_node_community_map()
    partition = {key: value[0] for key, value in partition.items()}

    return partition

def communities_to_labels(partition):
    # Convert a partition to cluster labels
    labels = []
    for node in partition:
        labels.append(partition[node])
    return labels


import igraph as ig
import community as community_louvain
from cdlib import algorithms
def convert_networkx_to_igraph(nx_graph):
    # Create a mapping from networkx nodes to sequential integers
    mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    # Create the igraph graph
    g_ig = ig.Graph(directed=nx_graph.is_directed())
    # Add vertices with the name attribute set as the original networkx node identifiers
    g_ig.add_vertices(len(mapping))
    g_ig.vs['name'] = list(nx_graph.nodes())
    # Add edges by converting networkx nodes to the integer identifiers used in igraph
    ig_edges = [(mapping[edge[0]], mapping[edge[1]]) for edge in nx_graph.edges()]
    g_ig.add_edges(ig_edges)
    # If the graph is weighted, transfer the weights
    if nx.is_weighted(nx_graph):
        #print('Wieghted graph')
        weights = [nx_graph[u][v]['weight'] for u,v in nx_graph.edges()]
        g_ig.es['weight'] = weights
    return g_ig
# %% stats for nonexpexp edges

edges_count = pd.read_csv(r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\unique_edges_noexpexp_count_in_top_1000_noexpexp.csv', index_col = 0)
# sort by count
edges_count = edges_count.sort_values('count', ascending = False)

# get the top 1000 edges    
topn = 5000
topn_edges = edges_count.iloc[:topn,:].reset_index(drop=True)

# plot the count of edges
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(topn_edges['count'])
ax.set_xlabel('Edge Index')
ax.set_ylabel('Count')
ax.set_title('Count of Unique Edges')
ax.set_ylim(bottom=0)  # Set the minimum limit of y-axis to 0
plt.show()



# %% load data
path_to_save = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300'
#topn = 1000
#LRP_pd = pd.read_csv(r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\top1000_5300\LRP_individual_top1000_filtered.csv', index_col = 0)
LRP_pd = pd.read_csv(r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\LRP_individual_top1000_noexpexp.csv', index_col = 0)
# set index  by 'edge' column and remove it
#LRP_pd = LRP_pd.set_index('edge')
# rename index as   'edge'
LRP_pd.index.name = 'edge'

# %% generate graphs
import community  # If using the Louvain method for community detection
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


LRP_to_graphs = LRP_pd.copy().reset_index()    
LRP_to_graphs['source_gene']  = LRP_to_graphs['edge'].str.split(' - ', expand = True)[0]
LRP_to_graphs['target_gene']= LRP_to_graphs['edge'].str.split(' - ', expand = True)[1]
# reorder columns so 'edge' is the first one, source_gene and target_gene are the second and third
LRP_to_graphs = LRP_to_graphs[['edge', 'source_gene', 'target_gene'] + list(LRP_to_graphs.columns[1:-2])]

# define list of number of max edges in the graph
n_edges = np.linspace(100, 150, 2).astype(int)[::-1]

def create_edges_from_lrp(LRP_to_graphs, i):
    """
    Create edges from LRP (Layer-wise Relevance Propagation) data.
    This function takes a DataFrame containing LRP data and an index, and creates a new DataFrame 
    representing edges between source and target genes, sorted by the LRP values in descending order.
    Parameters:
    LRP_to_graphs (pd.DataFrame): A DataFrame containing LRP data with columns 'source_gene', 'target_gene', 
                                  and LRP values of all samples in subsequent columns.
    i (int): An index indicating which LRP column (which sample) to use for creating edges.
    Returns:
    pd.DataFrame: A DataFrame with columns 'source_gene', 'target_gene', and 'LRP', sorted by 'LRP' in descending order.
    """
    lrp = LRP_to_graphs.iloc[:,3+i].values

    # create 'edges' dataframe
    edges = pd.DataFrame(
        {
            "source_gene": LRP_to_graphs["source_gene"],
            "target_gene": LRP_to_graphs["target_gene"],
            "LRP": lrp            
        }
    )

    # sort by LRP_norm
    edges = edges.sort_values('LRP', ascending = False).reset_index(drop=True)
    return edges

def get_edges_subset(edges_sample_i, n_):
    edges_temp = edges_sample_i.iloc[:n_,:].copy()
    return edges_temp

def normalize_lrp(edges_temp):
    edges_temp['LRP_norm'] = edges_temp['LRP'] / edges_temp['LRP'].max()


def get_adjacency_and_laplacian(G):
    # Get the weighted adjacency matrix
    A = nx.to_numpy_array(G, weight='LRP_norm')  # Use your edge attribute for weights
    
    # Compute the degree matrix
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    
    # Compute the Laplacian matrix
    L = D - A
    
    return A, L
def compute_laplacian_embedding(L, dim=10):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Select the eigenvectors corresponding to the smallest non-zero eigenvalues
    # Skip the first eigenvector (corresponding to eigenvalue zero)
    embedding = eigenvectors[:, 1:dim+1]
    return embedding

def create_fixed_size_embedding(embedding, all_nodes, graph_nodes, max_nodes, dim=10):
    # Create a zero matrix of maximum size
    fixed_embedding = np.zeros((max_nodes, dim))
    
    # Create mapping of node positions
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    
    # Fill in the embedding values for existing nodes
    for idx, node in enumerate(graph_nodes):
        if node in node_to_idx:
            fixed_embedding[node_to_idx[node]] = embedding[idx]
            
    return fixed_embedding

graphs_dict = {}

for i in range(988):
    print(i)
    edges_sample_i = create_edges_from_lrp(LRP_to_graphs, i)

    graphs_dict[i] = {}
    prev_partition = None

    # keep only top n edges
    for n_ in n_edges:
        graphs_dict[i][n_] = {}

        edges_temp = get_edges_subset(edges_sample_i, n_)
        # add column with nomaalized LRP
        normalize_lrp(edges_temp)

        # create nx graph from lists
        G = nx.from_pandas_edgelist(
            edges_temp,
            source="source_gene",
            target="target_gene",
            edge_attr=["LRP", "LRP_norm"]
        )

        # add weights to edges to G
        for (u, v, d) in G.edges(data=True):
            G[u][v]['weight'] = d['LRP_norm']
            
        graphs_dict[i][n_]['G'] = G

all_nodes_dict = {}
max_nodes_dict = {}

for n_ in n_edges:
    all_nodes = set()
    for i in graphs_dict:

        G = graphs_dict[i][n_]['G']
        all_nodes.update(G.nodes())

    all_nodes_dict[n_] = all_nodes
    max_nodes_dict[n_] = len(all_nodes)

embedding_dim = 2

for i in range(988):
    print(i)
    for n_ in n_edges: 
        G = graphs_dict[i][n_]['G']

        A, L = get_adjacency_and_laplacian(G)
        embedding = compute_laplacian_embedding(L, dim=embedding_dim)
        
        graphs_dict[i][n_]['A'] = A
        graphs_dict[i][n_]['L'] = L
        graphs_dict[i][n_]['laplacian_embedding'] = embedding
        fixed_embedding = create_fixed_size_embedding(
            embedding, 
            all_nodes_dict[n_] , 
            list(G.nodes()), 
            max_nodes_dict[n_],
            dim=embedding_dim
        )
        graphs_dict[i][n_]['fixed_laplacian_embedding'] = fixed_embedding
        graphs_dict[i][n_]['fixed_laplacian_embedding_flattened'] = fixed_embedding.ravel()


# %% get clinical features
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


# col colors
df_clinical_features_ = (
    LRP_to_graphs.T.reset_index()
    .iloc[:, 0:1]
    .merge(df_clinical_features, left_on="index", right_on="bcr_patient_barcode")
    .set_index("index")
)



# %% compute graph clustering based on Laplacian embedding
column_colors = pd.DataFrame(f.map_subtypes_to_col_color(df_clinical_features_)).T
column_colors = column_colors.rename(
    columns={"Estrogen_receptor": "ER", "Progesterone_receptor": "PR"}
)


# Initialize an empty list to store the rows of the DataFrame
# Initialize a dictionary to store the flattened embeddings
flattened_embeddings_dict = {}

for n_ in n_edges:
    flattened_embeddings_dict[n_] = {}
    for i in graphs_dict:
        flattened_embeddings_dict[n_][i] = graphs_dict[i][n_]['fixed_laplacian_embedding_flattened']

# Convert lists to DataFrames
for n_ in flattened_embeddings_dict:
    flattened_embeddings_dict[n_] = pd.DataFrame(flattened_embeddings_dict[n_])
    flattened_embeddings_dict[n_].columns = samples

cluster_labels_dict = {}
# plot clustermaps for each n_
for n_ in flattened_embeddings_dict:

    # copmute linkage using ward method 
    Z = linkage(flattened_embeddings_dict[n_].T, method='ward')
    # get clusters labels
    clusters = fcluster(Z, t=6, criterion='maxclust')
    # add clusters to the dataframe
    cluster_labels_dict[n_] = clusters
    
    # plot dendrogram
    plt.figure(figsize=(14, 6))
    dendrogram(Z, p=30, truncate_mode='level', orientation='top', leaf_rotation=90, leaf_font_size=8)
    plt.title(f"Dendrogram for n_ = {n_}")
    #plt.xlabel('Graph Index (i)')
    plt.ylabel('Distance')
    plt.show()

    



    # Create clustermap directly without creating a separate figure
    print(flattened_embeddings_dict[n_].shape)
    g = sns.clustermap(
        flattened_embeddings_dict[n_],
        col_linkage=Z,
        cmap="coolwarm",
        mask=flattened_embeddings_dict[n_] == 0,
        # set column colors
        col_colors=column_colors,
        cbar_kws={
            "label": "",
            "shrink": 0.8,  # Adjust the length of colorbar (0 to 1)
            "aspect": 30,   # Adjust the width of colorbar
            "pad": 0.02     # Adjust the distance between colorbar and plot
        },
        method='ward',
        figsize=(20, 20),
        vmin = -1,
        vmax = 1
    )
    
    # Add title
    g.fig.suptitle(f"Laplacian Embedding for n_ = {n_}")
    # Adjust layout to prevent title overlap
    #plt.tight_layout()
    #plt.subplots_adjust(top=0.9)
    # add x and y labels
    g.ax_heatmap.set_xlabel("Graph Index (i)")
    g.ax_heatmap.set_ylabel("Node Index")

# groups samples by cluster 
n= 100
df_clinical_features_['cluster'] = cluster_labels_dict[n_]


# get the number of samples in each cluster
df_clinical_features_['cluster'].value_counts()

# get weighted sum of edges connected to node in each cluster

# dictionary with dataframe for each cluster
nodes_LRP_sum_dict = {}
for n_ in n_edges:
    nodes_LRP_sum_dict[n_] = {}

    for cluster_i in np.unique(cluster_labels_dict[n_]):
        nodes_LRP_sum_dict[n_][cluster_i] = pd.DataFrame(list(all_nodes_dict[n_]), columns = ['node']).sort_values('node')
        print(nodes_LRP_sum_dict[n_][cluster_i].shape )

for i in range(988):
    print(i)
    # get cluster of the sample
    cluster_i = cluster_labels_dict[n_][i]


    for n_ in n_edges: 
        G = graphs_dict[i][n_]['G']

        # get degree of each node
        degrees = np.array(list(nx.degree_centrality(G).values()))
        # get the sum of LRP_norm for each node
        sum_LRP = np.array([np.sum([d['LRP_norm'] for u, v, d in G.edges(node, data=True)]) for node in G.nodes()])
        # store sum_LRP in the dataframe
        nodes_LRP_sum = pd.DataFrame({'node': list(G.nodes()), 'sum_LRP': sum_LRP})
        
        nodes_LRP_sum_dict[n_][cluster_i] = nodes_LRP_sum_dict[n_][cluster_i].merge(nodes_LRP_sum, on = 'node', how = 'left', suffixes = ('', '_'+str(i)))

LRP_sum_means = {}
# calculate mean of LRP sum explludeing nans
for n_ in n_edges:
    LRP_sum_means[n_] = pd.DataFrame()
    for cluster_i in np.unique(cluster_labels_dict[n_]):
        nodes_LRP_sum_dict[n_][cluster_i]['nanmean_{}'.format(cluster_i)] = np.nanmean(nodes_LRP_sum_dict[n_][cluster_i].iloc[:,1:], axis = 1)
        #nodes_LRP_sum_dict[n_][cluster_i] = nodes_LRP_sum_dict[n_][cluster_i].sort_values(by = 'nanmean', ascending = False)
        print(nodes_LRP_sum_dict[n_][cluster_i].shape)

        LRP_sum_means[n_]['node'] = nodes_LRP_sum_dict[n_][cluster_i]['node']
        LRP_sum_means[n_] = pd.concat([LRP_sum_means[n_], nodes_LRP_sum_dict[n_][cluster_i]['nanmean_{}'.format(cluster_i)]], axis = 1)




        # plot the sum of LRP for each node, use fig ax
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(nodes_LRP_sum_dict[n_][cluster_i]['node'], nodes_LRP_sum_dict[n_][cluster_i]['nanmean'])
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Sum of LRP_norm')
        ax.set_title(f'Sum of LRP_norm for cluster {cluster_i} n_ = {n_}')
        ax.set_ylim(bottom=0)  # Set the minimum limit of y-axis to 0
        plt.show()

for n_ in n_edges:
    temp = LRP_sum_means[n_].copy()
    temp = temp.set_index('node').sort_values(by = 'nanmean_3', ascending = False).fillna(0)
    # set size of the heatmap
    plt.figure(figsize=(20, 20))

    sns.heatmap(temp, cmap = 'coolwarm', vmax = 11, vmin = 0)

    plt.title(f'Heatmap of mean LRP for n_ = {n_}')
    plt.show()




# %% UMAP PCA
# 




import umap
# get umap embedding
umap_embeddings_dict = {}
for n_ in flattened_embeddings_dict:
    print('UMAP: ', n_)
    umap_embeddings_dict[n_] = umap.UMAP(n_components=2).fit_transform(flattened_embeddings_dict[n_].T)

# plot the umap embeddings
for n_ in umap_embeddings_dict:
    plt.figure(figsize=(10, 6))
    plt.scatter(
            umap_embeddings_dict[n_][:, 0],
            umap_embeddings_dict[n_][:, 1],
            c=cluster_labels_dict[n_],
            cmap='tab10',
            s=10,
            alpha=0.5
        )
    plt.title(f"UMAP Embedding for n_ = {n_}")
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.colorbar()
    plt.show()


from sklearn.decomposition import PCA

# get PCA embedding
pca_embeddings_dict = {}
for n_ in flattened_embeddings_dict:
    pca = PCA(n_components=2)
    pca_embeddings_dict[n_] = pca.fit_transform(flattened_embeddings_dict[n_].T)

# plot the PCA embeddings
for n_ in pca_embeddings_dict:
    plt.figure(figsize=(10, 6))
    plt.scatter(
        pca_embeddings_dict[n_][:, 0],
        pca_embeddings_dict[n_][:, 1],
        c=cluster_labels_dict[n_],
        s=10,
        alpha=0.5,
        cmap='tab10',
    )
    plt.title(f"PCA Embedding for n_ = {n_}")
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.colorbar()
    plt.show()





























# %%% plot and save all 988 networks by clusters
n_ = 1000
cluster_labels_df = pd.DataFrame(cluster_labels_dict[n_])

path_to_save = r'G:\My Drive\SAFE_AI\CCE_DART\scGeneRAI_results\model_BRCA_20230904\results_all_samples\202411_experiments'

pos_988 = []
for i in range(10):
    cluster_id = cluster_labels_df.loc[i, 0]
    print(i, 'cluster ', cluster_id)
    
    G = graphs_dict[i][n_]['G']
 
    degrees = np.array(list(nx.degree_centrality(G).values()))
    degrees_norm = degrees / np.max(degrees)
    widths = list(nx.get_edge_attributes(G, 'LRP_norm').values())
    widths = [x*2 for x in widths]
    
    edge_colors = plt.cm.Greys(widths)  # 'viridis' is a colormap, you can choose any
        
    node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}
        
    #pos = nx.spring_layout(G, weight='LRP_norm')
    
    fig, ax = plt.subplots(figsize=(7,7))     
        # Plot legend
    ls = list(node_color_mapper.keys())
    cl = list(node_color_mapper.values())
    for label, color in zip(ls, cl):
        ax.plot([], [], 'o', color=color, label=label)
    ax.legend(title = 'Nodes', loc='best')
    
    node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper)
    node_labels = {node: node.split('_')[0] for node in G.nodes()}   

    #pos = nx.spring_layout(G, weight='LRP_norm')
    if i ==0:
        pos = nx.shell_layout(G)   

        
    nx.draw(G, with_labels=True,
            labels=node_labels,  # Add the labels
            node_color=node_colors,
            width=widths*10,
            pos=pos,
            edge_color=edge_colors,
            ax=ax,
            node_size=degrees_norm * 500,
            edgecolors='white',
            linewidths=0.5,
            font_size=8) 
    
    title = 'Cluster: ' + str(cluster_id) + ' sample_i: ' + str(i) #+ '\n' + samples[i]
    ax.set_title(title)
    plt.tight_layout()
    name_to_save = str(n_) + '_network_cluster' + str(cluster_id) + '_' + str(i) 
    plt.savefig(os.path.join(path_to_save + '\\all_988_networks', name_to_save))
    plt.show()





































# %%

edges = LRP_pd.mean(axis=1).reset_index()
edges = edges.rename(columns = {edges.columns[1]:'LRP'})

edges['source_gene']  = edges['edge'].str.split('-', expand = True)[0]
edges['target_gene']= edges['edge'].str.split('-', expand = True)[1]
# edges['source_gene'] = edges['source_gene'].str.split('_', expand=True)[0]
# edges['target_gene'] = edges['target_gene'].str.split('_', expand=True)[0]
edges  = edges.rename(columns = {edges.columns[1]:'LRP'})
edges['LRP_norm'] = edges['LRP'] / edges['LRP'].max()
edges = edges.sort_values('LRP', ascending = False).reset_index(drop=True)

G = nx.from_pandas_edgelist(edges, source='source_gene', target='target_gene', edge_attr='LRP_norm')
degrees = np.array(list(nx.degree_centrality(G).values()))
degrees_norm = degrees / np.max(degrees)
# nx.draw(network, with_labels=True, node_color='white', width = edges['LRP']*100, node_size = network.degree)


widths = edges['LRP'] / edges['LRP'].max() 

edges_from_G = [i[0] + ' - ' + i[1] for i in list(G.edges)]
edge_colors = list(f.color_mapper(edges_from_G).values())

node_color_mapper = {'exp': 'gray', 'mut': 'red', 'amp': 'orange', 'del': 'green', 'fus': 'b'}

pos = nx.spring_layout(G, weight='LRP_norm')

fig, ax = plt.subplots(figsize=(20,20))   
# Plot legend
ls = list(node_color_mapper.keys())
cl = list(node_color_mapper.values())
for label, color in zip(ls, cl):
    ax.plot([], [], 'o', color=color, label=label)
ax.legend(title = 'Nodes', loc='best')

node_colors = pd.Series([i.split('_')[1] for i in list(G.nodes)]).map(node_color_mapper)

nx.draw(G, with_labels=False,
        node_color=node_colors,
        width=widths,
        pos=pos, # font_size=0,
        # cmap = colors,
        edge_color=edge_colors,
        ax=ax,
        # node_size = degrees,
        node_size=degrees_norm * 1000,
        edgecolors='white',  # This adds the white border
    linewidths=0.5)


ax.set_title('Top1000, all 5300 interactions, mean LRP')
plt.tight_layout()
# %%
import community  # If using the Louvain method for community detection
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


graphs_dict = {}

LRP_to_graphs = LRP_pd.copy().reset_index()    
LRP_to_graphs['source_gene']  = LRP_to_graphs['edge'].str.split('-', expand = True)[0]
LRP_to_graphs['target_gene']= LRP_to_graphs['edge'].str.split('-', expand = True)[1]
# reorder columns so 'edge' is the first one, source_gene and target_gene are the second and third
LRP_to_graphs = LRP_to_graphs[['edge', 'source_gene', 'target_gene'] + list(LRP_to_graphs.columns[1:-2])]

# define list of number of max edges in the graph
n_edges = np.linspace(100, 1500, 10).astype(int)[::-1]

for i in range(988):
    print(i)
    lrp = LRP_to_graphs.iloc[:,3+i].values

    # create 'edges' dataframe
    edges = pd.DataFrame(
        {
            "source_gene": LRP_to_graphs["source_gene"],
            "target_gene": LRP_to_graphs["target_gene"],
            "LRP": lrp            
        }
    )

    # sort by LRP_norm
    edges = edges.sort_values('LRP', ascending = False).reset_index(drop=True)

    graphs_dict[i] = {}
    prev_partition = None

    # keep only top n edges
    for n_ in n_edges:
        graphs_dict[i][n_] = {}

        edges_temp = edges.iloc[:n_,:].copy()
        # add column with nomaalized LRP
        edges_temp['LRP_norm'] = edges_temp['LRP'] / edges_temp['LRP'].max()

        # create nx graph from lists
        G = nx.from_pandas_edgelist(
            edges_temp,
            source="source_gene",
            target="target_gene",
            edge_attr=["LRP", "LRP_norm"]
        )

        # add weights to edges to G
        for (u, v, d) in G.edges(data=True):
            G[u][v]['weight'] = d['LRP_norm']


        # add weights to edges to G
        partition = detect_communities(G)

        # Calculate modularity
        modularity = community.modularity(partition, G, weight='LRP_norm')

        graphs_dict[i][n_]['G'] = G
        graphs_dict[i][n_]['partition'] = partition
        graphs_dict[i][n_]['n_nodes'] = G.number_of_nodes()
        graphs_dict[i][n_]['n_edges'] = G.number_of_edges()
        graphs_dict[i][n_]['n_communities'] = len(set(partition.values()))
        graphs_dict[i][n_]['modularity'] = modularity

        # Compute similarity with the previous partition if available
        if prev_partition is not None:
            nodes = list(set(prev_partition.keys()) & set(partition.keys()))
            labels_prev = [prev_partition[node] for node in nodes]
            labels_current = [partition[node] for node in nodes]

            ari = adjusted_rand_score(labels_prev, labels_current)
            nmi = normalized_mutual_info_score(labels_prev, labels_current)

            graphs_dict[i][n_]['ari'] = ari
            graphs_dict[i][n_]['nmi'] = nmi

        prev_partition = partition


# get dataframe with modularity, n_nodes, n_edges, n_communities, ari, nmi

# Initialize an empty list to store the rows of the DataFrame
rows = []

# Iterate over the graph indices
for i in graphs_dict:
    # Iterate over the different edge counts
    for n_ in graphs_dict[i]:
        data = graphs_dict[i][n_]
        
        # Prepare the row dictionary
        row = {
            'i': i,
            'n_edges_selected': n_,
            'n_nodes': data.get('n_nodes', None),
            'n_edges': data.get('n_edges', None),
            'n_communities': data.get('n_communities', None),
            'modularity': data.get('modularity', None),
            'ari': data.get('ari', None),
            'nmi': data.get('nmi', None),
            # Add other metrics if you have them
        }
        
        # Append the row to the list
        rows.append(row)

# Create the DataFrame
df_stats = pd.DataFrame(rows)

# Optionally, sort the DataFrame for better readability
df_stats.sort_values(['i', 'n_edges_selected'], ascending=[True, False], inplace=True)

# Reset index if needed
df_stats.reset_index(drop=True, inplace=True)

# %%% Plot graph stats - individual graphs

# List of metrics to plot
metrics = ['n_nodes', 'n_communities', 'modularity', 'ari', 'nmi']

# Ensure 'n_edges_selected' is sorted in descending order
df_stats.sort_values('n_edges_selected', ascending=False, inplace=True)

# Get the unique values of 'i' to iterate over
unique_i = df_stats['i'].unique()

# Generate a color map for different groups
# colors = cm.get_cmap('viridis', len(unique_i))

# Iterate over each metric to create separate figures
for metric in metrics:
    plt.figure(figsize=(10, 6))
    
    # Iterate over each group 'i' to plot their lines
    for idx, i in enumerate(unique_i):
        # Filter the DataFrame for the current group 'i'
        df_i = df_stats[df_stats['i'] == i]
        # Sort the DataFrame by 'n_edges_selected' to ensure correct plotting
        df_i = df_i.sort_values('n_edges_selected', ascending=False)
        
        # Drop NaN values for the current metric
        df_i = df_i.dropna(subset=[metric])
        
        # Plot the metric vs. number of edges selected with transparency 0.2
        plt.plot(df_i['n_edges_selected'], df_i[metric], alpha=0.05)#, color=colors(idx))
    
    # Set plot labels and title
    plt.xlabel('Number of Edges Selected (n_edges_selected)')
    plt.ylabel(metric)
    plt.title(f'{metric} vs. Number of Edges Selected')
    
    # Invert x-axis to show decreasing number of edges
    plt.gca().invert_xaxis()
    
    # Optionally, save the plot
    # plt.savefig(f'{metric}_vs_n_edges_selected.png')
    
    # Display the plot
    plt.show()
# %% analyse communities
import networkx as nx
import community  # For Louvain method
import pandas as pd
import numpy as np

# Assuming graphs_dict is already populated as per your previous code

# Define a function to extract communities with at least 5 nodes
def get_large_communities(partition):
    # Reverse the partition dictionary to get community labels as keys
    community_dict = {}
    for node, comm_id in partition.items():
        community_dict.setdefault(comm_id, set()).add(node)
    # Filter communities with at least 5 nodes
    large_communities = {comm_id: nodes for comm_id, nodes in community_dict.items() if len(nodes) >= 5}
    return large_communities

# Define a function to compare communities between two partitions
def match_communities(prev_communities, current_communities, threshold=0.5):
    # For each previous community, find the best matching current community
    matched_communities = {}
    for prev_comm_id, prev_nodes in prev_communities.items():
        best_match_id = None
        best_similarity = 0
        for curr_comm_id, curr_nodes in current_communities.items():
            # Compute Jaccard similarity
            intersection = prev_nodes & curr_nodes
            union = prev_nodes | curr_nodes
            similarity = len(intersection) / len(union)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = curr_comm_id
        # Check if the best similarity exceeds the threshold
        if best_similarity >= threshold:
            matched_communities[prev_comm_id] = {
                'current_comm_id': best_match_id,
                'similarity': best_similarity,
                'prev_nodes': prev_nodes,
                'current_nodes': current_communities[best_match_id]
            }
    return matched_communities

# Now, perform the analysis for each 'i'
threshold = 0.5  # Similarity threshold to consider communities as the same
n_edges_list = sorted(n_edges.tolist(), reverse=True)  # Ensure n_edges is sorted in descending order

# Store the results
community_persistence = {}

for i in graphs_dict:
    print(f"Analyzing graph group {i}")
    community_persistence[i] = {}
    
    # Get the partitions for all 'n_' values
    partitions = {}
    for n_ in n_edges_list:
        partitions[n_] = graphs_dict[i][n_]['partition']
    
    # Get large communities from the largest graph (highest n_)
    n_max = n_edges_list[0]
    prev_partition = partitions[n_max]
    prev_communities = get_large_communities(prev_partition)
    
    # Store the initial large communities
    community_persistence[i][n_max] = prev_communities
    
    # For decreasing 'n_' values, track communities
    for n_ in n_edges_list[1:]:
        print(f"Comparing communities at n_={n_} with n_={n_max}")
        current_partition = partitions[n_]
        current_communities = get_large_communities(current_partition)
        
        # Match communities from prev_communities to current_communities
        matched_communities = match_communities(prev_communities, current_communities, threshold=threshold)
        
        # Store the matched communities
        community_persistence[i][n_] = matched_communities
        
        # Update prev_communities for the next iteration
        prev_communities = {match['current_comm_id']: match['current_nodes'] for match in matched_communities.values()}
        # Update n_max for the next iteration
        n_max = n_
    
    print(f"Completed analysis for graph group {i}\n")

# Now, we can analyze whether the same communities persist across decreasing 'n_'

# Example: Print the persistence of communities for a specific 'i' and a community
i_example =1  # Replace with the desired 'i' value
print(f"Community persistence for graph group {i_example}:\n")
for n_ in n_edges_list:
    if n_ in community_persistence[i_example]:
        communities = community_persistence[i_example][n_]
        if isinstance(communities, dict):
            print(f"At n_={n_}, communities:")
            for comm_id, nodes in communities.items():
                print(f"  Community {comm_id} with {len(nodes)} nodes")
        else:
            print(f"At n_={n_}, matched communities:")
            for prev_comm_id, match_info in communities.items():
                curr_comm_id = match_info['current_comm_id']
                similarity = match_info['similarity']
                print(f"  Previous Community {prev_comm_id} matched with Current Community {curr_comm_id} (Similarity: {similarity:.2f})")
    else:
        print(f"No data for n_={n_}")
    print()


# %%
import matplotlib.pyplot as plt

# For a specific 'i' and community, plot the similarity over 'n_' values
i_example = 0  # Replace with your desired 'i'
community_ids = list(community_persistence[i_example][n_edges_list[0]].keys())

for comm_id in community_ids:
    similarities = []
    n_values = []
    n_max = n_edges_list[0]
    prev_comm_id = comm_id
    for n_ in n_edges_list[1:]:
        if n_ in community_persistence[i_example]:
            matched_communities = community_persistence[i_example][n_]
            if prev_comm_id in matched_communities:
                match_info = matched_communities[prev_comm_id]
                similarities.append(match_info['similarity'])
                n_values.append(n_)
                prev_comm_id = match_info['current_comm_id']
            else:
                # Community no longer matched
                similarities.append(0)
                n_values.append(n_)
                break  # Stop tracking this community
        else:
            break  # No data for this n_
    if similarities:
        plt.plot([n_max] + n_values, [1.0] + similarities, marker='o', label=f'Community {comm_id}')

plt.xlabel('Number of Edges Selected (n_edges_selected)')
plt.ylabel('Similarity')
plt.title(f'Community Persistence in Graph Group {i_example}')
plt.gca().invert_xaxis()
plt.legend()
plt.show()

# %%

import networkx as nx
from grakel import GraphKernel
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming graphs_dict is already populated as per your previous code
# and n_edges is a list or array of n_ values

# Ensure n_edges is a sorted list
n_edges_list = sorted(n_edges.tolist(), reverse=True)  # Ensure n_edges is sorted in descending order

# Iterate over each n_
for n_ in n_edges_list:
    print(f"Processing graphs with n_ = {n_}")
    
    # Collect graphs for this n_
    graph_list = []
    graph_labels = []  # To keep track of which graph is which
    for i in graphs_dict:
        # Check if the graph exists for this n_
        if n_ in graphs_dict[i]:
            G = graphs_dict[i][n_]['G']
            # Relabel nodes to integers starting from 0 for Grakel
            mapping = {node: idx for idx, node in enumerate(G.nodes())}
            G_relabelled = nx.relabel_nodes(G, mapping)
            # Extract edges and node labels if necessary
            edges = list(G_relabelled.edges())
            # Optionally, include node labels (attributes)
            # For WL kernel, node labels can be important
            node_labels = {idx: str(G.nodes[node].get('label', '')) for node, idx in mapping.items()}
            # Prepare the graph in Grakel format
            grakel_graph = (edges, node_labels)
            graph_list.append(grakel_graph)
            graph_labels.append({'i': i, 'n_edges_selected': n_})
        else:
            print(f"Graph {i} does not have n_ = {n_}")

    # Skip if no graphs are available for this n_
    if not graph_list:
        print(f"No graphs to process for n_ = {n_}")
        continue

    # Now compute the kernel matrix using the Weisfeiler-Lehman kernel
    # Initialize Weisfeiler-Lehman kernel
    gk = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": 5}, normalize=True)
    K = gk.fit_transform(graph_list)

    # K is the kernel (similarity) matrix between graphs
    # Now perform clustering using K

    # Decide on the number of clusters (you can adjust this)
    num_clusters = 5  # Adjust based on your data

    # Perform spectral clustering using the kernel matrix as affinity
    spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
    cluster_labels = spectral_clustering.fit_predict(K)

    # Assign cluster labels to the graphs
    df_graphs = pd.DataFrame(graph_labels)
    df_graphs['cluster_label'] = cluster_labels

    # Now analyze the clusters
    cluster_counts = df_graphs['cluster_label'].value_counts()
    print(f"Cluster counts for n_ = {n_}:\n{cluster_counts}\n")

    # Optionally, collect additional graph statistics for analysis
    modularity_list = []
    n_nodes_list = []
    n_edges_in_graph_list = []
    for idx, row in df_graphs.iterrows():
        i = row['i']
        n_edges_selected = row['n_edges_selected']
        data = graphs_dict[i][n_edges_selected]
        modularity_list.append(data.get('modularity', np.nan))
        n_nodes_list.append(data.get('n_nodes', np.nan))
        n_edges_in_graph_list.append(data.get('n_edges', np.nan))
    df_graphs['modularity'] = modularity_list
    df_graphs['n_nodes'] = n_nodes_list
    df_graphs['n_edges_in_graph'] = n_edges_in_graph_list

    # Visualize clusters based on modularity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_graphs, x='i', y='modularity', hue='cluster_label', palette='tab10', alpha=0.7)
    plt.title(f'Graph Clusters for n_ = {n_}')
    plt.xlabel('Graph Index (i)')
    plt.ylabel('Modularity')
    plt.legend(title='Cluster Label')
    plt.show()

    # Additional visualization: number of nodes vs. number of edges in graph
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_graphs, x='n_nodes', y='n_edges_in_graph', hue='cluster_label', palette='tab10', alpha=0.7)
    plt.title(f'Graph Size Distribution for n_ = {n_}')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Number of Edges in Graph')
    plt.legend(title='Cluster Label')
    plt.show()
