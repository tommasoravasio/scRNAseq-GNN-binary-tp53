"""
Graph and correlation matrix construction utilities for single-cell data.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats
import math
from torch_geometric.transforms import LargestConnectedComponents
import torch_geometric.utils as tg_utils
from torch_geometric.data import Data 
import networkx as nx
import torch
import os


def build_correlation_matrix(data, corr_threshold=0.1, p_value_threshold=0.05, p_val="yes"):
    """Build a filtered Spearman correlation matrix."""
    if p_val == "yes":
        corr, p = scipy.stats.spearmanr(data)
        alpha = p_value_threshold / math.comb(data.shape[1], 2)
        aus = np.where((p <= alpha) & (np.abs(corr) >= corr_threshold), corr, 0)
    else:
        corr, p = scipy.stats.spearmanr(data)
        aus = np.where((np.abs(corr) >= corr_threshold), corr, 0)
    np.fill_diagonal(aus, 0)
    return aus


def check_percentage_of_zeros(matrix):
    """Print percent of non-zero values in a matrix."""
    value_different_from_zero = np.sum(matrix != 0) / (matrix.shape[0]**2)
    print(f"Percentage of non-zero values in the correlation matrix: {value_different_from_zero }")


def plot_the_correlation_matrix(dataset_final, matrix):
    """Show a heatmap of the correlation matrix."""
    node_list = dataset_final.columns.to_list()
    num_nodes = len(node_list)
    tick_indices = np.arange(0, num_nodes, 100)
    tick_labels = [node_list[i] for i in tick_indices]
    plt.figure(figsize=(7, 7))
    plt.imshow(matrix, cmap='binary', interpolation='none')
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=90)
    plt.yticks(ticks=tick_indices, labels=tick_labels)
    plt.show()


def create_PyG_graph_from_df_cluster(df, matrix, label_column="mutation_status", label="train", graphs_per_batch=500, graphs_folder_ID=""):
    """Create and save PyG graphs from a DataFrame and correlation matrix."""
    edge_index = tg_utils.dense_to_sparse(torch.tensor(matrix, dtype=torch.float32))[0]
    graphs = []
    for i, obs in enumerate(df.itertuples(index=False)):
        x = torch.tensor(obs[:-1], dtype=torch.float32).view(-1, 1)
        y = int(getattr(obs, label_column) == "MUT")
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.long))
        graphs.append(data)
        if (i + 1) % graphs_per_batch == 0 or i == len(df) - 1:
            batch_index = i // graphs_per_batch
            folder = f"graphs{graphs_folder_ID}/{label}"
            os.makedirs(folder, exist_ok=True)
            filename = f"{folder}/batch_{batch_index:03d}.pt"
            torch.save(graphs, filename, pickle_protocol=5)
            print(f"Saved {len(graphs)} graphs to {filename}")
            graphs = []  
    return None


def check_graph_structure(dataframe_pyg):
    """Check if all PyG graphs have the same structure."""
    i, j = np.random.randint(0, len(dataframe_pyg), 2)
    graph1 = dataframe_pyg[i]
    graph2 = dataframe_pyg[j]
    return (graph1.edge_index == graph2.edge_index).all() 


def get_info_and_plot_graph(df_pyg):
    """Print info and plot the first PyG graph."""
    test = df_pyg[0] 
    print('=============================================================')
    print(f'Number of nodes: {test.num_nodes}') 
    print(f'Number of edges: {test.num_edges}') 
    print(f'Number of features per node: {test.num_node_features}') 
    print(f'Has isolated nodes: {test.has_isolated_nodes()}')
    print(f'Has self-loops: {test.has_self_loops()}')
    print(f'Is undirected: {test.is_undirected()}')
    print('=============================================================')
    plt.figure(figsize=(9,9))
    G = tg_utils.to_networkx(test, to_undirected=True)
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_size=5)
    plt.show()


def save_dataset(train_pyg, test_pyg, name_train="train_reteunica.pt", name_test="test_reteunica.pt"):
    """Save PyG datasets to .pt files."""
    torch.save(train_pyg, 'train_reteunica.pt')
    torch.save(test_pyg, 'test_reteunica.pt')


def plot_the_correlation_matrix_colored(dataset_final, matrix):
    """Show a colored heatmap of the correlation matrix."""
    node_list = dataset_final.columns.to_list()
    num_nodes = len(node_list)
    tick_indices = np.arange(0, num_nodes, 100)
    tick_labels = [node_list[i] for i in tick_indices]
    plt.figure(figsize=(7, 7))
    cmap = plt.get_cmap("magma")
    plt.imshow(matrix, cmap=cmap, interpolation='none')
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=90, fontsize=7)
    plt.yticks(ticks=tick_indices, labels=tick_labels, fontsize=7)
    plt.colorbar(label='Correlation coefficient (ρ)')
    plt.title('Correlation matrix', pad=12)
    plt.tight_layout()
    plt.show()


def main(feature_selection="HVG",batch_correction=None):
    df = pd.read_csv(f"notebooks/final_preprocessed_data_{feature_selection}_{batch_correction}.csv", index_col=0)
    train_df, test_df =train_test_split(df, test_size=0.2, random_state=42)
    mat = build_correlation_matrix(train_df.iloc[:, :-1], corr_threshold=0.2, p_value_threshold=0.05, p_val="yes")
    create_PyG_graph_from_df_cluster(train_df, mat, label_column="mutation_status",label="train",graphs_folder_ID=f"_{feature_selection}_{batch_correction}")
    create_PyG_graph_from_df_cluster(test_df, mat, label_column="mutation_status",label="test",graphs_folder_ID=f"_{feature_selection}_{batch_correction}")

if __name__ == "__main__":
    main(feature_selection="target",batch_correction="combat")  



