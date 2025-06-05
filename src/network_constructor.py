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
    """
    Creates the correlation matrix between the features of the dataset.
    CAMBIATA RISPETTO A PRIMA, ORA:
    p_val="yes" calcola anche p value
    p_val="no" non calcola p value (test per histogram correlation)
    """
    
    if p_val == "yes":
        #FILTRO SU P VALUE
        corr, p = scipy.stats.spearmanr(data)
        alpha = p_value_threshold / math.comb(data.shape[1], 2)
        aus = np.where((p <= alpha) & (np.abs(corr) >= corr_threshold), corr, 0)
    
    else:
        #NO FILTRO SU P VALUE
        corr, p = scipy.stats.spearmanr(data)
        aus = np.where((np.abs(corr) >= corr_threshold), corr, 0)

    np.fill_diagonal(aus, 0)
    
    return aus

def check_percentage_of_zeros(matrix):
    """
    Controlla la percentuale di valori diversi da zero nella matrice di correlazione."""
    value_different_from_zero = np.sum(matrix != 0) / (matrix.shape[0]**2)
    print(f"Percentage of non-zero values in the correlation matrix: {value_different_from_zero }")

def plot_the_correlation_matrix(dataset_final, matrix):
    """
    Plotta il grafo della matrice di correlazione."""
    node_list = dataset_final.columns.to_list()
    num_nodes = len(node_list)
    tick_indices = np.arange(0, num_nodes, 100)
    tick_labels = [node_list[i] for i in tick_indices]
    plt.figure(figsize=(7, 7))
    plt.imshow(matrix, cmap='binary', interpolation='none')
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=90)
    plt.yticks(ticks=tick_indices, labels=tick_labels)
    plt.show()

#VERSIONE PRIMA DI FARE IMPLEMENTAZIONE PER CLUSTER
def create_PyG_graph_from_df(df,matrix, label_column="mutation_status"):
    """
    Crea un grafo PyG da un dataframe e una matrice di correlazione."""
    df_pyg = []
    edge_index = tg_utils.dense_to_sparse(torch.tensor(matrix, dtype=torch.float32))[0]

    for obs in df.itertuples(index=False):
        #edge_index = tg_utils.dense_to_sparse(torch.tensor(matrix, dtype=torch.float32))[0]
        x = torch.tensor(obs[:-1],dtype=torch.float32).view(-1,1)
        y = int(getattr(obs, label_column) == "mut") #"mut":1 , "wt":0
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.long))

        transform = LargestConnectedComponents(num_components=1) #!!! Non dovremmo avere components separati ma cerca di capire
        data = transform(data)

        df_pyg.append(data)
    return df_pyg




#VERSIONE PER CLUSTER
def create_PyG_graph_from_df_cluster(df,matrix, label_column="mutation_status", label="train",graphs_per_batch=500, graphs_folder_ID=""):
    """
    ATTENTION: run this function only on the HPC cluster
    Crea un grafo PyG da un dataframe e una matrice di correlazione.
    There must exist a folder named graphs with two folders inside named train and test.
    Use label to indicate if we are building for train or for test"""
    #df_pyg = []
    edge_index = tg_utils.dense_to_sparse(torch.tensor(matrix, dtype=torch.float32))[0]
    graphs = []

    #for obs in df.itertuples(index=False):
    for i, obs in enumerate(df.itertuples(index=False)):

        #edge_index = tg_utils.dense_to_sparse(torch.tensor(matrix, dtype=torch.float32))[0]
        x = torch.tensor(obs[:-1],dtype=torch.float32).view(-1,1)
        y = int(getattr(obs, label_column) == "MUT") #"MUT":1 , "WT":0
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.long))
        

        # #!!! Non dovremmo avere components separati ma cerca di capire
        # transform = LargestConnectedComponents(num_components=1)
        # data = transform(data)
        graphs.append(data)

        #df_pyg.append(data)

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
    """
    Controlla se tutti i grafi nel dataframe PyG hanno la stessa struttura."""
    i,j = np.random.randint(0, len(dataframe_pyg), 2)
    graph1 = dataframe_pyg[i]
    graph2 = dataframe_pyg[j]
    return (graph1.edge_index == graph2.edge_index).all() 

def get_info_and_plot_graph(df_pyg):
    """
    Ottiene informazioni sul primo grafo e lo visualizza per vedere se formato va bene"""
    #TEST: test on the first network
    test=df_pyg[0] 
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {test.num_nodes}') 
    print(f'Number of edges: {test.num_edges}') 
    print(f'NUmber of features per node: {test.num_node_features}') 
    print(f'Has isolated nodes: {test.has_isolated_nodes()}')
    print(f'Has self-loops: {test.has_self_loops()}')
    print(f'Is undirected: {test.is_undirected()}')

    print('=============================================================')

    plt.figure(figsize=(9,9))
    G=tg_utils.to_networkx(test, to_undirected=True)
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_size = 5)
    plt.show()


def save_dataset(train_pyg, test_pyg, name_train="train_reteunica.pt", name_test="test_reteunica.pt"):
    """
    Salva i dataset PyG in file .pt"""
    #salva il dataset
    torch.save(train_pyg, 'train_reteunica.pt')
    torch.save(test_pyg, 'test_reteunica.pt')


def main():
    df = pd.read_csv("notebooks/final_preprocessed_data.csv", index_col=0)
    train_df, test_df =train_test_split(df, test_size=0.2, random_state=42)
    mat = build_correlation_matrix(train_df.iloc[:, :-1], corr_threshold=0.2, p_value_threshold=0.05, p_val="yes")
    create_PyG_graph_from_df_cluster(train_df, mat, label_column="mutation_status",label="train",graphs_folder_ID="_threshold_05_pval")
    create_PyG_graph_from_df_cluster(test_df, mat, label_column="mutation_status",label="test",graphs_folder_ID="_threshold_05_pval")

if __name__ == "__main__":
    main()   



