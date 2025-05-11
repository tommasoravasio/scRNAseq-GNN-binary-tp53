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


def build_correlation_matrix(data, corr_threshold=0.1):
    """
    Creates the correlation matrix between the features of the dataset using Spearman correlation."""
    corr,p = scipy.stats.spearmanr(data)  
    alpha = 0.05/ math.comb(data.shape[1], 2)
    aus = ( (p<alpha) & (np.absolute(corr) > corr_threshold) ).astype(int)
    np.fill_diagonal(aus,0)

    return aus

def check_percentage_of_zeros(matrix):
    """
    Controlla la percentuale di valori diversi da zero nella matrice di correlazione."""
    #controlliamo che percentuale delle caselle e diversa da 0
    value_different_from_zero = np.sum(matrix != 0) / (matrix.shape[0]**2)
    print(f"Percentage of non-zero values in the correlation matrix: {value_different_from_zero }")

def plot_the_correlation_matrix(dataset_final, matrix):
    """
    Plotta il grafo della matrice di correlazione."""
    #plottiamo il grafo
    node_list = dataset_final.columns.to_list()
    num_nodes = len(node_list)
    tick_indices = np.arange(0, num_nodes, 100)
    tick_labels = [node_list[i] for i in tick_indices]
    plt.figure(figsize=(7, 7))
    plt.imshow(matrix, cmap='binary', interpolation='none')
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=90)
    plt.yticks(ticks=tick_indices, labels=tick_labels)
    plt.show()


def create_PyG_graph_from_df(df,matrix):
    """
    Crea un grafo PyG da un dataframe e una matrice di correlazione."""
    df_pyg = []

    for obs in df.itertuples(index=False):
        edge_index = tg_utils.dense_to_sparse(torch.tensor(matrix, dtype=torch.float32))[0]
        x = torch.tensor(obs[:-1],dtype=torch.float32).view(-1,1)
        y = int(obs.label == "mut") #"mut":1 , "wt":0
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.long))

        transform = LargestConnectedComponents(num_components=1) #!!! Non dovremmo avere components separati ma cerca di capire
        data = transform(data)

        df_pyg.append(data)
    return df_pyg

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


def main(path):
    df = pd.read_csv(path)
    #split in train e test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"shape del train test: {train_df.shape} \nshape del test set: {test_df.shape}")
    
    #trova la matrice di correlazione
    mat=build_correlation_matrix(train_df.iloc[:,:-1])
    check_percentage_of_zeros(mat)
    plot_the_correlation_matrix(train_df, mat)

    #crea le strutture per pytorch geometric
    train_df_pyg = create_PyG_graph_from_df(train_df, mat)
    test_df_pyg = create_PyG_graph_from_df(test_df, mat)

    #check sul primo elemento del train e test set
    print(f"First element of train_df_pyg: {train_df_pyg[0]}")
    print(f"First element of test_df_pyg: {test_df_pyg[0]}")

    #check sulla struttura del grafo
    assert check_graph_structure(train_df_pyg), "The graphs in the train set do not have the same structure."
    assert check_graph_structure(test_df_pyg), "The graphs in the test set do not have the same structure."

    #get info and plot the first graph obtaied
    get_info_and_plot_graph(train_df_pyg)

    # #salva i dataset
    # save_dataset(train_df_pyg, test_df_pyg)   #UNCOMMENTA PER SALVARE IL DATASET


if __name__ == "__main__":
    main('/Users/tommasoravasio/Desktop/BSc Thesis/Mio/dataset_final_unica_rete.csv')   #Metti il tuo dataset qui



