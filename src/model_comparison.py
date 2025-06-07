import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import mygene
sys.path.append(os.path.abspath("../src"))
import importlib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats
import math
from torch_geometric.transforms import LargestConnectedComponents
import torch_geometric.utils as tg_utils
from torch_geometric.data import Data 
import networkx as nx
import torch
import seaborn as sns
import gc
from pathlib import Path
import scanpy.external as sce
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


import load_data
import preprocessing
import network_constructor
import model_constructor



def main():

    #BASELINE
    train_df_pyg = model_constructor.load_graphs("data/graphs_baseline/train")
    test_df_pyg = model_constructor.load_graphs("data/graphs_baseline/test")
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 25, batch_size = 16, ID_model = "baseline")

   #Combat
    train_df_pyg = model_constructor.load_graphs("data/graphs_combat/train")
    test_df_pyg = model_constructor.load_graphs("data/graphs_combat/test")
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 25, batch_size = 16, ID_model = "combat") 
    
    #harmony
    train_df_pyg = model_constructor.load_graphs("data/graphs_harmony/train")
    test_df_pyg = model_constructor.load_graphs("data/graphs_harmony/test")
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 25, batch_size = 16, ID_model = "harmony")
    
    #L2REG
    train_df_pyg = model_constructor.load_graphs("data/graphs_L2reg/train")
    test_df_pyg = model_constructor.load_graphs("data/graphs_l2reg/test")
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 25, batch_size = 16, ID_model = "L2reg")

    #Weight
    train_df_pyg = model_constructor.load_graphs("data/graphs_baseline/train")
    test_df_pyg = model_constructor.load_graphs("data/graphs_baseline/test")
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 25, batch_size = 16, ID_model = "weight", loss_weight=True)

    #Smalltest
    train_df_pyg = model_constructor.load_graphs("data/graphs_smalltest/train")
    test_df_pyg = model_constructor.load_graphs("data/graphs_smalltest/test")
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 25, batch_size = 16, ID_model = "smalltest")

    #Target tp53
    train_df_pyg = model_constructor.load_graphs("data/graphs_target/train")
    test_df_pyg = model_constructor.load_graphs("data/graphs_target/test")
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 25, batch_size = 16, ID_model = "target")

















if __name__ == "__main__":
    main()