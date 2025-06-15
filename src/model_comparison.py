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
import argparse


import load_data
import preprocessing
import network_constructor
import model_constructor



def main(feature_selection="HVG"):

    train_df_pyg = model_constructor.load_graphs(f"data/graphs_{feature_selection}_/train")
    test_df_pyg = model_constructor.load_graphs(f"data/graphs_{feature_selection}_/test")

    train_df_pyg_combat = model_constructor.load_graphs(f"data/graphs_{feature_selection}_combat/train")
    test_df_pyg_combat = model_constructor.load_graphs(f"data/graphs_{feature_selection}_combat/test")

    train_df_pyg_harmony = model_constructor.load_graphs(f"data/graphs_{feature_selection}_combat/train")
    test_df_pyg_harmony = model_constructor.load_graphs(f"data/graphs_{feature_selection}_combat/test")

    #BASELINE GCN
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 50, batch_size = 16, ID_model = "baseline", model_type="gcn",feature_selection=feature_selection)
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 50, batch_size = 16, ID_model = "baseline", model_type="gat",feature_selection=feature_selection)

    #Combat
    model = model_constructor.train_model(train_PyG=train_df_pyg_combat, test_PyG=test_df_pyg_combat, epochs = 50, batch_size = 16, ID_model = "combat", model_type="gcn",feature_selection=feature_selection) 
    model = model_constructor.train_model(train_PyG=train_df_pyg_combat, test_PyG=test_df_pyg_combat, epochs = 50, batch_size = 16, ID_model = "combat", model_type="gat",feature_selection=feature_selection) 

    #harmony
    model = model_constructor.train_model(train_PyG=train_df_pyg_harmony, test_PyG=test_df_pyg_harmony, epochs = 50, batch_size = 16, ID_model = "harmony", model_type="gcn",feature_selection=feature_selection)
    model = model_constructor.train_model(train_PyG=train_df_pyg_harmony, test_PyG=test_df_pyg_harmony, epochs = 50, batch_size = 16, ID_model = "harmony", model_type="gat",feature_selection=feature_selection)


    #Weight
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 50, batch_size = 16, ID_model = "weight", loss_weight=True, model_type="gcn",feature_selection=feature_selection)
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 50, batch_size = 16, ID_model = "weight", loss_weight=True, model_type="gat",feature_selection=feature_selection)


    #AdamW
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 50, batch_size = 16, ID_model = "AdamW", use_adamW=True, model_type="gcn",feature_selection=feature_selection)
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 50, batch_size = 16, ID_model = "AdamW", use_adamW=True, model_type="gat",feature_selection=feature_selection)


    #GraphNorm
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 50, batch_size = 16, ID_model = "GraphNorm", use_graphnorm=True, model_type="gcn",feature_selection=feature_selection)
    model = model_constructor.train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 50, batch_size = 16, ID_model = "GraphNorm", use_graphnorm=True, model_type="gat",feature_selection=feature_selection)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_selection",
        choices=["HVG", "target"],
        default="HVG",
    )
    args = parser.parse_args()

    main(feature_selection=args.feature_selection)