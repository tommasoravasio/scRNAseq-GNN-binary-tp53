import os
import sys
import json
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


def plot_training_curves(csv_path, model_name="Model"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    sns.lineplot(data=df, x="Epoch", y="Train Accuracy", label="Train Accuracy")
    sns.lineplot(data=df, x="Epoch", y="Test Accuracy", label="Validation Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    sns.lineplot(data=df, x="Epoch", y="Loss", label="Train Loss")
    if "Test Loss" in df.columns:
        sns.lineplot(data=df, x="Epoch", y="Test Loss", label="Validation Loss")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    f1_plotted = False
    if "Train F1" in df.columns:
        sns.lineplot(data=df, x="Epoch", y="Train F1", label="Train F1")
        f1_plotted = True
    if "Test F1" in df.columns:
        sns.lineplot(data=df, x="Epoch", y="Test F1", label="Validation F1")
        f1_plotted = True
    if f1_plotted:
        plt.title(f"{model_name} - F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.grid(True)
    else:
        plt.axis('off')

    plt.tight_layout()
    plt.show()




def main(config_path):
    # Inside main(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {config_path}. Details: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded configuration from: {config_path}")

    global_feature_selection = config.get("feature_selection", "HVG")
    base_data_path_prefix = config.get("base_data_path_prefix", "data/graphs_")
    dataset_variants_map = config.get("dataset_variants", {})
    default_train_params = config.get("default_train_model_params", {})
    comparison_runs = config.get("comparison_runs", [])

    if not comparison_runs:
        print("Warning: No comparison runs defined in config.", file=sys.stderr)
        return

    print(f"Global feature selection: {global_feature_selection}")

    for i, run_config in enumerate(comparison_runs):
        run_id = run_config.get("run_id")
        dataset_variant_key = run_config.get("dataset_variant_key")
        model_type = run_config.get("model_type")

        if not all([run_id, dataset_variant_key, model_type]):
            print(f"Run {i+1}: Skipping due to missing 'run_id', 'dataset_variant_key', or 'model_type'. Config: {run_config}", file=sys.stderr)
            continue

        print(f"\n--- Starting Run {i+1}/{len(comparison_runs)}: {run_id} ---")
        print(f"Model: {model_type}, Dataset Key: {dataset_variant_key}")

        variant_path_template = dataset_variants_map.get(dataset_variant_key)
        if not variant_path_template:
            print(f"Run '{run_id}': Dataset variant key '{dataset_variant_key}' not in 'dataset_variants'. Skipping.", file=sys.stderr)
            continue

        actual_variant_suffix = variant_path_template.replace("{feature_selection}", global_feature_selection)

        train_path = Path(f"{base_data_path_prefix}{actual_variant_suffix}/train")
        test_path = Path(f"{base_data_path_prefix}{actual_variant_suffix}/test")

        print(f"Loading train data from: {train_path}")
        # Assuming model_constructor.load_graphs is available and handles its own errors/exits
        train_pyg = model_constructor.load_graphs(str(train_path))
        if not train_pyg: # If load_graphs returns empty list (and didn't exit)
             print(f"Run '{run_id}': Failed to load training data from {train_path}. Skipping.", file=sys.stderr)
             continue

        print(f"Loading test data from: {test_path}")
        test_pyg = model_constructor.load_graphs(str(test_path))
        if not test_pyg: # If load_graphs returns empty list (and didn't exit)
             print(f"Run '{run_id}': Failed to load test data from {test_path}. Skipping.", file=sys.stderr)
             continue

        current_run_params = default_train_params.copy()
        current_run_params.update(run_config.get("train_model_params", {}))

        current_run_params["ID_model"] = run_id
        current_run_params["model_type"] = model_type
        # Pass the global_feature_selection to train_model, as it uses it for structuring results_dir
        current_run_params["feature_selection"] = global_feature_selection

        print(f"Run '{run_id}': Training with params: {current_run_params}")

        try:
            # Ensure model_constructor.train_model is correctly called
            model_constructor.train_model(
                train_PyG=train_pyg,
                test_PyG=test_pyg,
                **current_run_params # Unpack all parameters for train_model
            )
            print(f"--- Finished Run: {run_id} ---")
        except Exception as e:
            print(f"Run '{run_id}': Error during training: {e}", file=sys.stderr)
            # Optionally, add gc.collect() and torch.cuda.empty_cache() here if memory is an issue

    print("\nAll configured comparison runs attempted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file for model comparisons.")
    args = parser.parse_args()
    main(config_path=args.config)