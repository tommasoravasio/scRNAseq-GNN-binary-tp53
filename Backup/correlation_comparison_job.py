#correlation_comparison_job.py

import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import gc

sys.path.append(os.path.abspath("src"))
import load_data
import preprocessing
import network_constructor


def plot_frequency_of_correlation_values(matrices, bins=50, alpha=0.5, filename="plot.png"):

    plt.figure(figsize=(10, 6))
    for label, mat in matrices.items():
        values = mat.flatten()
        plt.hist(values, bins=bins, alpha=alpha, label=label, edgecolor='black', density=True)
        mean_val = np.mean(values)
        plt.axvline(mean_val, linestyle='--', linewidth=2, label=f"Mean {label}: {mean_val:.2f}")
    plt.yscale('log')
    plt.legend()
    plt.title('Hist of Correlation Values')
    plt.xlabel('Correlation Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(filename)
    plt.close('all')


def import_and_create_matrices_for_plotting(path, df_expression, col_name="Ensembl ID", verbosity=False):
    tab = pd.read_excel(path)
    tab_ensembl_ids = [gene for gene in tab[col_name] if gene in df_expression.columns]
    df_tab = df_expression[tab_ensembl_ids].copy()
    mat_tab = network_constructor.build_correlation_matrix(df_tab, corr_threshold=0., p_value_threshold=1)
    if verbosity:
        print(df_tab.head())
        network_constructor.check_percentage_of_zeros(mat_tab)
    return mat_tab


def plot_correlation_comparison(df_expression, path_list):

    if not isinstance(path_list, list):
        path_list = [path_list]

    mat = network_constructor.build_correlation_matrix(df_expression, corr_threshold=0., p_value_threshold=1)
    plot_frequency_of_correlation_values({"Original": mat}, bins=50, alpha=0.5, filename="correlation_hist_original.png")
    plt.close("all")
    del mat
    gc.collect()

    for path in path_list:
        label = path.split("/")[-1].split(".")[0]
        mat = import_and_create_matrices_for_plotting(path, df_expression, mat_name=label, col_name="Ensembl ID", verbosity=False)
        plot_frequency_of_correlation_values({label: mat}, bins=50, alpha=0.5, filename=f"correlation_hist_{label}.png")
        plt.close("all")
        del mat
        gc.collect()

adata = load_data.load_expression_data( "data/Expression_Matrix", verbosity=True)
df_expression=ad.AnnData.to_df(adata)
tp53_target_path = "data/Target_genes/41388_2017_BFonc2016502_MOESM5_ESM_tab1.xlsx"
plot_correlation_comparison(df_expression,tp53_target_path)