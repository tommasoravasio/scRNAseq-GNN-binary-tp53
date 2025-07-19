"""
This script provides utility functions for loading and preprocessing single-cell RNA-seq expression data.
"""

import scanpy as sc
import pandas as pd
import anndata as ad
import os


def one_to_three_columns_features_file(file_path):
    """Ensure features.tsv has 3 columns for Scanpy."""
    features = pd.read_csv(file_path, header=None, sep="\t")
    assert features.shape[1] == 1
    features[1] = features[0] 
    features[2] = "Gene Expression"
    features.to_csv(file_path, header=False, index=False, sep="\t")


def load_expression_data(file_path, verbosity=False): 
    """Load 10X expression data as AnnData."""
    #check on number of columns in features.tsv
    features_path = os.path.join(file_path, "features.tsv.gz")
    features = pd.read_csv(features_path, header=None, sep="\t", compression="gzip")
    if features.shape[1] == 1:
        one_to_three_columns_features_file(features_path)
    assert features.shape[1] == 3, f"features.tsv must have 3 columns, but has {features.shape[1]} columns"
   
    adata = sc.read_10x_mtx(file_path,
    var_names="gene_ids",
    cache=True)
    df_expression=ad.AnnData.to_df(adata)
    if verbosity:
        print(f"df_expression shape: {df_expression.shape}")
        print(f"df_expression columns: {df_expression.columns}")
        print(f"df_expression head: {df_expression.head()}")
    return adata


def load_mutation_data(file_path, verbosity=False):
    """Load mutation data from CSV."""
    df_mutation = pd.read_csv(file_path, sep=",", index_col=0)
    df_mutation['Sample_Name_cleaned'] = df_mutation['Sample_Name'].str.replace('-', '', regex=False)
    if verbosity:
        print(f"df_mutation shape: {df_mutation.shape}")
        print(f"df_mutation columns: {df_mutation.columns}")
        print(f"df_mutation head: {df_mutation.head()}")
    return df_mutation

def add_cleaned_column(df, column_name="Sample_Name"):
    """Remove hyphens from cell line IDs."""
    df[f"{column_name}_cleaned"] = df[column_name].str.replace('-', '', regex=False)
    return df

def test():
    """Test the functions in this module."""
    expression_data_path = "data/Expression_Matrix"
    df_expression = load_expression_data(expression_data_path, verbosity=True)
    mutation_data_path = "data/Mutation/CellLineDownload_r21.csv"
    df_mutation = load_mutation_data(mutation_data_path, verbosity=True)


if __name__ == "__main__":
    test()


