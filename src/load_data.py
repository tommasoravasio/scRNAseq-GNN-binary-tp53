import scanpy as sc
import pandas as pd
import anndata as ad
import os


def one_to_three_columns_features_file(file_path):
    """
    scanpy.read_10x_mtx requires the features.tsv file to have 3 columns. 
    This function creates two new useless columns to ensure the function does not return an error.
    """
    features = pd.read_csv(file_path, header=None, sep="\t")

    assert features.shape[1] == 1

    features[1] = features[0] 
    features[2] = "Gene Expression"

    features.to_csv(file_path, header=False, index=False, sep="\t")


def load_expression_data(file_path, verbosity=False): 
    """
    Loads expression data from a 10X Genomics file into an AnnData object and returns a pandas DataFrame.
    The expected format is a folder containing 3 files: matrix.mtx, barcodes.tsv, and features.tsv.
    IMPORTANT: THE FILES MUST BE COMPRESSED WITH GZIP, OTHERWISE scanpy.read_10x_mtx() WILL NOT WORK.
    """
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

    return df_expression


def load_mutation_data(file_path, verbosity=False):
    """
    Carica i dati delle mutazioni da un file CSV.
    """
    df_mutation = pd.read_csv(file_path, sep=",", index_col=0)
    df_mutation['Sample_Name_cleaned'] = df_mutation['Sample_Name'].str.replace('-', '', regex=False)
    
    if verbosity:
        print(f"df_mutation shape: {df_mutation.shape}")
        print(f"df_mutation columns: {df_mutation.columns}")
        print(f"df_mutation head: {df_mutation.head()}")

    return df_mutation

def add_cleaned_column(df, column_name="Sample_Name"):
    "remove the hypens from the cell lines identification codes"
    df[f"{column_name}_cleaned"] = df[column_name].str.replace('-', '', regex=False)
    return df

def check_on_cell_lines_correspondence(df_expression, df_mutation, mutation_column_name="Sample_Name_cleaned"):
    """
    Check how many cells in the expression data has their rispective cell lines in the mutation data.
    The cell lines are obtained by the barcode of the expression data.
    mutation_column_name is the name of the column in the mutation dataframe that contains the cell lines.
    ATTENTION: this returns the number of observations from expression data with the corresponding cell line 
    in the mutation data, not the number of cell lines that can be found in both dataframes.
    """

    if mutation_column_name not in df_mutation.columns:
        raise KeyError(f"The column '{mutation_column_name}' does not exist in the mutation DataFrame.")

    df_cell_lines = pd.DataFrame({"Cell Lines": df_expression.index.str.split('_').str[0]})

    matching_cell_lines = df_cell_lines[df_cell_lines["Cell Lines"].isin(df_mutation[mutation_column_name])]

    print(f"Number matching lines: {len(matching_cell_lines)}")

    print(f"Percentage of matching cell: {len(matching_cell_lines)/len(df_cell_lines)*100:.2f}%")


def test():
    """
    Test the functions in this module.
    """
    expression_data_path = "data/Expression_Matrix"
    df_expression = load_expression_data(expression_data_path, verbosity=True)

    
    mutation_data_path = "data/Mutation/CellLineDownload_r21.csv"
    df_mutation = load_mutation_data(mutation_data_path, verbosity=True)



    
    check_on_cell_lines_correspondence(df_expression, df_mutation)


if __name__ == "__main__":
    test()


