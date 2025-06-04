import scanpy as sc
import anndata as ad
import mygene
import pandas as pd

def get_genes_symbols(adata, EnsIDs_column, new_column_name='gene_symbols_mapped'):
    """
    Map Ensembl gene IDs to gene symbols in the AnnData object.
    EnsIDs_column is the name of the column in adata.var that contains the Ensembl IDs.
    The names are stored in a new column called 'gene_symbols_mapped'.
    """
    mg = mygene.MyGeneInfo()
    ensembl_ids = adata.var[EnsIDs_column].tolist()
    query = mg.querymany(ensembl_ids, scopes="ensembl.gene", fields="symbol,name", species="human")

    id_to_symbol = {r['query']: r.get('symbol', r['query']) for r in query}

    adata.var[new_column_name] = adata.var[EnsIDs_column].map(id_to_symbol)


def check_sparsity(adata):
    """
    Check the sparsity of the data in the AnnData object.
    """
    print(f"Number of cells: {adata.shape[0]}")
    print(f"Number of genes: {adata.shape[1]}")
    print(f"Number of non-zero entries: {adata.X.nnz}")
    print(f"Sparsity: {1 - (adata.X.nnz / (adata.shape[0] * adata.shape[1])):.2%}")


def show_qc_plots(adata, violin_cols=None, scatter_x=None, scatter_y=None):
    """
    Show quality control plots for the AnnData object.
    Violin must be a list of strings and scatter_x and scatter_y must be strings.
    """
    sc.pp.calculate_qc_metrics(
    adata,inplace=True, log1p=True)
    sc.pl.violin(adata,violin_cols,jitter=0.4,multi_panel=True)
    sc.pl.scatter(adata, scatter_x, scatter_y)

def add_mutation_column(adata, df_mutation, cell_lines_column_name = "Sample_Name_cleaned", mutation_status_column="TP53_mutated", new_obs_column="mutation_status"): 
    """ 
    Aggiunge una colonna ad adata.obs che indica se la cellula appartiene a una cell line mutata.
    Droppa le cellule che non hanno un corrispondente in df_mutation.
    """
    adata.layers["pre_mutation_match"] = adata.X.copy()

    adata.obs["cell_line"]=adata.obs_names.str.split('_').str[0]
    mutation_dict = df_mutation.set_index(cell_lines_column_name)[mutation_status_column].to_dict()
    adata.obs[new_obs_column] = adata.obs["cell_line"].map(mutation_dict)

    not_found = adata.obs.loc[adata.obs[new_obs_column].isna(), "cell_line"].unique()
    print(f"Cell lines not found in df_mutation: {not_found}")
    
    initial_n = adata.n_obs
    adata._inplace_subset_obs(~adata.obs[new_obs_column].isna())
    print(f"Removed {initial_n - adata.n_obs} cells with unknown mutation status.")
    print(f"Number matching lines: {adata.n_obs}")
    print(f"Percentage of matching cell: {adata.n_obs / initial_n * 100:.2f}%")


def main():
    adata = ad.read_h5ad("data/Expression_Matrix_raw.h5ad")
    df_mutation = pd.read_csv("data/Mutation_status_cleaned_column.csv")
    get_genes_symbols(adata,EnsIDs_column="gene_symbols")
    add_mutation_column(adata, df_mutation, cell_lines_column_name = "Sample_Name_cleaned", mutation_status_column="TP53status", new_obs_column="mutation_status")
    #normalization and log transformation
    adata.layers["raw_counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata) 
    #combat for batch effects
    adata.layers["pre_harmony"] = adata.X.copy()
    sc.pp.pca(adata, n_comps=1000)
    sc.external.pp.harmony_integrate(adata, key="cell_line")
    #HVG
    adata.layers["pre_feature_selection"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, min_mean=0.1, max_mean=3, min_disp=0.5 )
    adata = adata[:, adata.var.highly_variable]
    final_df = ad.AnnData.to_df(adata)
    final_df["mutation_status"] = adata.obs["mutation_status"].values
    final_df.to_csv("final_preprocessed_data_combat.csv")

if __name__ == "__main__":
    main()