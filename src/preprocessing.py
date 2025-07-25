"""
Preprocessing utilities for single-cell RNA-seq data.
"""
import scanpy as sc
import anndata as ad
import mygene
import pandas as pd
import numpy as np
from collections import defaultdict

def get_genes_symbols(adata, EnsIDs_column, new_column_name='gene_symbols_mapped'):
    """Map Ensembl IDs to gene symbols in AnnData."""
    mg = mygene.MyGeneInfo()
    ensembl_ids = adata.var[EnsIDs_column].tolist()
    query = mg.querymany(ensembl_ids, scopes="ensembl.gene", fields="symbol,name", species="human")
    id_to_symbol = {r['query']: r.get('symbol', r['query']) for r in query}
    adata.var[new_column_name] = adata.var[EnsIDs_column].map(id_to_symbol)

def check_sparsity(adata):
    """Print sparsity info for AnnData object."""
    print(f"Number of cells: {adata.shape[0]}")
    print(f"Number of genes: {adata.shape[1]}")
    print(f"Number of non-zero entries: {adata.X.nnz}")
    print(f"Sparsity: {1 - (adata.X.nnz / (adata.shape[0] * adata.shape[1])):.2%}")

def show_qc_plots(adata, violin_cols=None, scatter_x=None, scatter_y=None):
    """Show QC plots for AnnData object."""
    sc.pp.calculate_qc_metrics(
    adata,inplace=True, log1p=True)
    sc.pl.violin(adata,violin_cols,jitter=0.4,multi_panel=True)
    sc.pl.scatter(adata, scatter_x, scatter_y)

def add_mutation_column(adata, df_mutation, cell_lines_column_name = "Sample_Name_cleaned", mutation_status_column="TP53_mutated", new_obs_column="mutation_status"): 
    """Add mutation status column to AnnData.obs."""
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

def digitize(x, bins, side="both"):
    """Digitize values into bins (with random tie-breaking)."""
    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits
    right_digits = np.digitize(x, bins, right=True)
    rands = np.random.rand(len(x))
    digits = rands * (right_digits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits

def bin_data(x, n_bins):
    """Bin RNA-seq data at patient level."""
    binned_rows = []
    for row in x:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
        binned_rows.append(binned_row)
    binned_data = np.stack(binned_rows)
    return binned_data

def cdf_data(x):
    """Get CDF values for each row."""
    cdf_rows = []
    for row in x:
        inp_row = row[row != 0].copy()
        cdf_dict = {0.:0.}
        values, counts = np.unique(inp_row, return_counts=True)
        cdf = np.cumsum(counts) / len(inp_row)
        cdf_dict.update({values[i]:cdf[i] for i in range(len(values))})
        cdf_rows.append(np.vectorize(cdf_dict.get)(row))
    cdf_data = np.stack(cdf_rows)
    return cdf_data

def rank_data(x):
    """Get rank of each entry in each row (random for ties)."""
    rank_rows = []
    for row in x:
        inp_row = row[row != 0].copy()
        ranks = inp_row.argsort().argsort() + 1
        rank_dict = dict()
        for val in np.unique(inp_row):
            val_indices = np.where(inp_row == val)[0]
            val_ranks = ranks[val_indices]
            np.random.shuffle(val_ranks)
            rank_dict[val] = val_ranks.tolist()
        rank_row = np.zeros_like(row, dtype=float)
        n = len(inp_row)
        for i, val in enumerate(row):
            rank_row[i] = 0.
        rank_rows.append(rank_row)
    rank_data = np.stack(rank_rows)
    return rank_data

def avg_rank_data(x):
    """Get average rank of each entry in each row."""
    avg_rank_rows = []
    for row in x:
        inp_row = row[row != 0].copy()
        ranks_dict = defaultdict(list)
        ranks_dict[0.].append(0.)
        ranks = inp_row.argsort().argsort() + 1
        for i in range(len(inp_row)):
            ranks_dict[inp_row[i]].append(ranks[i])
        n = len(inp_row)
        avg_rank_dict = {rank:np.mean(ranks_dict[rank]) / n for rank in ranks_dict.keys()}
        avg_rank_rows.append(np.vectorize(avg_rank_dict.get)(row))
    avg_rank_data = np.stack(avg_rank_rows)
    return avg_rank_data

def main(feature_selection="HVG", batch_correction=None):  
    """Preprocess and save single-cell data for analysis."""
    target_genes=['ENSG00000144452', 'ENSG00000085563', 'ENSG00000107796', 'ENSG00000181026', 'ENSG00000081051', 'ENSG00000042286', 'ENSG00000011426', 'ENSG00000135046', 'ENSG00000169083', 'ENSG00000116017', 'ENSG00000162772', 'ENSG00000171791', 'ENSG00000121380', 'ENSG00000113916', 'ENSG00000168398', 'ENSG00000089685', 'ENSG00000176171', 'ENSG00000139618', 'ENSG00000159388', 'ENSG00000137752', 'ENSG00000003400', 'ENSG00000105974', 'ENSG00000134057', 'ENSG00000157456', 'ENSG00000026508', 'ENSG00000085117', 'ENSG00000117399', 'ENSG00000158402', 'ENSG00000170312', 'ENSG00000124762', 'ENSG00000147889', 'ENSG00000123975', 'ENSG00000116791', 'ENSG00000184371', 'ENSG00000101439', 'ENSG00000117984', 'ENSG00000006210', 'ENSG00000168824', 'ENSG00000196730', 'ENSG00000107984', 'ENSG00000136048', 'ENSG00000096696', 'ENSG00000120129', 'ENSG00000158050', 'ENSG00000120875', 'ENSG00000138166', 'ENSG00000139318', 'ENSG00000128951', 'ENSG00000165891', 'ENSG00000114346', 'ENSG00000127129', 'ENSG00000146648', 'ENSG00000122877', 'ENSG00000091831', 'ENSG00000181104', 'ENSG00000170345', 'ENSG00000116717', 'ENSG00000117228', 'ENSG00000130513', 'ENSG00000138271', 'ENSG00000233276', 'ENSG00000084207', 'ENSG00000113161', 'ENSG00000115738', 'ENSG00000137331', 'ENSG00000162783', 'ENSG00000163565', 'ENSG00000140443', 'ENSG00000146674', 'ENSG00000135709', 'ENSG00000049130', 'ENSG00000186847', 'ENSG00000170421', 'ENSG00000002834', 'ENSG00000131981', 'ENSG00000128342', 'ENSG00000134324', 'ENSG00000065833', 'ENSG00000105976', 'ENSG00000149573', 'ENSG00000182979', 'ENSG00000136997', 'ENSG00000116701', 'ENSG00000104419', 'ENSG00000117650', 'ENSG00000164867', 'ENSG00000136999', 'ENSG00000115758', 'ENSG00000228278', 'ENSG00000075891', 'ENSG00000138650', 'ENSG00000132646', 'ENSG00000112378', 'ENSG00000119630', 'ENSG00000121879', 'ENSG00000166851', 'ENSG00000145632', 'ENSG00000141682', 'ENSG00000170836', 'ENSG00000057657', 'ENSG00000007062', 'ENSG00000134222', 'ENSG00000073756', 'ENSG00000164611', 'ENSG00000103490', 'ENSG00000181467', 'ENSG00000133321', 'ENSG00000102760', 'ENSG00000115963', 'ENSG00000137393', 'ENSG00000196754', 'ENSG00000099194', 'ENSG00000012171', 'ENSG00000170099', 'ENSG00000206075', 'ENSG00000106366', 'ENSG00000118515', 'ENSG00000151012', 'ENSG00000112096', 'ENSG00000118007', 'ENSG00000196562', 'ENSG00000165025', 'ENSG00000059377', 'ENSG00000087510', 'ENSG00000160181', 'ENSG00000163235', 'ENSG00000198959', 'ENSG00000137462', 'ENSG00000187554', 'ENSG00000067369', 'ENSG00000115129', 'ENSG00000164938', 'ENSG00000132274', 'ENSG00000177169', 'ENSG00000038427', 'ENSG00000111424', 'ENSG00000090238', 'ENSG00000172667']
    adata = ad.read_h5ad("data/preprocessing/Expression_Matrix_raw.h5ad")
    df_mutation = pd.read_csv("data/preprocessing/Mutation_status_cleaned_column.csv")
    get_genes_symbols(adata,EnsIDs_column="gene_symbols")
    add_mutation_column(adata, df_mutation, cell_lines_column_name = "Sample_Name_cleaned", mutation_status_column="TP53status", new_obs_column="mutation_status")
    adata.layers["raw_counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata) 
    if feature_selection=="HVG":
        adata.layers["pre_feature_selection"] = adata.X.copy()
        sc.pp.highly_variable_genes(adata, min_mean=0.1, max_mean=3, min_disp=0.5 )
        adata = adata[:, adata.var.highly_variable]
    elif feature_selection=="target":
        adata.layers["pre_feature_selection"] = adata.X.copy()
        mask = adata.var_names.isin(target_genes)
        adata = adata[:, mask]
    else:
        raise KeyError("feature_selection can only be values from this list [\"HVG\" , \"target\"] ")
    if batch_correction=="harmony":
        adata.layers["pre_harmony"] = adata.X.copy()
        sc.pp.pca(adata, n_comps=1000)
        sc.external.pp.harmony_integrate(adata, key="cell_line")
    elif batch_correction=="combat":
        adata.layers["pre_combat"] = adata.X.copy()  
        sc.pp.combat(adata, key="cell_line")   
    elif batch_correction == None:
        pass
    else:
        raise KeyError("batch_correction can only be values from this list [None , \"combat\" , \"harmony\" ] ")          
    final_df = ad.AnnData.to_df(adata)
    final_df["mutation_status"] = adata.obs["mutation_status"].values
    suffix = f"{batch_correction}" if batch_correction else ""
    final_df.to_csv(f"final_preprocessed_data_{feature_selection}_{suffix}.csv")

if __name__ == "__main__":
    main(feature_selection="target", batch_correction="combat")