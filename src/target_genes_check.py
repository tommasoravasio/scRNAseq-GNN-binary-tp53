import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import network_constructor         


def plot_frequency_of_correlation_values(matrices, bins=50, alpha=0.5):
    """
    Plot the frequency of correlation values.
    """
    plt.figure(figsize=(10, 6))
    bin_edges = np.linspace(-1, 1, bins + 1)

    for label,mat in matrices.items():
        mat_copia = mat.copy()
        np.fill_diagonal(mat_copia, np.nan) 
        values = mat_copia.flatten()
        plt.hist(values, bins=bin_edges, alpha=alpha, label=label, edgecolor='black') #density=True
        mean_val = np.mean(values)
        plt.axvline(mean_val, linestyle='--', linewidth=2)
    plt.yscale('log')
    plt.legend()
    plt.title('Hist of Correlation Values')
    plt.xlabel('Correlation Value')
    plt.ylabel('Absolute frequence')
    plt.grid(True)
    plt.show()


def import_and_create_matrices_for_plotting(path,mat_name,col_name="Ensembl ID",verbosity=False):
    "Expect to have a column named 'Ensembl ID' in the excel file, can specify the name of the columns otherwise with col_name"
    tab = pd.read_excel(path)
    tab_ensembl_ids = [gene for gene in tab[col_name] if gene in df.columns]
    df_tab = df[tab_ensembl_ids].copy()
    mat_tab=network_constructor.build_correlation_matrix(df_tab, corr_threshold=0., p_value_threshold= 1)
    if verbosity==True:
        df_tab.head()
        network_constructor.check_percentage_of_zeros(mat_tab)
    return mat_tab


def plot_correlation_comparison(df_expression,path_list):
    if not isinstance(path_list, list):
        path_list = [path_list]

    
    mat=network_constructor.build_correlation_matrix(df_expression, corr_threshold=0., p_value_threshold= 1)
    
    plot_frequency_of_correlation_values({"Original":mat}, bins=50, alpha=0.5)
    plt.close("all")
    del mat
    gc.collect()
    
    for path in path_list:
        label= path.split("/")[-1].split(".")[0]
        mat = import_and_create_matrices_for_plotting(path, mat_name=label, col_name="Ensembl ID", verbosity=False)
        plot_frequency_of_correlation_values({label:mat}, bins=50, alpha=0.5)
        del mat
        gc.collect()

def plot_mean_expression_by_gene(df_mut, df_wt):
    df_mut_genes = df_mut.drop(columns=["mutation_type"]).copy()
    df_wt_genes = df_wt.drop(columns=["mutation_type"]).copy()
    means_mut = df_mut_genes.mean(axis=0)
    means_wt = df_wt_genes.mean(axis=0)
    total_means = means_mut + means_wt
    sorted_genes = total_means.sort_values(ascending=False).index
    means_mut = means_mut[sorted_genes]
    means_wt = means_wt[sorted_genes]

    x = range(len(sorted_genes))
    plt.figure(figsize=(12, 6))
    plt.bar(x, means_wt, label="WT", alpha=0.6)
    plt.bar(x, means_mut, bottom=means_wt, label="MUT", alpha=0.6)
    plt.xticks(ticks=x, labels=sorted_genes, rotation=90)
    plt.ylabel("Mean Expression per Gene")
    plt.title("Mean Gene Expression (normalized by group size)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()


