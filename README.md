# Predicting TP53 Mutation Status from Single-Cell RNA-seq with Graph Neural Networks

## Overview
This project explores the application of **graph neural networks (GNNs)** to predict the mutational status of the TP53 gene from **single-cell RNA sequencing (scRNA-seq)** data. The pipeline leverages advanced machine learning and network analysis techniques to address a key problem in cancer genomics: identifying TP53 mutations, which are highly relevant for cancer diagnostics and research.

## Motivation
TP53, known as the "guardian of the genome," is one of the most frequently mutated genes in human cancers. Accurate detection of its mutation status is crucial for understanding tumor biology and guiding therapeutic decisions. This project demonstrates how modern deep learning—specifically, GNNs—can be used to extract biologically meaningful patterns from high-dimensional, noisy single-cell data.

## Key Features & Technologies
- **Graph Neural Networks:** Implementation of Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) for graph classification tasks.
- **Gene Co-expression Networks:** Construction of gene-gene graphs from scRNA-seq data using Spearman correlation.
- **Feature Selection:** Comparison of statistically driven (Highly Variable Genes, HVG) and biologically driven (TP53 target genes) feature selection strategies.
- **Batch Correction & Regularization:** Integration of ComBat and Harmony for batch effect correction; use of GraphNorm and L2 regularization.
- **Hyperparameter Optimization:** Automated tuning with Optuna.
- **Baseline Models:** XGBoost classifiers for benchmarking.
- **Core Libraries:** PyTorch, PyTorch Geometric, Scanpy, scikit-learn, XGBoost, Optuna, pandas, numpy, matplotlib, seaborn.

## Data Sources
- **Single-cell RNA-seq:** [Single Cell Breast Cancer Cell-line Atlas (Gambardella, 2022)](https://doi.org/10.6084/m9.figshare.15022698.v2)
- **TP53 Mutation Status:** [The TP53 Database](https://tp53.cancer.gov)
- **Bulk RNA-seq:** [Cancer Cell Line Encyclopedia (CCLE), DepMap 22q2](https://depmap.org/portal/download/)
- **TP53 Target Genes:** [Fischer’s curated list of p53 targets](https://tp53.cancer.gov/target_genes)

**Note:** Data is not included in this repository. Please download the datasets from the above sources and follow the instructions below for preprocessing.

## Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/scRNAseq-GNN-binary-tp53.git
cd scRNAseq-GNN-binary-tp53
```

### 2. Set Up the Environment
It is recommended to use a virtual environment (e.g., `venv` or `conda`). Install dependencies:
```bash
pip install -r requirements.txt
```
*If `requirements.txt` is missing, install the following main packages:*
- torch, torch-geometric, scanpy, anndata, xgboost, optuna, scikit-learn, pandas, numpy, matplotlib, seaborn, mygene

### 3. Download and Prepare Data
- Download the scRNA-seq, mutation, and bulk RNA-seq data from the sources above.
- Place the raw data in the `data/` directory, following the structure expected by the scripts (see comments in `src/load_data.py` and `src/preprocessing.py`).
- Update file paths in the notebook or scripts if needed.

### 4. Run the Pipeline
The main workflow is provided in the Jupyter notebook:
```bash
jupyter notebook notebooks/main_experiment.ipynb
```
Follow the notebook cells to preprocess data, construct graphs, train models, and evaluate results.

Alternatively, you can use the provided shell scripts in the `jobs/` directory to run specific stages (preprocessing, model training, etc.).

## Results
- **XGBoost Baseline:** F1 score up to 0.995 (single-cell), 0.88 (bulk)
- **Best GNN Model:** GAT with ComBat batch correction on TP53 target genes, F1 score = 0.998
- **Interpretability:** Graph-based models capture gene–gene interactions, offering improved generalization and biological insight compared to classical ML approaches.

## Author & Contributions
This project was designed and implemented as a bachelor's thesis. All stages—from data preprocessing and graph construction to model development, hyperparameter optimization, and result analysis—were developed by the author.

## References
- Gambardella, G. et al. (2022). Single Cell Breast Cancer Cell-line Atlas. [Figshare](https://doi.org/10.6084/m9.figshare.15022698.v2)
- The TP53 Database: https://tp53.cancer.gov
- DepMap, Broad Institute. [CCLE](https://depmap.org/portal/download/)
- Fischer, M. (2017). [Curated list of p53 targets](https://tp53.cancer.gov/target_genes)

---
*For questions or collaboration opportunities, feel free to contact me.*
