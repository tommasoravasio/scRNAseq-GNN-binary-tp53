# network_job.py

import pandas as pd
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import network_constructor

df = pd.read_csv("final_preprocessed_data.csv", index_col=0)

train_df, test_df = network_constructor.train_test_split(df, test_size=0.2, random_state=42)
print(f"shape del train set: {train_df.shape} \nshape del test set: {test_df.shape}")


mat = network_constructor.build_correlation_matrix(train_df.iloc[:, :-1], corr_threshold=0.05)

train_df_pyg = network_constructor.create_PyG_graph_from_df(train_df, mat, label_column="mutation_status")
test_df_pyg = network_constructor.create_PyG_graph_from_df(test_df, mat, label_column="mutation_status")


# torch.save(train_df_pyg, "train_df_pyg.pt")
# torch.save(test_df_pyg, "test_df_pyg.pt")
