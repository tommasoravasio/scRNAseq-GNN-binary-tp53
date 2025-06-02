import numpy as np
import pandas as pd
import torch
import os
import csv
import json
from datetime import datetime
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Module, CrossEntropyLoss
from torch_geometric.loader import DataLoader
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


class GCN(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super(GCN,self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1=torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2=torch.nn.BatchNorm1d(out_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout( p=dropout_rate )

    def forward(self, x, edge_index,batch ):
        x=self.conv1(x, edge_index)
        x=self.bn1(x)
        x=F.relu(x)
        x=self.dropout(x)

        x=self.conv2(x,edge_index)
        x=self.bn2(x)
        x=F.relu(x)
        x=self.dropout(x)

        x= global_mean_pool(x, batch)
        x= self.lin(x)
        return x


def train(model,train_loader,optimizer,criterion,device):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    

def evaluate(model,loader,device,criterion,confusion_matrix=False):
        model.eval()
        y_true = []
        y_pred = []
        loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                loss += criterion(out, batch.y).item()
        acc = accuracy_score(y_true, y_pred)
        if confusion_matrix:
            mat = confusion_matrix(y_true, y_pred)
            return acc, loss / len(loader), mat
        else:
            return acc, loss / len(loader)


def train_model(train_PyG, test_PyG,batch_size=32, hidden_channels=64, dropout_rate=0.5,lr= 0.001,
                epochs=30):
    
    train_loader = DataLoader(train_PyG, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_PyG, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_channels=train_PyG[0].x.shape[1], hidden_channels=hidden_channels, out_channels=2,dropout_rate=dropout_rate).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    labels=torch.cat([data.y for data in train_PyG])
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    criterion = CrossEntropyLoss(weight=class_weights.to(device)).to(device)
    os.makedirs("results",exist_ok=True)
    
    log_path = "training_log.csv"
    with open(log_path,mode="w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Train Accuracy", "Test Accuracy"])

        for epoch in range(1,epochs+1):
            loss = train(model, train_loader, optimizer, criterion, device)
            train_acc = evaluate(model,train_loader,device, criterion,confusion_matrix=False)
            test_acc = evaluate(model,test_loader,device,criterion,confusion_matrix=False)
            print(f"Epoch: {epoch} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            writer.writerow([epoch, loss, train_acc, test_acc])
    
    model_path = "gcn_model.pt"
    torch.save(model.state_dict(),model_path)

    accuracy,avg_loss,mat = evaluate(model, test_loader, device, criterion, confusion_matrix=True)
    np.savetxt("results/confusion_matrix.csv", mat, delimiter=",", fmt="%d")
    summary_metrics = {
        "final_accuracy": accuracy,
        "final_loss": avg_loss,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open("results/summary_metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=4)

    return model



def load_graphs(path):
    graph_list = []
    for pt_file in sorted(Path(path).glob("*.pt")):
        graph = torch.load(pt_file, weights_only=False)

        if isinstance(graph, list):
            graph_list.extend(graph)
        else:
            graph_list.append(graph)

    return graph_list


def main():
    # IMPORTA GRAFI COME train_df_pyg test_df_pyg
    train_df_pyg = load_graphs("data/graphs/train")
    test_df_pyg = load_graphs("data/graphs/test")

    model = train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = 30, batch_size = 32)

if __name__ == "__main__":
    main()















