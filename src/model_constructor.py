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
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import GATConv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna
import argparse


class GCN(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate, use_graphnorm=False):
        super(GCN,self).__init__()
        self.use_graphnorm = use_graphnorm

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1=GraphNorm(hidden_channels) if use_graphnorm else torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2=GraphNorm(hidden_channels) if use_graphnorm else torch.nn.BatchNorm1d(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout( p=dropout_rate )
        

    def forward(self, x, edge_index,batch ):
        x=self.conv1(x, edge_index)
        x=self.bn1(x,batch) if self.use_graphnorm else self.bn1(x)
        x=F.relu(x)
        x=self.dropout(x)

        x=self.conv2(x,edge_index)
        x=self.bn2(x,batch) if self.use_graphnorm else self.bn2(x)
        x=F.relu(x)
        x=self.dropout(x)

        x= global_mean_pool(x, batch)
        x= self.lin(x)
        return x





class GAT(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate, use_graphnorm=False, heads=1,use_third_layer=False):
        super(GAT,self).__init__()
        self.use_graphnorm = use_graphnorm
        self.heads = heads
        self.use_third_layer = use_third_layer

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.bn1 = GraphNorm(hidden_channels * heads) if use_graphnorm else torch.nn.BatchNorm1d(hidden_channels * heads)

        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.bn2 = GraphNorm(hidden_channels * heads) if use_graphnorm else torch.nn.BatchNorm1d(hidden_channels * heads)

        if use_third_layer:
            self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
            self.bn3 = GraphNorm(hidden_channels * heads) if use_graphnorm else torch.nn.BatchNorm1d(hidden_channels * heads)

        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x, batch) if self.use_graphnorm else self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x, batch) if self.use_graphnorm else self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        if self.use_third_layer:
            x = self.conv3(x, edge_index)
            x = self.bn3(x, batch) if self.use_graphnorm else self.bn3(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
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
    

def evaluate(model,loader,device,criterion,compute_confusion_matrix=False):
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
        if compute_confusion_matrix:
            mat = confusion_matrix(y_true, y_pred)
            return acc, loss / len(loader), mat
        else:
            return acc, loss / len(loader)


def train_model(train_PyG, test_PyG, batch_size=32, hidden_channels=64, dropout_rate=0.2,lr= 0.0001,
                epochs=30, ID_model="baseline",loss_weight=False, use_graphnorm=False, use_adamW=False, 
                weight_decay=1e-4, model_type="gcn",heads=1, use_third_layer=False, feature_selection="HVG", early_stopping=False ):
    
    train_loader = DataLoader(train_PyG, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_PyG, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == "gcn":
        model = GCN(in_channels=train_PyG[0].x.shape[1], hidden_channels=hidden_channels, out_channels=2,dropout_rate=dropout_rate, use_graphnorm=use_graphnorm).to(device)
    elif model_type=="gat":
        model = GAT(in_channels=train_PyG[0].x.shape[1], hidden_channels=hidden_channels, out_channels=2,dropout_rate=dropout_rate, use_graphnorm=use_graphnorm, heads=heads, use_third_layer=use_third_layer).to(device)
    else:
        raise KeyError("model_type does not exist, has to be either \"gcn\" (default value) or \"gat\" ")
    
    if use_adamW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    labels=torch.cat([data.y for data in train_PyG])
    class_counts = torch.bincount(labels,minlength=2)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    if loss_weight:
        criterion = CrossEntropyLoss(weight=class_weights.to(device)).to(device)
    else:
        criterion = CrossEntropyLoss().to(device)
    
    results_dir = f"{feature_selection}/{model_type}_results/{ID_model}"
    os.makedirs(results_dir, exist_ok=True)
    log_path = f"{results_dir}/training_log.csv"
    with open(log_path,mode="w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Train Accuracy", "Train F1", "Test Accuracy", "Test F1", "Test Loss"])
        
        #Inizializza variabili per early stopping
        patience = 10  
        best_f1 = 0
        epochs_no_improve = 0
        best_model_state = None


        for epoch in range(1,epochs+1):
            loss = train(model, train_loader, optimizer, criterion, device)
            train_acc , train_loss = evaluate(model,train_loader,device, criterion,compute_confusion_matrix=False)
            test_acc , test_loss= evaluate(model,test_loader,device,criterion,compute_confusion_matrix=False)
            # Calcola F1 train
            model.eval()
            y_true_train, y_pred_train = [], []
            y_true_test, y_pred_test = [], []

            with torch.no_grad():
                for batch in train_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    preds = out.argmax(dim=1).cpu().numpy()
                    y_pred_train.extend(preds)
                    y_true_train.extend(batch.y.cpu().numpy())

                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    preds = out.argmax(dim=1).cpu().numpy()
                    y_pred_test.extend(preds)
                    y_true_test.extend(batch.y.cpu().numpy())

            train_f1 = f1_score(y_true_train, y_pred_train, zero_division=0)
            test_f1 = f1_score(y_true_test, y_pred_test, zero_division=0)
            print(f"Epoch: {epoch} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | Test Loss: {test_loss:.4f}")
            writer.writerow([epoch, loss, train_acc, train_f1, test_acc, test_f1, test_loss])


            if test_f1 > best_f1:
                best_f1 = test_f1
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if early_stopping and epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model_path = f"{results_dir}/{model_type}_model.pt"
    torch.save(model.state_dict(), model_path)
    accuracy,avg_loss,mat = evaluate(model, test_loader, device, criterion, compute_confusion_matrix=True )
    np.savetxt(f"{results_dir}/confusion_matrix.csv", mat, delimiter=",", fmt="%d")

    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index,batch.batch)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy() 
            preds = out.argmax(dim=1).cpu().numpy()
            y_prob.extend(probs if isinstance(probs, np.ndarray) else probs.cpu().numpy())
            y_pred.extend(preds)
            y_true.extend(batch.y.cpu().numpy())
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true,y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc=  roc_auc_score(y_true,y_prob)
    except ValueError:
        auc = None

    summary_metrics = {
        "final_accuracy": accuracy,
        "final_loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "number_of_epochs": epochs,
        "hidden_channels":hidden_channels,
        "dropout_rate":dropout_rate,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "heads": heads,
        "use_graphnorm": use_graphnorm,
        "use_third_layer": use_third_layer,
        "best_epoch": epoch - epochs_no_improve,
        "best_f1": best_f1,
        "early_stopping": early_stopping,
        "ID_model": ID_model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(f"{results_dir}/summary_metrics.json", "w") as f:
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



def main_optuna():

    epochs=100
    batch_size=16
    feature_selection="target"
    graphs_path = "graphs_baseline"
    
    def objective(trial):
        hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        heads = trial.suggest_categorical("heads", [1, 2, 4, 8, 16])
        loss_weight = trial.suggest_categorical("loss_weight", [True, False])
        use_third_layer = trial.suggest_categorical("use_third_layer", [True, False])

        train_df_pyg = load_graphs(f"data/{graphs_path}/train")    
        test_df_pyg = load_graphs(f"data/{graphs_path}/test")

        model = train_model(
            train_PyG=train_df_pyg,
            test_PyG=test_df_pyg,
            hidden_channels=hidden_channels,
            dropout_rate=dropout_rate,
            lr=lr,
            use_adamW=True,
            weight_decay=weight_decay,
            loss_weight= loss_weight,
            epochs=epochs,
            batch_size=batch_size,
            ID_model=f"optuna_{trial.number}",
            model_type = "gat",
            heads=heads,
            use_graphnorm=True,
            use_third_layer = use_third_layer,
            feature_selection=feature_selection,
            early_stopping=False
        )

        with open(f"{feature_selection}/gat_results/optuna_{trial.number}/summary_metrics.json") as f:
            metrics = json.load(f)

        return metrics["f1_score"]
        

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial
    print(f"F1: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")


def main_baseline():
    epochs = 50
    batch_size = 16
    ID_model = "GraphNorm_combat"
    use_adamW=True
    model_type="gat"
    use_graphnorm=True
    feature_selection="target"
    graphs_path = "graphs_target_combat"

    train_df_pyg = load_graphs(f"data/{graphs_path}/train")
    test_df_pyg = load_graphs(f"data/{graphs_path}/test")
    model = train_model(train_PyG=train_df_pyg, test_PyG=test_df_pyg, epochs = epochs, batch_size = batch_size, ID_model = ID_model, use_adamW=use_adamW, model_type=model_type, use_graphnorm=use_graphnorm, feature_selection=feature_selection, early_stopping=False)

# LOCAL TESTING
def test_run_baseline():
    train_df_pyg_big = load_graphs("data/graphs_baseline/train")
    test_df_pyg_big = load_graphs("data/graphs_baseline/test")
    train_df_pyg_small = train_df_pyg_big[:5]
    test_df_pyg_small = test_df_pyg_big[:5]
    
    model = train_model(
        train_PyG=train_df_pyg_small,
        test_PyG=test_df_pyg_small,
        hidden_channels=32,
        dropout_rate=0.3,
        lr=0.001,
        use_adamW=True,
        weight_decay=1e-4,
        loss_weight=False,
        epochs=2,
        batch_size=2,
        ID_model="test_run",
        model_type="gat",
        heads=2,
        use_graphnorm=True,
        use_third_layer=False,
        feature_selection="testing"
    )
    print("Test run completed successfully.")

# LOCAL TESTING
def main_optuna_test():
    train_df_pyg_big = load_graphs("data/graphs_baseline/train")
    test_df_pyg_big = load_graphs("data/graphs_baseline/test")
    train_df_pyg_small = train_df_pyg_big[:5]
    test_df_pyg_small = test_df_pyg_big[:5]

    def objective_test(trial):
        hidden_channels = trial.suggest_categorical("hidden_channels", [32])
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        heads = trial.suggest_categorical("heads", [2])
        loss_weight = trial.suggest_categorical("loss_weight", [False])
        use_third_layer = trial.suggest_categorical("use_third_layer", [False])

        model = train_model(
            train_PyG=train_df_pyg_small,
            test_PyG=test_df_pyg_small,
            hidden_channels=hidden_channels,
            dropout_rate=dropout_rate,
            lr=lr,
            use_adamW=True,
            weight_decay=weight_decay,
            loss_weight=loss_weight,
            epochs=2,
            batch_size=2,
            ID_model=f"optuna_test_{trial.number}",
            model_type="gat",
            heads=heads,
            use_graphnorm=True,
            use_third_layer=use_third_layer,
            feature_selection="testing"
        )

        with open(f"testing/gat_results/optuna_test_{trial.number}/summary_metrics.json") as f:
            metrics = json.load(f)

        return metrics["f1_score"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_test, n_trials=1)

    print("Optuna test run completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_tuning", choices=["yes","no"],default="no")
    args=parser.parse_args()

    if args.fine_tuning == "yes":
        main_optuna()
    else:
        main_baseline()

    # #FOR TESTING
    # test_run_baseline()
    # main_optuna_test()















