import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Module, CrossEntropyLoss
from torch_geometric.loader import DataLoader




class GCN(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super(GCN,self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index,batch ):
        x=self.conv1(x, edge_index)
        x=F.relu(x)
        x=self.conv2(x,edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        #x = self.lin(x)
        return x


def train_model(train_PyG, test_PyG,batch_size=32, hidden_channels=64, dropout_rate=0.5,lr= 0.01,
                epochs=100):
    train_loader = DataLoader(train_PyG, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_PyG, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_channels=train_PyG[0].x.shape[1], hidden_channels=hidden_channels, out_channels=2,dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss().to(device)
    #########################

    def train(model,train_loader):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def test(model, test_loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
        return correct / len(test_loader.dataset)
    
    for epoch in range(1,epochs+1):
        loss = train(model, train_loader)
        train_acc = test(model,train_loader)
        test_acc = test(model,test_loader)
        print(f"Epoch: {epoch} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    return model


    


















