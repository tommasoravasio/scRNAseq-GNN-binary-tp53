import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class GCN(torch.nn.Module):
    """Graph Convolutional Network for graph-level classification with Dropout and BatchNorm."""
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.classifier = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x

def train(model, loader, optimizer, criterion, device):
    """Train the model on the given data loader."""
    
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device, criterion=None, return_cm=False):
    """Evaluate the model on the given data loader."""
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            if criterion is not None:
                loss = criterion(out, batch.y)
                total_loss += loss.item()
    acc = accuracy_score(y_true, y_pred)
    if return_cm:
        cm = confusion_matrix(y_true, y_pred)
        return acc, cm
    else:
        if criterion is not None:
            return acc, total_loss / len(loader)
        else:
            return acc

def plot_confusion_matrix(cm, class_names):
    """Plot the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def plot_losses(train_losses, val_losses):
    """Plot training and validation losses."""

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_gnn(train_path='train_reteunica.pt', test_path='test_reteunica.pt', 
              input_dim=1, hidden_dim=64, output_dim=2, epochs=10, batch_size=32, lr=0.001,
              model_path='gnn_model.pt'):
    """
    Train a GNN model on the provided training and testing datasets """

    # Load data
    train_graphs = torch.load(train_path)
    test_graphs = torch.load(test_path)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(input_dim, hidden_dim, output_dim).to(device)

    # Calculate balanced class weights
    train_labels = [g.y.item() for g in train_graphs]
    class_counts = Counter(train_labels)
    

    num_samples = sum(class_counts.values())
    num_classes = 2
    weights = [num_samples / (num_classes * class_counts[c]) for c in [0, 1]]
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    # NEW: track loss
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_acc, val_loss = evaluate(model, test_loader, device, criterion=criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # NEW: plot losses
    plot_losses(train_losses, val_losses)

    # Final evaluation with confusion matrix
    acc, cm = evaluate(model, test_loader, device, return_cm=True)
    print(f"Final Test Accuracy: {acc:.4f}")
    plot_confusion_matrix(cm, class_names=['wt', 'mut'])