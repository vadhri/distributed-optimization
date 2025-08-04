from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Simple quadratic regression model: f(x) = a * x^2
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))  # quadratic term
        self.b = nn.Parameter(torch.randn(1))  # linear term
        self.c = nn.Parameter(torch.randn(1))  # constant term

    def forward(self, x):
        return self.a * x**2 + self.b * x + self.c

fds = None  # Cache FederatedDataset

# Generate synthetic data for f(x) = (partition+1)*x^2
def load_data(partition_id: int, num_partitions: int):
    np.random.seed(42)
    torch.manual_seed(42)
    print(f"Loading data for partition {partition_id} of {num_partitions}...", flush=True)
    n_samples = 200
    x = np.random.uniform((partition_id+1)*-2, (partition_id+2)*2, size=(n_samples, 1)).astype(np.float32)
    x = (x - x.mean()) / x.std()

    a = float(partition_id + 1)
    y = x ** 2

    # Split 80/20
    split = int(0.8 * n_samples)
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    trainloader = DataLoader(train_ds, batch_size=32, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=32)
    return trainloader, valloader

def train(net, trainloader, epochs, device):
    net.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    print("Training started...")
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

def test(net, testloader, device):
    net.to(device)
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            y_pred = net(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    # For regression, accuracy is not meaningful, so return R^2 score as 'accuracy'
    # Compute R^2
    y_true = []
    y_pred_all = []
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            y_pred = net(x)
            y_true.append(y.cpu())
            y_pred_all.append(y_pred.cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred_all = torch.cat(y_pred_all, dim=0)
    ss_res = ((y_true - y_pred_all) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return avg_loss, float(r2)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
