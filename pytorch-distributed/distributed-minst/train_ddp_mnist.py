import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torchvision.models as models
import os

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '12355'

class MNISTResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
    def forward(self, x):
        return self.model(x)

def load_digit_shard(digit):
    # Each process loads only its digit's data from its own shard file
    data = np.load(f'mnist_shards/mnist_{digit}.npz')
    X = torch.tensor(data['data'], dtype=torch.float32).unsqueeze(1) / 255.0
    y = torch.tensor(data['targets'], dtype=torch.long)
    return TensorDataset(X, y)

def train(rank, world_size):
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    print(f'Train on {rank} cpu')
    dataset = load_digit_shard(rank)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = MNISTResNet18()
    model = nn.parallel.DistributedDataParallel(model)
    device = torch.device('cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(3):
        loadercnt = 0
        for X, y in loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loadercnt += 1
            if loadercnt > 10:
                break
        if rank == 0:
            print(f"Digit {rank} Epoch {epoch+1} complete. Loss: {loss.item():.4f}")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 10
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
