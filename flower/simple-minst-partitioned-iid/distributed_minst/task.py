from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
from  distributed_minst.minst_hdf5 import MNISTToHDF5
from  distributed_minst.minst_hdf5 import HDF5Dataset
from torch.utils.data import DataLoader

mnist_hdf5 = MNISTToHDF5()
mnist_hdf5.download_and_save()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)    # [28x28] -> [24x24]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                       # [24x24] -> [12x12]
        self.conv2 = nn.Conv2d(6, 16, 5)                                        # [12x12] -> [8x8] -> pool -> [4x4]
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes

    def forward(self, x):
        # Ensure input is [batch, 1, 28, 28]
        if x.ndim == 3:
            x = x.unsqueeze(1)  # [B, 28, 28] → [B, 1, 28, 28]
        elif x.shape[1] != 1:
            x = x.permute(0, 2, 1, 3)  # e.g., [B, 28, 1, 28] → [B, 1, 28, 28]

        x = self.pool(F.relu(self.conv1(x)))  # → [B, 6, 12, 12]
        x = self.pool(F.relu(self.conv2(x)))  # → [B, 16, 4, 4]
        x = x.view(-1, 16 * 4 * 4)            # → [B, 256]
        x = F.relu(self.fc1(x))               # → [B, 120]
        x = F.relu(self.fc2(x))               # → [B, 84]
        return self.fc3(x)                    # → [B, 10]

def load_data(partition_id: int, num_partitions: int = 20):
    # Use global partitions (created earlier with HDF5Dataset)
    print (f"[INFO] Loading data for partition {partition_id} out of {num_partitions} partitions...", flush=True)
    # Get dataset for this partition
    full_dataset = mnist_hdf5.get_dataset(partition_id=partition_id, transform=None)

    # Split: 80% train, 20% test
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    test_len = total_len - train_len

    train_ds, test_ds = random_split(full_dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=20)
    testloader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=20)
    print(f"[INFO] Loaded {len(train_ds)} training samples and {len(test_ds)} test samples for partition {partition_id}.", flush=True)
    return trainloader, testloader

def train(net, trainloader, epochs,device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    print(f"[INFO] Training for {epochs} epochs...", flush=True)
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_trainloss = running_loss / len(trainloader)
    train_accuracy = correct / total
    print(f"[INFO] Training complete. Average loss: {avg_trainloss:.4f}, Accuracy: {train_accuracy:.4f}", flush=True)
    return avg_trainloss, train_accuracy

def test(net, testloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch 
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
