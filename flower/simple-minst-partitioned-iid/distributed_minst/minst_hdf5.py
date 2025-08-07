import os
import h5py
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader

class MNISTToHDF5:
    def __init__(self, output_dir="mnist_partitions", download=True):
        self.output_dir = output_dir
        self.download = download
        self.transform = transforms.ToTensor()
        os.makedirs(self.output_dir, exist_ok=True)

    def download_and_save(self):
        # Check if all 20 partition files already exist
        expected_files = [
            os.path.join(self.output_dir, f"mnist_partition_{i}.h5") for i in range(20)
        ]
        if all(os.path.exists(f) for f in expected_files):
            print(f"[INFO] All partition files already exist in {self.output_dir}. Skipping download and save.")
            return

        print("[INFO] Downloading MNIST and preparing partitions...")
        # Download MNIST dataset
        train_ds = datasets.MNIST(root="./data", train=True, download=self.download, transform=self.transform)
        test_ds = datasets.MNIST(root="./data", train=False, download=self.download, transform=self.transform)

        # Combine and normalize
        all_images = torch.cat([train_ds.data, test_ds.data], dim=0).unsqueeze(1).float() / 255.0
        all_labels = torch.cat([train_ds.targets, test_ds.targets], dim=0)

        # Shuffle the full dataset
        num_samples = len(all_labels)
        perm = torch.randperm(num_samples)
        all_images = all_images[perm]
        all_labels = all_labels[perm]

        # Split into 20 partitions
        num_partitions = 20
        partition_size = num_samples // num_partitions

        for pid in range(num_partitions):
            start = pid * partition_size
            end = (pid + 1) * partition_size if pid < num_partitions - 1 else num_samples

            imgs = all_images[start:end].numpy()
            labels = all_labels[start:end].numpy()

            fname = os.path.join(self.output_dir, f"mnist_partition_{pid}.h5")
            with h5py.File(fname, "w") as f:
                f.create_dataset("images", data=imgs, compression="gzip")
                f.create_dataset("labels", data=labels, compression="gzip")

            print(f"[INFO] Saved partition {pid} to {fname}")

    def get_dataset(self, partition_id, transform=None):
        file_path = os.path.join(self.output_dir, f"mnist_partition_{partition_id}.h5")
        return HDF5Dataset(file_path, transform=transform or self.transform)

    def download_full_train_test_dataloaders(self, batch_size=32, shuffle=True):
        print("[INFO] Downloading full MNIST train and test datasets...")

        train_raw = datasets.MNIST(root="./data", train=True, download=self.download, transform=self.transform)
        test_raw = datasets.MNIST(root="./data", train=False, download=self.download, transform=self.transform)

        # Normalize manually: [0,255] -> [0.0,1.0]
        train_imgs = train_raw.data.unsqueeze(1).float() / 255.0  # shape: [N, 1, 28, 28]
        test_imgs = test_raw.data.unsqueeze(1).float() / 255.0

        train_labels = train_raw.targets
        test_labels = test_raw.targets

        # Wrap into TensorDatasets and DataLoaders
        train_dataset = TensorDataset(train_imgs, train_labels)
        test_dataset = TensorDataset(test_imgs, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print("[INFO] Returned full MNIST DataLoaders.")
        return train_loader, test_loader
    
class HDF5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.file = None

        # Read only metadata
        with h5py.File(h5_path, "r") as f:
            self.length = len(f["labels"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        image = self.file["images"][idx]
        label = int(self.file["labels"][idx])

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, label

    def __del__(self):
        if self.file:
            self.file.close()
