import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.ports.data_loader import BaseDataLoader


class CIFAR100Dataset(Dataset):
    def __init__(self, file_path: Path, transform=None):
        if not file_path.exists():
            raise FileNotFoundError(f"CIFAR-100 data file not found at {file_path}")

        with open(file_path, "rb") as f:
            d = pickle.load(f, encoding="bytes")

        # Reshape to (N, 3, 32, 32) and normalize pixel values to [0, 1]
        self.data = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.labels = np.array(d[b"fine_labels"], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        x = torch.from_numpy(x)

        if self.transform:
            x = self.transform(x)

        y = torch.tensor(y, dtype=torch.long)
        return x, y


class CIFAR100DataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir: str = "data/cifar-100-python",
        batch_size: int = 64,
        num_workers: int = 2,
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = CIFAR100Dataset(self.data_dir / "train")
        self.test_dataset = CIFAR100Dataset(self.data_dir / "test")

    def get_train_data(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_test_data(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
