"""Datasets"""

from typing import Any

import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Unsupervised RGB image dataset stored in main memory"""
    def __init__(self, data: str, transform=None) -> None:
        """Create an RGB image dataset from '<data>.pt'"""
        super().__init__()

        if data.split(".")[-1] != ".pt":
            data += ".pt"

        self.images = torch.load(data)
        self.transform = transform

    def __len__(self):
        """Return the number of samples"""
        return len(self.images)

    def __getitem__(self, index) -> Any:
        """Return the sample at the given index"""
        if self.transform:
            return self.transform(self.images[index])
        return self.images[index]


# TODO: quick and dirty
class To01Transform:
    def __call__(self, batch: torch.Tensor):
        return batch / 255
