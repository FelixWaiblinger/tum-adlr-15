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


class CombineTransform:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, batch: torch.Tensor) -> Any:
        for transform in self.transforms:
            batch = transform(batch)
        return batch


class NormalizeTransform:
    def __init__(self, start: tuple, end: tuple) -> None:
        self.a, self.b = start
        self.c, self.d = end
        assert self.a != self.b, f"Start range must be a non-empty interval!"

    def __call__(self, batch: torch.Tensor):
        factor = (self.d - self.c) / (self.b - self.a)
        return self.c + factor * (batch - self.a)


class StandardizeTransform:
    def __init__(self, mean=None, std=None, dim: tuple=None) -> None:
        self.mean = mean
        self.std = std
        self.dim = dim

    def __call__(self, batch: torch.Tensor):
        if not self.mean:
            self.mean = batch.mean(dim=self.dim, keepdims=True)
        if not self.std:
            self.std = batch.std(dim=self.dim, keepdims=True)
        return (batch - self.mean) / self.std
