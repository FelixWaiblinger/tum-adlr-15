"""Datasets"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import create_env
from utils.constants import Observation


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
    """Transform to apply multiple transforms to data"""
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, batch: torch.Tensor) -> Any:
        for transform in self.transforms:
            batch = transform(batch)
        return batch


class NormalizeTransform:
    """Transform to scale data from one range to another"""
    def __init__(self, start: tuple, end: tuple) -> None:
        self.a, self.b = start
        self.c, self.d = end
        assert self.a != self.b, "Start range must be a non-empty interval!"

    def __call__(self, batch: torch.Tensor):
        factor = (self.d - self.c) / (self.b - self.a)
        return self.c + factor * (batch - self.a)


class StandardizeTransform:
    """Transform to remove data mean and scale inversly by data variance"""
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


def record_resets(save_dir: str, num_samples: int, options: dict):
    """Record image samples of the environment"""
    env = create_env(
        wrapper=[],
        render=False,
        obs_type=Observation.RGB,
        num_workers=1,
        options=options
    )

    dataset, temp = None, None
    for i in range(num_samples):
        print(f"\rGenerating dataset: {(float(i+1)/num_samples) * 100:.2f}%", end="")
        _ = env.reset()
        image = env.render()

        # change number of obstacles for each sample
        n_static = np.random.choice([0, 1, 2, 3, 4, 5])
        n_dynamic = np.random.choice([0, 1, 2, 3])
        setattr(env, "options['num_static_obstacles']", n_static)
        setattr(env, "options['num_dynamic_obstacles']", n_dynamic)

        # NOTE: use 128x128 as resolution instead of 512x512 to save memory
        image = image[2::4, 2::4, :].transpose([2, 0, 1])
        image = np.expand_dims(image, 0)

        temp = np.vstack([temp, image]) if temp is not None else image

        if (i+1) % 1000 == 0:
            dataset = np.vstack([dataset, temp]) if dataset is not None else temp
            temp = None

    torch.save(torch.from_numpy(dataset), save_dir + ".pt") # pylint: disable=E1101
    print("")
