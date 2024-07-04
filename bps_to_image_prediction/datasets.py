import torch
import os
from torch.utils.data import Dataset


class BpsToImgDataset(Dataset):
    def __init__(self, img_data: str, bps_data: str, img_transform=None, target_transform=None) -> None:
        """
        Create a Dataset that contains labels(k timesteps of bps) and images
        """

        self.images = torch.load(img_data)
        self.bps = torch.load(bps_data)
        self.img_transform = img_transform
        self.target_transform = target_transform

    def __len__(self):
        """Return the number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Return bps and image at the given Index"""
        img = self.images[index]
        if self.img_transform:
            img = self.img_transform(img)

        bps = self.bps[index]
        if self.target_transform:
            bps = self.target_transform(bps)

        return bps, img


class BinarizeTransform:

    def __call__(self, img: torch.Tensor):
        img = img.sum(0)
        mask = img == 765
        binary_img = torch.where(mask, 0, 1)
        return binary_img
