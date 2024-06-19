import numpy
import pandas as pd
import torch
import torch.nn as nn
from adlr_environments.constants import DATASET_PATH, TRAININGS_DATA_PATH
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader


class StateTransitionDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        """
        Arguments:
            data_dir (string): Directory with all data
            label_dir (string): Directory with all the labels
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.data = os.listdir(self.data_dir)
        self.labels = os.listdir(self.label_dir)

    def __len__(self):
        entries = os.listdir(self.data_dir)
        file_count = sum(os.path.isfile(os.path.join(self.data_dir, entry)) for entry in entries)
        return file_count

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_name = os.path.join(self.data_dir, self.data[idx])
        label_name = os.path.join(self.label_dir, self.labels[idx])
        data_sample = np.load(data_name).T
        label_sample = np.load(label_name).T
        label_sample = np.expand_dims(label_sample, axis=-1)
        sample = {"data": data_sample, "label": label_sample}
        if self.transform:
            sample = self.transform(sample)
        return sample
