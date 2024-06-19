from StateTransitionTrainer import dataset_creator
from constants import DATASET_PATH, TRAININGS_DATA_PATH
import numpy
import pandas as pd
import torch
import torch.nn as nn
from adlr_environments.constants import DATASET_PATH, TRAININGS_DATA_PATH
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset
from dataloader import StateTransitionDataset


class TemporalCNN(pl.LightningModule):
    def __init__(self, input_dim, num_filters, kernel_size, output_dim):
        super(TemporalCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(num_filters * ((50 - kernel_size + 1) // 2), output_dim)

    def forward(self, x):
        x = self.conv1(x)  # (batch_size, num_filters, new_seq_length)
        x = self.relu(x)
        x = self.pool(x)  # (batch_size, num_filters, pooled_seq_length)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x


data_directory = os.path.join(os.pardir, TRAININGS_DATA_PATH, "data/")
label_directory = os.path.join(os.pardir, TRAININGS_DATA_PATH, "label/")
dataset = StateTransitionDataset(data_dir=data_directory, label_dir=label_directory)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched["data"].size(), sample_batched["label"].size())
    if i_batch ==3:
        break
# model = TemporalCNN()
# trainer = pl.Trainer(fast_dev_run=100)
# trainer.fit(model=)