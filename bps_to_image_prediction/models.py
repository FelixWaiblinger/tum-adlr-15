import torch
from torch import nn, optim
import pytorch_lightning as L


class LitBpsToImgNetwork(L.LightningModule):
    def __init__(self, number_of_bps: int = 100, output_size: int = 64 * 64):
        super().__init__()

        # set hyperparams
        self.output_size = output_size
        self.input_size = number_of_bps * 3

        # build network
        self.network = nn.Sequential(
            nn.Linear(self.input_size, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.Linear(100, self.output_size),
            nn.BatchNorm1d(self.output_size),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        out_network = self.network(x)
        out_network = out_network.view(-1, 64, 64)
        return out_network

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        y = y.float()
        loss = nn.functional.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.float()
        y_hat = self.forward(x)
        loss = nn.functional.binary_cross_entropy(y_hat, y)
        self.log("val_loss", loss)
