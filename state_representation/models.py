"""Collection of pytorch models"""

from typing import OrderedDict

import torch
from torch import nn, optim
import torch.nn.functional as F

from utils.constants import DEVICE


class Encoder(nn.Module):
    """Encoder network transforms the input to a latent representation
    
    Hardcoded version (except latent size)
    """

    def __init__(self, latent_size: int) -> None:
        """Create a new encoder module"""
        super().__init__()

        layers = OrderedDict()
        # in: 128x128x3
        layers["c1"] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # in: 64x64x64
        layers["a1"] = nn.LeakyReLU(negative_slope=0.2)
        # in: 64x64x64
        layers["b1"] = nn.BatchNorm2d(64)
        # in: 64x64x64
        layers["c2"] = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        # in: 32x32x128
        layers["p2"] = nn.MaxPool2d(2, stride=2)
        # in: 16x16x128
        layers["a2"] = nn.LeakyReLU(negative_slope=0.2)
        # in: 16x16x128
        layers["b2"] = nn.BatchNorm2d(128)
        # in : 16x16x128
        layers["c3"] = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # in: 8x8x256
        layers["p3"] = nn.MaxPool2d(2, stride=2)
        # in: 4x4x256
        layers["a3"] = nn.LeakyReLU(negative_slope=0.2)
        # in: 4x4x256
        layers["b3"] = nn.BatchNorm2d(256)
        # in: 4x4x256
        layers["c4"] = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        # in: 2x2x512
        layers["a4"] = nn.LeakyReLU(negative_slope=0.2)
        # in: 2x2x512
        layers["b4"] = nn.BatchNorm2d(512)
        # in: 2x2x512
        layers["f"] = nn.Flatten()
        # in: 1x2048
        layers["l"] = nn.Linear(2048, latent_size)
        # in: 1xlatent
        layers["a"] = nn.Tanh()
        
        self.net = nn.Sequential(layers)
        self.latent_size = latent_size

    def forward(self, x):
        """Forward pass"""
        return self.net.forward(x)


class Decoder(nn.Module):
    """Decoder network transforms the latent representation to the format of
    the input

    Hardcoded version (except latent size)
    """

    def __init__(self, latent_size: int) -> None:
        """Create a new decoder module"""
        super().__init__()

        layers = OrderedDict()
        # in: 2x2x512
        layers["c1"] = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        # in: 4x4x256
        layers["a1"] = nn.LeakyReLU(negative_slope=0.2)
        # in: 4x4x256
        layers["b1"] = nn.BatchNorm2d(256)
        # in: 4x4x256
        layers["c2"] = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        # in: 8x8x128
        layers["u1"] = nn.Upsample(scale_factor=2, mode="bicubic")
        # in: 16x16x128
        layers["a2"] = nn.LeakyReLU(negative_slope=0.2)
        # in: 16x16x128
        layers["b2"] = nn.BatchNorm2d(128)
        # in: 16x16x128
        layers["c3"] = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        # in: 32x32x64
        layers["u3"] = nn.Upsample(scale_factor=2, mode="bicubic")
        # in: 64x64x64
        layers["a3"] = nn.LeakyReLU(negative_slope=0.2)
        # in: 64x64x64
        layers["b3"] = nn.BatchNorm2d(64)
        # in: 64x64x64
        layers["c42"] = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1)
        # in: 128x128x3
        layers["a4"] = nn.Sigmoid()

        self.linear = nn.Linear(latent_size, 2048)
        self.net = nn.Sequential(layers)

    def forward(self, x):
        """Forward pass"""
        x = self.linear(x).reshape(x.shape[0], -1, 2, 2)
        return self.net.forward(x)


class AutoEncoder(nn.Module):
    """Autoencoder module"""
    def __init__(self, latent_size: int) -> None:
        """Create a new autoencoder with a given latent size"""
        super().__init__()

        self.encoder = Encoder(latent_size=latent_size)
        self.decoder = Decoder(latent_size=latent_size)

        self.device = DEVICE
        self.optimizer, self.scheduler = None, None
        self.set_optimizer_scheduler()
        self.to(DEVICE)

    def forward(self, x):
        """Forward pass"""
        z = self.encoder.forward(x)
        return self.decoder.forward(z)

    def loss(self, batch, reconstruction):
        """Given a batch of images return the reconstruction loss (SSE)"""
        loss = F.mse_loss(batch, reconstruction, reduction="none") \
                .sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def set_optimizer_scheduler(self):
        """Configure optimizer and scheduler"""
        self.optimizer = optim.Adam([
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()}],
            lr=1e-3,
            weight_decay=0
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10000,
            gamma=0.5
        )

    def training_step(self, batch: torch.Tensor):
        """Perform a training step on a single batch"""
        self.train()
        self.optimizer.zero_grad()

        batch = batch.to(self.device)
        reconstruction = self.forward(batch)
        loss = self.loss(batch, reconstruction)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def validation_step(self, batch: torch.Tensor):
        """Perform a validation step on a single batch"""
        self.eval()

        batch = batch.to(self.device)
        reconstruction = self.forward(batch)
        loss = self.loss(batch, reconstruction)

        return loss


class CoordRegressorCNN(nn.Module):
    def __init__(self, feature_extractor, head, lr=0.005):
        super().__init__()
        self.features: nn.Sequential = feature_extractor
        self.head: nn.Sequential = head
        self.lr = lr
        self.device = DEVICE
        self.optimizer, self.scheduler = None, None
        self.set_optimizer_scheduler()
        self.to(DEVICE)

    def set_optimizer_scheduler(self):
        """Configure optimizer and scheduler"""
        self.optimizer = optim.Adam([
            {"params": self.features.parameters()},
            {"params": self.head.parameters()}],
            lr=self.lr,
            weight_decay=0
        )

        # self.scheduler = optim.lr_scheduler.StepLR(
        #     self.optimizer,
        #     step_size=10000,
        #     gamma=0.5
        # )

    def forward(self, batch):
        features = self.features.forward(batch)
        return self.head.forward(features)

    def loss(self, y_hat, y):
        """Given a batch of images return the reconstruction loss (SSE)"""
        return F.mse_loss(y_hat, y, reduction="none").sum(dim=1).mean(dim=0)

    def training_step(self, batch: torch.Tensor):
        self.train()
        self.optimizer.zero_grad()
        x, y = batch

        x = x.to(self.device)
        y = y.to(self.device)

        y_hat, _ = self.forward(x)
        loss = self.loss(y_hat, y)

        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

        return loss

    def validation_step(self, batch: torch.Tensor):
        x, y = batch
        self.eval()

        x = x.to(self.device)
        y = y.to(self.device)

        y_hat, _ = self.forward(x)
        loss = self.loss(y_hat, y)

        return loss
