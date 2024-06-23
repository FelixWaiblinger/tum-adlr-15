"""Collection of pytorch models"""

from typing import OrderedDict

import torch
from torch import nn, optim
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder network transforms the input to a latent representation"""

    def __init__(self,
        num_layers: int,
        channels: list,
        kernels: list,
        latent_size: int
    ) -> None:
        """Create a new encoder module"""
        super().__init__()

        assert len(channels) == num_layers + 1, \
            f"num_layers {num_layers} and channels {channels} do not fit!"
        assert len(kernels) == num_layers, \
            f"num_layers {num_layers} and channels {channels} do not fit!"

        layers = OrderedDict()
        for i in range(num_layers):
            layers["conv"+str(i)] = nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernels[i],
                padding=1,
                stride=2
            )
            layers["act"+str(i)] = nn.ReLU()

        layers["flat"] = nn.Flatten()
        # NOTE: 256 is calculated as input shrinks from 128 to 16 (squared)
        layers["latent"] = nn.Linear(256 * channels[-1], latent_size)

        self.net = nn.Sequential(layers)
        self.latent_size = latent_size

    def forward(self, x):
        """Forward pass"""
        return self.net.forward(x)


class Decoder(nn.Module):
    """Decoder network transforms the latent representation to the format of
    the input
    """

    def __init__(self,
        num_layers: int,
        channels: list,
        kernels: list,
        latent_size: int,
    ) -> None:
        """Create a new decoder module"""
        super().__init__()

        assert len(channels) == num_layers + 1, \
            f"num_layers {num_layers} and channels {channels} do not fit!"
        assert len(kernels) == num_layers, \
            f"num_layers {num_layers} and channels {channels} do not fit!"

        self.linear = nn.Sequential(OrderedDict(
            # NOTE: 256 is calculated as input shrinks from 128 to 16 (squared)
            latent=nn.Linear(latent_size, 256 * channels[0]),
            act=nn.ReLU()
        ))

        layers = OrderedDict()
        for i in range(num_layers):
            layers["tcnv"+str(i)] = nn.ConvTranspose2d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernels[i],
                output_padding=1,
                padding=1,
                stride=2
            )
            layers["act1"+str(i)] = nn.ReLU()
            layers["conv"+str(i)] = nn.Conv2d(
                in_channels=channels[i+1],
                out_channels=channels[i+1],
                kernel_size=kernels[i],
                padding=1,
            )
            if i+1 < num_layers:
                layers["act2"+str(i)] = nn.ReLU()
            else:
                layers["act2"+str(i)] = nn.Sigmoid()

        self.net = nn.Sequential(layers)

    def forward(self, x):
        """Forward pass"""
        x = self.linear(x)
        # NOTE: 256 is calculated as input shrinks from 128 to 16 (squared)
        x = x.reshape(x.shape[0], -1, 16, 16)
        return self.net.forward(x)


class AutoEncoder(nn.Module):
    """Autoencoder module"""
    def __init__(self,
        num_layers: int,
        channels: list,
        kernels: list,
        latent_size: int,
        device: str
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            num_layers=num_layers,
            channels=channels,
            kernels=kernels,
            latent_size=latent_size,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            channels=list(reversed(channels)),
            kernels=list(reversed(kernels)),
            latent_size=latent_size,
        )

        self.device = device
        self.optimizer, self.scheduler = None, None
        self.set_optimizer_scheduler()
        self.to(device)

    def forward(self, x):
        """Forward pass"""
        z = self.encoder.forward(x)
        return self.decoder.forward(z)

    def loss(self, batch, reconstruction):
        """Given a batch of images return the reconstruction loss (MSE)"""
        loss = F.mse_loss(batch, reconstruction, reduction="none") \
                .sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def set_optimizer_scheduler(self):
        """Configure optimizer and scheduler"""
        self.optimizer = optim.Adam([
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()}],
            lr=1e-3, # TODO: replace arbitrary values
            weight_decay=0 # TODO: replace arbitrary values
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000, # TODO: replace arbitrary values
            gamma=0.1 # TODO: replace arbitrary values
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
