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
            layers["conv1"+str(i)] = nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernels[i],
                padding=2,
                stride=2
            )
            layers["conv2"+str(i)] = nn.Conv2d(
                in_channels=channels[i+1],
                out_channels=channels[i+1],
                kernel_size=3,
                padding=2,
                stride=1
            )
            layers["pool"+str(i)] = nn.MaxPool2d(kernel_size=2, padding=0)
            layers["act"+str(i)] = nn.LeakyReLU(negative_slope=0.2)
            layers["bnrm"+str(i)] = nn.BatchNorm2d(channels[i+1])

        layers["flat"] = nn.Flatten()
        # NOTE: 4 is calculated as input shrinks from 128 to 2 (squared)
        layers["latent"] = nn.Linear(16 * channels[-1], latent_size)
        layers["lact"] = nn.Tanh()

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
            # NOTE: 4 is calculated as input shrinks from 128 to 2 (squared)
            latent=nn.Linear(latent_size, 4 * channels[0]),
        ))

        layers = OrderedDict()
        for i in range(num_layers):
            layers["tcnv"+str(i)] = nn.ConvTranspose2d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernels[i],
                output_padding=1,
                padding=1 if i+1 < num_layers else 2, # NOTE: caused by kernel
                stride=2
            )
            layers["upsm"+str(i)] = nn.Upsample(scale_factor=2, mode="bicubic")
            layers["act1"+str(i)] = nn.LeakyReLU(negative_slope=0.2)
            layers["bnrm"+str(i)] = nn.BatchNorm2d(channels[i+1])

        layers["outc"] = nn.ConvTranspose2d(
            in_channels=channels[-1],
            out_channels=channels[-1],
            kernel_size=3,
            padding=1
        )
        layers["outa"] = nn.Sigmoid()

        self.net = nn.Sequential(layers)

    def forward(self, x):
        """Forward pass"""
        x = self.linear(x)
        # NOTE: 2 is calculated as input shrinks from 128 to 2 (squared)
        x = x.reshape(x.shape[0], -1, 2, 2)
        return self.net.forward(x)


class Encoder2(nn.Module):
    """Encoder network transforms the input to a latent representation"""

    def __init__(self,
        num_layers: int,
        channels: list,
        kernels: list,
        latent_size: int
    ) -> None:
        """Create a new encoder module"""
        super().__init__()

        layers = OrderedDict()                                                  # input
        layers["c11"] = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)    # 128x128x3
        # layers["c12"] = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)   # 64x64x32
        layers["c13"] = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)   # ...
        layers["p1"] = nn.MaxPool2d(2, stride=2)                                # 64x64x64
        layers["a1"] = nn.LeakyReLU(negative_slope=0.2)                         # 32x32x64
        layers["b1"] = nn.BatchNorm2d(64)                                       # ...
        layers["c21"] = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)  # ...
        # layers["c22"] = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 16x16x128
        layers["c23"] = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # ...
        layers["p2"] = nn.MaxPool2d(2, stride=2)                                # 16x16x256
        layers["a2"] = nn.LeakyReLU(negative_slope=0.2)                         # 8x8x256
        layers["b2"] = nn.BatchNorm2d(256)                                      # ...
        layers["c31"] = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # ...
        # layers["c32"] = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1) # 4x4x512
        layers["p3"] = nn.MaxPool2d(2, stride=2)                                # 4x4x512
        layers["a3"] = nn.LeakyReLU(negative_slope=0.2)                         # 2x2x512
        layers["b3"] = nn.BatchNorm2d(512)                                      # ...
        layers["f"] = nn.Flatten()                                              # ...
        layers["l"] = nn.Linear(2048, latent_size)                              # 1x2048
        layers["a"] = nn.Tanh()                                                 # 1xlatent
        
        self.net = nn.Sequential(layers)
        self.latent_size = latent_size

    def forward(self, x):
        """Forward pass"""
        return self.net.forward(x)


class Decoder2(nn.Module):
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

        layers = OrderedDict()                                                                              # input
        layers["c11"] = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)                              # 2x2x512
        # layers["c12"] = nn.Conv2d(512, 512, kernel_size=3, padding=1)                                       # ...
        # layers["a11"] = nn.LeakyReLU(negative_slope=0.2)                                                    # ...
        layers["c13"] = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)  # ...
        layers["u1"] = nn.Upsample(scale_factor=2, mode="bicubic")                                          # 4x4x256
        layers["a12"] = nn.LeakyReLU(negative_slope=0.2)                                                    # 8x8x256
        layers["b1"] = nn.BatchNorm2d(256)                                                                  # ...
        layers["c21"] = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)                    # ...
        # layers["c22"] = nn.Conv2d(128, 128, kernel_size=3, padding=1)                                       # 8x8x128
        # layers["a21"] = nn.LeakyReLU(negative_slope=0.2)                                                    # ...
        layers["c23"] = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)   # ...
        layers["u2"] = nn.Upsample(scale_factor=2, mode="bicubic")                                          # 16x16x64
        layers["a22"] = nn.LeakyReLU(negative_slope=0.2)                                                    # 32x32x64
        layers["b2"] = nn.BatchNorm2d(64)                                                                   # ...
        layers["c31"] = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)                      # ...
        # layers["c32"] = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)                               # 32x32x32
        # layers["a31"] = nn.LeakyReLU(negative_slope=0.2)                                                    # ...
        layers["c33"] = nn.ConvTranspose2d(32, 3, kernel_size=7, stride=2, padding=3, output_padding=1)     # ...
        layers["u3"] = nn.Upsample(scale_factor=2, mode="bicubic")                                          # 64x64x3
        layers["a32"] = nn.Sigmoid()                                                                        # 128x128x3

        self.linear = nn.Linear(latent_size, 2048)
        self.net = nn.Sequential(layers)

    def forward(self, x):
        """Forward pass"""
        x = self.linear(x).reshape(x.shape[0], -1, 2, 2)
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

        self.encoder = Encoder2(
            num_layers=num_layers,
            channels=channels,
            kernels=kernels,
            latent_size=latent_size,
        )

        self.decoder = Decoder2(
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
            step_size=10000, # TODO: replace arbitrary values
            gamma=0.9 # TODO: replace arbitrary values
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
