import torch
from torch import nn, optim
import pytorch_lightning as L


class Encoder(nn.Module):

    def __init__(self, input_size=64 * 64, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim
        self.input_size = input_size
        # self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.Linear(100, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)


class Decoder(nn.Module):

    def __init__(self, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.BatchNorm1d(100),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(100, 500),
            nn.BatchNorm1d(500),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(500, output_size)
        )

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        batchsize = x.shape[0]
        width = x.shape[1]
        length = x.shape[2]

        x = x.view(x.shape[0], -1)
        z = self.encoder(x)
        o = self.decoder(z)
        o = o.view(batchsize, width, length)
        return o

    def set_optimizer(self):
        self.optimizer = optim.Adam([
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()}],
            lr=1e-3,  # TODO: replace arbitrary values
            weight_decay=0  # TODO: replace arbitrary values
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,  # TODO: replace arbitrary values
            gamma=0.1  # TODO: replace arbitrary values
        )

    def loss(self, batch, reconstruction):
        loss = nn.MSELoss()
        loss = loss(batch, reconstruction)
        return loss

    def training_step(self, batch: torch.Tensor):
        self.train()
        self.optimizer.zero_grad()

        batch = batch.to(self.device)
        reconstruction = self.forward(batch)
        loss = self.loss(batch, reconstruction)

        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()
        return loss

    def validation_step(self, batch: torch.Tensor):
        """Perform a validation step on a single batch"""
        self.eval()

        batch = batch.to(self.device)
        reconstruction = self.forward(batch)
        loss = self.loss(batch, reconstruction)

        return loss


class BpsToImgNetwork(nn.Module):
    def __init__(self, number_of_bps: int = 100, output_size: int = 64 * 64):
        super().__init__()

        # set hyperparams
        self.output_size = output_size
        self.input_size = number_of_bps * 3

        # set optimizer
        self.set_optimizer()

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
            nn.ReLU()
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        out_network = self.network(x)
        out_network = out_network.view(-1, 64, 64)
        return out_network

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)



    def training_step(self, batch: torch.Tensor):
        self.train()
        self.optimizer.zero_grad()

        batch = batch.to(self.device)
        reconstruction = self.forward(batch)
        loss = self.loss(batch, reconstruction)

        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()
        return loss

    def validation_step(self, batch: torch.Tensor):
        """Perform a validation step on a single batch"""
        self.eval()

        batch = batch.to(self.device)
        reconstruction = self.forward(batch)
        loss = self.loss(batch, reconstruction)

        return loss