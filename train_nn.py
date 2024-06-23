"""Unsupervised training using AutoEncoder"""

import copy

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from state_representation import AutoEncoder, ImageDataset, To01Transform
from adlr_environments.utils import create_env
from train_agent import ENV_OPTIONS, WRAPPER


DATA_PATH = "./state_representation/reset_image_data"
MODEL_PATH = "./state_representation/autoencoder"

CHANNELS = [3, 8, 16, 32]
KERNELS = [3, 3, 3]
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_tqdm_bar(iterable, desc):
    """Create a progress bar"""
    return tqdm(enumerate(iterable),total=len(iterable), ncols=150, desc=desc)


def training():
    """Train an autoencoder from scratch"""
    # load and prepare the data
    dataset = ImageDataset(DATA_PATH, transform=To01Transform())
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # create the model
    model = AutoEncoder(
        num_layers=3,
        channels=CHANNELS,
        kernels=KERNELS,
        latent_size=50,
        device=DEVICE
    )

    val_loss = 0
    for epoch in range(EPOCHS):
        # train the model using training data
        train_loss = 0
        train_loop = create_tqdm_bar(
            train_loader,
            desc=f"Training Epoch [{epoch + 1}/{EPOCHS}]"
        )

        for train_iteration, batch in train_loop:
            loss = model.training_step(batch)
            train_loss += loss.item()

            train_loop.set_postfix(
                curr_train_loss=f"{(train_loss / (train_iteration + 1)):.8f}",
                val_loss=f"{val_loss:.8f}"
            )

        # check performance using validation data
        val_loss = 0
        val_loop = create_tqdm_bar(
            val_loader,
            desc=f"Validation Epoch [{epoch + 1}/{EPOCHS}]"
        )

        with torch.no_grad():
            for val_iteration, batch in val_loop:            
                loss = model.validation_step(batch)
                val_loss += loss.item()

                val_loop.set_postfix(
                    val_loss=f"{(val_loss / (val_iteration + 1)):.8f}"
                )

        val_loss /= len(val_loader)

    torch.save(model, MODEL_PATH + ".pt")


def evaluate():
    """Show reconstruction of unseen sample"""
    options = ENV_OPTIONS
    options.pop("fork")

    model = torch.load(MODEL_PATH + ".pt").to(DEVICE)

    env = create_env(
        wrapper=WRAPPER,
        render=False,
        num_workers=1,
        options=options
    )

    transform = To01Transform()

    _ = env.reset()
    image = env.render()

    plt.figure()
    plt.imshow(image)

    # NOTE: this is ULTRA messy...
    image = image[2::4, 2::4, :]
    test = copy.deepcopy(image)
    x = torch.stack([torch.from_numpy(test.transpose([2, 0, 1]))]).to(DEVICE)
    reconstruction: torch.Tensor = model.forward(transform(x))[0].cpu()
    reconstruction = reconstruction.detach().numpy().transpose([1, 2, 0]) * 255

    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(reconstruction)
    plt.show()


def record_dataset(num_samples: int=100):
    """Record image samples of the environment"""
    options = ENV_OPTIONS
    options.pop("fork")

    env = create_env(
        wrapper=WRAPPER,
        render=False,
        num_workers=1,
        options=options
    )

    dataset = []
    for i in range(num_samples):
        # print(f"{(i/num_samples) * 100}%")
        _ = env.reset()
        image = env.render()

        # NOTE: use 128x128 as resolution instead of 512x512 to save memory
        image = image[2::4, 2::4, :].transpose([2, 0, 1])
        dataset.append(torch.from_numpy(image))

    dataset = torch.stack(dataset)
    torch.save(dataset, DATA_PATH + ".pt")


if __name__ == "__main__":
    # record_dataset(num_samples=10000)

    # training()

    evaluate()
