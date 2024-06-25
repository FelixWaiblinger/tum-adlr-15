"""Unsupervised training using AutoEncoder"""

import copy

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from state_representation import AutoEncoder, ImageDataset, CombineTransform, \
    NormalizeTransform, StandardizeTransform
from adlr_environments.utils import create_env
from train_agent import ENV_OPTIONS, WRAPPER


DATA_PATH = "./state_representation/reset_image_data"
MODEL_PATH = "./state_representation/autoencoder"
N_SAMPLES = 30000

LATENT_SIZE = 500
N_LAYERS = 3
CHANNELS = [3, 64, 128, 256]
KERNELS = [5, 3, 3]
BATCH_SIZE = 64
EPOCHS = 20
VAL_RATE = 0.1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_tqdm_bar(iterable, desc):
    """Create a progress bar"""
    return tqdm(enumerate(iterable),total=len(iterable), ncols=150, desc=desc)


def training():
    """Train an autoencoder from scratch"""
    # load and prepare the data
    transform = CombineTransform([
        NormalizeTransform(start=(0, 255), end=(0, 1)),
        # StandardizeTransform()
    ])
    dataset = ImageDataset(DATA_PATH, transform=transform)
    train_idx, val_idx = train_test_split(range(N_SAMPLES), test_size=VAL_RATE)

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
        num_layers=N_LAYERS,
        channels=CHANNELS,
        kernels=KERNELS,
        latent_size=LATENT_SIZE,
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

    transform = CombineTransform([
        NormalizeTransform(start=(0, 255), end=(0, 1)),
        # StandardizeTransform(dim=(2, 3))
    ])

    _ = env.reset()
    image_raw = env.render()

    # NOTE: this is ULTRA messy...
    image_tch = copy.deepcopy(image_raw[2::4, 2::4, :])
    image_tch = image_tch.transpose([2, 0, 1])
    image_tch = torch.stack([torch.from_numpy(image_tch)])
    image_tch = transform(image_tch)

    image_rec = model.forward(image_tch.to(DEVICE))
    image_rec = image_rec.cpu().detach().numpy()[0]
    image_rec = image_rec.transpose([1, 2, 0]) * 255

    image_npy = image_tch.numpy()[0].transpose([1, 2, 0])

    for img in [image_raw, image_npy, image_rec]:
        plt.figure()
        plt.imshow(img)
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
        print(f"\rGenerating dataset: {(i/num_samples) * 100:3.2f}%", end="")
        _ = env.reset()
        image = env.render()

        # NOTE: use 128x128 as resolution instead of 512x512 to save memory
        image = image[2::4, 2::4, :].transpose([2, 0, 1])
        dataset.append(torch.from_numpy(image))
        del image

    dataset = torch.stack(dataset)
    torch.save(dataset, DATA_PATH + ".pt")
    print("")


if __name__ == "__main__":
    # record_dataset(num_samples=N_SAMPLES)

    # training()

    evaluate()
