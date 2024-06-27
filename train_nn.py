"""Unsupervised training using AutoEncoder"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

import adlr_environments
from utils import arg_parse, create_env, create_tqdm_bar
from utils.constants import DEVICE
from state_representation import AutoEncoder, ImageDataset, record_resets
from utils.config import AE_CONFIG, TRANSFORM


DATA_PATH = "./state_representation/reset_image_data_random_obs"
MODEL_PATH = "./state_representation/autoencoder_random_obs_100.pt"
ARGUMENTS = [
    (("-r", "--record"), int, None),
    (("-t", "--train"), int, None),
    (("-e", "--eval"), str, None)
]

# model parameters
N_LAYERS = 3
CHANNELS = [3, 64, 128, 256]
KERNELS = [5, 3, 3]
LATENT_SIZE = 50

# training parameters
BATCH_SIZE = 64
VAL_RATE = 0.1


def training(epochs: int):
    """Train an autoencoder from scratch"""
    # load and prepare the data
    dataset = ImageDataset(DATA_PATH, transform=TRANSFORM)
    n_samples = len(dataset)
    train_idx, val_idx = train_test_split(range(n_samples), test_size=VAL_RATE)

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

    # model = AutoEncoder(
    #     num_layers=N_LAYERS,
    #     channels=CHANNELS,
    #     kernels=KERNELS,
    #     latent_size=LATENT_SIZE,
    #     device=DEVICE
    # )
    # create the model
    model = AutoEncoder(
        num_layers=N_LAYERS,
        channels=CHANNELS,
        kernels=KERNELS,
        latent_size=LATENT_SIZE,
    )

    val_loss = 0
    for epoch in range(epochs):
        # train the model using training data
        train_loss = 0
        train_loop = create_tqdm_bar(
            train_loader,
            desc=f"Training Epoch [{epoch + 1}/{epochs}]"
        )

        for train_iteration, batch in train_loop:
            loss = model.training_step(batch)
            train_loss += loss.item()

            train_loop.set_postfix(
                curr_train_loss=f"{(train_loss / (train_iteration + 1)):.5f}",
                val_loss=f"{val_loss:.5f}"
            )


        # check performance using validation data
        val_loss = 0
        val_loop = create_tqdm_bar(
            val_loader,
            desc=f"Validation Epoch [{epoch + 1}/{epochs}]"
        )

        with torch.no_grad():
            for val_iteration, batch in val_loop:
                loss = model.validation_step(batch)
                val_loss += loss.item()

                val_loop.set_postfix(
                    val_loss=f"{(val_loss / (val_iteration + 1)):.5f}"
                )

        val_loss /= len(val_loader)

    torch.save(model, MODEL_PATH)


def evaluate(show: str):
    """Show reconstruction of unseen sample"""
    model: AutoEncoder = torch.load(MODEL_PATH).to(DEVICE)

    env = create_env(
        wrapper=AE_CONFIG.wrapper,
        render=False,
        obs_type=AE_CONFIG.observation,
        num_workers=1,
        options=AE_CONFIG.env
    )

    _ = env.reset()
    image_raw = env.render()

    image_tch = image_raw[2::4, 2::4, :].transpose([2, 0, 1])
    image_tch = TRANSFORM(torch.from_numpy(np.expand_dims(image_tch, 0))) # pylint: disable=E1101
    image_rec = model.forward(image_tch.to(DEVICE)).cpu()

    if show == "image":
        image_rec = image_rec.detach().numpy()[0].transpose([1, 2, 0])
        image_npy = image_tch.numpy()[0].transpose([1, 2, 0])

        images = [image_raw, image_npy, image_rec]
        names = ["environment", "encoder input", "reconstruction"]
        for img, name in zip(images, names):
            plt.figure()
            plt.title(name)
            plt.imshow(img)
        plt.show()
    elif show == "loss":
        loss = torch.nn.functional.mse_loss(image_tch, image_rec)
        print(f"Test loss on one sample: {loss}")


if __name__ == "__main__":
    # parse arguments from cli
    args = arg_parse(ARGUMENTS)
    assert not all(arg is None for arg in [args.record, args.train, args.eval])

    # record reset dataset
    if args.record is not None:
        assert args.record > 0, \
            f"Number of samples ({args.record}) to record must be positive!"
        record_resets(DATA_PATH, args.record, AE_CONFIG.env)

    # train an autoencoder on the recorded data
    elif args.train is not None:
        assert args.train > 0, \
            f"Number of epochs ({args.train}) to train must be positive!"
        training(args.train)

    # evaluate the performance of the encoder by visual inspection
    elif args.eval is not None:
        evaluate(args.eval)
