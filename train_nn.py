"""Unsupervised training using AutoEncoder"""

import copy

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

from state_representation import AutoEncoder, ImageDataset, CombineTransform, \
    NormalizeTransform
from state_representation.datasets import GrayscaleTransform
from adlr_environments.utils import create_env
from train_agent import ENV_OPTIONS, WRAPPER
from train import environment_creation
from constants import OPTIONS
from state_representation.niklas_models import Autoencoder, Encoder, Decoder, BpsToImgNetwork
from StateTransitionTrainer.dataloader import BpsToImgDataset, BinarizeTransform

DATA_PATH = "./state_representation/reset_image_data"
MODEL_PATH = "./state_representation/autoencoder"
N_SAMPLES = 10000

LATENT_SIZE = 500
N_LAYERS = 3
CHANNELS = [3, 64, 128, 256]
KERNELS = [5, 3, 3]
BATCH_SIZE = 10
EPOCHS = 1000
VAL_RATE = 0.1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_tqdm_bar(iterable, desc):
    """Create a progress bar"""
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)


def show_dataset():
    transform = CombineTransform([
        GrayscaleTransform(),
        # NormalizeTransform(start=(0, 255), end=(0, 1)),
        # StandardizeTransform()
    ])
    dataset = ImageDataset(DATA_PATH, transform=transform)
    train_idx, val_idx = train_test_split(range(N_SAMPLES), test_size=VAL_RATE)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    img = dataset.__getitem__(0)
    image = img.numpy()
    # image = image.transpose(1, 2, 0)
    plt.imshow(image, cmap="gray")
    plt.show()
    print("test")


def training():
    """Train an autoencoder from scratch"""
    # load and prepare the data
    transform = CombineTransform([
        GrayscaleTransform(),
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

    # model = AutoEncoder(
    #     num_layers=N_LAYERS,
    #     channels=CHANNELS,
    #     kernels=KERNELS,
    #     latent_size=LATENT_SIZE,
    #     device=DEVICE
    # )
    # create the model
    hparams = {"device": "cpu"}
    encoder = Encoder(input_size=64 * 64)
    decoder = Decoder(output_size=64 * 64)
    model = Autoencoder(hparams=hparams, encoder=encoder, decoder=decoder)

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


def record_dataset(num_samples: int = 100):
    """Record image samples of the environment"""
    options = OPTIONS
    options.update({"render": "rgb_array"})
    env = environment_creation(options=OPTIONS)

    dataset = []
    for i in range(num_samples):
        print(f"\rGenerating dataset: {(i / num_samples) * 100:3.2f}%", end="")
        _ = env.reset()
        image = env.render()

        # NOTE: use 128x128 as resolution instead of 512x512 to save memory
        image = image[2::8, 2::8, :].transpose([2, 0, 1])

        dataset.append(torch.from_numpy(image))
        del image

    dataset = torch.stack(dataset)
    torch.save(dataset, DATA_PATH + ".pt")
    print("")


def evaluate_autoencoder():
    transform = CombineTransform([
        GrayscaleTransform(),
        NormalizeTransform(start=(0, 255), end=(0, 1)),
        # StandardizeTransform()
    ])
    dataset = ImageDataset(DATA_PATH, transform=transform)
    train_loader = DataLoader(dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True
                              )
    data_iter = iter(train_loader)
    images = next(data_iter)
    model = torch.load(MODEL_PATH + ".pt").to(DEVICE)
    reconstruction = model.forward(images)
    num_comparisons = 3
    fig, ax = plt.subplots(num_comparisons, 2)
    for i in range(num_comparisons):
        # plot both images
        img = images[i]
        img1 = img.numpy()
        # image = image.transpose(1, 2, 0)

        # get old input range back
        # reconstruction = reconstruction * 255
        img2 = reconstruction[i]
        img2 = img2.detach().numpy()

        # Display first image
        ax[i, 0].imshow(img1.squeeze(), cmap='gray')  # Use squeeze to remove channel dimension
        ax[i, 0].set_title(f'Label: Original')
        ax[i, 0].axis('off')  # Turn off axis

        # Display second image
        ax[i, 1].imshow(img2.squeeze(), cmap='gray')  # Use squeeze to remove channel dimension
        ax[i, 1].set_title(f'Label: Reconstruction')
        ax[i, 1].axis('off')  # Turn off axis

    # Show the plot
    plt.show()


def check_dataset():
    img_data_location = "./StateTransitionTrainer/bps_img_img.pt"
    bps_data_location = "./StateTransitionTrainer/bps_img_bps.pt"
    dataset = BpsToImgDataset(img_data=img_data_location, bps_data=bps_data_location, img_transform=BinarizeTransform())

    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_labels[4].numpy()
    plt.imshow(img, cmap="gray")
    plt.show()


if __name__ == "__main__":
    # record_dataset(num_samples=N_SAMPLES)
    # show_dataset()
    # training()
    # evaluate_autoencoder()
    # evaluate()
    check_dataset()
