import numpy as np

from bps_to_image_prediction.datasets import BpsToImgDataset, BinarizeTransform
from bps_to_image_prediction.models import LitBpsToImgNetwork
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as L
import matplotlib.pyplot as plt
from train import environment_creation
from adlr_environments.constants import OPTIONS
import queue
import torch

BATCH_SIZE = 128
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
EPOCHS = 200
N_SAMPLES = 50000
IMG_DATA_LOCATION = "./bps_img_img.pt"
BPS_DATA_LOCATION = "./bps_img_bps.pt"
CHECKPOINT_LOCATION = r"D:\uni\adl4r\code\tum-adlr-15\bps_to_image_prediction\lightning_logs\version_18\checkpoints\epoch=199-step=70400.ckpt"

def training_lightning():
    """Trains a nn that maps from bps to images"""

    # load dataset
    dataset = BpsToImgDataset(img_data=IMG_DATA_LOCATION, bps_data=BPS_DATA_LOCATION, img_transform=BinarizeTransform())

    # split up into training and test dataset
    train_idx, val_idx = train_test_split(range(N_SAMPLES), test_size=0.1)
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
    model = LitBpsToImgNetwork()
    trainer = L.Trainer(accelerator=DEVICE, max_epochs=EPOCHS)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def eval_ligthning():
    # load dataset
    dataset = BpsToImgDataset(img_data=IMG_DATA_LOCATION, bps_data=BPS_DATA_LOCATION, img_transform=BinarizeTransform())


    # split up into training and test dataset
    train_idx, val_idx = train_test_split(range(N_SAMPLES), test_size=0.1)
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


    model = LitBpsToImgNetwork.load_from_checkpoint(CHECKPOINT_LOCATION).cpu()
    model.eval()
    data_iter = iter(val_loader)
    for l in data_iter:

        features, labels, original_labels = l

        prediction = model(features)
        num_comparisons = 3
        fig, ax = plt.subplots(num_comparisons, 3)
        for i in range(num_comparisons):
            # plot both images
            img = labels[i]
            img1 = img.numpy()
            # image = image.transpose(1, 2, 0)

            # get old input range back
            # reconstruction = reconstruction * 255
            img2 = prediction[i]
            img2 = img2.detach().numpy()

            # Display first image
            ax[i, 0].imshow(img1, cmap='gray')  # Use squeeze to remove channel dimension
            ax[i, 0].set_title(f'Label: Original')
            ax[i, 0].axis('off')  # Turn off axis

            # Display second image
            ax[i, 1].imshow(img2, cmap='gray')  # Use squeeze to remove channel dimension
            ax[i, 1].set_title(f'Label: Reconstruction')
            ax[i, 1].axis('off')  # Turn off axis

            img3 = original_labels[i]
            img3 = img3.detach().numpy().transpose(1, 2, 0)
            ax[i, 2].imshow(img3)  # Use squeeze to remove channel dimension
            ax[i, 2].set_title(f'Label: RGB_Image')
            ax[i, 2].axis('off')  # Turn off axis

        plt.show()


def ingame_evaluation():
    model = LitBpsToImgNetwork.load_from_checkpoint(CHECKPOINT_LOCATION).cpu()
    model.eval()
    options = OPTIONS
    options.update({"render": "rgb_array"})
    env = environment_creation(options=OPTIONS)
    _ = env.reset()
    lq = queue.LifoQueue()

    for i in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if lq.qsize() == 3:
            bps_sample = []
            for i in range(3):
                bps_sample.append(lq.get())
            bps_sample = torch.from_numpy(np.array(bps_sample)).unsqueeze(0)
            prediction = model(bps_sample)
            prediction = prediction.detach().numpy()
            prediction = prediction.squeeze()
            # plt.imshow(prediction, cmap="gray")

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(prediction, cmap='gray')
            img = env.render()
            img = img[2::8, 2::8, :]
            ax[1].imshow(img)
            plt.show()
            _ = env.reset()
            continue

        if not terminated:
            lq.put(observation["state"])

        if terminated:
            while not lq.empty():
                lq.get()


def check_dataset():
    img_data_location = "./bps_img_img.pt"
    bps_data_location = "./bps_img_bps.pt"
    dataset = BpsToImgDataset(img_data=img_data_location, bps_data=bps_data_location, img_transform=BinarizeTransform())

    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    for i in range(64):

        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        img = train_labels[i].numpy()
        plt.imshow(img, cmap="gray")
        plt.show()





if __name__ == "__main__":
    #training_lightning()
    #eval_ligthning()
    ingame_evaluation()
    #check_dataset()