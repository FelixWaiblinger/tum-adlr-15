import matplotlib.pyplot as plt

from train import environment_creation
from adlr_environments.constants import OPTIONS
import torch
import numpy as np
from utils import create_env
from utils.config import BPS_CONFIG

IMG_DATA_PATH = "./bps_img_img_70k"
BPS_DATA_PATH = "./bps_img_bps_70k"
CONFIG = BPS_CONFIG


def record_dataset(num_samples: int, timesteps: int):
    """
    Goal: from the k timesteps use the bps to predict an occupancy map of the field
    1. run the simulation
    2. collect k timesteps and the image from the k+1 timestep
        2.1 be careful with resets:
            a. when one of the elements between [0-k+1] resets the environment then delete an restart collection
        2.2 saved images have to be without agent and target
    """

    options = OPTIONS
    options.update({"render": "rgb_array"})

    env = create_env(
        wrapper=CONFIG.wrapper,
        render=True,
        obs_type=CONFIG.observation,
        num_workers=1,
        options=CONFIG.env
    )

    bps_dataset = []
    img_dataset = []  # for k bps timesteps we have 1 image
    _ = env.reset()
    bps_sample = []
    i = 0
    while i < num_samples:
        print(f"\rGenerating dataset: {(i / num_samples) * 100:3.2f}%", end="")

        # generate samples by stepping through environment
        action = env.action_space.sample()
        action = np.array([[action[0], action[1]], [action[0], action[1]]])
        observation, reward, done, info = env.step(action)
        img = observation["image"][0]

        if len(bps_sample) == timesteps:
            # when k timesteps where collected without termination we can save the k+1 image

            # plt.imshow(img)
            # plt.show()
            # save original image(512 x 512) at (64 x 64)
            img = img[2::8, 2::8, :].transpose([2, 0, 1])
            img_dataset.append(torch.from_numpy(img))
            bps_dataset.append(torch.from_numpy(np.array(bps_sample)))
            bps_sample = []
            i += 1
            _ = env.reset()
            continue

        if not done and len(bps_sample) < timesteps:
            bps_sample.append(observation["state"][0])

        if done:
            bps_sample = []  # reset list to 0 elements if environment terminated

    img_dataset = torch.stack(img_dataset)
    bps_dataset = torch.stack(bps_dataset)
    print("")
    print(f"The image dataset has the following shape: {img_dataset.size()}")
    print(f"The image dataset has the following shape: {bps_dataset.size()}")
    torch.save(img_dataset, IMG_DATA_PATH + ".pt")
    torch.save(bps_dataset, BPS_DATA_PATH + ".pt")


if __name__ == "__main__":
    record_dataset(num_samples=50000, timesteps=3)
