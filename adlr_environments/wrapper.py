"""Environment wrappers"""

from typing import Tuple, Any

import torch
import pygame
from pygame import K_UP, K_DOWN, K_LEFT, K_RIGHT # pylint: disable=E0611
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from utils.constants import *
from state_representation import BPS, AutoEncoder


class BPSWrapper(gym.Wrapper):
    """Wrapper to transform pointclouds of obstacles to a Basis Point Set"""

    def __init__(self, env: gym.Env, num_points: int):
        """Create a wrapper that converts a pointcloud of obstacles to a set
        of basis points

        Args:
            ``env``: Environment to wrap
            ``num_points``: Number of basis points to use

        IMPORTANT: This wrapper should be placed before FlattenObservation!
        """
        super().__init__(env)

        self.bps = BPS(num_points=num_points)
        self.observation_space = Dict({
            "agent": Box(-1, 1, shape=(4,), dtype=DTYPE),
            "target": Box(-1, 1, shape=(2,), dtype=DTYPE),
            "state": Box(0, 2*np.sqrt(2), shape=(num_points,), dtype=DTYPE)
        })

    def reset(self, *,
        seed: int=None,
        options: dict=None
    ) -> Tuple[Any, dict]:
        """Reset the environment"""
        obs, info = super().reset(seed=seed, options=options)
        obs = self._encode(obs)

        return obs, info

    def step(self, action):
        """Step the environment once"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._encode(obs)

        return obs, reward, terminated, truncated, info
    
    def _encode(self, obs):
        """Encode the image observation as a latent representation"""
        assert isinstance(obs, dict), \
            f"Type of observation must be an 'OrderedDict', was {type(obs)}!"

        # encode into basis point set
        obs["state"] = self.bps.encode(obs["state"]).astype(DTYPE)

        return obs


class AEWrapper(gym.Wrapper):
    """Wrapper to transform a RGB image to a learned latent representation"""

    def __init__(self, env: gym.Env, model_path: str, transform=None):
        """Create a wrapper that converts a RGB image to a learned latent
        representation

        Args:
            ``env``: Environment to wrap
            ``model_path``: file path of the autoencoder model to use
            ``transform``: optional transforms applied to the image observation

        IMPORTANT: This wrapper should be placed before FlattenObservation!
        """
        super().__init__(env)

        self.ae: AutoEncoder = torch.load(model_path)
        self.transform = transform
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(-1, 1, shape=(4,), dtype=DTYPE),
            "target": gym.spaces.Box(-1, 1, shape=(2,), dtype=DTYPE),
            "state": gym.spaces.Box(
                -1, 1, shape=(self.ae.encoder.latent_size,), dtype=DTYPE
            )
        })

    def reset(self, *,
        seed: int=None,
        options: dict=None
    ) -> Tuple[Any, dict]:
        """Reset the environment"""
        obs, info = super().reset(seed=seed, options=options)

        obs = self._encode(obs)

        return obs, info

    def step(self, action):
        """Step the environment once"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self._encode(obs)

        return obs, reward, terminated, truncated, info
    
    def _encode(self, obs):
        """Encode the image observation as a latent representation"""
        assert isinstance(obs, dict), \
            f"Type of observation must be an 'OrderedDict', was {type(obs)}!"

        # image transformations
        image = obs.pop("image")

        # remove agent from image
        blue = np.all(image == Color.BLUE.value, axis=-1)
        image[blue] = Color.WHITE.value
        
        # remove target from image
        red = np.all(image == Color.RED.value, axis=-1)
        image[red] = Color.WHITE.value

        image = image[2::4, 2::4, :].transpose([2, 0, 1])
        image = torch.from_numpy(np.expand_dims(image, 0))
        if self.transform is not None:
            image = self.transform(image)

        # encode into latent representation
        state: torch.Tensor = self.ae.encoder.forward(image.to(self.ae.device))
        state = state.cpu().detach().numpy()[0].astype(DTYPE)
        obs["state"] = state

        # NOTE: only for debugging
        # reconstruction = self.model.decoder.forward(state)
        # reconstruction = reconstruction.cpu().detach().numpy()[0]
        # reconstruction = reconstruction.transpose([2, 1, 0])
        # self.plot = plt.imshow(reconstruction)
        # plt.show()

        return obs


class RewardWrapper(gym.Wrapper):
    """Customize the reward function of the environment"""

    def __init__(self, env: gym.Env, playmode: bool=False) -> None:
        """Create reward function wrapper"""
        super().__init__(env)

        self.r_target = 10
        self.r_collision = -10
        self.r_time = -0.05
        self.r_wall = -0.1
        self.max_steps = 500 if playmode else MAX_EPISODE_STEPS

    def step(self, action):
        """Perform one step in the environment"""

        obs, _, terminated, truncated, info = self.env.step(action)

        # penalize long simulation times
        r_time = self.r_time * info["timestep"] / self.max_steps

        # reward target reaching and obstacle avoidance and penalize wall hits
        r_win = self.r_target if info["win"] else 0
        r_crash = self.r_collision if info["collision"] else 0
        r_wall = self.r_wall if info["wall_collision"] else 0

        reward = r_time + r_win + r_crash + r_wall

        return obs, reward, terminated, truncated, info


class PlayWrapper(gym.Wrapper):
    """Gamification of our experiments
    The user is able to control the agent using either mouse, keyboard or an
    external controller
    Comparing against the latest benchmark DRL agent is also possible
    """

    def __init__(self, env: gym.Env, control: Input=Input.MOUSE):
        """Create a Play Wrapper for an environment with given input device"""
        super().__init__(env)
        assert control in [Input.MOUSE, Input.KEYBOARD, Input.CONTROLLER, Input.AGENT]
        self.player_pos = None
        self.control = control
        pygame.init() # pylint: disable=no-member

    def reset(self, *,
        seed: int=None,
        options: dict=None
    ) -> Tuple[Any, dict]:
        """Reset the environment"""
        obs, info = super().reset(seed=seed, options=options)
        self.player_pos = obs[:2]
        return obs, info

    def step(self, action):
        """Perform one step in the environment"""
        # player controls the action using a mouse
        if self.control == Input.MOUSE:
            # get mouse position in world coordinates
            mouse_pos = np.array(pygame.mouse.get_pos())
            mouse_pos = (mouse_pos - 256) * (1. / 256)
            # action as clipped direction of player to mouse
            player_action = np.clip(mouse_pos - self.player_pos, -1, 1)

        # player controls the action using a keyboard
        elif self.control == Input.KEYBOARD:
            keys = pygame.key.get_pressed()
            player_action = np.zeros(2)
            # each key adds maximal input to direction vector
            player_action[0] += -1 if keys[K_LEFT] else 0
            player_action[0] += +1 if keys[K_RIGHT] else 0
            player_action[1] += -1 if keys[K_UP] else 0
            player_action[1] += +1 if keys[K_DOWN] else 0

        # player controls the action using a controller
        elif self.control == Input.CONTROLLER:
            stick = pygame.joystick.Joystick(0)
            player_action = np.array([stick.get_axis(0), stick.get_axis(1)])

        # agent controls the action
        elif self.control == Input.AGENT:
            player_action = action

        obs, reward, terminated, truncated, info = self.env.step(player_action)
        self.player_pos = obs[:2]

        return obs, reward, terminated, truncated, info


class HParamCallback(BaseCallback):
    """Saves the hyperparameters at the beginning of training and logs them to
    Tensorboard.
    """

    def __init__(self, env_params : dict = {}, agent_params : dict={}):
        super().__init__()
        self.env_params = env_params
        self.agent_params = agent_params

    def _on_training_start(self) -> None:
        def to_python_type(value):
            if isinstance(value, np.generic):
                return value.item()
            return value

        hparam_dict = {}
        hparam_dict.update(self.env_params)
        hparam_dict.update(self.agent_params)

        #transform dictionary to python types
        hparam_dict = {k: to_python_type(v) for k, v in hparam_dict.items()}

        # define the metrics that will appear in the HPARAMS Tensorboard tab
        # by referencing their tag
        # Tensorboard will find & display metrics from the SCALARS tab
        metric_dict = {
            "rollout/ep_rew_mean": 0.0,
            "rollout/ep_len_mean": 0.0
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
