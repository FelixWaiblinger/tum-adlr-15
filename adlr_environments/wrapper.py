"""Environment wrappers"""

from typing import Tuple, Dict, Any
import pygame
from pygame import K_UP, K_DOWN, K_LEFT, K_RIGHT # pylint: disable=E0611
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from adlr_environments.constants import MAX_EPISODE_STEPS, Input


class RewardWrapper(gym.Wrapper):
    """Customize the reward function of the environment"""

    def __init__(self, env: gym.Env, playmode: bool=False) -> None:
        """Create reward function wrapper"""
        super().__init__(env)

        self.r_target = 10
        self.r_collision = -10
        self.r_time = -0.05
        self.r_distance = 2
        self.r_wall = -0.1
        self.max_steps = 500 if playmode else MAX_EPISODE_STEPS

    def step(self, action):
        """Perform one step in the environment"""

        obs, _, terminated, truncated, info = self.env.step(action)

        #######################################################################
        # PPO reward function
        #######################################################################
        # # reward minimizing distance to target
        # r_dist = np.exp(-info["distance"])
        # r_dist = self.r_distance * np.clip(r_dist, 0, self.r_target * 0.1)

        # # reward maximizing distance to obstacles
        # r_obs = -np.exp(-info["obs_distance"])
        # r_obs = self.r_distance * np.clip(r_obs, self.r_collision * 0.1, 0)

        # # scale distance rewards by simulation time
        # time_factor = np.exp(-0.01 - info["timestep"] / MAX_EPISODE_STEPS)
        # time_factor = self.r_time * time_factor

        # # reward target reaching and obstacle avoidance
        # r_win = self.r_target if info["win"] else 0
        # r_crash = self.r_collision if info["collision"] else 0

        # reward = ((r_dist + r_obs) * time_factor) + r_win + r_crash
        #######################################################################

        #######################################################################
        # SAC reward function
        #######################################################################
        # penalize long simulation times
        r_time = self.r_time * info["timestep"] / self.max_steps

        # reward target reaching and obstacle avoidance and penalize wall hits
        r_win = self.r_target if info["win"] else 0
        r_crash = self.r_collision if info["collision"] else 0
        r_wall = self.r_wall if info["wall_collision"] else 0

        reward = r_time + r_win + r_crash + r_wall
        #######################################################################

        return obs, reward, terminated, truncated, info


class PlayWrapper(gym.Wrapper):
    """Player"""

    def __init__(self, env: gym.Env, control: Input=Input.MOUSE):
        super().__init__(env)
        assert control in [Input.MOUSE, Input.KEYBOARD, Input.JOYSTICK, Input.AGENT]
        self.player_pos = None
        self.control = control
        pygame.init() # pylint: disable=no-member

    def reset(self, *,
        seed: int=None,
        options: Dict[str, Any]=None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment"""
        obs, info = super().reset(seed=seed, options=options)
        self.player_pos = obs[:2]
        return obs, info

    def step(self, action):
        """Perform one step in the environment"""
        if self.control == Input.MOUSE:
            # get mouse position in world coordinates
            mouse_pos = np.array(pygame.mouse.get_pos())
            mouse_pos = (mouse_pos - 256) * (1. / 256)

            # action as clipped direction of player to mouse
            player_action = np.clip(mouse_pos - self.player_pos, -1, 1)

        elif self.control == Input.KEYBOARD:
            keys = pygame.key.get_pressed()
            player_action = np.zeros(2)
            player_action[0] += -1 if keys[K_LEFT] else 0
            player_action[0] += +1 if keys[K_RIGHT] else 0
            player_action[1] += -1 if keys[K_UP] else 0
            player_action[1] += +1 if keys[K_DOWN] else 0

        elif self.control == Input.JOYSTICK:
            stick = pygame.joystick.Joystick(0)
            player_action = np.array([stick.get_axis(0), stick.get_axis(1)])
        elif self.control == Input.AGENT:
            player_action = action

        obs, reward, terminated, truncated, info = self.env.step(player_action)
        self.player_pos = obs[:2]

        return obs, reward, terminated, truncated, info


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters at the beginning of training and logs them to Tensorboard.
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
