"""Environment wrappers"""

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from adlr_environments.constants import MAX_EPISODE_STEPS


class RewardWrapper(gym.Wrapper):
    """Customize the reward function of the environment"""

    def __init__(self, env: gym.Env) -> None:
        """Create reward function wrapper"""
        super().__init__(env)

        self.r_target = 10
        self.r_collision = -10
        self.r_time = -0.05
        self.r_distance = 2
        self.r_wall = -0.1

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
        r_time = self.r_time * info["timestep"] / MAX_EPISODE_STEPS

        # reward target reaching and obstacle avoidance and penalize wall hits
        r_win = self.r_target if info["win"] else 0
        r_crash = self.r_collision if info["collision"] else 0
        r_wall = self.r_wall if info["wall_collision"] else 0

        reward = r_time + r_win + r_crash + r_wall
        #######################################################################

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
