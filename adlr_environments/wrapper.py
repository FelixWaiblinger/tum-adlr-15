"""Environment wrappers"""

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from adlr_environments.utils import MAX_EPISODE_STEPS, eucl


class NormalizeObservationWrapper(gym.Wrapper):  # VecEnvWrapper):
    """Normalize bps distance observations into the range of [0, 1]"""

    def __init__(self, env):
        """Create normalization wrapper"""

        super().__init__(env)

    def step(self, action):
        """Perform one step in the environment"""

        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = (obs - np.min(obs)) / (np.max(obs) - np.min(obs))

        return obs, reward, terminated, truncated, info


class RewardWrapper(gym.Wrapper):  # VecEnvWrapper):
    """Customize the reward function of the environment"""

    def __init__(self, env: gym.Env, options):  #: dict | None=None) -> None:
        """Create reward function wrapper"""

        self.r_target = options.get("r_target", 1)
        self.r_collision = options.get("r_collision", -1)
        self.r_time = options.get("r_time", 0)
        self.r_distance = options.get("r_distance", 0)
        self.r_wall_collision = options.get("wall_collision", 0)
        super().__init__(env)

    def step(self, action):
        """Perform one step in the environment"""

        obs, _, terminated, truncated, info = self.env.step(action)

        reward = 0
        reward += self.r_target if info["win"] else 0
        reward += self.r_collision if info["collision"] else 0
        reward += self.r_time
        reward += self.r_distance * info["distance"]
        reward += self.r_wall_collision * info["wall_collision"]

        if info["win"]:
            print("winnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
        # if terminated:
        #     print("terminated")
        # if info["collision"]:
        #     print("collision")
        # if info["wall_collision"]:
        #     print("wall_collision")

        return obs, reward, terminated, truncated, info


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters at the beginning of training and logs them to Tensorboard.
    """

    def __init__(self, env_params: dict = {}, agent_params: dict = {}):
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

        # transform dictionary to python types
        hparam_dict = {k: to_python_type(v) for k, v in hparam_dict.items()}

        # define the metrics that will appear in the HPARAMS Tensorboard tab by referencing their tag
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
