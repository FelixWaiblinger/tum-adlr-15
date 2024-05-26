"""Environment wrappers"""

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class NormalizeObservationWrapper(gym.Wrapper): #VecEnvWrapper):
    """Normalize bps distance observations into the range of [0, 1]"""

    def __init__(self, env):
        """Create normalization wrapper"""

        super().__init__(env)

    def step(self, action):
        """Perform one step in the environment"""

        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = (obs - np.min(obs)) / (np.max(obs) - np.min(obs))

        return obs, reward, terminated, truncated, info


class RewardWrapper(gym.Wrapper): #VecEnvWrapper):
    """Customize the reward function of the environment"""

    def __init__(self, env: gym.Env, options): #: dict | None=None) -> None:
        """Create reward function wrapper"""

        self.r_target = options.get("r_target", 1)
        self.r_collision = options.get("r_collision", -1)
        self.r_time = options.get("r_time", 0)
        self.r_distance = options.get("r_distance", 0)
        super().__init__(env)

    def step(self, action):
        """Perform one step in the environment"""

        obs, _, terminated, truncated, info = self.env.step(action)

        reward = 0
        reward += self.r_target if info["win"] else 0
        reward += self.r_collision if info["collision"] else 0
        reward += self.r_time
        reward += self.r_distance * info["distance"]

        return obs, reward, terminated, truncated, info


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters at the beginning of training and logs them to Tensorboard.
    """

    def __init__(self, params_dictionary):
        super().__init__()
        self.params_dictionary = params_dictionary

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "r_target": int(self.params_dictionary["r_target"]),
            "r_collision": int(self.params_dictionary["r_collision"]),
            "r_time": float(self.params_dictionary["r_time"]),
            "r_distance": float(self.params_dictionary["r_distance"]),
            "bps_size": int(self.params_dictionary["bps_size"])

        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorboard will find & display metrics from the `SCALARS` tab
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
