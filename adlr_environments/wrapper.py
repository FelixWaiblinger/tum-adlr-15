"""Environment wrappers"""

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from adlr_environments.utils import MAX_EPISODE_STEPS, eucl


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
        self.pos_buffer = []
        self.buffer_length = 10
        super().__init__(env)

    def step(self, action):
        """Perform one step in the environment"""

        obs, _, terminated, truncated, info = self.env.step(action)

        # # store most recent 10 agent positions
        # pos = obs[:2]
        # self.pos_buffer.append(pos)
        # if len(self.pos_buffer) > self.buffer_length:
        #     self.pos_buffer.pop(0)
        
        # # penalize low activity
        # r_active = 0.1 * sum(-1 for p in self.pos_buffer if eucl(p, pos) < 0.1)

        # reward minimizing distance to target
        r_dist = np.exp(-info["distance"])
        r_dist = self.r_distance * np.clip(r_dist, 0, self.r_target * 0.2)

        # reward maximizing distance to obstacles
        r_obs = -np.exp(-info["obs_distance"])
        r_obs = self.r_distance * np.clip(r_obs, 0, -self.r_collision * 0.9)

        # scale distance rewards by simulation time
        time_factor = np.exp(-0.1 - 2 * info["timestep"] / MAX_EPISODE_STEPS)
        time_factor = self.r_time * time_factor

        # reward target reaching and obstacle avoidance
        r_win = self.r_target if info["win"] else 0
        r_crash = self.r_collision if info["collision"] else 0

        reward = ((r_dist + r_obs) * time_factor) + r_win + r_crash #+ r_active

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
