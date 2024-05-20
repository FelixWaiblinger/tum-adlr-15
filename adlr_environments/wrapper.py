"""Environment wrappers"""

import numpy as np
import gymnasium as gym
# from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper


class NormalizeObservationWrapper(gym.Wrapper): #VecEnvWrapper):
    """Normalize bps distance observations into the range of [0, 1]"""

    def __init__(self, env):
        """Create normalization wrapper"""

        super().__init__(env)

    # def reset(self):
    #     return self.venv.reset()

    # def step_async(self, actions):
    #     """Perform one step in the environment"""

    #     obs, reward, done, info = self.venv.step(actions)

    #     obs = (obs - np.min(obs)) / (np.max(obs) - np.min(obs))

    #     return obs, reward, done, info

    def step(self, action):
        """Perform one step in the environment"""

        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = (obs - np.min(obs)) / (np.max(obs) - np.min(obs))

        return obs, reward, terminated, truncated, info

    # def step_wait(self):
    #     return self.venv.step_wait()


class RewardWrapper(gym.Wrapper): #VecEnvWrapper):
    """Customize the reward function of the environment"""

    def __init__(self, env: gym.Env, options: dict | None=None) -> None:
        """Create reward function wrapper"""

        self.r_target = options.get("r_target", 1)
        self.r_collision = options.get("r_collision", -1)
        self.r_time = options.get("r_time", 0)
        self.r_distance = options.get("r_distance", 0)
        super().__init__(env) #, venv.observation_space, venv.action_space)

    # def reset(self):
    #     return self.venv.reset()

    # def step_async(self, actions):
    #     """Perform one step in the environment"""

    #     obs, _, done, info = self.venv.step_async(actions)

    #     reward = 0
    #     reward += self.r_target if info["win"] else 0
    #     reward += self.r_collision if info["collision"] else 0
    #     reward += self.r_time
    #     reward += self.r_distance * info["distance"]

    #     return obs, reward, done, info

    def step(self, action):
        """Perform one step in the environment"""

        obs, _, terminated, truncated, info = self.env.step(action)

        reward = 0
        reward += self.r_target if info["win"] else 0
        reward += self.r_collision if info["collision"] else 0
        reward += self.r_time
        reward += self.r_distance * info["distance"]

        return obs, reward, terminated, truncated, info

    # def step_wait(self):
    #     return self.venv.step_wait()
