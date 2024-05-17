import numpy as np

import gymnasium as gym


def min_max_normalize(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr


class NormalizedObservation(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        obs = min_max_normalize(obs)
        return obs, rews, terminateds, truncateds, infos
