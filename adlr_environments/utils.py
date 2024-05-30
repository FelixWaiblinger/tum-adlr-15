"""Utility functions"""

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


MAX_EPISODE_STEPS = 200


def draw_policy(
    agent,
    observation: np.ndarray,
    world_size: float,
    resolution: int=10
) -> None:
    """Plot a vector field representing the actions chosen by the policy for
    any position in the environment
    """

    # start arrows in the middle of "their box"
    offset = 0.5 * (world_size / resolution)
    for i in np.linspace(offset, world_size - offset, resolution):
        for j in np.linspace(offset, world_size - offset, resolution):
            observation[0, :2] = np.array([i, j], dtype=np.float32)
            target, _ = agent.predict(observation, deterministic=True)
            target = 0.5 * target[0] # shorten for visibility
            plt.arrow(i, j, target[0], -target[1], head_width=0.1)
    plt.show()


def eucl(x: np.ndarray, y: np.ndarray) -> float:
    """Return euclidean distance between positions x and y"""
    return np.linalg.norm(x - y, ord=2)


def linear(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule (current learning rate depending on
    remaining progress)
    """

    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0"""

        return progress_remaining * initial_value

    return func


def to_py_dict(dictionary: dict):
    """Convert values in dictionary to python built-in types"""

    result = {}
    for k, v in dictionary.items():
        if isinstance(v, np.float64):
            result[k] = float(v)
        elif isinstance(v, np.int64):
            result[k] = int(v)
        elif isinstance(v, Callable):
            result[k] = str(v)
        else:
            result[k] = float(v)
    return result
