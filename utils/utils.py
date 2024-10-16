"""Utility functions"""

from typing import Callable, Dict
from argparse import ArgumentParser

import numpy as np
import gymnasium as gym
from tqdm.auto import tqdm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from utils.constants import Observation


def create_env(
    wrapper: list=None,
    render: bool=False,
    obs_type: Observation=Observation.POS,
    num_workers: int=1,
    options: Dict=None
):
    """Create a vectorized environment
    
    Args:
        ``wrapper``: list of (wrapper, kwargs_dict)
        ``render``: render flag
        ``obs_type``: type of observations (position, image or all)
        ``num_workers``: number of parallel environments
        ``options``: dict of environment options
    """

    def env_factory(id, render):
        env = gym.make(
            id=id,
            render_mode=render,
            observation_type=obs_type,
            options=options
        )

        for wrap, kwargs in wrapper:
            env = wrap(env, **kwargs)

        return env

    fork = options.pop("fork", False)
    env_name = options.pop("env", None)

    # single/multi threading
    env = make_vec_env(
        env_factory,
        n_envs=num_workers,
        env_kwargs={
            "id": ("World2D-v0" if env_name is None else env_name),
            "render": ("human" if render else "rgb_array"),
        },
        vec_env_cls=DummyVecEnv if num_workers == 1 else SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"} if fork else {}
    )

    return env


def eucl(x: np.ndarray, y: np.ndarray) -> float:
    """Return Euclidean distance between positions x and y"""
    return np.linalg.norm(x - y, ord=2)


def linear_schedule(initial_value: float, slope: float=1) -> Callable[[float], float]:
    """Linear learning rate schedule (current learning rate depending on
    remaining progress)
    """

    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0"""
        return initial_value * (1 - slope + slope * progress_remaining)

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


def arg_parse(arguments: list):
    """Parse commandline arguments"""
    parser = ArgumentParser()
    for flags, arg_type, dft in arguments:
        if arg_type is None and dft is None:
            parser.add_argument(*flags, action="store_true")
        else:
            parser.add_argument(*flags, type=arg_type, default=dft)

    args = parser.parse_args()
    return args


def create_tqdm_bar(iterable, desc):
    """Create a progress bar"""
    return tqdm(enumerate(iterable),total=len(iterable), ncols=150, desc=desc)


__all__ = [
    "create_env",
    "eucl",
    "linear_schedule",
    "to_py_dict",
    "arg_parse",
    "create_tqdm_bar",
]