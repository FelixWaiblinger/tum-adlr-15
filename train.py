"""Training"""

import os
import json

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation #, NormalizeObservation
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import adlr_environments  # pylint: disable=unused-import
from adlr_environments.wrapper import RewardWrapper, HParamCallback
from adlr_environments.utils import to_py_dict, linear_schedule, draw_policy


AGENT = "./agents/"
NAME = "test"
RESULT_PATH = "./agents/random_search_results.json"
MODEL_PATH = "./agents/random_search_model"


def environment_creation(num_workers: int=1, options=None): #: dict | None=None):
    """Create environment"""

    def env_factory(render):
        # create environment
        env = gym.make(id="World2D-v0", render_mode=render, options=options)

        # flatten observations
        env = FlattenObservation(env)

        # normalize observations NOTE: tests indicate normalizing is bad
        # env = NormalizeObservation(env)

        # custom reward function
        env = RewardWrapper(env, options)

        return env

    render = options.pop("render")
    fork = options.pop("fork")

    # single/multi threading
    env = make_vec_env(
        env_factory,
        n_envs=num_workers,
        env_kwargs={"render": ("human" if render else "rgb_array")},
        vec_env_cls=DummyVecEnv if num_workers == 1 else SubprocVecEnv,
        # vec_env_kwargs={"start_method": "fork"} if fork else None
    )

    return env


def start_training(
    num_steps: int,
    name: str,
    num_workers: int=1
):
    """Train a new agent from scratch"""

    options = {
        "r_target": 10,
        "r_collision": -10,
        "r_time": -0.01,
        "r_distance": -0.01,
        "world_size": 10,
        "num_static_obstacles": 5,
        "bps_size": 50,
        "fork": True,
        "render": False,
    }
    logger = "./logs/" + NAME
    env = environment_creation(num_workers=num_workers, options=options)
    # NOTE: consider multi input policy in combination with raw pointcloud
    model = PPO("MlpPolicy", env, tensorboard_log=logger,
                batch_size=128, learning_rate=linear_schedule(0.001))
    model.learn(total_timesteps=num_steps, progress_bar=True,
                callback=HParamCallback(env_params=options))
    model.save(AGENT + NAME)


def continue_training(
    num_steps: int,
    name: str,
    new_name: str=None,
    num_workers: int=1
):
    """Resume training aka. perform additional training steps and update an
    existing agent
    """

    if not new_name:
        new_name = name

    options = {
        "r_target": 10,
        "r_collision": -10,
        "r_time": -0.01,
        "r_distance": -0.01,
        "world_size": 8,
        "num_static_obstacles": 3,
        "bps_size": 50,
        "fork": True,
        "render": False,
    }

    # if os.path.exists(RESULT_PATH):
    #     with open(RESULT_PATH, 'r', encoding='utf-8') as f:
    #         parameters = json.load(f)
    #         options.update(parameters)
    # else:
    #     return

    logger = "./logs/" + new_name
    env = environment_creation(num_workers=num_workers, options=options)
    model = PPO.load(name, env=env, tensorboard_log=logger)
    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(new_name)


def evaluate(name: str, num_steps: int=1000):
    """Evaluate a trained agent"""

    options = {
        "r_target": 10,
        "r_collision": -10,
        "r_time": -0.01,
        "r_distance": -0.01,
        "world_size": 10,
        "num_static_obstacles": 5,
        "bps_size": 50,
        "fork": False,
        "render": True,
    }

    # if os.path.exists(RESULT_PATH):
    #     with open(RESULT_PATH, 'r', encoding='utf-8') as f:
    #         parameters = json.load(f)
    #         options.update(parameters)
    # else:
    #     return

    env = environment_creation(num_workers=1, options=options)

    # load the trained agent
    model = PPO.load(name, env)

    rewards, episodes = 0, 0
    obs = env.reset()

    # draw policy
    draw_policy(model, obs, options["world_size"])

    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        rewards += reward
        env.render("human")

        if done:
            episodes += 1

    print(f"Average reward over {episodes} episodes: {rewards / episodes}")


def random_search(
    # search_space: dict,
    num_tests: int=100,
    num_train_steps: int=10000,
    num_workers: int=1
):
    """Select hyperparameters in search space randomly"""

    # specify search space
    # agent_space = {
    #     "learning_rate": [linear_schedule(0.001), linear_schedule(0.01)],
    #     "n_steps": [2048],
    #     "batch_size": [32, 64, 128],
    #     "n_epochs": [5],
    #     "gamma": [0.95, 0.99],
    #     "gae_lambda": [0.95, 0.99],
    # }
    agent_space = {
        "learning_rate": [linear_schedule(0.001), linear_schedule(0.01)],
        "buffer_size": [500000, 1000000],
        "learning_starts": [50, 100, 200],
        "batch_size": [128, 256, 512],
        "tau": [0.01, 0.005, 0.001],
        "gamma": [0.95, 0.99],
        "gradient_steps": [1, 2, 3],
        "policy_delay": [1, 2, 3],
    }
    env_space = {
        "r_target": [100],
        "r_collision": [-10],
        "r_time": [-0.01],
        "r_distance": [-0.01],
        "world_size": [8],
        "num_static_obstacles": [3],
        "bps_size": [15, 20, 25],#, 60, 90],
        "fork": [True],
        "render": [False],
    }
    print(str(agent_space["learning_rate"][0]))
    print(str(agent_space["learning_rate"][1]))

    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, 'r', encoding='utf-8') as f:
            best_parameters = json.load(f)
            best_avg_reward = best_parameters["reward"]
    else:
        best_parameters = {}
        best_avg_reward = float("-inf")

    for i in range(num_tests):
        print(f"Running test {i}...")

        # select hyperparameters at random
        agent_params = {k: np.random.choice(v) for k, v in agent_space.items()}
        env_params = {k: np.random.choice(v) for k, v in env_space.items()}

        # create environment
        env = environment_creation(num_workers=num_workers, options=env_params)

        # create and train an agent
        model = PPO("MlpPolicy", env, **agent_params)

        model.learn(total_timesteps=num_train_steps, progress_bar=True)

        # evaluate trained agent
        rewards = np.zeros(num_workers)
        episodes = np.zeros(num_workers)

        obs = env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            episodes[done] += 1

        avg_reward = np.mean(rewards / episodes)

        # save best result
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_parameters.update(agent_params)
            best_parameters.update(env_params)
            best_parameters.update({"reward": avg_reward})

            best_parameters = to_py_dict(best_parameters)
            print(best_parameters)

            # store results
            model.save(MODEL_PATH)
            with open(RESULT_PATH, 'w', encoding="utf-8") as f:
                json.dump(best_parameters, f)


if __name__ == '__main__':
    # random_search(num_tests=50, num_train_steps=200000, num_workers=6)

    # start_training(num_steps=500000, name=NAME, num_workers=8)

    # continue_training(
    #     num_steps=500000,
    #     name=AGENT + NAME,
    #     num_workers=8
    # )

    evaluate(AGENT + NAME)
