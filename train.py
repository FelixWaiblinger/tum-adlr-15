"""Training"""

import os
import json

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, NormalizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import adlr_environments  # pylint: disable=unused-import
from adlr_environments.wrapper import RewardWrapper, HParamCallback
from adlr_environments.utils import to_py_dict, linear_schedule, draw_policy


AGENT = "./agents/"
NAME = "ppo_static_obstacles"
RESULT_PATH = "./agents/random_search_results.json"
MODEL_PATH = "./agents/random_search_model"


def environment_creation(num_workers: int=1, options: dict | None=None):
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
        vec_env_kwargs={"start_method": "fork"} if fork else None
    )

    return env


def start_training(
    num_steps: int,
    name: str,
    num_workers: int=1
):
    """Train a new agent from scratch"""

    options = {
        "r_target": 500,
        "r_collision": -10,
        "r_time": -0.005,
        "r_distance": -0.01,
        "bps_size": 75
    }
    logger = "./logs/" + name
    env = environment_creation(num_workers=num_workers, options=options)
    # NOTE: consider multi input policy in combination with raw pointcloud
    model = SAC("MlpPolicy", env, tensorboard_log=logger)
    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(AGENT + name)


def continue_training(
    num_steps: int,
    name: str,
    new_name: str=None,
    num_workers: int=1
):
    """Resume training aka. perform additional training steps and update an
    existing agent
    """

    options = {
        "r_target": 500,
        "r_collision": -10,
        "r_time": -0.005,
        "r_distance": -0.01,
        "world_size": 5,
        "num_static_obstacles": 1,
        "bps_size": 5,
        "fork": True,
        "render": False,
    }

    if not new_name:
        new_name = name

    logger = "./logs/" + new_name
    env = environment_creation(num_workers=num_workers, options=options)
    model = PPO.load(name, env=env, tensorboard_log=logger)
    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(new_name)


def evaluate(num_steps: int=1000):
    """Evaluate a trained agent"""

    options = {
        "r_target": 100,
        "r_collision": -50,
        "r_time": -0.01,
        "r_distance": -0.01,
        "world_size": 5,
        "num_static_obstacles": 1,
        "bps_size": 5,
        "fork": False,
        "render": True,
    }

    env = environment_creation(num_workers=1, options=options)

    # load the trained agent
    model = PPO.load(MODEL_PATH + "_resume", env)

    rewards, episodes = 0, 0
    obs = env.reset()

    # draw policy
    draw_policy(model, obs, options["world_size"])

    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        rewards += reward
        env.render()

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
    agent_space = {
        # "algorithm": [PPO], #, A2C, DDPG, SAC, TD3],
        "learning_rate": [linear_schedule(0.001)],
        "n_steps": [1024, 2048, 4096],
        "batch_size": [32, 64, 128],
        "n_epochs": [5, 10, 20],
        "gamma": [0.95, 0.99, 1],
        "gae_lambda": [0.9, 0.95, 0.99],
    }
    env_space = {
        "r_target": [100],
        "r_collision": [-50],
        "r_time": [-0.01],
        "r_distance": [-0.01],
        "world_size": [5],
        "num_static_obstacles": [1],
        "bps_size": [5],#, 60, 90],
        "fork": [True],
        "render": [False],
    }

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

        try:
            model.learn(total_timesteps=num_train_steps, progress_bar=True)
        except EOFError:
            print("EOF occurred!")

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
    # perform random search
    # random_search(num_tests=5, num_train_steps=200000, num_workers=8)

    # start_training(num_steps=100000, name=NAME)
    evaluate()
    # continue_training(num_steps=1000000, name=MODEL_PATH, new_name=MODEL_PATH + "_resume", num_workers=8)
