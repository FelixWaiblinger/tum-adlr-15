"""Training"""

import os
import json
import time
from typing import Dict
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, NormalizeObservation
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from adlr_environments.wrapper import RewardWrapper, HParamCallback, NormalizeObservationWrapper
from adlr_environments.utils import to_py_dict, linear, draw_policy
from adlr_environments.constants import OPTIONS, AGENT, AGENT_PATH, RESULT_PATH, LOG_PATH, MODEL_PATH


def environment_creation(num_workers: int = 1, options: Dict = None,
                         vector_environment: bool = False):
    """Create environment"""

    def env_factory(render_mode: str = None):
        env = gym.make(id="World2D-v0", render_mode=render_mode, options=options)
        env = FlattenObservation(env)
        env = NormalizeObservationWrapper(env)
        env = RewardWrapper(env, options)
        return env

    if vector_environment:
        render = options["render"]
        environment = make_vec_env(
            env_factory,
            n_envs=num_workers,
            env_kwargs={"render_mode": ("human" if render else "rgb_array")},
            vec_env_cls=DummyVecEnv if num_workers == 1 else SubprocVecEnv,
        )
        return environment

    else:
        render = options["render"]
        render = "human" if render else "rgb_array"
        environment = env_factory(render)
        return environment


def start_training(
        num_steps: int,
        num_workers: int = 1,
        vector_environment: bool = False
) -> None:
    """Train a new agent from scratch"""

    env = environment_creation(num_workers=num_workers, options=OPTIONS, vector_environment=False)
    model = SAC("MlpPolicy", env, tensorboard_log=LOG_PATH)  # learning_rate=linear(0.001))
    model.learn(total_timesteps=num_steps, progress_bar=True, callback=HParamCallback(env_params=OPTIONS))
    model.save(AGENT_PATH)


def continue_training(
        num_steps: int,
        new_name: str = None,
        num_workers: int = 1
) -> None:
    """Resume training aka. perform additional training steps and update an
    existing agent
    """

    if not new_name:
        new_name = AGENT_PATH

    options = OPTIONS
    options.update({"fork": True, "render": False})
    logger = "./logs/" + new_name

    env = environment_creation(num_workers=num_workers, options=options)
    model = PPO.load(AGENT_PATH, env=env, tensorboard_log=logger)
    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(new_name)


def evaluate(name: str, num_steps: int = 1000) -> None:
    """Evaluate a trained agent"""

    options = OPTIONS
    options.update({"render": True})

    env = environment_creation(num_workers=1, options=options, vector_environment=False)
    model = SAC.load(name, env)

    rewards, episodes, wins, crashes, stuck = 0, 0, 0, 0, 0
    observation, _ = env.reset()

    # draw_policy(model, observation, options["world_size"])

    for _ in range(num_steps):
        action, test = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        env.render()

        if terminated:
            observation, info = env.reset()
            episodes += 1
            if info["win"]:
                wins += 1
            elif info["collision"]:
                crashes += 1
            else:
                stuck += 1

        time.sleep(0.05)

    print(f"Average reward over {episodes} episodes: {rewards / episodes}")
    print(f"Successrate: {100 * (wins / episodes):.2f}%")
    print(f"Crashrate: {100 * (crashes / episodes):.2f}%")
    print(f"Stuckrate: {100 * (stuck / episodes):.2f}%")


def random_search(
        # search_space: dict,
        num_tests: int = 100,
        num_train_steps: int = 10000,
        num_workers: int = 1
) -> None:
    """Select hyperparameters in search space randomly"""

    # specify search space
    # agent_space = {
    #     "learning_rate": [linear(0.001), linear(0.01)],
    #     "n_steps": [2048],
    #     "batch_size": [32, 64, 128],
    #     "n_epochs": [5],
    #     "gamma": [0.95, 0.99],
    #     "gae_lambda": [0.95, 0.99],
    # }
    agent_space = {
        "learning_rate": [linear(0.001), linear(0.01)],
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
        "bps_size": [15, 20, 25],  # , 60, 90],
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

    start_training(num_steps=1000000, num_workers=6, vector_environment=False)

    # continue_training(num_steps=1000000, num_workers=8)

    #evaluate(AGENT_PATH)
