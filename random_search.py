"""Random search for parameter optimization"""

import os
import json

import numpy as np
from numpy.random import choice
from stable_baselines3 import SAC
from gymnasium.wrappers import FlattenObservation, FrameStack

from adlr_environments.wrapper import RewardWrapper
from utils import create_env, to_py_dict, linear


RESULT_PATH = "./agents/random_search_results.json"
MODEL_PATH = "./agents/random_search_model"

NUM_TESTS: int=100,
TRAIN_STEPS: int=100000,
N_WORKERS: int=8

WRAPPER = [
    (FlattenObservation, {}),
    (FrameStack, {"num_stack": 3}),
    (RewardWrapper, {})
]

AGENT_SPACE_PPO = {
    "learning_rate": [linear(0.001), linear(0.01)],
    "n_steps": [2048],
    "batch_size": [32, 64, 128],
    "n_epochs": [5],
    "gamma": [0.95, 0.99],
    "gae_lambda": [0.95, 0.99],
}
AGENT_SPACE_SAC = {
    "learning_rate": [linear(0.001), linear(0.01)],
    "buffer_size": [500000, 1000000],
    "learning_starts": [50, 100, 200],
    "batch_size": [128, 256, 512],
    "tau": [0.01, 0.005, 0.001],
    "gamma": [0.95, 0.99],
    "gradient_steps": [1, 2, 3],
    "policy_delay": [1, 2, 3],
}
ENV_SPACE = {
    "r_target": [100],
    "r_collision": [-10],
    "r_time": [-0.01],
    "r_distance": [-0.01],
    "num_static_obstacles": [3],
    "bps_size": [15, 20, 25],#, 60, 90],
    "fork": [True],
}

if os.path.exists(RESULT_PATH):
    with open(RESULT_PATH, 'r', encoding='utf-8') as f:
        best_parameters = json.load(f)
        best_avg_reward = best_parameters["reward"]
else:
    best_parameters = {}
    best_avg_reward = float("-inf")

for i in range(NUM_TESTS):
    print(f"Running test {i}...")

    # select hyperparameters at random
    agent_params = {k: choice(v) for k, v in AGENT_SPACE_SAC.items()}
    env_params = {k: choice(v) for k, v in ENV_SPACE.items()}

    # create environment
    env = create_env(
        wrapper=WRAPPER,
        render=False,
        num_workers=N_WORKERS,
        options=env_params
    )

    # create and train an agent
    model = SAC("MlpPolicy", env, **agent_params)

    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)

    # evaluate trained agent
    rewards = np.zeros(N_WORKERS)
    episodes = np.zeros(N_WORKERS)

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
