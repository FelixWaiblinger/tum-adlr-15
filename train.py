"""Training"""

import json
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import adlr_environments  # pylint: disable=unused-import
from adlr_environments.wrapper import NormalizeObservationWrapper, RewardWrapper, HParamCallback

AGENT = "./agents/"
NAME = "ppo_static_obstacles"
RESULT_PATH = "./agents/random_search_results.json"
MODEL_PATH = "./agents/random_search_model"


def environment_creation(num_workers: int=1, options: dict | None=None):
    """Create environment"""

    def env_factory():
        # create environment
        env = gym.make(id="World2D-v0", render_mode="rgb_array", options=options)

        # flatten observations
        env = FlattenObservation(env)

        # normalize observations
        # env = NormalizeObservationWrapper(env)

        # custom reward function
        env = RewardWrapper(env, options)

        return env

    # check_env(env)

    # single/multi threading
    env = make_vec_env(
        env_factory,
        n_envs=num_workers,
        vec_env_cls=DummyVecEnv if num_workers == 1 else SubprocVecEnv
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

    if not new_name:
        new_name = name

    logger = "./logs/" + new_name
    env = environment_creation(num_workers=num_workers)
    model = PPO.load(AGENT + name, env=env, tensorboard_log=logger)
    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(AGENT + new_name)
    # wrap the environment into observation wrapper


def evaluate(num_steps: int=1000, num_workers: int=1):
    """Evaluate a trained agent"""

    options = {
        "r_target": 500,
        "r_collision": -10,
        "r_time": -0.005,
        "r_distance": -0.01,
        "bps_size": 75,
    }
    env = environment_creation(num_workers=num_workers, options=options)
    # load the trained agent
    model = SAC.load(AGENT + NAME, env)

    rewards, episodes = 0, 0
    obs, _ = env.reset()
    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        rewards += reward
        env.render()#"human")

        if done or truncated:
            obs, _ = env.reset()
            episodes += 1

    print(f"Average reward over {episodes} episodes: {rewards / episodes}")


def random_search(
    # search_space: dict,
    num_tests: int=100,
    num_train_steps: int=10000,
    num_workers: int=1
) -> dict:
    """Select hyperparameters in search space randomly"""

    # specify search space
    agent_space = {
        # "algorithm": [PPO], #, A2C, DDPG, SAC, TD3],
        "learning_rate": [0.001, 0.0003, 0.0001],
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
        "r_distance": [-0.005],
        "bps_size": [30, 60, 90],
    }

    best_parameters = {}
    best_avg_reward = float("-inf")

    def clean(d: dict):
        result = {}
        for k, v in d.items():
            if isinstance(v, np.float64):
                result[k] = float(v)
            if isinstance(v, np.int64):
                result[k] = int(v)
            else:
                result[k] = float(v)
        return result

    for i in range(num_tests):
        print(f"Running test {i}...")
        # select hyperparameters at random
        agent_params = {k: np.random.choice(v) for k, v in agent_space.items()}
        env_params = {k: np.random.choice(v) for k, v in env_space.items()}

        # create environment
        env = environment_creation(num_workers=num_workers, options=env_params)

        # create and train an agent
        model = PPO("MlpPolicy", env, **agent_params)
        model.learn(total_timesteps=num_train_steps)#, progress_bar=True)

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

            best_parameters = clean(best_parameters)
            print(best_parameters)

            # store results
            model.save(MODEL_PATH)
            with open(RESULT_PATH, 'w', encoding="utf-8") as f:
                json.dump(best_parameters, f)


if __name__ == '__main__':
    # perform random search
    random_search(num_tests=100, num_train_steps=300000, num_workers=10)

    # start_training(num_steps=100000, name=NAME)
    # evaluate()
