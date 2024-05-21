"""Training"""

import json
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv #, SubprocVecEnv
import adlr_environments  # pylint: disable=unused-import
from adlr_environments.wrapper import NormalizeObservationWrapper, RewardWrapper, HParamCallback

# scenario_name = "ppo_static_obstacles_200k_normalized"
AGENT = "./agents/"
NAME = "ppo_no_obstacles"
RESULT_PATH = "./agents/random_search_results.json"


def environment_creation(num_workers: int=1, options: dict | None=None):
    """Create environment"""

    def env_factory():
        # create environment
        env = gym.make(id="World2D-v0", render_mode="rgb_array", options=options)

        # flatten observations
        env = FlattenObservation(env)

        # normalize observations
        env = NormalizeObservationWrapper(env)

        # custom reward function
        env = RewardWrapper(env, options)

        # check_env(env)

        return env

    # single/multi threading
    env = make_vec_env(
        env_factory,
        n_envs=num_workers,
        vec_env_cls=DummyVecEnv # if num_workers == 1 else SubprocVecEnv
    )

    return env


def start_training(
    num_steps: int,
    name: str,
    num_workers: int=1
):
    """Train a new agent from scratch"""

    logger = "./logs/" + name
    env = environment_creation(num_workers=num_workers)
    # NOTE: consider multi input policy in combination with raw pointcloud
    model = PPO("MlpPolicy", env, tensorboard_log=logger)
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

    env = environment_creation(num_workers=num_workers)
    # load the trained agent
    model = PPO.load(AGENT, env)

    obs = env.reset()
    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render("human")


def random_search(
    search_space: dict,
    num_tests: int=100,
    num_train_steps: int=10000
) -> dict:
    """Select hyperparameters in search space randomly"""

    results = {
        "best_parameters": {},
        "best_avg_reward": float("-inf")
    }

    for _ in range(num_tests):
        # select hyperparameters at random
        parameters = {k: np.random.choice(v) for k, v in search_space.items()}
        print("these are the current parameters : ", parameters)

        # create environment
        env = environment_creation(num_workers=8, options=parameters)

        # create and train an agent
        logger = "./logs/" + NAME
        model = parameters["algorithm"]("MlpPolicy", env, tensorboard_log=logger)
        model.learn(total_timesteps=num_train_steps, progress_bar=True, callback=HParamCallback(parameters))

        # evaluate trained agent
        episode_rewards = [0]
        obs = env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)

        #     if done:
        #         episode_rewards.append(reward)
        #     else:
        #         episode_rewards[-1] += reward
        #
        # avg_reward = np.mean(episode_rewards)

        # save best result
        # if avg_reward > results["best_avg_reward"]:
        #     results["best_avg_reward"] = avg_reward
        #     results["best_parameters"] = parameters

    return results


if __name__ == '__main__':
    # specify search space
    space = {
        "algorithm": [PPO, A2C, DDPG, SAC, TD3],
        "r_target": [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
        "r_collision": [-1, -2, -5, -10, -25, -50, -100, -250, -500, -1000],
        "r_time": [-0.5, -0.1, -0.05, -0.01, -0.005, -0.001, -0.0005, -0.0001],
        "r_distance": [0.001, -0.1, -0.05, -0.01, -0.005, -0.001, -0.0005, -0.0001, -0.00001],
        "bps_size": [10, 20, 30, 50, 75, 100, 150, 200]
    }

    # perform random search
    best_agent = random_search(space, num_tests=10, num_train_steps=10000)
    best_agent["best_parameters"] = \
        {k: str(v) for k, v in best_agent["best_parameters"].items()}

    print(best_agent)

    # store results
    with open(RESULT_PATH, 'w', encoding="utf-8") as f:
        json.dump(best_agent, f)


# start_training(num_steps=300000, name=NAME)
#evaluate_environment()


# continue_training(training_steps=200000, old_name="ppo_static_obstacles_200k_adapted_loss",
#                   new_name="ppo_static_obstacles_200k_adapted_loss_continued")
