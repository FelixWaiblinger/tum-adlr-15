"""Training"""

# import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import FlattenObservation
import adlr_environments  # pylint: disable=unused-import


AGENT = "./agents/ppo_static_obstacles"


def train_environment(training_steps):
    # create environment
    env = gym.make(id='World2D-v0', render_mode='rgb_array')

    # wrap the environment into observation wrapper
    env = FlattenObservation(env)

    # check if environment satisfies stable baselines3
    check_env(env=env)

    # instantiate the agent
    model = PPO("MlpPolicy", env)

    # train the agent
    model.learn(total_timesteps=training_steps, progress_bar=True)

    # save the agents behaviour
    model.save(AGENT)


def evaluate_environment():
    # create environment
    env = gym.make(id='World2D-v0', render_mode='rgb_array')

    # wrap the environment into observation wrapper
    env = FlattenObservation(env)

    # load the trained agent
    model = PPO.load(AGENT, env=env)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")


# train_environment(100000)
evaluate_environment()
