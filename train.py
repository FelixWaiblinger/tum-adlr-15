"""Training"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import adlr_environments # pylint: disable=unused-import


env = gym.make('World2D-v0', render_mode='rgb_array')
check_env(env)

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=1000, progress_bar=True)
model.save("./agent")
