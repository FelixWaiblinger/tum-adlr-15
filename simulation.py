"""Simulation"""

import gymnasium as gym
import adlr_environments # pylint: disable=unused-import
from stable_baselines3.common.env_checker import check_env

options = {
    "num_static_obstacles": 5,
    "num_dynamic_obstacles": 0,
    "world_size" : 5
}

env = gym.make('World2D-v0', render_mode='human', options=options)
check_env(env)


observation, info = env.reset(seed=42)

print(observation)
for i in range(300):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        observation, info = env.reset()
env.close()

print("Simulation finished!")
