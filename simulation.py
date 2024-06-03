"""Simulation"""

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from adlr_environments.constants import OPTIONS


env = gym.make('World2D-v0', render_mode='human', options=OPTIONS)
check_env(env)

observation, info = env.reset(seed=42)
print("Observation:", observation)
print("Info:", info)
for i in range(300):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        observation, info = env.reset()
env.close()

print("Simulation finished!")
