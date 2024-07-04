"""Simulation"""

import time

import gymnasium as gym
from utils.constants import Observation

options = {
    "num_static_obstacles": 5,
    "num_dynamic_obstacles": 0,
    "uncertainty": True
}

env = gym.make(
    'World2D-v0',
    render_mode='human',
    observation_type=Observation.RGB,
    options=options
)

observation, info = env.reset(seed=42)

print(observation)
for i in range(300):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    time.sleep(0.1)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

print("Simulation finished!")
