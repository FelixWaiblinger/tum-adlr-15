"""Simulation"""

import gymnasium as gym
import adlr_environments # pylint: disable=unused-import
from adlr_environments.constants import Observation

options = {
    "num_static_obstacles": 5,
    "num_dynamic_obstacles": 0,
}

env = gym.make('World2D-v0', render_mode='human', observation_type=Observation.RGB, options=options)

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
