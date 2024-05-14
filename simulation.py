"""Simulation"""

import gymnasium as gym
import adlr_environments # pylint: disable=unused-import

options = {
    "num_static_obstacles": 20,
    "num_dynamic_obstacles": 3,
    "min_size": 0.8,
    "max_size": 3
}

env = gym.make('World2D-v0', render_mode='human') #, options=options)

observation, info = env.reset(seed=42)

print(observation)
for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

print("Simulation finished!")
