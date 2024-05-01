"""Simulation"""

import gymnasium as gym
import adlr_environments # pylint: disable=unused-import

env = gym.make('World2D-v0', render_mode='human')
options = {
    "num_static_obstacles": 5,
    "num_dynamic_obstacles": 2,
    "min_size": 0.8,
    "max_size": 2
}

observation, info = env.reset(seed=42, options=options)

print(observation)
for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset(options=options)
env.close()

print("Simulation finished!")
