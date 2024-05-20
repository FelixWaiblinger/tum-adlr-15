"""Environment registration"""

from gymnasium.envs.registration import register

register(
     id="World2D-v0",
     entry_point="adlr_environments.environments:World2D",
     max_episode_steps=200,
)
