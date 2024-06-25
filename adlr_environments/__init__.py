"""Environment registration"""

from gymnasium.envs.registration import register

from adlr_environments.constants import MAX_EPISODE_STEPS
from adlr_environments.environments.levels import LEVEL1, LEVEL2, LEVEL3


register(
    id="World2D-v0",
    entry_point="adlr_environments.environments:World2D",
    max_episode_steps=MAX_EPISODE_STEPS,
)

register(
    id="World2D-Play-v0",
    entry_point="adlr_environments.environments:World2D",
    max_episode_steps=500,
)
