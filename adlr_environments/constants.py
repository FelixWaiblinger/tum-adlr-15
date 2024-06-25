"""Constant values"""

from enum import Enum


MAX_EPISODE_STEPS = 200
MAX_PLAYMODE_STEPS = 500

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

class Input(Enum):
    """Input type for pygame"""
    MOUSE=0
    KEYBOARD=1
    CONTROLLER=2
    AGENT=3
