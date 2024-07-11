"""Constant values"""

from enum import Enum

import torch
import numpy as np


MAX_EPISODE_STEPS = 200
MAX_PLAYMODE_STEPS = 500
DTYPE = np.float32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PIXELS = 512


class Color(Enum):
    """Basic colors"""
    WHITE=(255, 255, 255)
    RED=(255, 0, 0)
    GREEN=(0, 255, 0)
    BLUE=(0, 0, 255)
    BLACK=(0, 0, 0)


class Observation(Enum):
    """Observation types for the environment"""
    POS=0
    RGB=1,
    ALL=2


class Input(Enum):
    """Input type for pygame"""
    MOUSE=0
    KEYBOARD=1
    CONTROLLER=2
    AGENT=3


__all__ = [
    "MAX_EPISODE_STEPS",
    "MAX_PLAYMODE_STEPS",
    "DTYPE",
    "DEVICE",
    "PIXELS",
    "Color",
    "Observation",
    "Input"
]