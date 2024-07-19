"""Configuration classes"""

from gymnasium.wrappers import FlattenObservation

from utils.constants import Observation
from adlr_environments.wrapper import BPSWrapper, AEWrapper, RewardWrapper
from state_representation.datasets import CombineTransform, NormalizeTransform


AGENT_PATH = "./agents/"
LOG_PATH = "./logs/"
AUTOENCODER = "./state_representation/ae50_random_obs.pt"
TRANSFORM = CombineTransform([
    NormalizeTransform(start=(0, 255), end=(0, 1)),
])


class BPS_CONFIG:
    wrapper = [
        (BPSWrapper, {"num_points": 50}),
        (FlattenObservation, {}),
        (RewardWrapper, {})
    ]
    observation = Observation.POS
    env = {
        "step_length": 0.1,
        "size_agent": 0.075,
        "size_target": 0.1,
        "num_static_obstacles": 5,
        "size_static": 0.1,
        "num_dynamic_obstacles": 3,
        "size_dynamic": 0.075,
        "min_speed": -0.05,
        "max_speed": 0.05,
        "uncertainty": False
    }


class AE_CONFIG:
    wrapper = [
        (AEWrapper, {"model_path": AUTOENCODER, "transform": TRANSFORM}),
        (FlattenObservation, {}),
        (RewardWrapper, {})
    ]
    observation = Observation.ALL
    env = {
        "step_length": 0.1,
        "size_agent": 0.075,
        "size_target": 0.1,
        "num_static_obstacles": 5,
        "size_static": 0.1,
        "num_dynamic_obstacles": 3,
        "size_dynamic": 0.075,
        "min_speed": -0.05,
        "max_speed": 0.05,
        "uncertainty": False
    }
