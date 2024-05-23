"""Utility functions"""

import numpy as np
import pygame as pg


def draw_policy(
    agent,
    observation: np.ndarray,
    surface: pg.Surface,
    world_size: float,
    resolution: int=10
) -> None:
    """Test"""

    offset = 0.5 * (world_size / resolution)
    for i in np.linspace(offset, world_size - offset, resolution):
        for j in np.linspace(offset, world_size - offset, resolution):
            position = np.array([i, j], dtype=np.float32)
            observation["agent"] = position
            target, _ = agent.predict(observation, deterministic=True)

            pg.draw.line(
                surface,
                color=(0, 0, 255),
                start_pos=position,
                end_pos=target
            )
