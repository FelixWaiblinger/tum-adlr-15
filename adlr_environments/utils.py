"""Utility functions"""

from typing import Callable

import pygame
import numpy as np
import matplotlib.pyplot as plt


def draw_policy(
    agent,
    observation: np.ndarray,
    arrows_per_row: int=10
) -> None:
    """Plot a vector field representing the actions chosen by the policy for
    any position in the environment
    """

    # start arrows in the middle of "their box"
    offset = 2 / (arrows_per_row + 1)
    for i in np.linspace(-1 + offset, 1 - offset, arrows_per_row):
        for j in np.linspace(-1 + offset, 1 - offset, arrows_per_row):
            observation[0, :2] = np.array([i, j], dtype=np.float32)
            target, _ = agent.predict(observation, deterministic=True)
            target = 0.5 * target[0] # shorten for visibility
            plt.arrow(i, j, target[0], -target[1], head_width=0.1)
    plt.show()


def draw_arrow(
    surface: pygame.Surface,
    start: pygame.Vector2,
    end: pygame.Vector2,
    color: pygame.Color,
    body_width: int = 2,
    head_width: int = 4,
    head_height: int = 2,
):
    """Draw an arrow between start and end with the arrow head at the end.

    Args:
        surface (pygame.Surface): The surface to draw on
        start (pygame.Vector2): Start position
        end (pygame.Vector2): End position
        color (pygame.Color): Color of the arrow
        body_width (int, optional): Defaults to 2.
        head_width (int, optional): Defaults to 4.
        head_height (float, optional): Defaults to 2.
    """
    arrow = start - end
    angle = arrow.angle_to(pygame.Vector2(0, -1))
    body_length = arrow.length() - head_height

    # Create the triangle head around the origin
    head_verts = [
        pygame.Vector2(0, head_height / 2),  # Center
        pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
        pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
    ]
    # Rotate and translate the head into place
    translation = pygame.Vector2(0, arrow.length() - (head_height / 2))
    translation = translation.rotate(-angle)

    for vert in head_verts:
        vert.rotate_ip(-angle)
        vert += translation
        vert += start

    pygame.draw.polygon(surface, color, head_verts)

    # Stop weird shapes when the arrow is shorter than arrow head
    if arrow.length() >= head_height:
        # Calculate the body rect, rotate and translate into place
        body_verts = [
            pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
            pygame.Vector2(body_width / 2, body_length / 2),  # Topright
            pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
            pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
        ]
        translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
        for vert in body_verts:
            vert.rotate_ip(-angle)
            vert += translation
            vert += start

        pygame.draw.polygon(surface, color, body_verts)


def eucl(x: np.ndarray, y: np.ndarray) -> float:
    """Return euclidean distance between positions x and y"""
    return np.linalg.norm(x - y, ord=2)


def linear(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule (current learning rate depending on
    remaining progress)
    """

    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0"""

        return progress_remaining * initial_value

    return func


def to_py_dict(dictionary: dict):
    """Convert values in dictionary to python built-in types"""

    result = {}
    for k, v in dictionary.items():
        if isinstance(v, np.float64):
            result[k] = float(v)
        elif isinstance(v, np.int64):
            result[k] = int(v)
        elif isinstance(v, Callable):
            result[k] = str(v)
        else:
            result[k] = float(v)
    return result
