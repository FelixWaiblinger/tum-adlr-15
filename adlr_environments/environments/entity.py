"""All types of objects in a 2D environment"""

from abc import ABC

import pygame
import numpy as np
from numpy.random import Generator

from utils import eucl, draw_arrow, draw_uncertainty
from utils.constants import Color, PIXELS


class Entity(ABC):
    """2D entity"""

    position: np.ndarray
    color: Color
    size: float

    def reset(self, entities: list, generator: Generator):
        """Reset the entity for the next episode"""
        while True:
            # NOTE: spawn entities on a bit smaller area to avoid world edges
            self.position = generator.uniform(-0.9, 0.9, 2)
            if not any(self.collision(e) for e in entities):
                entities.append(self)
                break

    def collision(self, other) -> bool:
        """Check for a collision"""
        distance = eucl(self.position, other.position)
        sizes = self.size + other.size
        return distance < sizes

    def draw(self, canvas: pygame.Surface):
        """Draw the entity on the canvas"""
        pygame.draw.circle(
            surface=canvas,
            color=self.color.value,
            center=((self.position + 1) * canvas.get_width() / 2).tolist(),
            radius=self.size * canvas.get_width() / 2
        )


class Agent(Entity):
    """2D agent"""

    def __init__(self, size: float=0.1, vision: float=0.5) -> None:
        """Create a new agent"""
        self.position = np.zeros(2, dtype=np.float32)
        self.speed = np.zeros(2, dtype=np.float32)
        self.color = Color.BLUE
        self.size = size
        self.vision = vision

    def reset(self, entities: list, generator: Generator):
        """Reset the entity for the next episode"""
        super().reset(entities, generator)
        self.speed = np.zeros(2, dtype=np.float32)

    def draw(self,
        canvas: pygame.Surface,
        draw_vision: bool=False,
        draw_direction: bool=True
    ):
        """Draw the agent and its direction on the canvas"""
        super().draw(canvas)
        scale = canvas.get_width() / 2
        start = ((self.position + 1) * scale).tolist()
        end = ((self.position + self.speed * 0.3 + 1) * scale).tolist()

        if draw_vision:
            draw_uncertainty(canvas, pygame.Vector2(start), self.vision)

        if draw_direction:
            draw_arrow(
                canvas,
                pygame.Vector2(start),
                pygame.Vector2(end),
                color=Color.BLACK.value,
                body_width=5,
                head_width=20,
                head_height=12
            )


class Target(Entity):
    """2D Target box"""

    def __init__(self, size: float=0.1) -> None:
        """Create a new target"""
        self.position = 0.7 * np.ones(2, dtype=np.float32)
        self.color = Color.RED
        self.size = size


class StaticObstacle(Entity):
    """Static obstacle"""

    def __init__(self, size: float=0.1) -> None:
        """Create a new static obstacle"""
        self.position = np.zeros(2)
        self.color = Color.BLACK
        self.size = size


class DynamicObstacle(Entity):
    """Dynamic obstacle"""

    def __init__(self, size: float=0.1, speed: np.ndarray=np.ones(2)) -> None:
        """Create a new dynamic obstacle"""
        self.position = np.zeros(2)
        self.color = Color.GREEN
        self.size = size
        self.speed = speed

    def move(self):
        """Move the dynamic obstacle"""
        self.position = np.clip(self.position + self.speed, -1, 1)
