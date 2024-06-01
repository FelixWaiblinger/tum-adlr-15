"""All types of objects in a 2D environment"""

from abc import ABC

import pygame
import numpy as np
from numpy.random import Generator

from adlr_environments.utils import eucl, draw_arrow


class Entity(ABC):
    """2D entity"""

    position: np.ndarray
    color: tuple
    size: float

    def reset(self, world_size: float, entities: list, generator: Generator):
        """Reset the entity for the next episode"""
        while True:
            self.position = generator.uniform(0, world_size, 2).astype(np.float32)
            if not any(self.collision(e) for e in entities):
                entities.append(self)
                break

    def collision(self, other) -> bool:
        """Check for a collision"""
        distance = eucl(self.position, other.position)
        sizes = self.size + other.size
        return distance < sizes

    def draw(self, canvas: pygame.Surface, world2canvas: float):
        """Draw the entity on the canvas"""
        pygame.draw.circle(
            canvas,
            self.color,
            (self.position * world2canvas),
            self.size * world2canvas
        )


class Agent(Entity):
    """2D agent"""

    def __init__(self) -> None:
        """Create a new agent"""
        self.position = np.zeros(2, dtype=np.float32)
        self.speed = np.zeros(2, dtype=np.float32)
        self.color = (0, 0, 255)
        self.size = 0.3

    def reset(self, world_size: float, entities: list, generator: Generator):
        """Reset the entity for the next episode"""
        super().reset(world_size, entities, generator)
        self.speed = np.zeros(2, dtype=np.float32)

    def draw(self, canvas: pygame.Surface, world2canvas: float):
        """Draw the agent and its direction on the canvas"""
        super().draw(canvas, world2canvas)
        start = (world2canvas * self.position).tolist()
        end = (world2canvas * (self.position + self.speed)).tolist()
        draw_arrow(
            canvas,
            pygame.Vector2(start),
            pygame.Vector2(end),
            color=(0, 0, 0),
            body_width=5,
            head_width=20,
            head_height=12
        )


class Target(Entity):
    """2D Target box"""

    def __init__(self, world_size: float) -> None:
        """Create a new target"""
        self.position = 0.7 * world_size * np.ones(2, dtype=np.float32)
        self.color = (255, 0, 0)
        self.size = 0.5


class StaticObstacle(Entity):
    """Static obstacle"""

    def __init__(self) -> None:
        """Create a new static obstacle"""
        self.position = np.zeros(2)
        self.color = (0, 0, 0)
        self.size = 0.5


class DynamicObstacle(Entity):
    """Dynamic obstacle"""

    def __init__(self, speed: tuple=(1, 1)) -> None:
        """Create a new dynamic obstacle"""
        self.position = np.zeros(2)
        self.color = (0, 255, 0)
        self.size = 0.5
        self.speed = np.array(speed)

    def move(self, bounds: tuple=None):
        """Move the dynamic obstacle"""
        self.position += self.speed
        if bounds:
            self.position = np.clip(self.position, 0, np.array(bounds))
        self.position = self.position.astype(np.float32)
