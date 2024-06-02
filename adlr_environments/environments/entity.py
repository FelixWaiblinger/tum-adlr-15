"""All types of objects in a 2D environment"""

from abc import ABC

import pygame
import numpy as np


class Entity(ABC):
    """2D entity"""

    position: np.ndarray
    color: tuple
    size: float
    start_position: np.ndarray
    random: bool

    def reset(self, world_size: float, illegal_positions: list, generator):
        """Reset the entity for the next episode"""

        def too_close(p1, p2):
            return np.all(np.abs(p1 - p2) < self.size)

        if self.random:
            while True:
                position = generator.uniform(
                    low=0, high=world_size, size=2
                ).astype(np.float32)

                illegal = any(too_close(position, p) for p in illegal_positions)

                if not illegal:
                    self.position = position
                    illegal_positions.append(self.position)
                    break
        else:
            self.position = self.start_position

    def collision(self, other) -> bool:
        """Check for a collision"""

        sizes = self.size + other.size
        distance = np.linalg.norm(self.position - other.position, ord=2)

        return distance < sizes

    def draw(self, canvas: pygame.Surface, world2canvas: float):
        """Draw the entity on the canvas"""

        pygame.draw.circle(
            canvas,
            self.color,
            (self.position * world2canvas).tolist(),
            self.size * world2canvas
        )


class Agent(Entity):
    """2D agent"""

    def __init__(self, position, random) -> None:
        """Create a new agent"""

        self.position = np.zeros(2, dtype=np.float32)
        self.speed = np.zeros(2, dtype=np.float32)
        self.color = (0, 0, 255)
        self.size = 0.1
        self.start_position = position
        self.random = random

    def reset(self, world_size: float, illegal_positions: list, generator):
        """Reset the entity for the next episode"""

        super().reset(world_size, illegal_positions, generator)
        self.speed = np.zeros(2, dtype=np.float32)

    def wall_collision(self, world_size: float):
        x_position = self.position[0]
        y_position = self.position[1]

        if x_position < self.size:
            return True
        elif x_position > world_size - self.size:
            return True
        elif y_position < self.size:
            return True
        elif y_position > world_size - self.size:
            return True
        else:
            return False


class Target(Entity):
    """2D Target box"""

    def __init__(self, random, position) -> None:
        """Create a new target"""

        self.position = np.zeros(2)
        self.color = (255, 0, 0)
        self.size = 0.5
        self.start_position = position
        self.random = random


class StaticObstacle(Entity):
    """Static obstacle"""

    def __init__(self, random, position) -> None:
        """Create a new static obstacle"""

        self.position = np.zeros(2)
        self.color = (0, 0, 0)
        self.size = 0.5
        self.start_position = position
        self.random = random


class DynamicObstacle(Entity):
    """Dynamic obstacle"""

    def __init__(self, speed: tuple = (1, 1)) -> None:
        """Create a new dynamic obstacle"""

        self.position = np.zeros(2)
        self.color = (0, 255, 0)
        self.size = 0.5
        self.speed = np.array(speed)

    def move(self, bounds=None):  #: tuple | None=None):
        """Move the dynamic obstacle"""

        # bounded movement
        if bounds:
            self.position = np.clip(
                self.position + self.speed, 0, np.array(bounds)
            )
        # unbounded movement
        else:
            self.position += self.speed
