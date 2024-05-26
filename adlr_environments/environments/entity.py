"""All types of objects in a 2D environment"""

from abc import ABC

import pygame
import numpy as np


class Entity(ABC):
    """2D entity"""

    position: np.ndarray
    visual: pygame.Rect
    color: tuple
    size: float

    def reset(self, world_size: float, illegal_positions: list, generator):
        """Reset the entity for the next episode"""

        def too_close(p1, p2):
            return np.all(np.abs(p1 - p2) < self.size)

        while True:
            position = generator.uniform(
                low=0, high=world_size, size=2
            ).astype(np.float32)

            illegal = any(too_close(position, p) for p in illegal_positions)

            if not illegal:
                self.position = position
                illegal_positions.append(self.position)
                break

    def collision(self, other) -> bool:
        """Check for a collision"""

        return self.visual.colliderect(other.visual)

    def draw(self, canvas: pygame.Surface, world2canvas: float):
        """Draw the entity on the canvas"""

        self.visual = pygame.draw.rect(
            canvas,
            self.color,
            pygame.Rect(
                (self.position - 0.5 * np.array(self.size)) * world2canvas,
                self.size * world2canvas * np.ones(2)
            )
        )


class Agent(Entity):
    """2D agent"""

    def __init__(self) -> None:
        """Create a new agent"""

        self.position = np.zeros(2)
        self.color = (0, 0, 255)
        self.size = 0.3

    def draw(self, canvas: pygame.Surface, world2canvas: float):
        """Draw the agent on the canvas"""

        self.visual = pygame.draw.circle(
            canvas,
            self.color,
            (self.position * world2canvas).tolist(),
            self.size * world2canvas
        )


class Target(Entity):
    """2D Target box"""

    def __init__(self) -> None:
        """Create a new target"""

        self.position = np.zeros(2)
        self.color = (255, 0, 0)
        self.size = np.array((1, 1))


class StaticObstacle(Entity):
    """Static obstacle"""

    def __init__(self, size: tuple=(1, 1)) -> None:
        """Create a new static obstacle"""

        self.position = np.zeros(2)
        self.color = (0, 0, 0)
        self.size = np.array(size)


class DynamicObstacle(Entity):
    """Dynamic obstacle"""

    def __init__(self, size: tuple=(1, 1), speed: tuple=(1, 1)) -> None:
        """Create a new dynamic obstacle"""

        self.position = np.zeros(2)
        self.color = (0, 255, 0)
        self.size = np.array(size)
        self.speed = np.array(speed)

    def move(self, bounds=None): #: tuple | None=None):
        """Move the dynamic obstacle"""

        # bounded movement
        if bounds:
            self.position = np.clip(
                self.position + self.speed, 0, np.array(bounds)
            )
        # unbounded movement
        else:
            self.position += self.speed
