"""Basis Point Sets"""

import numpy as np
from scipy.spatial.distance import cdist


STATIC_COLOR = (0, 0, 0)
DYNAMIC_COLOR = (0, 255, 0)


class BPS:
    """Basis Point Set"""

    def __init__(self,
        seed: int,
        num_points: int=100,
        world_size: float=1,
        origin: tuple[float, float]=(0, 0)
    ) -> None:
        """Create a set of basis points"""

        generator = np.random.default_rng(seed)
        self.points: np.ndarray = generator.uniform(
            0, world_size, size=(num_points, 2)
        )
        self.points -= origin

    def encode(self, pointcloud: np.ndarray) -> np.ndarray:
        """Compute minimal distances between basis points and a pointcloud"""

        distances = cdist(self.points, pointcloud, "euclidean")

        return np.min(distances, axis=1).astype(np.float32)

def img2pc(image: np.ndarray, world_size: float=1) -> np.ndarray:
    """Convert an image (w, h, 3) of obstacles to a 2D pointcloud (p, 2)"""

    image_size = min(image.shape[:2])

    static_pixels = np.where(np.all(image == STATIC_COLOR, axis=-1))
    dynamic_pixels = np.where(np.all(image == DYNAMIC_COLOR, axis=-1))

    static_coordinates = list(zip(static_pixels[0], static_pixels[1]))
    dynamic_coordinates = list(zip(dynamic_pixels[0], dynamic_pixels[1]))

    pointcloud = np.array(static_coordinates + dynamic_coordinates)
    pointcloud = (pointcloud / image_size) * world_size # scale to world coords

    return pointcloud.astype(np.float32)