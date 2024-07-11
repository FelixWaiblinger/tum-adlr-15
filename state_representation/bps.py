"""Basis Point Sets"""

import numpy as np
from scipy.spatial.distance import cdist

from utils.constants import Color
from utils.visualization import draw_bps


BPS_SEED = 42


class BPS:
    """Basis Point Set"""

    def __init__(self, seed: int=BPS_SEED, num_points: int=100) -> None:
        """Create a set of basis points"""

        generator = np.random.default_rng(seed)
        self.points: np.ndarray = generator.uniform(-1, 1, (num_points, 2))

    def encode(self, pointcloud: np.ndarray, show: bool=False) -> np.ndarray:
        """Compute minimal distances between basis points and a pointcloud"""

        assert pointcloud.size > 0, "The pointcloud must not be empty!"
        distances = cdist(self.points, pointcloud, "euclidean")

        if show:
            draw_bps(self.points, pointcloud)

        return np.min(distances, axis=1)

    @staticmethod
    def img2pc(image: np.ndarray, image_size: int) -> np.ndarray:
        """Convert an image (w, h, 3) of obstacles to a 2D pointcloud (p, 2)"""

        static_pixels = np.where(np.all(image == Color.BLACK.value, axis=-1))
        dynamic_pixels = np.where(np.all(image == Color.GREEN.value, axis=-1))

        static_coordinates = list(zip(static_pixels[0], static_pixels[1]))
        dynamic_coordinates = list(zip(dynamic_pixels[0], dynamic_pixels[1]))

        pointcloud = np.array(static_coordinates + dynamic_coordinates)
        pointcloud = (pointcloud / (image_size / 2)) - 1 # scale to world coords

        return pointcloud.astype(np.float32)
