"""Basis Point Sets"""

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from adlr_environments.constants import BLACK, GREEN


class BPS:
    """Basis Point Set"""

    def __init__(self,
        seed: int,
        num_points: int=100,
    ) -> None:
        """Create a set of basis points"""

        generator = np.random.default_rng(seed)
        self.points: np.ndarray = generator.uniform(-1, 1, (num_points, 2))

    def encode(self, pointcloud: np.ndarray) -> np.ndarray:
        """Compute minimal distances between basis points and a pointcloud"""

        assert pointcloud.size > 0, "The pointcloud must not be empty!"
        distances = cdist(self.points, pointcloud, "euclidean")

        # ====================================================================
        # NOTE: uncomment to show bps for debugging purposes
        # test = []
        # plt.scatter(self.points[:, 0], self.points[:, 1], color='r')
        # plt.scatter(pointcloud[:, 0], pointcloud[:, 1])
        # for p in self.points:
        #     diffs = np.subtract(pointcloud, p)
        #     dists = np.linalg.norm(diffs, ord=2, axis=1)
        #     match = np.argmin(dists)
        #     test.append(dists[match])
        #     plt.arrow(p[0], p[1], diffs[match, 0], diffs[match, 1])
        # plt.gca().invert_yaxis()
        # plt.show()

        # res = np.min(distances, axis=1).astype(np.float32)
        # print(np.array(test))
        # print(res)
        # ====================================================================

        return np.min(distances, axis=1)

def img2pc(image: np.ndarray, image_size: int) -> np.ndarray:
    """Convert an image (w, h, 3) of obstacles to a 2D pointcloud (p, 2)"""

    static_pixels = np.where(np.all(image == BLACK, axis=-1))
    dynamic_pixels = np.where(np.all(image == GREEN, axis=-1))

    static_coordinates = list(zip(static_pixels[0], static_pixels[1]))
    dynamic_coordinates = list(zip(dynamic_pixels[0], dynamic_pixels[1]))

    pointcloud = np.array(static_coordinates + dynamic_coordinates)
    pointcloud = (pointcloud / (image_size / 2)) - 1 # scale to world coords

    return pointcloud.astype(np.float32)
