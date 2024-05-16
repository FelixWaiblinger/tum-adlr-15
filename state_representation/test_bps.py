"""Test"""

import time

import numpy as np
import matplotlib.pyplot as plt

from bps import BPS, img2pc

SEED = 42
N_POINTS = 100
SIZE = 10
IMG_SHAPE = (512, 512, 3)
STATIC_COLOR = (0, 0, 0)
DYNAMIC_COLOR = (0, 255, 0)

bps = BPS(SEED, N_POINTS, SIZE)

image = np.random.uniform(0, 255, size=IMG_SHAPE).astype(int)
image[123, 123] = np.array(STATIC_COLOR)
image[213, 213] = np.array(DYNAMIC_COLOR)

start = time.time()
pointcloud = img2pc(image, SIZE)
end = time.time()

print(f"Time for pointcloud creation: {end - start}")

start = time.time()
distances = bps.encode(pointcloud)
end = time.time()

print(f"Time for encoding: {end - start}")

plt.scatter(pointcloud[:, 0], pointcloud[:, 1], color='r')
plt.scatter(bps.points[:, 0], bps.points[:, 1])
plt.show()
