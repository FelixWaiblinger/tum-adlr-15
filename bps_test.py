import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bps import bps
import time
if __name__ == "__main__":
    # load array
    arr = np.load(r"F:\Uni\master\Semester 2\ADL4R\project\code\tum-adlr-15\arr_1.npy")
    # plt.imshow(arr)
    # plt.show()

    # sum along first axis
    arr = np.sum(arr, axis=2)
    t = arr

    t[arr < 765] = 0
    t[arr == 765] = 1

    indices = np.where(t == 0)
    pointcloud = np.array(indices)
    #pointcloud = np.concatenate((pointcloud, np.zeros((1, np.shape(pointcloud)[1]))))
    pointcloud = np.transpose(pointcloud)
    pointcloud = pointcloud[np.newaxis, :, :]

    pointcloud = bps.normalize(pointcloud)

    start = time.time()
    for i in range(0, 100):
        x_norm= bps.encode(pointcloud, bps_arrangement="random", n_bps_points=100, bps_cell_type="dists", n_jobs=1)

    print("time" + str((time.time() - start)/100))



    # pointcloud = pointcloud[0, :,:]
    # x = pointcloud[:, 0]
    # y = pointcloud[:, 1]
    # #z = pointcloud[:, 2]
    #
    # a = bps_basis[:, 0]
    # b = bps_basis[:, 1]
    # #c = bps_basis[:, 2]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(x, y)
    # ax.scatter(a, b, c="b")
    #
    #
    #
    # # Customization
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # ax.set_title('3D Scatter Plot')
    # ax.view_init(elev=20, azim=135)
    #
    # # Show Plot
    # plt.show()
    # #
    # #
    # #
    # # # plt.imshow(t, cmap="gray")
    # # # plt.show()
    # # #
    # # # print(arr)
    # # # print("hello")