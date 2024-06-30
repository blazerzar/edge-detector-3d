from os import listdir

import matplotlib.pyplot as plt
import numpy as np
from open3d import geometry, utility, visualization

files = [f for f in listdir('results') if f != '.gitkeep']
files = sorted([(int(f.split('.')[0]), f) for f in files])
files = [f[1] for f in files][::-1]
linked = [plt.imread('results/' + file)[:, :, 0] for file in files]

pc = np.array(linked)
pointcloud = np.argwhere(pc == 1).astype(np.float32)
pointcloud[:, 0] *= 8

pcd1 = geometry.PointCloud()
pcd1.points = utility.Vector3dVector(pointcloud)
visualization.draw_geometries([pcd1])
