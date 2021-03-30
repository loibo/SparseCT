import data
import os
import numpy as np
import visualization

"""
This file contains an example on how to generate the dataset from the functions in this library.
"""

PATH = './SparseCT/'
LOAD = False

if not LOAD:
    ellipsoid_data = data.get_data((1, 600, 450, 32))
    data.to_tif(ellipsoid_data, PATH)
    data.to_tif(ellipsoid_data, PATH, type='parallel')
    data.to_np(ellipsoid_data, PATH)
else:
    ellipsoid_data = np.load(PATH + 'ellipsoid_dataset.npy')
    print(ellipsoid_data.shape)

visualization.show3d(ellipsoid_data[0])
