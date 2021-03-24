import data
import transform
import os
import numpy as np

"""
This file contains an example on how to generate the dataset from the functions in this library.
"""

PATH = './SparseCT/'
LOAD = False

if not LOAD:
    ellipsoid_data = data.get_data((1, 450, 600, 32))
    data.to_tif(ellipsoid_data, PATH)
    data.to_tif(ellipsoid_data, PATH, type='parallel')
    data.to_np(ellipsoid_data, PATH)
else:
    ellipsoid_data = np.load(PATH + 'ellipsoid_dataset.npy')
    print(ellipsoid_data.shape)

s = transform.img_to_sino(np.transpose(ellipsoid_data[0], (2, 0, 1)))