import data
import transform
import os
import numpy as np

"""
This file contains an example on how to generate the dataset from the functions in this library.
"""

PATH = './SparseCT/'
LOAD = True

if not LOAD:
    ellipsoid_data = data.get_data((1, 450, 600, 32))
    data.to_tif(ellipsoid_data, PATH)
    data.to_tif(ellipsoid_data, PATH, type='parallel')
    data.to_np(ellipsoid_data, PATH)
else:
    ellipsoid_data = np.load(PATH + 'ellipsoid_dataset.npy')
    # print(ellipsoid_data.shape)

img = ellipsoid_data[0]
astra_img = np.transpose(img, (1, 2, 0)) # (x, y, z) = (600, 32, 450)
print('Data Shape:', astra_img.shape)

obj_id, vol_geom = transform.get_volume_geom(astra_img)