import data
import transform
import os
import numpy as np

"""
This file contains an example on how to generate the dataset from the functions in this library.
"""

PATH = './SparseCT/data/'
LOAD = True

if not LOAD:
    ellipsoid_data = data.get_data()
    data.to_np(ellipsoid_data, PATH)
else:
    ellipsoid_data = np.load(PATH + 'ellipsoid_dataset.npy')
    print(ellipsoid_data.shape)

img = ellipsoid_data[0]
s = transform.img_to_sino(img)
print(s.shape)


