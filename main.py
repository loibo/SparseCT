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
s_noise = transform.add_noise_to_sino(s, sigma=0.1)
img_recon = transform.sino_to_img(s_noise)

angles = transform.get_proj_angle()

# get projection geometry
proj_geom = astra.create_proj_geom('cone', det_spacing_x, det_spacing_y,
                                   det_row_count, det_col_count, angles,
                                   src_orig_dist, orig_det_dist)
proj_id = astra.data3d.create('-sino', proj_geom, projection)

# get the volume geometry
vol_geom = (64, 512, 512)

# FDK RECONSTRUCTION
reconstruction = reconstruct(proj_id, vol_geom)


