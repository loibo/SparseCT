import numpy as np
import astra

import os
from PIL import Image, ImageOps
import cv2
import time
import matplotlib.pyplot as plt


def img_to_sino(img):
    """
    Computes the projection of img, transformed with the cone-beam geometry and angles theta.

    :param img: ndarray, the input image of dimension (H, W, D)
    :param theta: ndarray or list or tuple, a list of the angles
    :return: ndarray, the projection of img of dimension (Hs, Ws, Ds)
    """
    # hor. and ver. spacing between adjacent pixels in the detector surface
    det_spacing_x = 1
    det_spacing_y = 1
    det_dims = get_dims_from_phantom(img)

    # angles at which projections will be taken
    proj_amplitude = 100
    proj_no = 101
    angles = get_proj_angle(-50, proj_amplitude, proj_no)

    # distance between source-origin/origin detector
    src_orig_dist = 1000
    orig_det_dist = 10
    proj_geom = get_projection_geom(angles, det_dims, det_spacing_x, det_spacing_y, src_orig_dist, orig_det_dist)

    # compute projections
    print('Calculating Projections..')
    obj_id, vol_geom = get_volume_geom(img)

    # get sinograms performing the backprojection
    s = project(obj_id, vol_geom, proj_geom)

    return s


def get_proj_angle(starting_angle, proj_amplitude, proj_no):
    # return the angles converted into radians
    end_angle = starting_angle + proj_amplitude
    angles = np.linspace(starting_angle, end_angle, num=proj_no)
    return np.deg2rad(angles)


# instantiate the volume inside astra
def get_volume_geom(volume_data):
    vol_geom = astra.creators.create_vol_geom(volume_data.shape[1], volume_data.shape[2], volume_data.shape[0])
    obj_id = astra.data3d.create('-vol', vol_geom, data=volume_data)
    return obj_id, vol_geom


# build projection geometry
def get_projection_geom(angles, det_dims, det_spacing_x, det_spacing_y, src_orig_dist, orig_det_dist):
    det_row_count, det_col_count = det_dims
    return astra.create_proj_geom('cone', det_spacing_x, det_spacing_y,
                                  det_row_count, det_col_count, angles,
                                  src_orig_dist, orig_det_dist)


# actual projection work
def project(obj_id, vol_geom, proj_geom):
    projections_id, projections = astra.creators.create_sino3d_gpu(obj_id, proj_geom, vol_geom)
    return projections


def get_dims_from_phantom(phantom):
    shape_arr = np.array([phantom.shape])
    shape_arr.flatten()
    dims = np.around(shape_arr[0] * np.sqrt(2))

    return int(dims[0]), int(dims[1])


def add_noise_to_sino(s, sigma, type="Gaussian"):
    """
    Add noise of type "type" to the sinogram s, with noise level sigma.

    :param s: ndarray, sinogram of dimension (H, W, D)
    :param sigma: float32, noise level
    :param type: str, type of noise
    :return: ndarray, corrupted sinogram with the same dimension of s
    """
    if type == "Gaussian":
        eta = np.random.normal(size=s.shape)
        eta /= np.linalg.norm(eta, 'fro')
        eta *= sigma * np.linalg.norm(s, 'fro')
    else:
        eta = 0
        print('Noise type not found. No noise added to img.')

    return s + eta
