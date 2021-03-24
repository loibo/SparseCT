import numpy as np
import astra

import os
from PIL import Image, ImageOps
import cv2
import time
import matplotlib.pyplot as plt


def img_to_sino(img):
    """
    Computes the projection of img, transformed with the cone-beam geometry and angles theta. The parameters are defined
    in the code.

    We assume that 1 unit length (the dimension of a side of each voxel) is equivalent to 100 micron. So, if we take the
    detector to have det_spacing_x = det_spacing_y = 1, we have the same accuracy for both the volume and the detector.

    Usually the detector is around 6cm = 600 unit length far from the origin, while the source is 70cm = 7000 unit length
    far from the origin.

    :param img: ndarray, the input image of dimension (T, X, Y)
    :return: ndarray, the projection of img of dimension (N_t, x, y)
    """
    # hor. and ver. spacing between adjacent pixels in the detector surface
    det_spacing_x = 1
    det_spacing_y = 1
    det_dims = get_dims_from_phantom(img)

    # angles at which projections will be taken
    start_angle = -50 # in degrees
    end_angle = 50    # in degrees
    proj_no = 101
    angles = get_proj_angle(start_angle, end_angle, proj_no) # in radians

    # distance between source-origin/origin detector
    src_orig_dist = 7000  # 70cm
    det_orig_dist = 600   # 6cm
    proj_geom = get_projection_geom(angles, det_dims, det_spacing_x, det_spacing_y, src_orig_dist, det_orig_dist)

    # compute projections
    print('Calculating Projections..')
    obj_id, vol_geom = get_volume_geom(img)

    # get sinograms performing the backprojection
    s = project(obj_id, vol_geom, proj_geom)

    return s


# actually reconstruct the object from the projections
def sino_to_img(s, volume_shape, alg='FDK_CUDA', iter_no=1):
    """
    Computes the reconstruction of rec_proj_id, compute with the algorithm alg after iter_no iterations.

    :param s: ndarray, the projection data
    :param volume_shape: list or tuple, a list that contains information about the dimension of the rec. image
    :param alg: str, the name of the algorithm. More information in the documentation (DEFAULT FDK_CUDA)
    :param iter_no: int, the number of iterations that will be performed (DEFAULT 1)
    :return: ndarray, the reconstructed image
    """
    # hor. and ver. spacing between adjacent pixels in the detector surface
    det_spacing_x = 1
    det_spacing_y = 1
    det_dims = get_dims_from_projection(s)

    # angles at which projections will be taken
    start_angle = -50 # in degrees
    end_angle = 50    # in degrees
    proj_no = 101
    angles = get_proj_angle(start_angle, end_angle, proj_no) # in radians

    # distance between source-origin/origin detector
    src_orig_dist = 7000  # 70cm
    det_orig_dist = 600   # 6cm
    proj_geom = get_projection_geom(angles, det_dims, det_spacing_x, det_spacing_y, src_orig_dist, det_orig_dist)
    proj_id = get_sinogram(s, proj_geom)

    # instantiate volume shapes
    rec_vol_geom = astra.creators.create_vol_geom(volume_shape[1], volume_shape[2], volume_shape[0])
    rec_vol_id = astra.data3d.create('-vol', rec_vol_geom, data=0)

    # reconstruction algorithm configurations
    alg_cfg = astra.astra_dict(alg)
    alg_cfg['ProjectionDataId'] = proj_id
    alg_cfg['ReconstructionDataId'] = rec_vol_id
    alg_id = astra.algorithm.create(alg_cfg)

    # run the algorithm
    astra.algorithm.run(alg_id, iter_no)

    return astra.data3d.get(rec_vol_id)


def get_proj_angle(starting_angle, end_angle, proj_no):
    """
    Get the projection angles needed to perform projection.

    :param starting_angle: float32, the angle theta_0
    :param end_angle: float32, the angle theta_end
    :param proj_no: int, the number of angular projections
    :return: ndarray, an array that contains the angles
    """
    angles = np.linspace(starting_angle, end_angle, num=proj_no)
    return np.deg2rad(angles)


def get_volume_geom(volume_data):
    """
    Returns a volume object from AstraToolbox
    :param volume_data: ndarray, an array that contains the image data (32, 512, 512)
    :return: (int, astra_volume), the id of the volume object and the object itself
    """
    vol_geom = astra.creators.create_vol_geom(volume_data.shape[1], volume_data.shape[2], volume_data.shape[0])
    obj_id = astra.data3d.create('-vol', vol_geom, data=volume_data)
    return obj_id, vol_geom


def get_projection_geom(angles, det_dims, det_spacing_x, det_spacing_y, src_orig_dist, orig_det_dist):
    """
    Get the projection geometry. Needed to perform FBP.

    :param angles: ndarray, output of the function get_proj_angles
    :param det_dims: list or tuple, a list that contains the dimension of the detector for each angle
    :param det_spacing_x: int, hor. spacing in the detector
    :param det_spacing_y: int, ver. spacing in the detector
    :param src_orig_dist: int, distance between the source and the origin
    :param orig_det_dist: int, distance between the origin and the detector
    :return: astra_projection, an object that contains the projection geometry
    """
    det_row_count, det_col_count = det_dims
    return astra.create_proj_geom('cone', det_spacing_x, det_spacing_y,
                                  det_row_count, det_col_count, angles,
                                  src_orig_dist, orig_det_dist)


def project(obj_id, vol_geom, proj_geom):
    """
    Runs the projection b = Ax, where x is the input phantom and A is the projector matrix.

    :param obj_id: int, the first output of the function get_volume_geom
    :param vol_geom: volume_object, the second output of the function get_volume_geom
    :param proj_geom: proj_geom, the output of the function get_projection_geom
    :return: ndarray, an array of dimension (N, det_dims[0], det_dims[1], det_dims[2]) that contains the sinogram.
    """
    projections_id, projections = astra.creators.create_sino3d_gpu(obj_id, proj_geom, vol_geom)
    return projections


def get_dims_from_phantom(phantom):
    """
    Compute the value of det_dims from a phantom data.

    :param phantom: ndarray, an array that contains the phantom image. (T, X, Y)
    :return: tuple, a tuple that contains the dimension of the detector
    """
    phantom_shape = phantom.shape
    return int(phantom_shape[1]*np.sqrt(2)), int(phantom_shape[2]*np.sqrt(2))


def add_noise_to_sino(s, sigma, type="Gaussian"):
    """
    Add noise of type "type" to the sinogram s, with noise level sigma.

    :param s: ndarray, sinogram of dimension (H, W, D)
    :param sigma: float32, noise level
    :param type: str, type of noise
    :return: ndarray, corrupted sinogram with the same dimension of s
    """
    if type == "Gaussian":
        eta = np.random.normal(size=s.shape, scale=sigma)
    else:
        eta = 0
        print('Noise type not found. No noise added to img.')

    return s + eta


def get_dims_from_projection(proj):
    """
    Get det_dim from the projection (724, 101, 724)

    :param proj: ndarray, an array that contains the projection data
    :return: tuple, a tuple with the dimension of the detector
    """
    dims = proj.shape
    return int(dims[0]), int(dims[2])


def get_sinogram(projections, proj_geom):
    """
    Get the sinogram object of AstraToolbox from the projections and the projection geometry

    :param projections: ndarray, the projection data
    :param proj_geom: projection_obj, the output of the function get_projection_geom
    :return: int, the ID of an AstraToolbox sinogram
    """
    return astra.data3d.create('-sino', proj_geom, projections)
