import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage

from pyellipsoid import drawing

import visualization

# ellipsoid dataset (2000 grayscale images 512x512, normalized in 0-1)

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import random
import pickle


def add_ellipse(fig, center_range=(100, 100, 10), axes_range=(30, 30, 3)):
    H, W, D = fig.shape

    ell_center = (random.randrange(center_range[0], H - center_range[0]),
                  random.randrange(center_range[1], W - center_range[1]),
                  random.randrange(center_range[2], D - center_range[2]))

    ell_axes = (random.randrange(axes_range[0], 2*axes_range[0]),
                random.randrange(axes_range[1], 2*axes_range[1]),
                random.randrange(axes_range[2], 2*axes_range[2]))

    ell_angle = np.deg2rad([random.randrange(90), random.randrange(90), random.randrange(90)])

    ell_opacity = random.random()

    overlay = drawing.make_ellipsoid_image((D, W, H), ell_center, ell_axes, ell_angle)
    overlay = np.transpose(overlay, (2, 1, 0))
    overlay = overlay.astype('float32')

    fig = unify(fig, overlay, ell_opacity)
    return fig


def add_circle(fig, center_range=(100, 100, 10)):
    H, W, D = fig.shape

    point_center = (random.randrange(center_range[0], H - center_range[0]),
                    random.randrange(center_range[1], W - center_range[1]),
                    random.randrange(center_range[2], D - center_range[2]))
    point_axes = (3, 3, 3)
    point_angle = (0, 0, 0)
    point_opacity = 0.9

    overlay = drawing.make_ellipsoid_image((D, W, H), point_center, point_axes, point_angle)
    overlay = np.transpose(overlay, (2, 1, 0))
    overlay = overlay.astype('float32')

    fig = unify(fig, overlay, point_opacity)
    return fig


def add_line(fig, center_range=(100, 100, 10), length_range=80):
    H, W, D = fig.shape

    line_center = (random.randrange(center_range[0], H - center_range[0]),
                  random.randrange(center_range[1], W - center_range[1]),
                  random.randrange(center_range[2], D - center_range[2]))

    line_axes = (random.randrange(length_range, 2*length_range), 1, 1)
    line_angle = np.deg2rad([random.randrange(90), random.randrange(90), random.randrange(90)])
    line_opacity = 0.9

    overlay = drawing.make_ellipsoid_image((D, W, H), line_center, line_axes, line_angle)
    overlay = np.transpose(overlay, (2, 1, 0))
    overlay = overlay.astype('float32')

    fig = unify(fig, overlay, line_opacity)
    return fig


def unify(fig, overlay, opacity):
    H, W, D = fig.shape

    fig[overlay >= fig] = overlay[overlay >= fig] * opacity
    return fig


def get_data():
    N, H, W, D = 3, 512, 512, 32  # Shape of the dataset.
    N_ell = 10  # Number of ellipses in each image.
    N_circle = 15 # Number of circles of opacity 0.9.
    N_lines = 10 # Number of lines of opacity 0.9
    center_range = (50, 50, 0)  # Possible centers of the ellipses: (c_range, dim - c_range)
    axes_range = (100, 50, 30)  # Possible radius of the ellipses: (r_range, 100 - r_range)

    ellipsoid_dataset = np.empty((N, H, W, D), dtype=np.float32)

    for i in range(N):
        print('Drawing the ', str(i), '-th Ellipsoid', end='')
        fig = np.zeros((H, W, D))

        for n_ell in range(N_ell):
            fig = add_ellipse(fig, center_range, axes_range)
            print('.', end='')
        print('/', end='')

        for n_circle in range(N_circle):
            fig = add_circle(fig, center_range)
            print('.', end='')
        print('/', end='')

        for n_line in range(N_lines):
            fig = add_line(fig)
            print('.', end='')

        fig = fig / fig.max()
        ellipsoid_dataset[i, :, :, :] = fig
        print('')

    return ellipsoid_dataset


def to_npz(data, PATH='./'):
    np.savez(PATH + 'ellipsoid_dataset.npz', data)


def to_np(data, PATH='./'):
    np.save(PATH + 'ellipsoid_dataset.npy', data)


def to_tif(data, PATH='./'):
    import tifffile as tif
    for i in range(data.shape[0]):
        tif.imsave(PATH + 'ellipsoid_dataset_'+ str(i) + '.tif', np.transpose(data[0], (2, 0, 1)))
