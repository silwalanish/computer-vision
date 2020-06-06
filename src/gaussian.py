import math
import numpy as np

from convolution import apply_filter


def gaussian(x, var):
    return math.exp(-(x ** 2) / (2 * var)) / (math.sqrt(2 * math.pi * var))


def gaussian_xx(x, var):
    return gaussian(x, var) * ((x ** 2 - var) / var ** 2)


def gen_kernel(size, var, func):
    h = math.floor(size / 2)
    kernel = np.array([[func(i, var) for i in range(-h, h + 1)]])

    return kernel


def gaussian_filter(img, sig, size):
    var = sig * sig
    kernel = gen_kernel(size, var, gaussian)

    filter_x = apply_filter(img, kernel)
    filter_y = apply_filter(img, kernel.T)
    return filter_x + filter_y


def lapacian_gaussian_filter(img, sig, size):
    var = sig * sig
    g_x_kernel = gen_kernel(size, var, gaussian)
    g_y_kernel = g_x_kernel.T

    g_xx_kernel = gen_kernel(size, var, gaussian_xx)
    g_yy_kernel = g_xx_kernel.T

    filter_xx = apply_filter(img, g_xx_kernel)
    filter_x = apply_filter(filter_xx, g_x_kernel)

    filter_yy = apply_filter(img, g_yy_kernel)
    filter_y = apply_filter(filter_yy, g_y_kernel)

    return filter_x + filter_y


def zero_crossing(image):
    z_c_image = np.zeros(image.shape)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [
                image[i + 1, j - 1],
                image[i + 1, j],
                image[i + 1, j + 1],
                image[i, j - 1],
                image[i, j + 1],
                image[i - 1, j - 1],
                image[i - 1, j],
                image[i - 1, j + 1],
            ]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h > 0:
                    positive_count += 1
                elif h < 0:
                    negative_count += 1

            z_c = (negative_count > 0) and (positive_count > 0)

            if z_c:
                if image[i, j] > 0:
                    z_c_image[i, j] = image[i, j] + np.abs(e)
                elif image[i, j] < 0:
                    z_c_image[i, j] = np.abs(image[i, j]) + d

    z_c_norm = z_c_image / z_c_image.max() * 255
    z_c_image = np.uint8(z_c_norm)

    return z_c_image
