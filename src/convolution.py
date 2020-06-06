import math
import numpy as np


def convolve(input, kernel):
    y, x = kernel.shape

    sum = 0
    for i in range(0, y):
        for j in range(0, x):
            sum += input[(i + y - 1) % y, (j + x - 1) % x] * kernel[i, j]
    return sum


def apply_filter(img, kernel):
    h, w = img.shape

    filtered = np.zeros(img.shape, dtype=np.int)
    y, x = kernel.shape

    halfX = math.floor(x / 2)
    halfY = math.floor(y / 2)

    print("Applying the following filter...")
    print(kernel)

    for i in range(halfY, h - halfY):
        for j in range(halfX, w - halfX):
            input = img[i - halfY : i + halfY + 1, j - halfX : j + halfX + 1]
            filtered[i, j] = convolve(input, kernel)

    return filtered
