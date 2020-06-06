import os
import cv2
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt

from gaussian import gaussian_filter, lapacian_gaussian_filter, zero_crossing

IMAGE_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../images")


def main():
    filename = "unsplash1.jpg"
    filepath = os.path.join(IMAGE_DIR_PATH, filename)

    print("Reading image...")
    img = cv2.imread(filepath, 0)
    img = cv2.resize(img, (512, 512))

    print("Applying Gaussian filter...")
    start = datetime.datetime.now()
    gaus_img = gaussian_filter(img, 1, 7)
    end = datetime.datetime.now()
    print("Completed IN: {}".format(end - start))

    print("Applying LoG filter...")
    start = datetime.datetime.now()
    log_img = lapacian_gaussian_filter(img, 1, 7)
    end = datetime.datetime.now()
    print("Completed IN: {}".format(end - start))

    log_img = zero_crossing(log_img)

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(gaus_img, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(log_img, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
