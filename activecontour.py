import numpy as np
import os
from os import listdir
from os.path import isfile, join
from copy import deepcopy
from scipy.linalg import norm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import scipy.io
import cv2
import skfuzzy as fuzz

from skimage.filters import gaussian
from skimage.segmentation import active_contour


from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)

IMG_PATH = '/Users/raghuveerbhat/Downloads/cv/FinalAssignment/Data'
files = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))] # read all files in IMG_PATH

for file in files:
    target_img_path = os.path.join(IMG_PATH,file)
    if(target_img_path.endswith('.png')):
        img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)

        s = np.linspace(0, 2*np.pi, 400)
        r = 100 + 300*np.sin(s)
        c = 200 + 300*np.cos(s)
        init = np.array([r, c]).T

        snake = active_contour(gaussian(img, 3, preserve_range=False),
                            init, alpha=0.015, beta=10, gamma=0.001)
        snake2 = active_contour(gaussian(img, 3, preserve_range=False),
                            snake, alpha=0.015, beta=10, gamma=0.001)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.plot(snake2[:, 1], snake2[:, 0], '-g', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])

        plt.show()
