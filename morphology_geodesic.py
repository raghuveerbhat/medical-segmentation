from tkinter.tix import Balloon
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

from skimage.util import img_as_float

from skimage.filters import gaussian
from skimage.segmentation import active_contour


from skimage.segmentation import (morphological_chan_vese,
                                  checkerboard_level_set,morphological_geodesic_active_contour,inverse_gaussian_gradient)

from skimage.filters import threshold_multiotsu

IMG_PATH = '/Users/raghuveerbhat/Downloads/cv/FinalAssignment/Data'
files = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))] # read all files in IMG_PATH

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


for file in files:
    target_img_path = os.path.join(IMG_PATH,file)
    if(target_img_path.endswith('.png')):
        image = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        backup_img = deepcopy(image)
        # init_ls = checkerboard_level_set(image.shape, 6)
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = store_evolution_in(evolution)
        image = img_as_float(image)
        gradient = inverse_gaussian_gradient(image)
        init_ls = np.zeros(image.shape, dtype=np.int8)
        init_ls[10:-10, 10:-10] = 1
        ls = morphological_geodesic_active_contour(gradient, num_iter=35, init_level_set=init_ls,
                                    smoothing=3, balloon=-10,iter_callback=callback)
        evolution = np.array(evolution)

        plt.imshow(ls)
        plt.show()

