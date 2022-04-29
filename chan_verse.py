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
import math
from scipy.spatial import ConvexHull

from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.measure import label, regionprops


from skimage.segmentation import (morphological_chan_vese,
                                  checkerboard_level_set,chan_vese)

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
        print(target_img_path)
        # init_ls = checkerboard_level_set(image.shape, 6)
        # List with intermediate results for plotting the evolution
        # evolution = []
        # callback = store_evolution_in(evolution)
        # ls = morphological_chan_vese(image, num_iter=35, init_level_set=init_ls,
        #                             smoothing=3, iter_callback=callback)
        cv = chan_vese(image, mu=0.40, lambda1=1, lambda2=1, tol=1e-5,
               max_num_iter=2000, dt=0.5, init_level_set="checkerboard",
               extended_output=True)

        ls =np.uint8(cv[0]*255)
        ret, im = cv2.threshold(ls, 127, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy  = cv2.findContours(ls, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # segement1 = np.zeros((362,434))
        print(contours[2].shape)
        generators = contours[2].reshape(contours[2].shape[0],2)
        hull = ConvexHull(points=generators)
        # print(hull.vertices)
        # print(hull.vertices.shape)
        hull_vertices = generators[hull.vertices]
        
        hull_vertices = np.array([hull_vertices])
        im = np.zeros([362,434],dtype=np.uint8)
        cv2.fillPoly( im, hull_vertices, 255 )

        plt.imshow(im)
        plt.show()
        # cv2.drawContours(image, contours, 2, color=255, thickness=cv2.FILLED)




        # cv2.drawContours(image, contours, 1, (255,0,0), 3)
        plt.imshow(image)
        plt.show()

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        ax = axes.flatten()

        ax[0].imshow(image, cmap="gray")
        ax[0].set_axis_off()
        ax[0].set_title("Original Image", fontsize=12)

        ax[1].imshow(cv[0], cmap="gray")
        ax[1].set_axis_off()
        title = f'Chan-Vese segmentation - 3{len(cv[2])} iterations'
        ax[1].set_title(title, fontsize=12)

        ax[2].imshow(cv[1], cmap="gray")
        ax[2].set_axis_off()
        ax[2].set_title("Final Level Set", fontsize=12)

        ax[3].plot(cv[2])
        ax[3].set_title("Evolution of energy over iterations", fontsize=12)

        fig.tight_layout()
        plt.show()
                
