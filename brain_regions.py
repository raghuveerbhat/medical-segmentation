# University of Birmingham
# Author: Raghuveer Bhat R

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

from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.measure import label, regionprops, find_contours

from scipy.spatial import ConvexHull

from skimage.segmentation import (morphological_chan_vese,
                                  checkerboard_level_set, chan_vese)

from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa

from skimage.filters import threshold_multiotsu


def performSegmentation(img, method="multiotsu", no_of_class=4):
    vectorized = img.reshape((1,-1))
    segmented_region = None
    if method == "multiotsu":
        # Applying multi-Otsu threshold for the default value, generating
        # four classes.
        thresholds_brain = threshold_multiotsu(img,classes=no_of_class)
        # Using the threshold values, we generate the three regions.
        segmented_region = np.digitize(img, bins=thresholds_brain)
    elif method == "fuzzycmeans":
        ncenters = no_of_class
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        vectorized, ncenters, 2, error=0.005, maxiter=1000, init=None)
        cluster_membership = np.argmax(u, axis=0)
        segmented_region = cluster_membership.reshape(img.shape).astype('int')

    return segmented_region

def show_image(image):
    plt.imshow(image)
    plt.show()

def show_all_masks(b):
    plt.imshow(b.or1_mask)
    plt.show()
    plt.imshow(b.or2_mask)
    plt.show()
    plt.imshow(b.br_mask)
    plt.show()
    plt.imshow(b.or1br_mask)
    plt.show()
    plt.imshow(b.or2br_mask)
    plt.show()
    plt.imshow(b.or1or2_mask)
    plt.show()

def show_all_regions(b):
    plt.imshow(b.or1)
    plt.show()
    plt.imshow(b.or2)
    plt.show()
    plt.imshow(b.br)
    plt.show()
    plt.imshow(b.or1br)
    plt.show()
    plt.imshow(b.or2br)
    plt.show()
    plt.imshow(b.or1or2)
    plt.show()


class BrainRegions:
    def __init__(self, image, method="chanvese", convex_hull=False, dilate_and_threshold=True):
        self.backup_img = deepcopy(image)
        self.ls = None
        # image = self.increaseContrast(image, 1.75)
        self.or1_mask = None
        self.or2_mask = None
        self.br_mask = None
        self.getDifferentRegionsMask(image, method=method, use_convex_hull=convex_hull, dilate_and_threshold=dilate_and_threshold)
        self.or1br_mask = (self.or1_mask | self.br_mask)
        self.or2br_mask = self.or2_mask | self.br_mask
        self.or1or2_mask = self.or1_mask | self.or2_mask
        self.or1 = self.imageFromMask(self.or1_mask)
        self.or2 = self.imageFromMask(self.or2_mask)
        self.br = self.imageFromMask(self.br_mask)
        self.or1br = self.imageFromMask(self.or1br_mask)
        self.or2br = self.imageFromMask(self.or2br_mask)
        self.or1or2 = self.imageFromMask(self.or1or2_mask)

    def getDifferentRegionsMask(self, image, method="chanvese", use_convex_hull=False, dilate_and_threshold=True):
        if method == "chanvese":
            cv = chan_vese(image, mu=0.40, lambda1=1, lambda2=1, tol=1e-5,
               max_num_iter=2000, dt=0.5, init_level_set="checkerboard",
               extended_output=True)
            self.ls = np.uint8(cv[0]*255)
        elif method == "m-chanvese":
            init_ls = checkerboard_level_set(image.shape, 6)
            self.ls = morphological_chan_vese(image, num_iter=35, init_level_set=init_ls,
                                        smoothing=2)
            self.ls = np.uint8(self.ls*255)

        contours, hierarchy = self.findContour(self.ls)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # getOuter1Region
        segcontour = contours[0:2]
        self.or1_mask = np.zeros((362,434), dtype=np.uint8)
        cv2.fillPoly(self.or1_mask,pts=segcontour,color=255)

        # getOuter2Region
        segcontour = contours[1:3]
        self.or2_mask = np.zeros((362,434), dtype=np.uint8)
        cv2.fillPoly(self.or2_mask,pts=segcontour,color=255)

        # getBrainRegion
        self.br_mask = np.zeros((362,434), dtype=np.uint8)
        cv2.drawContours(self.br_mask, contours, 2, color=255, thickness=cv2.FILLED)

        if use_convex_hull is True:
            # Get the brain segmented coordinates
            generators = contours[2].reshape(contours[2].shape[0],2)
            hull = ConvexHull(points=generators)
            # print(hull.vertices)
            # print(hull.vertices.shape)
            hull_vertices = generators[hull.vertices]
        
            hull_vertices = np.array([hull_vertices])
            self.br_mask = np.zeros((362,434),dtype=np.uint8)
            cv2.fillPoly(self.br_mask, hull_vertices, 255)

            self.or2_mask = (self.or2_mask & self.br_mask) ^ self.or2_mask
        if dilate_and_threshold is True:
            footprint_d = disk(7)
            footprint_o = disk(2)
            footprint_c = disk(4)
            dilated = dilation(self.br_mask,footprint_d)
            image = self.imageFromMask(dilated)
            # show_image(image)
            ret, im = cv2.threshold(image, 45, 255, cv2.THRESH_BINARY)
            # show_image(im)
            opened = opening(im, footprint_o)
            # show_image(opened)
            closed = closing(opened, footprint_c)
            # show_image(closed)
            contours, hierarchy = self.findContour(np.uint8(opened))
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            self.br_mask = np.zeros((362,434), dtype=np.uint8)
            cv2.drawContours(self.br_mask, contours, 0, color=255, thickness=cv2.FILLED)

            self.or2_mask = (self.or2_mask & self.br_mask) ^ self.or2_mask


    def findContour(self, image):
        ret, im = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        return cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    def imageFromMask(self, mask):
        im = np.zeros((362,434))
        for i in range(0,362):
            for j in range(0,434):
                if mask[i][j] == 255.0:
                    im[i][j] = self.backup_img[i][j]
        return im

    def increaseContrast(self, image, val):
        im = image
        for i in range(0, 362):
            for j in range(0, 434):
                if im[i][j] * val <= 255:
                    im[i][j] = np.uint8(im[i][j] * val)
                else:
                    im[i][j] = 255
        return im                
        