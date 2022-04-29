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
from skimage.measure import label, regionprops

from scipy.spatial import ConvexHull

from skimage.segmentation import (morphological_chan_vese,
                                  checkerboard_level_set)

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

#colors for visualization
class_0 = np.array([0,0,102]) 
class_1 = np.array([75,75,255]) 
class_2 = np.array([102,255,255])
class_3 = np.array([255,255,0])
class_4 = np.array([255,75,75]) 
class_5 = np.array([125,0,0]) 


for file in files:
    target_img_path = os.path.join(IMG_PATH,file)
    if(target_img_path.endswith('.png')):
        image = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        backup_img = deepcopy(image)
        init_ls = checkerboard_level_set(image.shape, 2)
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = store_evolution_in(evolution)
        ls = morphological_chan_vese(image, num_iter=70, init_level_set=init_ls,
                                    smoothing=11, iter_callback=callback)
        evolution = np.array(evolution)
        # print(np.max(ls))
        ls =np.uint8(ls*255)
        # plt.imshow(ls)
        # plt.show()
        # exit()
        ret, im = cv2.threshold(ls, 127, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy  = cv2.findContours(ls, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # print(contours.shape)
        # print(hierarchy.shape)
        res_img = ls
        cv2.drawContours(image, contours, 1, (255,0,0), 3)

        seg1contour = contours[0:3:2]
        segement1 = np.zeros((362,434))
        cv2.fillPoly(segement1,pts=seg1contour,color=255)
        print(np.max(segement1))

        seg2contour = contours[0:3]
        segement2 = np.zeros((362,434))
        cv2.fillPoly(segement2,pts=seg2contour,color=255)

        seg3contour = contours[0:2]
        segement3 = np.zeros((362,434))
        cv2.fillPoly(segement3,pts=seg3contour,color=255)

        seg4contour = contours[1]
        segement4 = np.zeros((362,434))
        cv2.drawContours(segement4, contours, 1, color=255, thickness=cv2.FILLED)

        plt.imshow(segement4)
        plt.show()

        needtoseg2_1 = segement2
        for i in range(0,362):
            for j in range(0,434):
                if segement3[i][j] == 255.0:
                    needtoseg2_1[i][j] = 0

        #Brain region
        needtoseg2 = np.zeros((362,434))
        for i in range(0,362):
            for j in range(0,434):
                if needtoseg2_1[i][j] == 255.0:
                    needtoseg2[i][j] = backup_img[i][j]

        

        #Outer region
        needtoseg1 = np.full((362,434), 245)
        # needtoseg1 = np.zeros((362,434))
        for i in range(0,362):
            for j in range(0,434):
                if segement1[i][j] == 255.0:
                    needtoseg1[i][j] = backup_img[i][j]

        needtoseg3 = np.full((362,434), 250)
        for i in range(0,362):
            for j in range(0,434):
                if segement4[i][j] == 255.0:
                    needtoseg3[i][j] = backup_img[i][j]
        
        plt.imshow(needtoseg3)
        plt.show()


        vectorized1 = needtoseg1.reshape((1,-1))
        vectorized2 = needtoseg2.reshape((1,-1))
        vectorized3 = needtoseg3.reshape((1,-1))
        vectorized4 = backup_img.reshape((1,-1))

        # ncenters = 3
        # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        # vectorized1, ncenters, 2, error=0.005, maxiter=1000, init=None)
        # print(u[0])
        # cluster_membership = np.argmax(u, axis=0)
        # segmented_outer = cluster_membership.reshape(needtoseg1.shape).astype('int')

        # ncenters = 4
        # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        # vectorized2, ncenters, 2, error=0.005, maxiter=1000, init=None)
        # print(u[0])
        # cluster_membership = np.argmax(u, axis=0)
        # segmented_brain = cluster_membership.reshape(needtoseg2.shape).astype('int')


        # Applying multi-Otsu threshold for the default value, generating
        # four classes.
        thresholds_brain = threshold_multiotsu(needtoseg2,classes=4)
        # Using the threshold values, we generate the three regions.
        regions_brain = np.digitize(needtoseg2, bins=thresholds_brain)


        thresholds_outerbrain = threshold_multiotsu(needtoseg3,classes=5)
        # Using the threshold values, we generate the three regions.
        regions_outerbrain = np.digitize(needtoseg3, bins=thresholds_brain)

        plt.imshow(regions_outerbrain)
        plt.show()

        # Applying multi-Otsu threshold for the default value, generating
        # four classes.
        thresholds_outer = threshold_multiotsu(needtoseg1, classes=3)
        # Using the threshold values, we generate the three regions.
        regions_outer = np.digitize(needtoseg1, bins=thresholds_outer)

        # ncenters = 5
        # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        # vectorized4, ncenters, 2, error=0.005, maxiter=1000, init=None)
        # # print(u[0])
        # cluster_membership = np.argmax(u, axis=0)
        # segmented_full = cluster_membership.reshape(backup_img.shape).astype('int')

        # plt.imshow(segmented_full)
        # plt.show()
        brain_final_segement = np.zeros((362,434))
        regions = regionprops(regions_brain)
        def sort_by_perimeter(s):
            return s.perimeter_crofton
        def sort_by_area(s):
            return s.area
        regions = sorted(regions, key=sort_by_area)
        regions[1:] = sorted(regions[1:], key=sort_by_perimeter)
        for idx, region in enumerate(regions):
            # for prop in region:
            #     print(prop, region[prop])
            print("Region_label=", region.label)
            print("Region area=",region.area)
            print("coords=",region.coords.shape)
            for i,j in region.coords:
                brain_final_segement[i][j] = idx+1
            # plt.imshow(brain_final_segement)
            # plt.show()
            # print("num_pixels=",region.num_pixels)
            print("Perimeter = ",region.perimeter_crofton)
        
        brain_final_segement = np.stack((brain_final_segement,)*3, axis=-1)

        for i in range(0,362):
            for j in range(0,434):
                if brain_final_segement[i][j][0] == 0:
                    brain_final_segement[i][j] = class_0
                elif brain_final_segement[i][j][0] == 1:
                    brain_final_segement[i][j] = class_3
                elif brain_final_segement[i][j][0] == 2:
                    brain_final_segement[i][j] = class_5
                elif brain_final_segement[i][j][0] == 3:
                    brain_final_segement[i][j] = class_4

        # print("brain_final_segement=",brain_final_segement)

        # print(type(brain_final_segement))
        # plt.imshow(brain_final_segement.astype(int))
        # plt.show()



        # fig, ax = plt.subplots()
        # ax.imshow(image, cmap=plt.cm.gray)

        # regions = regionprops(regions_brain)
        # for props in regions:
        #     y0, x0 = props.centroid
        #     print("y0,x0=",y0,x0)
        #     orientation = props.orientation
        #     x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        #     y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        #     x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        #     y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        #     ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        #     ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        #     ax.plot(x0, y0, '.g', markersize=15)

        #     minr, minc, maxr, maxc = props.bbox
        #     bx = (minc, maxc, maxc, minc, minc)
        #     by = (minr, minr, maxr, maxr, minr)
        #     ax.plot(bx, by, '-b', linewidth=2.5)

        # ax.axis((0, 600, 600, 0))
        # plt.show()

        # plt.imshow(ls_brain)
        # plt.show()


        # cv2.namedWindow('Contours1',cv2.WINDOW_NORMAL)
        # cv2.imshow('Contours1', segement3)
        # if cv2.waitKey(0):
        #     cv2.destroyAllWindows()

        # print(res_img.shape)
        # plt.imshow(segmented_outer)
        # plt.show()

        # plt.imshow(segmented_brain)
        # plt.show()

        # plt.imshow(regions_brain)
        # plt.show()

        # plt.imshow(regions_outer)
        # plt.show()
        
