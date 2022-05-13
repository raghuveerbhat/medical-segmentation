from typing import final
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import cv2
from torch import xlogy_
from brain_regions import BrainRegions,performSegmentation,show_image, show_all_masks, show_all_regions, show_each_segment
import skfuzzy as fuzz

from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa

from skimage.filters import threshold_multiotsu

from skimage import measure

from skimage.measure import label, regionprops

from visualization import show_seg_regions

from evaluation import *

import scipy.io

footprint1 = disk(1)
footprint2 = disk(2)
footprint3 = disk(3)
footprint4 = disk(4)
footprint5 = disk(5)
iou_scores = []
file_names = []
mssim_2d = []

IMG_PATH = '/Users/raghuveerbhat/Downloads/cv/FinalAssignment/Data'
files = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))] # read all files in IMG_PATH

def get_corresponding_label(img_path):
    idx = int(img_path[-5])
    mat = scipy.io.loadmat('Brain.mat')
    labels = mat['label']
    label = labels[:,:,idx]
    return label

def write_segmented_image(final_segmented, fname):
    visualize_segmented = show_seg_regions(final_segmented)
    # show_image(visualize_segmented)
    fname = "Output/" + fname
    cv2.imwrite(fname, cv2.cvtColor(visualize_segmented, cv2.COLOR_BGR2RGB))
    
def write_scores(scores, fname, file_names, header):
    t = np.column_stack((file_names, scores))
    t = np.row_stack((header, t))
    np.savetxt(fname, t, delimiter=" ", fmt="%s")

def sort_by_perimeter(s):
        return s.perimeter_crofton

def sort_by_area(s):
        return s.area

def label_regions_brain(regions_brain):
    brain_final_segement = np.zeros((362,434))
    regions = regionprops(regions_brain)
    regions = sorted(regions, key=sort_by_area)
    if regions[-1].area>70000:
        # Background class detected so making it all 0s and running regionprops again
        regions_brain[np.where(regions_brain == 0)] = 5
        regions_brain[np.where(regions_brain == regions[-1].label)] = 0
        regions = regionprops(regions_brain)
        regions = sorted(regions, key=sort_by_area)
    regions[1:] = sorted(regions[1:], key=sort_by_perimeter)

    for idx, region in enumerate(regions):
        # for prop in region:
        #     print(prop, region[prop])
        # print("Region_label=", region.label)
        # print("Region area=",region.area)
        # print("coords=",region.coords.shape)
        # print("Label = ",idx+1)
        for i,j in region.coords:
            brain_final_segement[i][j] = idx+1
        # plt.imshow(brain_final_segement)
        # plt.show()
        # print("num_pixels=",region.num_pixels)
        # print("Perimeter = ",region.perimeter_crofton)

    brain_final_segement[np.where(brain_final_segement == 3)] = 4
    brain_final_segement[np.where(brain_final_segement == 2)] = 5
    brain_final_segement[np.where(brain_final_segement == 1)] = 3
    
    return brain_final_segement

for file in files:
    target_img_path = os.path.join(IMG_PATH,file)
    if(target_img_path.endswith('.png')):
        print(f'Processing {file} ...')
        file_names.append(target_img_path[-10:])
        image = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        b = BrainRegions(image, method="chanvese", convex_hull=True, dilate_and_threshold=True)

        vec_br = b.br.reshape((1,-1))
        print(vec_br.shape)
        # vec_or2 = b.or2.reshape((1,-1))
        # show_all_regions(b)
        regions_brain = None
        segment_brain_method = "multiotsu"



        if segment_brain_method == "multiotsu":
            # Applying multi-Otsu threshold for the default value, generating
            # four classes.
            thresholds_brain = threshold_multiotsu(b.br,classes=4)
            # Using the threshold values, we generate the three regions.
            regions_brain = np.digitize(b.br, bins=thresholds_brain)
        if segment_brain_method == "fuzzyc":
            ncenters = 4
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            vec_br, ncenters, 1.5, error=0.005, maxiter=1000, init=None)
            # print(u[0])
            cluster_membership = np.argmax(u, axis=0)
            regions_brain = cluster_membership.reshape(b.br.shape).astype('uint8')

        # Applying multi-Otsu threshold for the default value, generating
        # four classes.
        thresholds_or2 = threshold_multiotsu(b.or2,classes=3)
        # Using the threshold values, we generate the three regions.
        regions_or2 = np.digitize(b.or2, bins=thresholds_or2)

        
        brain_final_segement = label_regions_brain(regions_brain)

        or2_class1_segment = np.zeros((362,434))
        regions = regionprops(regions_or2)
        regions = sorted(regions, key=sort_by_area, reverse=True)

        #Get the class 1 segment in or2 region
        npc = np.array(regions[-1].coords)
        or2_class1_segment[(npc[:,0],npc[:,1])] = 1

        region_or1_or2 = b.or1_mask
        region_or1_or2[np.where(region_or1_or2 == 255)] = 1
        region_or1_or2 = region_or1_or2 | b.or2_mask
        region_or1_or2[np.where(region_or1_or2 == 255)] = 2
        region_or1_or2[np.where(or2_class1_segment == 1)] = 1

        
        
        # visualize_brain = show_seg_regions(brain_final_segement)
        # show_image(visualize_brain)

        final_segmentation = np.uint8(brain_final_segement) | np.uint8(region_or1_or2)
        # show_all_masks(b)
        # show_image(show_seg_regions(final_segmentation))
        # show_each_segment(final_segmentation,3)


        # write_segmented_image(final_segmentation, file_names[-1])
        score = getIOUScores(get_corresponding_label(target_img_path),final_segmentation)
        mssim = getSSIMScores(get_corresponding_label(target_img_path),final_segmentation)
        mssim_2d.append(mssim.tolist())
        print("IOU SCORE: ", score)
        print("SSIM: ", mssim)
        iou_scores.append(score.tolist())



iou_scores = np.array(iou_scores)
iou_scores = np.vstack((iou_scores, np.average(iou_scores,axis=0)))
iou_scores = np.hstack((iou_scores, np.array([np.average(iou_scores,axis=1)]).T))
mssim_2d = np.array(mssim_2d)
mssim_2d = np.vstack((mssim_2d, np.average(mssim_2d,axis=0)))
mssim_2d = np.hstack((mssim_2d, np.array([np.average(mssim_2d,axis=1)]).T))
print("Average IOU: ",iou_scores[-1])
print("Average SSIM: ",mssim_2d[-1])
file_names.append("Average   ")
header = np.array(['File names', '  Class0  ', '  Class1  ', '  Class2  ', '  Class3  ', '  Class4  ', '  Class5  ', 'Overall'])
# write_scores(iou_scores,'Output/iou_scores.txt',np.array(file_names), header)
        # show_image(regions_brain)
        # show_image(region_or1_or2)

        # show_all_masks(b)
        # ret, im = cv2.threshold(b.or2br, 53, 255, cv2.THRESH_BINARY)
        # eroded = erosion(im, footprint1)
        # eroded = erosion(eroded, footprint1)
        # eroded = erosion(eroded, footprint1)
        # eroded = erosion(eroded, footprint1)
        # # eroded = erosion(im, footprint)
        # closed2 = closing(eroded, footprint4)
        # closed2 = closing(closed2, footprint4)
        # closed2 = closing(closed2, footprint4)

        # show_image(b.br_mask)
        # dil = dilation(b.br_mask,footprint5)
        # show_image(dil)
        
        # contours = measure.find_contours(closed2, 0.8)

        # def contourArea(contours):
        #     # Expand numpy dimensions
        #     c = np.expand_dims(contours.astype(np.float32), 1)
        #     # Convert it to UMat object
        #     c = cv2.UMat(c)
        #     area = cv2.contourArea(c)
        #     return area
        # contours = sorted(contours, key=contourArea, reverse=True)

        # contours = measure.approximate_polygon(contours, tolerance=2.5)
        # fig, ax = plt.subplots()
        # ax.imshow(closed2)

        # for contour in contours:
        #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        #     break

        # ax.axis('image')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # plt.show()


        # # show_image(closed)
        # # show_image(eroded)
        # # show_image(closed2)
        # # show_image(b.or2br)

        # exit()

