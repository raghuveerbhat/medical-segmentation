'''
File			    : main3d.py
Application		    : 2D and 3D brain MRI segmentation (6 classes and 10 slices)
Author			    : Raghuveer Bhat R
Last Modified Date	: 13/05/2022
description		    : This file is the main entry point for 2D as well as 3D segmentation. Before running this file create "Output1" folder in the current directory. 
                      This file requires "Brain.mat" to run it.
Libraries required  : skimage, numpy, matplotlib, scipy, skfuzzy, cv2(opencv), sklearn
Usage			    : python3 main3d.py [3D/2D] [fuzzyc/multiotsu (brain segmentation method)] [default/otsu (foreground background segmentation)] [write/show]
'''

import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import cv2
from skimage.filters import threshold_multiotsu
from skimage import measure
import skfuzzy as fuzz
import sys


from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat, cube, octahedron, ball, disk)

from brain_regions import BrainRegions3D
from evaluation import *

from utils import *

'''
	Load the "Brain.mat" file and normalize it in the range(0-255)
'''
def load_dataset(file):
    mat = scipy.io.loadmat(file)
    image3d = mat['T1']
    for i in range(0,10):
        image = image3d[:,:,i]
        im = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im = np.uint8(im*255)
        image3d[:,:,i] = im
    labels = mat['label']
    return image3d, labels

'''
	Run 3D segmentation algorithm on all the slices simultaneuosly
'''
def segmentation_3D(image3d, labels, segmentation_method="default", segment_brain_method="fuzzyc", write_or_show="show"):
    print("3D segmentation in progress ...")
    # Get all the masks for performing segmentation 
    b = BrainRegions3D(image3d,method=segmentation_method)
    print("Regions extracted.")
    # Class1 in class 2 region
    # Applying multi-Otsu threshold for the default value, generating three classes.
    thresholds_or2 = threshold_multiotsu(image3d,classes=3)
    # Using the threshold values, we generate the three regions.
    regions_or2 = np.digitize(image3d, bins=thresholds_or2)
    t = b.or2 > 44

    print("Brain region segmentation in progress ...")
    regions_brain = None
    if segment_brain_method == "fuzzyc":
        vec_br = b.br.reshape((1,-1))
        ncenters = 4
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        vec_br, ncenters, 2.0, error=0.005, maxiter=1000, init=None)
        cluster_membership = np.argmax(u, axis=0)
        regions_brain = cluster_membership.reshape(b.br.shape).astype('uint8')
    else:
        # Applying multi-Otsu threshold for the default value, generating four classes.
        thresholds_or2 = threshold_multiotsu(b.br,classes=4)
        # Using the threshold values, we generate the three regions.
        regions_brain = np.digitize(b.br, bins=thresholds_or2)
    print("Processing completed.")
    # Class 3, 4 and 5
    region_brain_final = label_regions_brain_3d(regions_brain)
    # Class 0
    final_segmentation = np.zeros(image3d.shape, dtype=np.uint8)
    # Class 1
    final_segmentation[np.where(b.or1_mask == 255)] = 1
    # Class 2
    final_segmentation[np.where(b.or2_mask == 255)] = 2
    # Class 1 inside class 2
    final_segmentation[np.where(t == 1)] = 1
    final_segmentation = final_segmentation | region_brain_final
    # Scores
    iou_scores3d = getIOUScores(labels,final_segmentation)
    mssim_3d = getSSIMScores(labels,final_segmentation,channel_axis=2)
    msqe_3d = getMeanSqError(labels,final_segmentation)
    print("IOU SCORES: ",iou_scores3d)
    print("SSIM: ",mssim_3d)
    show_3D(final_segmentation)


'''
	Run 2D segmentation algorithm on each slice one by one.
'''
def segmentation_2D(image3d, labels, segmentation_method="default", segment_brain_method="fuzzyc", write_or_show="show"):
    iou_scores_2d = []
    mssim_2d = []
    msqe_2d = []
    file_names = ["Slice1 ", "Slice2 ", "Slice3 ", "Slice4 ", "Slice5 ", "Slice6 ", "Slice7 ", "Slice8 ", "Slice9 ", "Slice10 "]
    file_names.append("Average ")
    for idx in range(0,image3d.shape[2]):
        print(f'Processing {file_names[idx]} ...')
        # Get all the masks for performing segmentation 
        b = BrainRegions3D(image3d[:,:,idx], type="2D", method=segmentation_method)
        vec_br = b.br.reshape((1,-1))
        regions_brain = None
        if segment_brain_method == "multiotsu":
            # Applying multi-Otsu threshold for the default value, generating
            # four classes.
            b.br = b.increaseContrast(b.br, 0.8)
            thresholds_brain = threshold_multiotsu(b.br,classes=4)
            # Using the threshold values, we generate the three regions.
            regions_brain = np.digitize(b.br, bins=thresholds_brain)
        if segment_brain_method == "fuzzyc":
            ncenters = 4
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            vec_br, ncenters, 2.0, error=0.005, maxiter=1000, init=None)
            cluster_membership = np.argmax(u, axis=0)
            regions_brain = cluster_membership.reshape(b.br.shape).astype('uint8')

        # Segmenting class 1 in class 2 region
        # Applying multi-Otsu threshold for the default value, generating three classes.
        b.or2 = b.increaseContrast(b.or2, 0.048)
        thresholds_or2 = threshold_multiotsu(b.or2,classes=3)
        # Using the threshold values, we generate the three regions.
        regions_or2 = np.digitize(b.or2, bins=thresholds_or2)

        brain_final_segement = label_regions_brain(regions_brain)
        
        or2_class1_segment = np.zeros(b.or2.shape, dtype=np.uint8)
        regions = measure.regionprops(regions_or2)
        regions = sorted(regions, key=sort_by_area, reverse=True)

        #Get the class 1 segment in or2 region
        npc = np.array(regions[-1].coords)
        or2_class1_segment[(npc[:,0],npc[:,1])] = 1

        region_or1_or2 = b.or1_mask
        region_or1_or2[np.where(region_or1_or2 == 255)] = 1
        region_or1_or2 = region_or1_or2 | b.or2_mask
        region_or1_or2[np.where(region_or1_or2 == 255)] = 2
        region_or1_or2[np.where(regions_or2 == 2)] = 1
        final_segmentation = np.uint8(brain_final_segement) | np.uint8(region_or1_or2)

        # Calculate scores (metrics)
        score = getIOUScores(labels[:,:,idx],final_segmentation)
        iou_scores_2d.append(score.tolist())
        mssim = getSSIMScores(labels[:,:,idx],final_segmentation)
        mssim_2d.append(mssim.tolist())
        mse = getMeanSqError(labels[:,:,idx],final_segmentation)
        msqe_2d.append(mse)

        if write_or_show == "write":
            write_segmented_image(final_segmentation, file_names[idx])
        elif write_or_show == "show":
            print("IOU SCORE: ", score)
            print("SSIM: ", mssim)
            show_image(show_seg_regions(final_segmentation))

    iou_scores_2d = np.array(iou_scores_2d)
    mssim_2d = np.array(mssim_2d)
    msqe_2d = np.array(msqe_2d)
    iou_scores_2d = np.vstack((iou_scores_2d, np.average(iou_scores_2d,axis=0)))
    iou_scores_2d = np.hstack((iou_scores_2d, np.array([np.average(iou_scores_2d,axis=1)]).T))
    mssim_2d = np.vstack((mssim_2d, np.average(mssim_2d,axis=0)))
    mssim_2d = np.hstack((mssim_2d, np.array([np.average(mssim_2d,axis=1)]).T))
    msqe_2d = np.append(msqe_2d, np.average(msqe_2d))
   

    if write_or_show == "write":
        header = np.array(['File names', '  Class0  ', '  Class1  ', '  Class2  ', '  Class3  ', '  Class4  ', '  Class5  ', 'Overall'])
        write_scores(iou_scores_2d,'Output1/iou_scores.txt',np.array(file_names), header)
    else:
         print("Average IOU scores:\n", iou_scores_2d[-1])
         print("Average SSIM: ", mssim_2d[-1])


if __name__ == "__main__":
    args = sys.argv[1:]
    arglen = len(args)

    type = "2D"
    segment_brain_method = "fuzzyc"
    segmentation_method = "otsu"
    write_or_show = "show"

    # Parse arguments
    if arglen == 1:
        if args[0] == "--help":
            print("python3 main3d.py [3D/2D] [fuzzyc/multiotsu (brain segmentation method)] [default/otsu (foreground background segmentation)] [write/show]")
            exit()
        type = args[0]
    elif arglen == 2:
        type = args[0]
        segment_brain_method = args[1]
    elif arglen == 3:
        type = args[0]
        segment_brain_method = args[1]
        segmentation_method = args[2]
    elif arglen == 4:
        type = args[0]
        segment_brain_method = args[1]
        segmentation_method = args[2]
        write_or_show = args[3]
    else:
        print("Running with defaults - 2D, fuzzyc, otsu and show options")

    #Load Brain.mat datatset
    image3d, labels = load_dataset('Brain.mat')

    if type == "3D":
        segmentation_3D(image3d,labels,segmentation_method,segment_brain_method,write_or_show)
    else:
        segmentation_2D(image3d,labels,segmentation_method,segment_brain_method,write_or_show)
    
