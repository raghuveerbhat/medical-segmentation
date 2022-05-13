import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import cv2
from skimage.filters import threshold_multiotsu
from skimage import measure
import skfuzzy as fuzz

from visualization import *

from skimage.morphology import (erosion, dilation, opening, closing, white_tophat, cube, octahedron, ball)


def show_each_segment(image, seg_class = 3):
    seg_region = np.zeros((362,434), dtype=np.uint8)
    seg_region[np.where(image == seg_class)] = 1
    plt.imshow(seg_region, cmap='gray')
    plt.show()

def show_each_segment(image, seg_class = 3):
    seg_region = np.zeros((362,434), dtype=np.uint8)
    seg_region[np.where(image == seg_class)] = 1
    plt.imshow(seg_region, cmap='gray')
    plt.show()

def write_segmented_image(final_segmented, fname):
    visualize_segmented = show_seg_regions(final_segmented)
    fname = "Output1/" + fname
    print(fname)
    fname = fname + ".png"
    cv2.imwrite(fname, cv2.cvtColor(visualize_segmented, cv2.COLOR_BGR2RGB))
    
def write_scores(scores, fname, file_names, header):
    t = np.column_stack((file_names, scores))
    t = np.row_stack((header, t))
    np.savetxt(fname, t, delimiter=" ", fmt="%s")

def show_image(image):
    plt.imshow(image,  cmap='gray')
    plt.show()

def show_3D(image,show_only8 = True):
    fig, axes = plt.subplots(3, 3, figsize=(10,7))
    if show_only8 == False:
        fig, axes = plt.subplots(3, 4, figsize=(10,7))
    for i in range(0,3):
        for j in range(0,3):
            axes[i][j].imshow(image[:,:,3*i+j])
            axes[i][j].set_title(f'#{3*i+j}')
    if show_only8 == False:
        axes[0][3].imshow(image[:,:,9])
        axes[0][3].set_title(f'#9')
    fig.tight_layout()
    plt.show()

def sort_by_perimeter(s):
    return s.perimeter_crofton

def sort_by_area(s):
    return s.area

def label_regions_brain(regions_brain):
    brain_final_segement = np.zeros((362,434))
    regions = measure.regionprops(regions_brain)
    regions = sorted(regions, key=sort_by_area)
    if regions[-1].area>70000:
        # Background class detected so making it all 0s and running regionprops again
        regions_brain[np.where(regions_brain == 0)] = 5
        regions_brain[np.where(regions_brain == regions[-1].label)] = 0
        regions = measure.regionprops(regions_brain)
        regions = sorted(regions, key=sort_by_area)
    regions[1:] = sorted(regions[1:], key=sort_by_perimeter)

    for idx, region in enumerate(regions):
        for i,j in region.coords:
            brain_final_segement[i][j] = idx+1

    brain_final_segement[np.where(brain_final_segement == 3)] = 4
    brain_final_segement[np.where(brain_final_segement == 2)] = 5
    brain_final_segement[np.where(brain_final_segement == 1)] = 3
    
    return brain_final_segement


def label_regions_brain_3d(regions_brain):
    brain_final_segement = np.zeros(regions_brain.shape, dtype=np.uint8)
    regions = measure.regionprops(regions_brain)
    regions = sorted(regions, key=sort_by_area)
    if regions[-1].area>700000:
        # Background class detected so making it all 0s and running regionprops again
        regions_brain[np.where(regions_brain == 0)] = 8
        regions_brain[np.where(regions_brain == regions[-1].label)] = 0
        regions = measure.regionprops(regions_brain)
        regions = sorted(regions, key=sort_by_area)

    for idx, region in enumerate(regions):
        for i,j,k in region.coords:
            brain_final_segement[i][j][k] = idx+6

    brain_final_segement[np.where(brain_final_segement == 6)] = 3
    brain_final_segement[np.where(brain_final_segement == 7)] = 5
    brain_final_segement[np.where(brain_final_segement == 8)] = 4
    return brain_final_segement