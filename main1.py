import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import cv2
from torch import xlogy_
from brain_regions import BrainRegions,performSegmentation,show_image, show_all_masks, show_all_regions

from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa

from skimage.filters import threshold_multiotsu

from skimage import measure

from skimage.measure import label, regionprops

from visualization import show_seg_regions

footprint1 = disk(1)
footprint2 = disk(2)
footprint3 = disk(3)
footprint4 = disk(4)
footprint5 = disk(5)

IMG_PATH = '/Users/raghuveerbhat/Downloads/cv/FinalAssignment/Data'
files = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))] # read all files in IMG_PATH

for file in files:
    target_img_path = os.path.join(IMG_PATH,file)
    if(target_img_path.endswith('.png')):
        image = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        b = BrainRegions(image, method="chanvese", convex_hull=False, dilate_and_threshold=True)

        # vec_br = b.br.reshape((1,-1))
        # vec_or2 = b.or2.reshape((1,-1))
        # show_all_regions(b)
        
        # Applying multi-Otsu threshold for the default value, generating
        # four classes.
        thresholds_brain = threshold_multiotsu(b.br,classes=4)
        # Using the threshold values, we generate the three regions.
        regions_brain = np.digitize(b.br, bins=thresholds_brain)

        # Applying multi-Otsu threshold for the default value, generating
        # four classes.
        thresholds_or2 = threshold_multiotsu(b.or2,classes=3)
        # Using the threshold values, we generate the three regions.
        regions_or2 = np.digitize(b.or2, bins=thresholds_or2)



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

        visualize_brain = show_seg_regions(brain_final_segement)

        # show_image(visualize_brain)
        or2_class1_segment = np.zeros((362,434))
        regions = regionprops(regions_or2)
        regions = sorted(regions, key=sort_by_area, reverse=True)
        # for idx, region in enumerate(regions):
        # for i,j in regions[-1].coords:
        #     or2_class1_segment[i][j] = 1
        # print(regions[-1].coords)
        x = np.array(regions[-1].coords)
        print(x.shape)
        y = (x[:,0],x[:,1])
        or2_class1_segment[y] = 1


        

        # show_image(regions_brain)
        show_image(or2_class1_segment)

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

