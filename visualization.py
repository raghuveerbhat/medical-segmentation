import numpy as np

#colors for visualization
class_0 = np.array([0,0,102]) 
class_1 = np.array([50,50,255]) 
class_2 = np.array([102,255,255])
class_3 = np.array([255,255,0])
class_4 = np.array([255,0,0]) 
class_5 = np.array([100,0,0]) 

def show_seg_regions(segmented_image):
    final_segement = np.stack((segmented_image,)*3, axis=-1)

    for i in range(0,segmented_image.shape[0]):
        for j in range(0,segmented_image.shape[1]):
            if final_segement[i][j][0] == 0:
                final_segement[i][j] = class_0
            elif final_segement[i][j][0] == 1:
                final_segement[i][j] = class_1
            elif final_segement[i][j][0] == 2:
                final_segement[i][j] = class_2
            elif final_segement[i][j][0] == 3:
                final_segement[i][j] = class_3
            elif final_segement[i][j][0] == 4:
                final_segement[i][j] = class_4
            elif final_segement[i][j][0] == 5:
                final_segement[i][j] = class_5

    return final_segement.astype(np.uint8)