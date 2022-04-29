import numpy as np

#colors for visualization
class_0 = np.array([0,0,102]) 
class_1 = np.array([75,75,255]) 
class_2 = np.array([102,255,255])
class_3 = np.array([255,255,0])
class_4 = np.array([255,0,0]) 
class_5 = np.array([100,0,0]) 


def show_seg_regions(segmented_image):
    brain_final_segement = np.stack((segmented_image,)*3, axis=-1)
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
    return brain_final_segement.astype(np.uint8)
    # print("brain_final_segement=",brain_final_segement)

    # print(type(brain_final_segement))
    # plt.imshow(brain_final_segement.astype(int))
    # plt.show()