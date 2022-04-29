import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import cv2
from skimage.filters import threshold_multiotsu

mat = scipy.io.loadmat('Brain.mat')
# images = mat['T1']
# for i in range(0,10):
#     image = images[:,:,i]
#     im = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     im = np.uint8(im*255)
#     fname = "brain"+str(i)+".png"
#     cv2.imwrite(fname, im)
labels = mat['label']
label = labels[:,:,0]
# cv2.imwrite("brain_label.png", label)
for i in range(0,7):
    segment = np.zeros(label.shape)
    segment[np.where(label == i)] = i
    fname = "brain_label_" + str(i) + ".png"
    plt.imshow(segment)
    plt.savefig(fname, dpi=300)
    plt.close()





   
def makedirs(path):
    import os
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)
# =============================================================================
# Standard Fuzzy C-means algorithm 
# (https://en.wikipedia.org/wiki/Fuzzy_clustering.)
# =============================================================================

import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np 
from scipy.signal import convolve2d


def np_save(x,filename,f):
    np.savetxt(filename, x, fmt=f)

def class_wise_segmentation(result,OUTPUT_PLOT_PATH,file):
    for i in range(1,7):
        segment = np.zeros(result.shape)
        segment[np.where(result == i)] = i
        makedirs(OUTPUT_PLOT_PATH)       
        fname = "%s"+str(i)+".png"  
        seg_result_path = os.path.join(OUTPUT_PLOT_PATH,fname%(os.path.splitext(file)[0]))
        plt.imshow(segment)
        plt.savefig(seg_result_path, dpi=300)
        plt.close()



class FCM():
    def __init__(self, image, image_bit, n_clusters, m, epsilon, max_iter):
        '''Modified Fuzzy C-means clustering

        <image>: 2D array, grey scale image.
        <n_clusters>: int, number of clusters/segments to create.
        <m>: float > 1, fuzziness parameter. A large <m> results in smaller
             membership values and fuzzier clusters. Commonly set to 2.
        <max_iter>: int, max number of iterations.
        '''

        #-------------------Check inputs-------------------
        if np.ndim(image) != 2:
            raise Exception("<image> needs to be 2D (gray scale image).")
        if n_clusters <= 0 or n_clusters != int(n_clusters):
            raise Exception("<n_clusters> needs to be positive integer.")
        if m < 1:
            raise Exception("<m> needs to be >= 1.")
        if epsilon <= 0:
            raise Exception("<epsilon> needs to be > 0")

        self.image = image
        self.image_bit = image_bit
        self.n_clusters = n_clusters
        self.m = m
        self.epsilon = epsilon
        self.max_iter = max_iter

        self.shape = image.shape # image shape
        self.X = image.flatten().astype('float') # flatted image shape: (number of pixels,1) 
        self.numPixels = image.size
       
    #--------------------------------------------- 
    def initial_U(self):
        U=np.zeros((self.numPixels, self.n_clusters))
        idx = np.arange(self.numPixels)
        for ii in range(self.n_clusters):
            idxii = idx%self.n_clusters==ii
            U[idxii,ii] = 1      
        return U
    
    def update_U(self):
        '''Compute weights'''
        c_mesh,idx_mesh = np.meshgrid(self.C,self.X)
        power = 2./(self.m-1)
        p1 = abs(idx_mesh-c_mesh)**power
        p2 = np.sum((1./abs(idx_mesh-c_mesh))**power,axis=1)
        
        return 1./(p1*p2[:,None])

    def update_C(self):
        '''Compute centroid of clusters'''
        numerator = np.dot(self.X,self.U**self.m)
        denominator = np.sum(self.U**self.m,axis=0)
        return numerator/denominator
                       
    def form_clusters(self):      
        '''Iterative training'''        
        d = 100
        self.U = self.initial_U()
        if self.max_iter != -1:
            i = 0
            while True:             
                self.C = self.update_C()
                old_u = np.copy(self.U)
                self.U = self.update_U()
                d = np.sum(abs(self.U - old_u))
                print("Iteration %d : cost = %f" %(i, d))

                if d < self.epsilon or i > self.max_iter:
                    break
                i+=1
        else:
            i = 0
            while d > self.epsilon:
                self.C = self.update_C()
                old_u = np.copy(self.U)
                self.U = self.update_U()
                d = np.sum(abs(self.U - old_u))
                print("Iteration %d : cost = %f" %(i, d))

                if d < self.epsilon or i > self.max_iter:
                    break
                i+=1
        print(self.C.shape)
        print(self.U.shape)
        self.segmentImage()


    def deFuzzify(self):
        return np.argmax(self.U, axis = 1)

    def segmentImage(self):
        '''Segment image based on max weights'''

        result = self.deFuzzify()
        self.result = result.reshape(self.shape).astype('int')

        return self.result
    
    
def main():
    IMG_PATH = '/Users/raghuveerbhat/Downloads/cv/FinalAssignment/Data'
    OUTPUT_PATH = '/Users/raghuveerbhat/Downloads/cv/FinalAssignment/Output'
    OUTPUT_PLOT_PATH = os.path.join(OUTPUT_PATH,'segmentation') # path for output (plot directory)
    
    IS_PLOT = False
    IS_SAVE = True
    
    files = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))] # read all files in IMG_PATH
    it = 0
    for file in files:
        target_img_path = os.path.join(IMG_PATH,file)
        if(target_img_path.endswith('.png')):
            try:
                #--------------Lord image file--------------  
                img= cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE) # cf. 8bit image-> 0~255
                #--------------Clustering--------------  
                cluster = FCM(img, image_bit=8, n_clusters=4, m=2, epsilon=0.05, max_iter=200)
                cluster.form_clusters()
                result=cluster.result
                result_up = result
                result_up[np.where(result == 0)] = 4 
                if False:
                    fname = "brain0" + str(it)
                    it+=1
                    np_save(result,fname,'% 4d')
                if True:
                    class_wise_segmentation(result_up,OUTPUT_PLOT_PATH,file)
                  
                #-------------------Plot and save result------------------------
                if IS_PLOT:      
                    
                    fig=plt.figure(figsize=(12,8),dpi=100)
                
                    ax1=fig.add_subplot(1,2,1)
                    ax1.imshow(img,cmap='gray')
                    ax1.set_title('image')
                
                    ax2=fig.add_subplot(1,2,2)
                    ax2.imshow(result)
                    ax2.set_title('segmentation')
                    
                    plt.show(block=False)
                    plt.close()
                    
                if IS_SAVE:
                    makedirs(OUTPUT_PLOT_PATH)            
                    seg_result_path = os.path.join(OUTPUT_PLOT_PATH,"%s.png"%(os.path.splitext(file)[0]))
                    plt.imshow(result)
                    plt.savefig(seg_result_path, dpi=300)
                    plt.close()
                    
                
            except IOError:
                print("Error")
        else:
            print("skipping as it is not an image....")

if __name__ == '__main__':
    main()
