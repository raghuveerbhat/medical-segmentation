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


mat = scipy.io.loadmat('Brain.mat')
# images = mat['T1']    
# for i in range(0,10):
#     image = images[:,:,i]
#     im = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     im = np.uint8(im*255)
#     fname = "brain"+str(i)+".png"
#     cv2.imwrite(fname, im)
# labels = mat['label']
# label = labels[:,:,0]
# cv2.imwrite("brain_label.png", label)
# for i in range(0,7):
#     segment = np.zeros(label.shape)
#     segment[np.where(label == i)] = i
#     fname = "brain_label_" + str(i) + ".png"
#     plt.imshow(segment)
#     plt.savefig(fname, dpi=300)
#     plt.close()

class Fuzzy_Clustering:
    def __init__(self, n_clusters=4, max_iter=150, fuzzines=2, error=1e-5, random_state=42, dist="euclidean", method="Cmeans"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.fuzzines = fuzzines
        self.error = error
        self.random_state = random_state
        self.dist = dist
        self.method = method
        
    def fit(self, X):
        memberships = self._init_mem(X)
              
        iteration = 0
        while iteration < self.max_iter:
            membershipsNew = deepcopy(memberships)
            new_class_centers = self._update_centers(X, memberships)
            distance = self._calculate_dist(X,memberships,new_class_centers)
            memberships = self._update_memberships(X, memberships, new_class_centers, distance)
            iteration += 1
            if norm(memberships - membershipsNew) < self.error:
                break
            
        return memberships, new_class_centers
    
    def _init_mem(self,X):
        n_samples = X.shape[0]
        n_clusters = self.n_clusters

        #initialize memberships
        rnd = np.random.RandomState(self.random_state)
        memberships = rnd.rand(n_samples,n_clusters)

        #update membership relative to classes
        summation = memberships.sum(axis=1).reshape(-1,1)
        denominator = np.repeat(summation,n_clusters,axis=1)
        memberships = memberships/denominator
        
        return memberships

    def _update_centers(self, X, memberships):
        fuzzyMem = memberships ** self.fuzzines
        new_class_centers = (np.dot(X.T,fuzzyMem)/np.sum(fuzzyMem,axis=0)).T
        return new_class_centers
    
    def _calculate_fuzzyCov(self,X,memberships,new_class_centers):
        #calculating covariance matrix in its fuzzy form  
        fuzzyMem = memberships ** self.fuzzines
        n_clusters = self.n_clusters
        FcovInv_Class = []
        dim = X.shape[1]
        for i in range(n_clusters): 
            diff = X-new_class_centers[i]
            left = np.dot((fuzzyMem[:,i].reshape(-1,1)*diff).T,diff)/np.sum(fuzzyMem[:,i],axis=0)
            Fcov = (np.linalg.det(left)**(-1/dim))*left
            FcovInv = np.linalg.inv(Fcov)
            FcovInv_Class.append(FcovInv)
        return FcovInv_Class

    def _calculate_dist(self,X,memberships,new_class_centers):
        
        if self.method == "Gustafson–Kessel":
            n_clusters = self.n_clusters
            FcovInv_Class = self._calculate_fuzzyCov(X,memberships,new_class_centers)

            #calculating mahalanobis distance
            mahalanobis_Class = []
            for i in range(n_clusters): 
                diff = X-new_class_centers[i]
                left = np.dot(diff,FcovInv_Class[i])    
                mahalanobis = np.diag(np.dot(left,diff.T))
                mahalanobis_Class.append(mahalanobis)
            distance = np.array(mahalanobis_Class).T
            return distance
        
        elif self.method == "Cmeans":
            distance = cdist(X, new_class_centers,metric=self.dist)
            return distance

    def _update_memberships(self, X, memberships, new_class_centers, distance):
        fuzziness = self.fuzzines
        n_clusters = self.n_clusters
        n_samples = X.shape[0]
        
        power = float(2/(fuzziness - 1))
        distance = distance**power
        arr = np.zeros((n_samples,n_clusters))
        for i in range(n_clusters):
            for ii in range(n_clusters):
                arr[:,ii] = ((distance[:,i]/distance[:,ii]))
            memberships[:,i] = 1/np.sum(arr,axis=1)   
        return memberships
    


IMG_PATH = '/Users/raghuveerbhat/Downloads/cv/FinalAssignment/Data'
files = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))] # read all files in IMG_PATH

for file in files:
    target_img_path = os.path.join(IMG_PATH,file)
    if(target_img_path.endswith('.png')):
        img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        # print(img.shape)
        vectorized = img.reshape((1,-1))
        print(vectorized.shape)
        
        # fuzzy = Fuzzy_Clustering()
        # imgflatten = img.flatten().astype(float)
        # imgflatten = np.insert(np.reshape(imgflatten,(imgflatten.shape[0],1)),0,np.arange(0,len(imgflatten)),axis=1)    
        # print(imgflatten.shape)
        ncenters = 5
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        vectorized, ncenters, 2, error=0.005, maxiter=1000, init=None)
        print(u[0])
        cluster_membership = np.argmax(u, axis=0)
        segmented = cluster_membership.reshape(img.shape).astype('int')
        print(segmented.shape)



        

        # memberships, new_class_centers = fuzzy.fit(imgflatten)
        # segmented = np.argmax(memberships,axis=1)
        # segmented = np.reshape(segmented,(dim1,dim2))
        plt.imshow(segmented)
        plt.savefig("FUZZY", dpi=300)
        plt.close()
    
        # print(segmented.shape)
        # print(memberships[0])
        # print(memberships.shape)
        # print(new_class_centers.shape)