import sklearn.metrics as metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import numpy as np

def getIOUScores(y_true, y_pred):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    x = metrics.jaccard_score(y_true, y_pred, average=None)
    return x

def getSSIMScores(im1, im2, channel_axis=None, classes=6):
    scores = []
    for i in range(0,classes):
        c_g = np.full(im1.shape, 10,dtype=np.uint8)
        c_r = np.full(im2.shape, 10,dtype=np.uint8)
        c_g[np.where(im1 == i)] = i
        c_r[np.where(im2 == i)] = i
        x = ssim(c_g, c_r, data_range=c_g.max() - c_r.min(), channel_axis=channel_axis)
        scores.append(x)
    # x = ssim(im1, im2, data_range=im2.max() - im2.min(), channel_axis=channel_axis)
    return np.array(scores)

def getMeanSqError(im1, im2):
    x = mean_squared_error(im1, im2)
    return x