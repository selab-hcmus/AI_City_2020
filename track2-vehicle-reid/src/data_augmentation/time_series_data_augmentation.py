import numpy as np
import ipdb
from transforms3d.axangles import axangle2mat  # for rotation
import torch
class RandomRotation(object):
    def __init__(self):
        np.random.seed(0)

    def rotate(self,X):
        axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        return np.matmul(X , axangle2mat(axis,angle))
    
    def __call__(self,sample):
        return np.concatenate(
            (sample[:,:1],
            self.rotate(sample[:,1:])),
            axis=1).astype(np.float32)

class RandomPermutation(object):
    def __init__(self, nPerm=4, minSegLength=10):
        self.nPerm = nPerm
        self.minSegLength = minSegLength
    
    def permutation(self, X):
        X_new = np.zeros(X.shape)
        idx = np.random.permutation(self.nPerm)
        bWhile = True
        while bWhile == True:
            segs = np.zeros(self.nPerm+1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(self.minSegLength, X.shape[0]-self.minSegLength, self.nPerm-1))
            segs[-1] = X.shape[0]
            if np.min(segs[1:]-segs[0:-1]) > self.minSegLength:
                bWhile = False
        pp = 0
        for ii in range(self.nPerm):
            x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
            X_new[pp:pp+len(x_temp),:] = x_temp
            pp += len(x_temp)
        return(X_new)
    def __call__(self,sample):
        return np.concatenate(
            (sample[:,:1],
            self.permutation(sample[:,1:])),
            axis=1).astype(np.float32)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}