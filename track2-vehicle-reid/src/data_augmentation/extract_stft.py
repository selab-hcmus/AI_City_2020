import numpy as np
import ipdb
import torch
from scipy.signal import stft

class STFT(object):
    def __init__(self):
        np.random.seed(0)

    def rotate(self,X):
        axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        return np.matmul(X , axangle2mat(axis,angle))
    
    def get_stft(self, samples, sampling_freq):
        stfts = []
        for j in range(1, 4):
            stft_freq, stft_time, stft_resp = stft(samples[:, j], fs=sampling_freq)
            # stft_resp = (np.log(np.abs(stft_resp)))  # log scale of real part
            stfts.append(stft_resp[...,np.newaxis])
        return np.concatenate(stfts, axis = 2)
    

    def __call__(self,sample):
        stft_smpls = []
        for smpl in sample:
            n_samples = len(smpl)
            duration = smpl[-1, 0] - smpl[0, 0]
            if (duration == 0):
                stft = np.zeros((129,487,3))
            else:
                sampling_freq = 1. * n_samples / duration
                stft = self.get_stft(smpl, duration)
            stft_smpls.append(stft[np.newaxis, ...])
        return np.concatenate(stft_smpls, axis = 0).astype(np.float32)