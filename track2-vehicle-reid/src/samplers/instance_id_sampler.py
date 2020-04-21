"""This code is adapted from 
https://github.com/CoinCheung/triplet-reid-pytorch/blob/master/batch_sampler.py

"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
import random
import logging
import sys
import ipdb


class InstanceIdSampler(Sampler):
    '''
    This sampler returns a triplet of (anchor, positive, negative) for training
    triplet-reid with batch hard. Given p (n_class) and k (n_num) 

    Function: __iter__ is called every batch iteration
    '''
    def __init__(self, dataset, n_class, n_num, *args, **kwargs):
        super(InstanceIdSampler, self).__init__(dataset, *args, **kwargs)
        self.n_class = n_class
        self.n_num = n_num
        self.batch_size = self.n_class * self.n_num
        self.dataset    =   dataset.get_nclasses()
        self.labels     =   dataset.get_list_of_labels()
        self.labels_uniq =  dataset.get_unique_labels()
        self.len = len(dataset) // self.batch_size
        self.lb_img_dict = dataset.get_inst2imgs_dict()
        self.iter_num = len(self.labels_uniq) // self.n_class

    def __iter__(self):
        curr_p = 0
        np.random.shuffle(self.labels_uniq)
        for k, v in self.lb_img_dict.items():
            random.shuffle(self.lb_img_dict[k])
        for i in range(self.iter_num):
            label_batch = self.labels_uniq[curr_p: curr_p + self.n_class]
            curr_p += self.n_class
            idx = []
            for lb in label_batch:
                if len(self.lb_img_dict[lb]) > self.n_num:
                    idx_smp = np.random.choice(self.lb_img_dict[lb],
                            self.n_num, replace = False)
                else:
                    idx_smp = np.random.choice(self.lb_img_dict[lb],
                            self.n_num, replace = True)
                idx.extend(idx_smp.tolist())
            yield idx

    def __len__(self):
        return self.iter_num