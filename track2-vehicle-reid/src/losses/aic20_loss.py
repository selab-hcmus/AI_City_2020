from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import src.utils.logging as logging
logger = logging.get_logger(__name__)

import torch
import numpy as np
import torch.nn as nn
from src.utils.reid_metrics import pdist_torch
import src.config as cfg

from src.losses.triplet_loss import TripletLoss
from src.losses.cross_entro_lbl_smooth import CrossEntropyLabelSmooth
import ipdb

class AIC20Loss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin, num_classes = 230, epsilon=0.1):
        super(AIC20Loss, self).__init__()
        self.triplet_loss = TripletLoss(margin)
        self.celsmth_loss = CrossEntropyLabelSmooth(
            num_classes,
            epsilon=epsilon,
            use_gpu = True)
        self.weight = [1.0, 1.0]

    def forward(self, inputs, labels):
        trip_lss = self.triplet_loss(inputs, labels)
        entro_lss= self.celsmth_loss(inputs['cls'], labels)
        # logger.info("Triplet loss       : %.4f" % trip_lss)
        # logger.info("Cross entropy loss : %.4f" % entro_lss)
        final_lss = self.weight[0] * trip_lss + self.weight[1] * entro_lss
        return final_lss
