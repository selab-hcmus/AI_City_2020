from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.utils.reid_metrics import pdist_torch
import src.config as cfg
from .triplet_loss_online_utils import AllTripletSelector,HardestNegativeTripletSelector,\
            RandomNegativeTripletSelector, SemihardNegativeTripletSelector
import ipdb

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        if (selector == "hardest"):
            triplet_selector = HardestNegativeTripletSelector(margin)
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        embeddings = embeddings['feat']
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.to(cfg.device)
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean() #, len(triplets)