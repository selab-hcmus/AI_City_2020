"""Base model"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os

import torch
from torch import nn


class BaseModel(nn.Module):

    def __init__(self, device):
        """Initialize the model

        Args:
            device: the device (cpu/gpu) to place the tensors
        """
        super().__init__()

        self.device = device
        self.layer_dict = None
        self.model = None

    def save_model(self, path):
        """Save model state dict to a give path

        Args:
            path: path to save the model to
        """
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load_model(self, path):
        """Load model state dict from a given path

        Args:
            path: path to load model from
        """
        assert os.path.isfile(path), 'Not found: {}'.format(path)
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
