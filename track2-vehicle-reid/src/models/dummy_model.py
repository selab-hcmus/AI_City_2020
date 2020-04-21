"""Dummy model"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from torch import nn

from src.models.base_model import BaseModel
import ipdb

class DummyModel(BaseModel):

    def __init__(self, device, low, high):
        """Initialize the model

        Args:
            device: the device (cpu/gpu) to place the tensors
            low: (float) low range
            high: (float) high range
            seq_len: (int) len of the sequece
        """
        super().__init__(device)

        self.low = low
        self.high = high
        self.seq_len = seq_len = 20

        self.build_model()

    def build_model(self):
        """Build model architecture
        """
        self.layer_dict = {
            'w1': nn.Linear(self.seq_len*4, 512),
            'w2': nn.Linear(512, 1),
        }

        self.model = nn.Sequential(
            self.layer_dict['w1'],
            nn.ReLU(),
            self.layer_dict['w2'],
            nn.Sigmoid(),
        )

    def forward(self, input_tensor):
        """Forward function of the model

        Args:
            input_tensor: pytorch input tensor
        """
        x_in = input_tensor.view(-1, self.seq_len*4).float()
        y_pred = self.model(x_in).squeeze()

        # Rescale to the range of (self.low, self.high)
        y_pred = (y_pred + self.low) * self.high

        return y_pred
