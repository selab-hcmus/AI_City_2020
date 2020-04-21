"""Dummy model"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from torch import nn
#immport base model here:
import torchvision.models as models 
from src.models.lib.efficientnet.efficientnet import EfficientNet

from src.models.base_model import BaseModel
import ipdb

class TripletNet(BaseModel):

    def __init__(self, device, base, pretrained = True, n_classes=230, multi_gpus = False, head = None):
        """Initialize the model

        Args:
            device: the device (cpu/gpu) to place the tensors
            base: (str) name of base model
            n_classes: (int) number of classes
        """
        super().__init__(device)
        self.n_classes = n_classes
        self.base = base
        self.head = head
        self.pretrained = pretrained
        self.multi_gpus = multi_gpus
        self.build_model()

        
    def set_parameter_requires_grad(self, is_features_extracter):
        """Set requires grad for all model's parameters
        Args:
            is_feature_extracter: (bool) if we only use base model as feature extractor (no training)
        """
        if is_features_extracter:
            for param in self.model.parameters():
                param.requires_grad = False

    def build_model(self):
        """Build model architecture
        """
        if (self.base == 'resnet101'):
            resnet = models.resnet101(pretrained = self.pretrained)
            self.model = nn.Sequential(*list(resnet.children())[:-1])
            self.ln    = nn.Linear(2048, self.n_classes)
        elif (self.base == 'efficientnet-b0'):
            self.model = EfficientNet.from_pretrained(self.base)
            if (self.head != None):
                self.head =   nn.Sequential(
                    nn.Linear(1280, self.head),
                    nn.BatchNorm1d(self.head)
                )
            self.ln    = nn.Linear(1280, self.n_classes)
        elif (self.base == 'efficientnet-b2'):
            self.model = EfficientNet.from_pretrained(self.base)
            self.ln    = nn.Linear(1408, self.n_classes)
        elif (self.base == 'efficientnet-b4'):
            self.model = EfficientNet.from_pretrained(self.base)
            self.ln    = nn.Linear(1792, self.n_classes)
        elif (self.base == 'densenet161'):
            self.model = models.resnet101(pretrained = self.pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif (self.base == 'resnet50'):
            self.model = models.resnet50(pretrained = self.pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        else:
            raise ValueError("Model {} is not supported! ".format(self.base))
        if (self.multi_gpus):
            self.model = nn.DataParallel(self.model)   #multi-gpus

    def forward(self, input_tensor):
        """Forward function of the model
        Args:
            input_tensor: pytorch input tensor
        """
        y_feat = self.model(input_tensor).squeeze()
        y_cls  = self.ln(y_feat)
        if (self.head):
            y_feat = self.head(y_feat)
        return {'feat': y_feat, 'cls': y_cls}
