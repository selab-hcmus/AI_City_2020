from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import os.path as osp

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import numpy as np
import pandas as pd
import torch
from src.inferences.base_infer import BaseInfer

class ImgClsInfer(BaseInfer):
    """    
       Note: this class is used for image classification task
    """
    def __init__(self, **kwargs):
        pass

    def init_metric(self, **kwargs):
        """
            This function is called once before evaluation
        """
        self.logger = kwargs['logger']
        #calculate accuracy metric
        self.n_correct, self.n_samples = 0, 0
        self.all_predictions = []
        self.all_labels = []
    
    def batch_update(self, outputs, labels):
        """
            This function is called every batch
        """ 
        # Predicting
        _, predicted = torch.max(outputs.data, 1)
        # Statistics
        self.n_samples += labels.size(0)
        self.n_correct += (predicted == labels.data).sum()
        # Save predictions to list
        self.all_predictions += list(predicted)
        self.all_labels += list(labels)

    def finalize_metric(self, test_loss):
        """
            This function is called at the end of evaluation process
            Final statistic results are given
        """
        self.test_acc = 1.0 * self.n_correct / self.n_samples
        self.logger.info('Validation accuracy: %.4f' % self.test_acc)
        return self.test_acc
    
        

    # # Save predictions to txt files
    #     if (args.pred_path is not None):
    #         dir = os.path.dirname(args.pred_path)
    #         os.makedirs(dir, exist_ok=True)
    #         with open(args.pred_path, "w") as fo:
    #             for pred in predictions:
    #                 fo.write("{},{}\n".format(pred[0], pred[1]))
    #             fo.close()