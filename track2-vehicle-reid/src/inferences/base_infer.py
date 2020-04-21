"""Image Base dataset"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import os.path as osp
from PIL import Image

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import numpy as np
import pandas as pd
from src.utils.misc import MiscUtils
import torch
import ipdb

class BaseInfer():
    """Defines further steps afer getting outputs from model, e.g. 
       calculate additional metrics: accuracy, plot confusion mat, ...
       export prediction results to txt file
    """
    def __init__(self, **kwargv):
        self.output_dir = kwargv['output']
        self.write_output = False

    def init_metric(self, **kwargs):
        self.model      = kwargs['model']
        self.device     = kwargs['device']
        self.criterion  = kwargs['criterion']
        self.loaders    = kwargs['loaders']
        # Switch to eval mode
        self.model.eval()
        # Init some variables
        self.eval_loader = self.loaders['test'] if 'test' in self.loaders else None
        self.eval_mess   = 'Evaluate: '
        self.test_loss   = 0.0 

    def batch_evaluation(self, outputs, labels):
        pass
        """
            This function is called every batch
        """
    
    def __call__(self, **kwargs):
        self.init_metric(**kwargs)
        self.evaluation_loop()
        return self.evaluation_result()

    def evaluation_loop(self):
        """
            This function loop through every testing epoches
        """ 
        assert self.eval_loader is not None, "Evaluation loader is not specified"
        # Setup progressbar
        pbar = MiscUtils.gen_pbar(max_value=len(self.eval_loader), msg=self.eval_mess)
        with torch.no_grad():
            for i, (samples, labels) in enumerate(self.eval_loader):
                # Evaluating for the current batch
                self.batch_evaluation(samples, labels)
                # Monitor progress
                pbar.update(i+1)
        pbar.finish()
    
    def batch_evaluation(self, samples, labels):
        """
            This function is called every batch evaluation
        """ 
         # Place data on the corresponding device
        samples = samples.to(self.device)
        labels = labels.to(self.device)

        # Forwarding
        outputs = self.model(samples)
        # ipdb.set_trasce()
        loss = 0.0 #self.criterion(outputs, labels)
        self.test_loss += loss
        return outputs

    def evaluation_result(self):
        """
        This function is called at the end of evaluation process
        Final statistic results are given
        """
        eval_loss  = self.finalize_loss()
        eval_scr   = self.finalize_metric()
        self.export_output()
        return eval_loss, eval_scr

    def finalize_loss(self):
        self.test_loss /= len(self.eval_loader)
        self.logger.info('Validation loss: %.4f' % self.test_loss)
        return self.test_loss

    def finalize_metric(self, test_loss):
        return 0.0

    def init_best_model_score(self):
        self.best_score = 0.0
    
    def is_better_model(self, eval_loss, eval_scr):
        new_score = eval_scr
        better = new_score > self.best_score
        if (better):
            self.logger.info('Current best score: %.2f' % new_score)
            self.best_score = new_score
        return better
    
    def export_output(self):
        if (self.output_dir == None or self.write_output == False):
            return 
        os.makedirs(self.output_dir, mode=0o777, exist_ok=True)

    def write_enable(self):
        self.write_output = True