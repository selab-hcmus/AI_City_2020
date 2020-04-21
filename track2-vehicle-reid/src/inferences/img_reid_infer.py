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
from src.utils.reid_metrics import reid_evaluate
from src.utils.misc import MiscUtils
import ipdb
class ImgReIdInfer(BaseInfer):
    """    
       Note: this class is used for image classification task
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def init_metric(self, **kwargs):
        """
            This function is called once before evaluation
        """
        #getting some new parameters #TODO: change base and cls class
        super().init_metric(**kwargs)
        loaders = kwargs['loaders']
        self.gal_ld = loaders['gallery']
        self.que_ld = loaders['query']
        self.logger = kwargs['logger']
        
        self.hard_eval_set = False
        if ('h_gallery' in loaders and 'h_query' in loaders):
            self.hard_eval_set = True
            self.gal_ld2 = loaders['h_gallery']
            self.que_ld2 = loaders['h_query']

        #Easy query set:
        self.que_emb, self.que_lbl = self.embed_imgs(self.que_ld, 'easy queries')
        self.que_emb = torch.cat(self.que_emb, dim = 0)
        self.que_lbl = torch.cat(self.que_lbl, dim = 0).detach().numpy()

        #Hard query set:
        if self.hard_eval_set:
            self.que_emb2, self.que_lbl2 = self.embed_imgs(self.que_ld2, 'hard queries')
            self.que_emb2 = torch.cat(self.que_emb2, dim = 0)
            self.que_lbl2 = torch.cat(self.que_lbl2, dim = 0).detach().numpy()

        self.eval_mess = "Embedding easy gallery: " 
        self.eval_loader = self.gal_ld
        self.gal_lbl = []
        self.gal_emb = []
        
    
    def batch_evaluation(self, samples, labels):
        """
            This function is called every batch evaluation
        """ 
        outputs = super().batch_evaluation(samples, labels)
        # Collecting all gallery embeddings
        self.gal_emb.append(outputs['feat'])
        self.gal_lbl.append(labels)

    def evaluation_loop(self):
        super().evaluation_loop()
        if (self.hard_eval_set):
            self.gal_emb2, self.gal_lbl2 = self.embed_imgs(self.gal_ld2, "hard gallery")
    
    def embed_imgs(self, imgloader, name = ""):
        """
            This function embeds all images in given query set to vectors
        """
        # Setup progressbar
        pbar = MiscUtils.gen_pbar(max_value=len(imgloader), msg='Embedding %s: ' % name)
        que_emb = []
        que_lbl = []
        with torch.no_grad():
            for i, (samples, labels) in enumerate(imgloader):
                samples = samples.to(self.device)
                que_emb.append(self.model(samples)['feat'])
                que_lbl.append(labels)
                #Monitor progress
                pbar.update(i+1)
        pbar.finish()
        return que_emb, que_lbl

    def finalize_metric(self):
        """
            This function is called at the end of evaluation process
            Final statistic results are given
        """
        self.gal_lbl = torch.cat(self.gal_lbl, dim = 0).cpu().detach().numpy()
        self.gal_emb = torch.cat(self.gal_emb, dim = 0)
        self.idcs, mAP, cmc, _ = reid_evaluate(self.que_emb, self.gal_emb, self.que_lbl, self.gal_lbl)
        self.logger.info('$$$ Validation mAP (easy): %.4f' % mAP)
        self.logger.info('$$$ Validation cmc (easy): %.4f' % cmc)
        self.logger.info('-' * 50)

        if self.hard_eval_set:
            self.gal_lbl2 = torch.cat(self.gal_lbl2, dim = 0).cpu().detach().numpy()
            self.gal_emb2 = torch.cat(self.gal_emb2, dim = 0)
            self.idcs2, mAP2, cmc2, _ = reid_evaluate(self.que_emb2, self.gal_emb2, self.que_lbl2, self.gal_lbl2)
            self.logger.info('$$$ Validation mAP (hard): %.4f' % mAP2)
            self.logger.info('$$$ Validation cmc (hard): %.4f' % cmc2)           
        
        return mAP

    def export_output(self):
        if (self.output_dir == None or self.write_output == False):
            return 
        super().export_output()
        #write embs
        self.logger.info("Reranking results")
        idcs,mAP, cmc, dist = reid_evaluate(self.que_emb, self.gal_emb, \
                        self.que_lbl, self.gal_lbl, is_reranking = True)
        self.logger.info('Reranked - Validation mAP: %.4f' % mAP)
        self.logger.info('Reranked - Validation cmc (hard): %.4f' % cmc)   
        
        self.gal_emb = self.gal_emb.cpu().detach().numpy()
        self.que_emb = self.que_emb.cpu().detach().numpy()
        np.save(osp.join(self.output_dir, "que_emb.npy"), self.que_emb)
        np.save(osp.join(self.output_dir, "gal_emb.npy"), self.gal_emb)
        #generate submission file
        que_fname = self.que_ld.dataset.get_img_names()
        gal_fname = self.gal_ld.dataset.get_img_names()
        self.logger.info("Saving distance matrix")
        np.save(osp.join(self.output_dir, "dist.npy"), dist)
        self.logger.info("Saving embeddings")
        que_fname = np.array([int(i) for i in que_fname]).astype(np.int32)
        gal_fname = np.array([int(i) for i in gal_fname]).astype(np.int32)
        self.logger.info("Saving submission file")
        out_file  = osp.join(self.output_dir, "track2.txt")
        np.savetxt(out_file, gal_fname[idcs], 
                delimiter = " ", fmt = "%d", newline='\n')
