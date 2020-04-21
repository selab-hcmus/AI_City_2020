""" This code is adapted from 
https://github.com/CoinCheung/triplet-reid-pytorch/blob/master/eval.py
"""
import torch

import pickle
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from src.utils.misc import MiscUtils
from tools.aic20_re_ranking import re_ranking
import ipdb
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()

    return dist_mtx

def reid_evaluate(emb_query, emb_gallery, lb_ids_query, lb_ids_gallery, \
                cmc_rank=1, top_k=100, is_reranking=None):
    #Calculate distance matrix between query images and gallery images
    dist_mtx = pdist_torch(emb_query,emb_gallery).cpu().detach().numpy()
    if (is_reranking):
        print ("Reranking is applied!")
        dist_mtx = re_ranking(emb_query, emb_gallery)
    n_q, n_g = dist_mtx.shape
    #sort "gallery index" in "distance" ascending order 
    indices = np.argsort(dist_mtx, axis = 1)[:,:top_k]
    matches = lb_ids_gallery[indices] == lb_ids_query[:, np.newaxis]
    matches = matches.astype(np.int32)
    all_aps = []
    all_cmcs = []
    # Setup progressbar
    pbar = MiscUtils.gen_pbar(max_value=n_q, msg="Evaluating: ")

    for qidx in range(n_q):
        qpid = lb_ids_query[qidx]
        # qcam = lb_cams_query[qidx]

        order = indices[qidx]
        pid_diff = lb_ids_gallery[order] != qpid
        # cam_diff = lb_cams_gallery[order] != qcam
        useful = lb_ids_gallery[order] != -1
        # keep = np.logical_or(pid_diff, cam_diff)
        # keep = np.logical_and(keep, useful)
        # match = matches[qidx][keep]
        match = matches[qidx]
        if not np.any(match): continue
        cmc = match.cumsum()
        cmc[cmc > 1] = 1
        #basically count all correct prediction < cmc_rannk
        all_cmcs.append(cmc[:cmc_rank]) 

        num_real = match.sum()
        match_cum = match.cumsum()
        match_cum = [el / (1.0 + i) for i, el in enumerate(match_cum)]
        match_cum = np.array(match_cum) * match
        ap = match_cum.sum() / num_real
        all_aps.append(ap)
        
        # Monitor progress
        pbar.update(qidx+1)
    pbar.finish()
    assert len(all_aps) > 0, "NO QUERY MATCHED"
    mAP = sum(all_aps) / len(all_aps)
    all_cmcs = np.array(all_cmcs, dtype = np.float32)
    cmc = np.mean(all_cmcs, axis = 0)

    return indices, mAP, cmc, dist_mtx