import cv2
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import os.path as osp
from pandas import DataFrame
from encoding.utils.files import *


import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from typing import List


class RS_SegmentationMetric(object):
    '''
    This class is a calculator for segmentation metrics via confusion matrix(cm)
    TODO this class should be base-class
    The cm calculation relies on bit operation by torch, which maintain the high
    performance.
    
    The input predicted and target is:
    torch.Tensor [B, ncls, w, h], torch.float, cuda normally
    torch.Tensor [B, w, h], torch.int64, cuda normally
    
    For total metric, inspired by scikit-learn, we give several specification for 
    getting the global metric:
        @binary:    int >> return metric for specific class
        @micro:     Calculate metrics globally by counting the total true positives, 
                    false negatives and false positives.
        @macro:     Calculate metrics for each label, and find their unweighted mean.
                    This does not take label imbalance into account.
        @weighted:  Calculate metrics for each label, and find their average 
                    weighted by support (the number of true instances for each 
                    label). This alters 'macro' to account for label imbalance; 
                    it can result in an F-score that is not between precision 
                    and recall.
    it requires the self.classN for accmulating store the pixel amount per each
    class while programm goes.
    
    
    '''
    
    NCLASS = -1
    
    average = 'macro'
    class_rep_total = None
    classN:torch.LongTensor
    cm = None
    
    pixAcc= OA= -1.
    PA = None
    UA = None
    
    recall = None
    precision = None
    
    IoU:torch.Tensor
    mIoU:float
    F1:float
    AUC:float
    ODS:float
    OIS:float
    AP:List[float]
    mAP:float
    
    total_metirc = dict()
    class_metric = dict()
    
    roc_precision:int = 100
    
    
    
    _eps = torch.Tensor([np.spacing(1)]) #tensor([2.2204e-16])
    
    def __init__(self, nclass, defect_metric, defect_filter, ignore_index=[0], com_f1=True, device='cuda', average='',roc_precision=100, bin_cls=None):
        
        self.NCLASS = nclass
        #----------------------------------------------------------------------
        # initialize
        
        if average in [ 'binary', 'micro', 'macro', 'weighted']:
            self.average = average
        
        if average == 'weighted':
            self.classN = torch.zeros(nclass, device=device, dtype=torch.long)        
        
        if average == 'binary':
            self.class_rep_total = bin_cls
            if self.class_rep_total is None: raise
        
        self.precision  = torch.zeros(self.NCLASS, dtype=torch.float32)
        self.recall     = torch.zeros(self.NCLASS, dtype=torch.float32)
        self.F1         = torch.zeros(self.NCLASS, dtype=torch.float32)
        self.IoU        = torch.zeros(self.NCLASS, dtype=torch.float32)
        self.pixAcc     = torch.zeros(self.NCLASS, dtype=torch.float32)
        
        
        #----------------------------------------------------------------------
        # for interval calculation for sparse thresholds
        
        self.roc_precision = roc_precision
        
        self.p_score_count = torch.zeros(self.roc_precision, dtype=torch.int64).to(device)
        self.n_score_count = torch.zeros(self.roc_precision, dtype=torch.int64).to(device)
        
        #----------------------------------------------------------------------
        # for binary cal
        
        
        
        if device.find('cuda') != -1:
            self._eps = self._eps.to(device=device)
    
    def update_batch_metrics(self, predict, target, **kwargs):
        
        psc,  nsc = batch_positive_score_seg(predict, target)
        
        self.p_score_count += psc
        self.n_score_count += nsc

        
        if self.average == 'weighted':
            self.classN += torch.bincount(target, minlength=self.NCLASS)
            
        
        # binary
        _, predict = torch.max(predict, 1)
        
        if self.cm is None:
            self.cm =  _calComfusionMatrix(predict, target, self.NCLASS)
        else:
            self.cm = merge_cm(
                self.cm, _calComfusionMatrix(predict, target, self.NCLASS)
            )
    
    
    def get_epoch_results(self):
        
        self._getNPTF()
        
        self._get_class_metric()
        self._get_total_metric()
        
        
        self.AUC = get_seg_auc(
            self.p_score_count, self.n_score_count, self.roc_precision
        )
        
        self.mIoU = self.IoU.mean()
        
        return  self.total_pixAcc, self.pixAcc, self.mIoU, self.IoU, self.AUC, \
                self.total_recall, self.recall, self.total_precision, \
                self.precision, self.total_F1, self.F1
        
    
    def _getNPTF(self, input_cm=None):
        if input_cm is None:
            cm = self.cm
        else:
            cm = input_cm
            
        cm_sum = torch.sum(self.cm)
        clslist = list(range(cm.shape[0]))
            
        self.pred_sum = [torch.sum(cm[:,i]) for i in clslist]
        self.ref_sum = [torch.sum(cm[i,:]) for i in clslist]
        
        self.TP = [cm[i,i] for i in clslist]
        self.FP = [self.pred_sum[i]- self.TP[i] for i in clslist]
        self.FN = [self.ref_sum[i] - self.TP[i] for i in clslist]
        self.TN = [cm_sum - self.TP[i] - self.FP[i] - self.FN[i] for i in clslist]
        

    def _get_class_metric(self, input_cm=None):
        
        if input_cm is None:
            cm = self.cm
        else:
            cm = input_cm
            
        assert cm is not None, 'Got a empty Confusion matrix'
               
        clslist = list(range(cm.shape[0]))
        
        for i in clslist:
            
            _der_precision = (self.TP[i] + self.FP[i])
            self.precision[i] = self.TP[i] / _der_precision if _der_precision > 0 \
                else 0
            
            # cal recall
            _der_recall = (self.TP[i] + self.FN[i])
            self.recall[i] = self.TP[i] / _der_recall if _der_recall > 0 \
                else 0

            # call F1
            _der_F1 = (self.precision[i] + self.recall[i])
            self.F1[i] = 2 * self.precision[i] * self.recall[i] / _der_F1 if _der_F1 > 0 \
                else 0
            
            # call IoU
            _der_IoU = (self.pred_sum[i] + self.ref_sum[i] - self.TP[i])
            self.IoU[i] = self.TP[i] / _der_IoU if _der_IoU > 0 \
                else 0
            
            self.pixAcc[i] = (
                self.TP[i] + self.TN[i]) / (self.TP[i] + self.TN[i] + self.FP[i] + self.TN[i])
        
    def _get_total_metric(self):
        
        '''
        mIoU, pixAcc, recall, precision, F1
        '''
                
        if self.average == 'binary':
            self._get_total_metric_binary()
        elif self.average == 'macro':
            self._get_total_metric_macro()
        elif self.average == 'micro':
            self._get_total_metric_micro()
        elif self.average == 'weighted':
            self._get_total_metric_weighted()
        
        
    def _get_total_metric_macro(self, input_cm=None):
        if input_cm is None:
            cm = self.cm
        else:
            cm = input_cm
            
        assert cm is not None, 'Got a empty Confusion matrix'
        
        
        tTP = torch.sum(torch.Tensor(self.TP))
        tFP = torch.sum(torch.Tensor(self.FP))
        tFN = torch.sum(torch.Tensor(self.FN))
        tTN = torch.sum(torch.Tensor(self.TN))
        
        _der_precision = tTP + tFP
        self.total_precision = tTP / _der_precision if _der_precision > 0 else 0
        
        _der_recall = tTP + tFN
        self.total_recall = tTP / _der_recall if _der_recall > 0 else 0
        
        _der_F1 = self.total_precision + self.total_recall
        self.total_F1 = 2 * self.total_precision * self.total_recall/ _der_F1\
            if _der_F1 > 0 else 0
        
        _der_IoU = tTP + tFP + tFN
        self.total_IoU = tTP / _der_IoU if _der_IoU >0 else 0
        
        _der_pixacc = tTP + tTN + tFP + tTN
        #denominator here cannot be zero, but to be ensure, write like this
        self.total_pixAcc = tTN / _der_pixacc if _der_pixacc > 0 else 0
        
    def _get_total_metric_micro(self):
        raise NotImplementedError
    
    def _get_total_metric_weighted(self):
        raise NotImplementedError
    
    def _get_total_metric_binary(self):
        raise NotImplementedError
    
    def reset(self):
        self.total_pixAcc = None
        self.pixAcc = torch.zeros(self.NCLASS, dtype=torch.float32)
        self.mIoU = None
        self.IoU = torch.zeros(self.NCLASS, dtype=torch.float32)
        self.AUC = None
        self.total_recall = None
        self.recall = torch.zeros(self.NCLASS, dtype=torch.float32)
        self.total_precision = None
        self.precision = torch.zeros(self.NCLASS, dtype=torch.float32)
        self.total_F1 = None
        self.F1 = torch.zeros(self.NCLASS, dtype=torch.float32)

        self.cm= None
        
        torch.cuda.empty_cache()






    
def _calComfusionMatrix(predict, target, nclass):
    
    available_idx= (predict >= 0) & (predict <= nclass-1)

    return torch.bincount(
        nclass * target[available_idx].int() + predict[available_idx], \
        minlength= nclass ** 2 \
    ).view(nclass, nclass)
    
def merge_cm(target_cm, input_cm):
    
    assert target_cm.shape == input_cm.shape
    
    return target_cm + input_cm


def batch_positive_score_seg(output, target, mode=-1, precision=100):
    """Batch Score and reference
    Args:
        predict: input 4D tensor, b,cls,w,h
        target:  label 3D tensor, b,    w,h
        nclass:  number of categories (int)
    
    mode:int
        eq -1 >> calculate all class except class 0 which always is regarded as 
                 background.
        ge 0  >> calculate specific class auc
        
    precesion:int 
        roc scale
    """
    
    if mode == -1:
        output_score = 1 - output[:, 0, :, :]
        positive_mask = target != 0
    else:
        output_score = output[:, mode, :, :]
        positive_mask = target == mode

    output_score = (output_score * precision).long()
    
    positive_score_count = torch.bincount(
        output_score[positive_mask], minlength=precision
    )
    
    negative_score_count = torch.bincount(
        output_score[~positive_mask], minlength=precision
    )
    
    # print(positive_score_count.shape, negative_score_count.shape)
    return positive_score_count, negative_score_count


def get_seg_auc(psc, nsc, precision):
    eps = np.spacing(1)
    TP, FP, FN, TN = [], [], [], []
    
    for i in range(precision):
        
        TP.append( torch.sum(psc[:i]).cpu().detach().item())
        FP.append( torch.sum(psc[i:]).cpu().detach().item())
        TN.append( torch.sum(nsc[:i]).cpu().detach().item())
        FN.append( torch.sum(nsc[i:]).cpu().detach().item())
        
    TP = np.asarray(TP).astype('float32')
    FP = np.asarray(FP).astype('float32')
    FN = np.asarray(FN).astype('float32')
    TN = np.asarray(TN).astype('float32')
    
    _fpr = FP+TN
    _fpr[_fpr == 0] = eps
    FPR = TP / _fpr
    
    _tpr = TP+FN
    _tpr[_tpr == 0] = eps
    TPR = TP / _tpr
    
    auc = 0
    
    for i in range(len(FPR)):
        auc = auc + TPR[i]
    auc  = auc / len(FPR)
    
    
    
    return auc