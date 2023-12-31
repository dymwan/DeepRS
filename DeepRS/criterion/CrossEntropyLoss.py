import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import List, Optional, Union

from .criterion_utils import compute_weight_from_label_2D
from .ohem import OHEM



def Cross_Entropy(target, input, weight:Union[str, List]=None, ohem_filter=None, ignore_index=None):
    if ohem_filter is not None:
        target = ohem_filter(input, target, ignore_index)
    
    if weight is None:
        ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif isinstance(weight, list):
        ce_loss = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index
        )
    elif weight == 'auto':
        batch_weight = compute_weight_from_label_2D(target)
        ce_loss = nn.CrossEntropyLoss(
            weight=batch_weight, ignore_index=ignore_index)

    ce_loss = ce_loss.forward(input, target)
    return ce_loss


class Cross_Entropy_Loss(nn.Module):
    
    AUTO_WEIGHT:bool                = False
    CUSTOM_WEIGHT:Optional[List]    = None
    IGNORE_INDEX:Optional[int]      = None
    OHEM_MODE:int                   = 0
    
    ce_loss = None #Key executor
    
    ohem   = None
    
    def __init__(self, cfg) -> None:
        super(Cross_Entropy_Loss, self).__init__()
        self._parse_config(cfg)
        
    def _parse_config(self, cfg):
        
        
        '''
        >>> LOSS.CrossEntropyLoss = CN()
        >>> LOSS.CrossEntropyLoss.AUTO_WEIGHT = False
        >>> LOSS.CrossEntropyLoss.CUST_WEIGHT = None
        >>> LOSS.CrossEntropyLoss.IGNORE_INDEX= -100
        '''
        
        #TODO Watch out for type conversions!
        self.AUTO_WEIGHT  =  cfg.LOSS.CrossEntropyLoss.AUTO_WEIGHT
        self.CUSTOM_WEIGHT= cfg.LOSS.CrossEntropyLoss.CUST_WEIGHT
        self.IGNORE_INDEX = cfg.LOSS.CrossEntropyLoss.IGNORE_INDEX
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.CUSTOM_WEIGHT, ignore_index=self.IGNORE_INDEX
        )
        
        self.OHEM_MODE = cfg.LOSS.OHEM.MODE
        self.OHEM_THRE = cfg.LOSS.OHEM.THRESHOLD
        self.OHEM_KEEP = cfg.LOSS.OHEM.KEEP
        
        if self.OHEM_MDOE:
            self.ohem = OHEM(cfg)
        
        
    
    def forward(self, pred, target):
        '''
        In this very simple version, CE loss only support the base implementation,
        more utilities will/can be add at the TODO mark below.
        '''
        loss = 0
        
        if self.OHEM_MODE:
            target = self.ohem.filter(pred, target, self.IGNORE_INDEX)
        
        if self.AUTO_WEIGHT:
            batch_weight = compute_weight_from_label_2D(target)
            self.ce_loss = nn.CrossEntropyLoss(
                weight=batch_weight, ignore_index=self.IGNORE_INDEX)
        
        ce_loss = self.ce_loss.forward(pred, target)
        
        loss += ce_loss
        
        return loss
