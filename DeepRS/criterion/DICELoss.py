import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
from .ohem import OHEM

def _dice_coef( target=None, input=None, pt=None, smooth=1e-5, ohem_mask=None):
    
    assert target is not None
    
    if pt is None:
        assert input is not None, "Error dice-err-init-001"
        logpt = F.log_softmax(input, dim=1)
        pt = Variable(logpt.data.exp())
        
    N,H,W = pt.size()
    
    target_onehot = torch.zeros(pt.size()).cuda()  # N,C,H,W
    target_onehot.scatter_(1, target.view(N,1,H,W), 1)  # N,C,H,W

    intersection = torch.sum(pt * target_onehot, dim=1)  # N,H,W
    union = torch.sum(pt, dim=1) + torch.sum(target_onehot, dim=1)
    
    if ohem_mask is not None:
        intersection = torch.masked_select(intersection, ohem_mask)
        union = torch.masked_select(union, ohem_mask)

    Dice_coef = torch.mean((2*torch.sum(intersection) + smooth) / (torch.sum(union) + smooth))
    
    return Dice_coef

    

def Dice(target, input=None, pt=None, smooth=1e-5, ohem_mask=None):
    
    dice_coef = _dice_coef(target, input=input, pt=pt, smooth=smooth, ohem_mask=ohem_mask)
    Dice_loss = 1 - dice_coef
    
    return Dice_loss


'''
CONFIG.LOSS.FUN_LIST=[ 'Dice' ]

CONFIG.LOSS.OHEM.MODE = 0
CONFIG.LOSS.OHEM.THRESHOLD = 0.60
CONFIG.LOSS.OHEM.KEEP = 2e5

'''

#TODO there should be an unified caller to get ohem mask

class dice_loss(nn.Module): #As Individual loss be called
    
    '''
    This class will be called when use single loss named as "Dice".
    '''
    
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.ohem_mode = cfg.LOSS.OHEM.MODE
        
        if self.ohem_mode:
            self.ohem = OHEM(cfg)
        
        
        
    def forwar(self, input:torch.Tensor, target:torch.Tensor):
        
        N,C,H,W = input.size()
        
        logpt = F.log_softmax(input, dim=1)

        pt = Variable(logpt.data.exp())
        
        #----------------------------TODO---------------------------------------
        if self.ohem_mode:
            
            if logpt.dim() >2:
                logpt_view = logpt.view(N,C,-1)
                logpt_view = logpt_view.transpose(1,2)
                logpt_view = logpt_view.contiguous().view(-1, C)
            else:
                logpt_view = logpt
            
            torch.Tensor.gather()
            
            target_view = target.view(-1, 1).long()
            
            logtpt_gather =  logpt_view.gather(1, target_view) 
            pt_gather = Variable(logtpt_gather.data.exp())
            ohem_keep_mask = self.ohem.focal(pt_gather)
        else:
            ohem_keep_mask = None
        #--------------Make the block unified-----------------------------------
            
        dice_coef = _dice_coef(target=target, pt=pt, ohem_mask=ohem_keep_mask)
        
        _dice_loss = 1 - dice_coef
        
        
        return _dice_loss
    
    
    