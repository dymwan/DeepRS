'''
Design philosophy:

    -multi-loss supported
    -multi-stage supported

@Multi-loss 
multiple losses at same node

@Multi-stage loss
multiple losses at different nodes, such as auxiliary, and rcf-loss etc.
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Dict

#local losses

PREPARED_LOSSES = {
    
}



class BaseLoss(nn.Module):
    '''
    demo:
        cfg.LOSS.LOSS_NAMES: [
            BCE, BCE, BCE, [CE, DICE]
        ]
        the yaml definition above shows the criterion for training RCF model.
        Accordingly, the input of forward() function should be in the same form:
        
            YourModel.forward(x)-> [bout1, bout2, bout3, final_out]          (1)
        
        And, the target should be claimed in:

            #which means the mission of all branches are same
                target:torch.Tensor                                        (2-1)
            #different taskes driven.
                target::List[torch.Tensor, List[torch.Tensor]]             (2-2)
            
        Furthermore, the loss weight should be claimed carefully. Even there are
        multiple types of loss implementations and combinations, all loss finally 
        will be added together. So, given the loss weights in same form as def-1:
        
            loss_weight = [ w1, w2, w3, [w4, w5]]                            (3)
            
        in which:
            sum([w[i] for i in range(5)]) = 1                                (4)
            
        for convinent, you can define each wi as any number (even gt 1), the
        program will normalize all the weights making them has the sum as 1.
    
    Attention plz:
        @Even you chose only one loss, it should be in the List container.
        @This loss should not be the futher-class for specific loss class.
        @Use it in training programm directly.
        @If needed, more auxiliary losses can be added in this->subclass
    '''
    
    _ORIGINAL_LOSS_NAMES:List[List[str] or str]
    
    def __init__(self, cfg) -> None:
        super().__init__()
        
        self.cfg = cfg
        
        
    
    def init_individual_loss(self):
        self._ORIGINAL_LOSS_NAMES = self.cfg.LOSS.LOSS_NAMES
        
        self.losses = []
        
        for lossname in self._ORIGINAL_LOSS_NAMES:
            if isinstance(lossname, str):
                self.losses.append(get_single_loss(self.cfg, lossname))
            elif isinstance(lossname, list):
                self.mlosses = []
                for sublossname in lossname:
                    self.mlosses.append(get_single_loss(self.cfg, sublossname))
                self.losses.append(self.mlosses)

def get_single_loss(cfg, lossname):
    
    global PREPARED_LOSSES
    
    if lossname in PREPARED_LOSSES.keys():
        return PREPARED_LOSSES.get(lossname)(cfg)
    else:
        raise KeyError(f'Loss name [{lossname}] are not supported for now.')