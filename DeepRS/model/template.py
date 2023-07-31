'''
illustrates the template of a model class, all the model should be defined 
according this template
'''

import torch.nn as nn


class customize_model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    
    
    def forward(self, inputs):
        
        return
    
    @classmethod
    def from_name(cls, model_name, **overide_params):
        
        return cls() # do initialization here

    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        
        return #model_instance
    
    