import torch


__all__ =  [
    "get_optm",
    "get_param_list", 
]


OPTMS = {
    'sgd': torch.optim.SGD,
    'adamax': torch.optim.Adamax,
}



def get_param_list(cfg, model):
    params_list = list()
    param_pretrained = False
    base_lr = cfg.SOLVER.LR.BASE_LR
    
    if hasattr(model, 'pretrained'):
        param_pretrained = True
        params_list.append({
            'params': model.pretrained.parameters(), 'lr': base_lr
        })
    if hasattr(model, 'exclusive'):
        for module in model.exclusive:
            params_list.append({
                'param': getattr(model, module).parameters(), 'lr': base_lr
            })
    
    if hasattr(model, 'scratch'):
        params_list.append({
            'params': model.parameters(), 'lr': base_lr
        })        
    
    return params_list



def get_optm(cfg):
    
    optm_name = cfg.SOLVER.OPT.OPTIMIZER
    base_lr = cfg.SOLVER.LR.BASE_LR
    adjust_lr = cfg.SOLVER.LR.ADJUST_LR
    weight_decay = cfg.SOLVER.OPT.WEIGHT_DECAY
    momentum = cfg.SOLVER.LR.POLY.MOMENTUM
    
    optm = OPTMS.get(optm_name)
    
    optm_kwargs = {
        'lr': base_lr,
        'weight_decay': weight_decay,
        'momentum': momentum,
        
    }
    
    return optm, optm_kwargs