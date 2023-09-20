import torch
from tqdm import tqdm
from torch.utils import data as torchdata

from DeepRS.utils.files import *
from DeepRS.data import *
from DeepRS.model import *
from DeepRS.optm import *



class Trainer:
    
    train_loader_kwargs = {'num_workers': 1, 'pin_memory': True, 
                           'drop_last': True, 'shuffle': True,}
    val_loader_kwargs = {'num_workers': 1, 'pin_memory': True, 
                           'drop_last': True, 'shuffle': True,}
    
    def __init__(self, cfg) -> None:
        '''
        @Input: cfg -> yaml
        initiate:
            @ Dataset
            @ Model
            @ Hyper Parameters (Solver)
            @ Optimizer
            @ Learning Scheduler
            @ Criterion
            @ SummaryWriter (Recoder)
            
        notes: multi-datasets, multi-input/output should define separately.
        
        '''
        self.cfg = cfg
        
        self.workplace= build_trianing_workplace(cfg)
        self.printer = printer_writer(self.workplace)
        
        
        
        self.train_batch_size = cfg.TRAIN.BATCH_SIZE
        self.val_batch_size = cfg.VAL.BATCH_SIZE
        
        
        #-------------------------dataset def----------------------------------
        if cfg.DATASETS.NUM_WORKERS > 1:
            self.train_loader_kwargs['num_workers'] = cfg.DATASETS.NUM_WORKERS
            self.val_loader_kwargs['num_workers'] = cfg.DATASETS.NUM_WORKERS
        
        self.trian_dataset = get_dataset(cfg, split='train')
        self.train_loader  = torchdata.DataLoader(self.trian_dataset, batch_size=self.train_batch_size, **self.train_loader_kwargs)
        
        self.val_dataset = get_dataset(cfg, split='val')
        self.val_loader = torchdata.DataLoader(self.val_dataset, batch_size=self.val_batch_size, **self.val_loader_kwargs)
        
        
        #-----------------------model definition--------------------------------
        self.model = get_model(cfg)
        self.printer(self.model)
        
        
        
        #------------------- optimizer defeinitio-------------------------------
        # from DeepRS.optm.__init__
        _optm, self.optm_kwargs = get_optm()
        param_list = get_param_list()
        self.optm = self._optm(param_list, **self.optm_kwargs)
        
        
        self.scheduler_mode = cfg.SOLVER.LR.UPDATE_POLICY = "LR_Scheduler_Epoch"
        para_pretrained = False
        
        
        self.scheduler = get_lr_scheduler(
            self.scheduler_mode, cfg, self.iter_per_epoch, para_pretrained
            )
        
        #-------------------criterion (loss def) -------------------------------
        
        
        #-------------------const hyper-parameters -----------------------------
        self.best_epoch:int     = 0
        self.reset_num:int      = 0
        self.reduce_lr = 1
        
        
        
        
        
    def training(self, epoch):
        
        # log the up-to-now training status
        self.printer(
            f"[INFO]:Current epoch: {epoch}, \
                Best epoch: {self.best_epoch}, \
                Reset_num: {self.reset_num}"
        )
        
        train_loss = 0.0
        
        self.model.train()
        self.optm.zero_grad()
        
        if self.scheduler_mode == 'LR_Scheduler_Epoch':
            self.scheduler(self.optm, epoch, reduce_lr=self.redece_lr)
        
        train_prefetcher = data_prefetcher(self.train_loader)
        
        tbar = tqdm(range(self.iter_per_epoch), unit='batch', desc='==>Train', ncols=90, ascii=True)

        for i in tbar:
            
            image, target = train_prefetcher.next()
            
            if self.scheduler_mode == 'LR_Scheduler_Batch':
                self.scheduler(self.optm, i, epoch)
            
            predict = self.model(image)

            loss = self.crite


if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    
    C = CN()
    
    C.DATASETS = CN()
    C.DATASETS.ROOT = r"D:\torchTest\test_dataset"
    C.DATASETS.IMAGE_FOLDER_NAMES = ['image']
    C.DATASETS.IMAGE_FOLDER_INDEX = [0]
    C.DATASETS.LABEL_FOLDER_NAMES = ['bound_label', 'field_label']
    C.DATASETS.LABEL_FOLDER_INDEX = [1, 2]
    C.DATASETS.TYPE = 'multitask_geodataset'
    C.DATASETS.CLASSES = [[0,1], [0,1]]
    C.DATASETS.CLASS_NAMES = [['bound_back', 'bound'],['field_back', 'field']]
    
    C.DATASETS.WORKS = 2
    C.DATASETS.PATCH_SIZE = 256
    C.DATASETS.BATCH_SIZE = 1
    C.DATASETS.INCHANNEL = 3 
    C.DATASETS.MEAN = [0.485,0.456,0.406]
    C.DATASETS.STD = [0.229,0.224,0.225]
    C.DATASETS.STATIONS = []
    
    
    C.LOG = CN()
    C.LOG.WORK_DIR = './'
    C.LOG.LOG_NAME = ''
    C.LOG.LOG_WRITE_MODE = 'a'