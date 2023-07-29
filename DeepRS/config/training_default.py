'''
@Author: Yaming Duan, Beijing Normal University, Beijing.
@Date: 20191211
@Purpose: 
    Guides the construction of training code.
@Contact: dymwan@gmail.com
'''
from yacs.config import CfgNode as CN

_C = CN()

# ########################## Dataset Definition ################################
_C.DATASETS = CN()
_C.DATASETS.ROOT = r''
_C.DATASETS.IMAGE_FOLDER_NAMES= []
_C.DATASETS.IMAGE_FOLDER_INDEX= []
_C.DATASETS.LABEL_FOLDER_NAMES = []
_C.DATASETS.LABEL_FOLDER_INDEX = []
_C.DATASETS.TYPE = 'geodataset'
_C.DATASETS.CLASS = []
_C.DATASETS.CLASS_NAMES = []

_C.DATASETS.WORKS = 2
_C.DATASETS.PATCH_SIZE = 256
_C.DATASETS.BATCH_SIZE = 1
_C.DATASETS.IN_CHANNEL = 2
_C.DATASETS.MEAN = [0.5, 0.5, 0.5]
_C.DATASETS.STD = [1, 1, 1]

# ########################## Model Definition ##################################
_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = 'inception_v3_demo'
_C.MODEL.PRETRAINED = r''

# ########################## Training Setting ##################################
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 1
_C.TRAIN.END_EPOCH = 400
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.DROPOUT_RATE = 1.0
_C.TRAIN.RESUME = CN()
_C.TRAIN.RESUME.CONTINUE_TRAIN = False
_C.TRAIN.RESUME.FT = False
_C.TRAIN.RESUME.CHECK_POINT = r''
_C.TRAIN.ADAP_LR = CN()
_C.TRAIN.ADAP_LR.LOSS_ERROR_THR = 20.0
_C.TRAIN.ADAP_LR.LR_DECAY = 0.8
_C.TRAIN.CHECKNAME = 'log'

# ########################## Validating Setting ################################
_C.VAL = CN()
_C.VAL.BATCH_SIZE = 4
_C.VAL.VAL_FREQUENCY = 1
_C.VAL.VAL_START_EPOCH = 1
_C.VAL.COM_F1 = True
_C.VAL.METRIC = CN()
_C.VAL.BEST_TYPE = 'mIoU'
_C.VAL.METRIC.TYPE = 'pix_iou'
_C.VAL.METRIC.THRESHOLD = []


# ##################### Solver Setting (Hyper-Params.)##########################
_C.SOLVER = CN()
_C.SOLVER.OPT = CN()
_C.SOLVER.OPT.OPTIMIZER = 'adamax'
_C.SOLVER.OPT.MOMENTUM = 0.9
_C.SOLVER.OPT.WEIGHT_DECAY = 5e-6

_C.SOLVER.LR = CN()
_C.SOLVER.LR.BASE_LR = 0.001
_C.SOLVER.LR.ADJUST_LR = 0.1
_C.SOLVER.LR.UPDATE_POLICY = 'LR_Scheduler_Epoch'
_C.SOLVER.LR.LR_SCHEDULER  = 'cos'
_C.SOLVER.LR.CYCLE_LR = False
_C.SOLVER.LR.CYCLE_LR_STEP = 40

_C.SOLVER.LR.POLY = CN()
_C.SOLVER.LR.POLY.POWER = 0.9

_C.SOLVER.LR.STEP = CN()
_C.SOLVER.LR.STEP.LR_STEP = 1
_C.SOLVER.LR.STEP.LR_DECAY = 0.8

_C.SOLVER.OPT = CN()
_C.SOLVER.OPT.OPTIMIZER = 'adamax'
_C.SOLVER.OPT.MOMENTUM= 0.9
_C.SOLVER.OPT.WEIGHT_DECAY= 5e-6

# ########################## Criterion Definition###############################
_C.LOSS = CN()
_C.LOSS.FUN_LIST = ['BCE', 'Dice'] # support mix-loss
_C.LOSS.WEIGHT_LIST= []
_C.LOSS.AUTO_WEIGHT= False
_C.LOSS.FOCAL = CN()
_C.LOSS.FOCAL.GAMA = []
_C.LOSS.FOCAL.ALPHA= []
_C.LOSS.FOCAL.SIZE_AVERAGE= True
_C.LOSS.LovaszLoss = CN()
_C.LOSS.LovaszLoss.PER_IMAGE = False
_C.LOSS.CrossEntropyLoss= CN()
_C.LOSS.CrossEntropyLoss.WEIGHT = []
_C.LOSS.ICNETLOSS= CN()
_C.LOSS.ICNETLOSS.WEIGHT= []
_C.LOSS.OHEM = CN()
_C.LOSS.OHEM.MODE = -1
_C.LOSS.OHEM.THRESHOLD = -1
_C.LOSS.OHEM.KEEP = -1
_C.LOSS.AUX = False
_C.LOSS.AUX_WEIGHT = -1

# ########################## Deployment Setting ################################
_C.GPU = [0]
_C.CUDA = True
_C.SAVE_MODE = 'best' #['epoch', 'best']
_C.SEED = 1

_C.CUDNN = CN()
# NotImplement
