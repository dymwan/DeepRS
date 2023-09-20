from .dataloader_utils import data_prefetcher
from .geodatasets import *



__all__ = [
    "data_prefetcher",
    "get_dataset"
]


_dataloaders = {
    'geodataset': geodatasets,
    'multitask_geodataset': multitask_georasterDataset
}



def get_dataset(cfg, split='train'):
    
    return