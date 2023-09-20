import os
from time import time
from dataclasses import dataclass


from typing import Any


__all__ = [
    "printer_writer",
    "build_trianing_workplace",
    ]


@dataclass
class workspace:
    root:       str
    log_name:   str
    write_mdoe: str




class printer_writer:
    WRITE_MODE_NAME_MAPPING = {
        'a': 'append',
        'w': 'write/overwrite'
    }
    def __init__(self, workspace:workspace) -> None:
        
        self.printer_writer
        
        log_path = os.path.join(workspace.root, workspace.log_name)
        
        if log_path is None:
            print("**[WARNING]** using print-only mode due to no log path decalred.")
            time.sleep(3)
            self.write = False
        else:
            folder, filename = os.path.split(log_path)
            
            if not os.path.isdir(folder):
                raise FileExistsError(f'>> Folder {folder} dosent exist')
            elif not os.path.isfile(log_path):
                self.log_file = open(log_path, 'w')
            else:
                print(f"**[WARNING]** logfilew {log_path} exist, \
                    and the writing mode is\
                    {self.WRITE_MODE_NAME_MAPPING.get(workspace.write_mdoe)}"
                    )
                time.sleep(3)
                self.log_file = open(log_path, workspace.write_mdoe)
            
            self.write = True
        
    
    def __call__(self, input:str=None) -> Any:
        print(input)
        
        if self.write:
            print(input, file=self.log_path)
            self.log_path.flush()


def build_trianing_workplace(cfg):
    '''
    C.LOG.WORK_DIR = './'
    C.LOG.LOG_NAME = ''
    C.LOG.LOG_WRITE_MODE = 'a'
    '''
    
    root = cfg.LOG.WORK_DIR
    log_name = 'auto.log' if cfg.LOG.LOG_NAME == '' else cfg.LOG.LOG_NAME
    log_write_mode = cfg.LOG.LOG_WRITE_MODE
    assert log_write_mode in ['a', 'w'] # More to use TODO
    
    if not os.path.isdir(root):
        os.makedirs(root)
    else:
        print(f"**[WARNING]** work_dir {root} exists, not safe if continue.")
        time.sleep(4)
    
    return workspace(root, log_name, log_write_mode)
    
    


