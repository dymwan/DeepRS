###########################################################################
# Created by: Yaming Duan
# Email:  bnu.dym@mail.bnu.edu.cn
# Copyright (c) 2017
###########################################################################
import os
# from re import I
import numpy as np
from PIL import Image

from tqdm import tqdm

import gdal
# from osgeo import gdal
import torch
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms as T
print(torch.__version__)
from PIL import Image
import random

from .base import BaseDataset
from encoding.utils.files import *
from encoding.transforms.augmentation import augment
from encoding.utils.georaster_processor import georaster_loader, loadGeoRasterToImage


# def convert2OpencvImage(I):
#     if 

# def rgb2gray(I, method='hsv'):
#     if method == 'hsv':
        


class georasterDataset(BaseDataset):
    def __init__(self, cfg, split='train', mode=None, repeat=1):
        super(georasterDataset, self).__init__(cfg, split, mode)
        # self.patch_size
        self.classes= cfg.DATASETS.CLASSES
        self.labels = cfg.DATASETS.LABELS
        self.stations = cfg.DATASETS.STATIONS
        # self.auto_weight = cfg.LOSS.AUTO_WEIGHT
        self.patch_size = cfg.DATASETS.PATCH_SIZE
        # TODO this param indicates the size of the original patches
        self.original_size = 1024
        self.max_anchor = self.original_size-self.patch_size
        # self.augment = augment(cfg)

        self.images = []
        self.masks = []
        self.stations_list = []

        # TODO defectSimulate is None
        self.ds = None

        self._transpose_idx = [0,1,2,3,4,5,6]

        _split_file = os.path.join(self.root, '{}.csv'.format(self.split))
        

        self.jitter =  T.Compose({
            T.ColorJitter(brightness=[0.8,1.2], contrast=0.3, saturation=0.3),
            T.ToTensor(),
            })
        self.to_tensor = T.Compose({
            T.ToTensor()
        })
        # self.normalize = T.Normalize([0.2638, 0.3007, 0.2999],[0.1292, 0.1126, 0.1007])#google
        
        self.normalize = T.Normalize([0.3722, 0.4498, 0.3964],[0.1220, 0.1139, 0.1064]) # gf2 zj
        
        self.PILToTensor = T.PILToTensor()
        
        


        if not os.path.isfile(_split_file):
            raise RuntimeError('Unkown dataset _split_file: {}'.format(_split_file))
        


        with open(os.path.join(_split_file), 'r') as lines:
            lines = list(lines)
            
            for line in tqdm(lines):
                try:
                    relative_img_path, relative_mask_path = line.rstrip('\n').split(',')
                except:
                    relative_img_path, relative_mask_path,_ = line.rstrip('\n').split(',')
                    # print(line.split(r'\n'))
                    raise
                img_name = os.path.basename(relative_img_path).split('.')[0]

                self.update_stations(img_name)
                
                self.update_images(relative_img_path)
                self.update_masks(relative_mask_path)

        
        if repeat > 1:
            self.images = self.images * repeat
            self.masks = self.masks * repeat

        self.samples_num = len(self.images)
        print('Number of samples: {}'.format(self.samples_num))

    def get_station(self, img_name):
        station = 'no'
        return station
    
    def update_stations(self, img_name):
        _station = self.get_station(img_name)
        if self.stations == []:
            pass
        
        elif _station not in self.stations:
            raise RuntimeError('Unknown station: {}'.format(_station))
        
        else:
            self.stations_list.append(_station)

    def update_images(self, relative_img_path):
        _image = os.path.join(self.root, relative_img_path)
        if os.path.isfile(_image):
            self.images.append(_image)
        else:
            raise RuntimeError('image is not a file: {}'.format(_image))

    def update_masks(self, relative_mask_path):
        _mask = os.path.join(self.root, relative_mask_path)
        if os.path.isfile(_mask):
            self.masks.append(_mask)
        else:
            raise RuntimeError('mask is not a file : {}'.format(_mask))
    
    def _rotateAndFlip(self, img, mask):
        
        # x_flip = 1 if random.random() < 0.5 else -1
        # y_flip = 1 if random.random() < 0.5 else -1

        rotate = [45, 45]
        angle = random.randint(-rotate[0], rotate[1])
        img = img.rotate(angle, Image.BILINEAR, expand=0)
        mask = mask.rotate(angle, Image.NEAREST, expand=0)
        # print(img.size)
        # exit()
        return img, mask

    def __check_out_init(self):
        
        self.__check_out_folder = r'/home2/wjx/guangxi_comp/exp4_test_rotate/test_out'
        self.__check_out_folder_image = os.path.join( self.__check_out_folder, 'image')
        self.__check_out_folder_label = os.path.join( self.__check_out_folder, 'label')
        
    def __check_out(self, img, label, name):
        print('check out', img.shape)
        print('check out', label.shape)
        img_numpy = img.squeeze(0).cpu().numpy()
        label_numpy = label.squeeze(0).squeeze(0).cpu().numpy()
        
        # img_numpy *= 255
        #sugercane
        # label_numpy[label_numpy == 1] == 255

        img_Image = Image.fromarray(img_numpy.transpose(1,2,0).astype(np.uint8))
        label_Image = Image.fromarray(label_numpy.astype(np.uint8))
        label_Image.putpalette([
            0,  0,   0,
            218, 60, 60,
            214, 136, 35,
            # 188, 196, 78,
            188, 196, 78,
            110, 218, 140,
            95, 163, 251
        ])
        
        img_Image.save(os.path.join(self.__check_out_folder_image, name))
        label_Image.save(os.path.join(self.__check_out_folder_label, name))
    
    def _colorJitter(self, img):
        # print(torch.mean(img), 'check')
        img /= 255.
        tsfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1),
            transforms.ToTensor()
        ])
        
        return tsfm(img).float() * 255  #.unsqueeze(0)

    @staticmethod
    def _do_rotate(x, theta, _mode='bilinear'):
        
        # _dtype = torch.cuda.FloatTensor if x.device != 'cpu' else torch.FloatTensor
        _dtype = torch.FloatTensor
        original_dtype = x.dtype
        x *=10
        rot_mat = theta[None, ...].type(_dtype).repeat(x.shape[0],1,1)
        grid = F.affine_grid(rot_mat, x.size()).type(_dtype)
        x = F.grid_sample(x.type(_dtype), grid, mode=_mode)
        x /= 10
        return x.type(original_dtype)

    def _rotate(self, img, label, rotate_range=[-np.pi/2, np.pi/2]):
        
        rotate_lower = torch.Tensor([rotate_range[0]])
        rotate_upper = torch.Tensor([rotate_range[1]])
        
        theta = torch.rand((1), dtype=img.dtype, device=img.device)
        theta = theta * (rotate_upper - rotate_lower) + rotate_lower
        
        theta_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

        return self._do_rotate(img.unsqueeze(0), theta_mat).squeeze(0), \
            self._do_rotate(label.unsqueeze(0).unsqueeze(0), theta_mat, _mode='nearest').squeeze(0).squeeze(0)
    
    def _rotate_low(self, img, label, rotate_range=[-np.pi/4, np.pi/4]):
        rotate_lower = torch.Tensor([rotate_range[0]])
        rotate_upper = torch.Tensor([rotate_range[1]])
        
        theta = float(torch.rand([1]))
        
        #img: torch.Tensor, angle: float, resample: int = 0, expand: bool = False, center: Optional[List[int]] = None, fill: Optional[int] = None
        r_img = transforms.functional.rotate(img, theta)
        r_label = transforms.functional.rotate(label, theta)
        
        return r_img, r_label

    def new_rotate(self, img, label, rotate_range=[0, np.pi/2]):
        
        
        theta = random.uniform(*rotate_range)

        # theta = torch.rand((1), dtype=img.dtype, device=img.device)
        # theta = theta * (rotate_upper - rotate_lower) + rotate_lower

        # gather = torch.cat([img, label.unsqueeze(0)], dim = 0)
        img = transforms.functional.rotate(img, theta, resample=transforms.InterpolationMode.BILINEAR)
        label = transforms.functional.rotate(label, theta, resample=transforms.InterpolationMode.NEAREST)

        return img, label

    def _standardTsf(self, img, label, size=256):

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()

        

        #TODO test rotate
        # img, label = self.new_rotate(img, label)
        if random.uniform(0,1) > 0.9:
            img, label = self._rotate(img, label)
        # print(torch.mean(img), 'rotate')
            
        
        if random.uniform(0,1) > 0.7:
            img = self._colorJitter(img.squeeze(0))#.squeeze(0))

        # print(img.shape)
        b,w,h = img.shape

        if w > size:
            anchor_w_max = w - size
        if h > size:
            anchor_h_max = h - size                
        
        
        anchor_w = np.random.randint(0, anchor_w_max)
        anchor_h = np.random.randint(0, anchor_h_max)
        
        img = img[:, anchor_w: anchor_w+size, anchor_h: anchor_h+size]
        label = label[anchor_w: anchor_w+size, anchor_h: anchor_h+size]

        return img, label
        
    
    def _get_fetch_param(self):
        #TODO original_size should be automated decided
        anchor_w = np.random.randint(0, self.max_anchor)
        anchor_h = np.random.randint(0, self.max_anchor)
        
        fetch_param = [anchor_w, anchor_h, self.patch_size, self.patch_size]
        
        return fetch_param
        # self.patch_size

    # augmentation for Image instances
    def _transpose(self, *inputs):
        transpose_idx = np.random.choice(self._transpose_idx)
        output = []
        for input in inputs:
            output.append(input.transpose(transpose_idx))
        return output
        

    def __getitem__(self, index):
        
        # TODO affine transform
        # image, im_ds = georaster_loader.loadGeoRaster(self.images[index], dt=gdal.GDT_Byte)
        # target, msk_ds = georaster_loader.loadGeoRaster(self.masks[index], dt=gdal.GDT_Byte)
        
        
        fetch_param = self._get_fetch_param()
        
        image = loadGeoRasterToImage(self.images[index],fetch_param)
        target = loadGeoRasterToImage(self.masks[index],fetch_param, np.uint8)

        # rotate and flip
        image, target = self._transpose(image, target)

        

        image = self.jitter(image) 
        # image = self.to_tensor(image)
        image = self.normalize(image)

        target = self.PILToTensor(target)
        
        # # self.__check_out_init()
        # # self.__check_out(image, target, f'i_{index}.png')

        # image /= 255.
        # print(torch.mean(image), f'i_{index}.png')
        
        return image, target.squeeze(0).long()
    
    
    def __len__(self):
        return self.samples_num





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