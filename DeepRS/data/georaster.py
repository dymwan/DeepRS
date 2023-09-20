'''
@Author: Yaming Duan, Beijing Normal University, Beijing.
@Date: 20200106
@Purpose: 
    solving the io requirement or remote sensing raster data.
@Contact: dymwan@gmail.com
'''
import os
from osgeo import gdal
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import random




        
class inference_loader:
    
    def __init__(
                    self, 
                    src_dir, 
                    dst_dir, 
                    patch_size=256, 
                    buffer_size=32, 
                    out_channel=1, 
                    odtype=gdal.GDT_Byte, 
                    overwrite_out=False,
                    load_approach=1,
                    selected_band=[0,1,2],
                    add_suffix = None
                ) -> None:

        
        self.dst_dir = dst_dir
        if add_suffix is not None:
            self._add_suffix(add_suffix)

        self.ps = patch_size
        self.bs = buffer_size
        
        self.stride = self.ps - 2* self.bs
        if self.stride <= 0:
            raise ValueError('buffersize should lt ps/2')
        
        self.src_dir = src_dir
        self.ds = gdal.Open(self.src_dir)
        
        self.rx = self.ds.RasterXSize
        self.ry = self.ds.RasterYSize

        self.clip_map = []
        self.ods = None
        self.out_channels = out_channel

        
        self.out_dtype = odtype
        self.overwrite_out = overwrite_out
        
        if self.out_channels == 1:
            self.write_fun = self.save_patch2d
        else:
            self.write_fun = self.save_patch3d
        
        self.station= 0

        self.approach = load_approach

        self.get_shifting_map()

        self.another_outs = {}
        
        self.selected_band = selected_band
        
        # self.add_suffix = add_suffix
    
    def _add_suffix(self, add_suffix):
        assert len(self.dst_dir.split('.')) > 1
        
        suffix = self.dst_dir.split('.')[-1]
        
        self.dst_dir = self.dst_dir.replace(
            '.'+suffix, '.' + add_suffix.rstrip('.') + '.' + suffix
        )
    
    def Create_out_file(self):
        
        

        driver = gdal.GetDriverByName('GTiff')
        
        self.ods = driver.Create(
            self.dst_dir, self.rx, self.ry, self.out_channels, self.out_dtype,
            options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"]
        )
        
        self.ods.SetProjection(self.ds.GetProjection())
        self.ods.SetGeoTransform(self.ds.GetGeoTransform())



    def get_shifting_map(self):

        
        nx = get_shifting_steps(self.rx, self.stride)
        ny = get_shifting_steps(self.ry, self.stride)
        
        for xi in range(nx):
            for yi in range(ny):

                xs = xi * self.stride - self.bs
                ys = yi * self.stride - self.bs
                
                lpadx, xs, oxs = nagetive_pad_check(xs, self.bs)
                upady, ys, oys = nagetive_pad_check(ys, self.bs)
                
                
                rpadx, xoff, oxoff = positive_pad_check(
                    xs, self.ps, self.rx, self.bs, self.stride, lpadx)
                bpady, yoff, oyoff = positive_pad_check(
                    ys, self.ps, self.ry, self.bs, self.stride, upady)
                
                self.clip_map.append(
                    [xs, ys, xoff, yoff, oxs, oys, oxoff, oyoff, \
                     lpadx, rpadx, upady, bpady]
                )
    
    @staticmethod
    def data_processor(sub_patch:np.ndarray, approach=2) -> torch.Tensor:
        assert approach in [1,2,3]
        if approach == 1:
            sub_patch = torch.from_numpy(sub_patch).float().unsqueeze(dim=0)
        elif approach == 2:
            sub_patch = Image.fromarray(sub_patch.astype(np.uint8).transpose(1,2,0))
            sub_patch = transforms.ToTensor()(sub_patch).float().unsqueeze(dim=0)
        elif approach ==3:
            sub_patch = Image.fromarray(sub_patch.astype(np.uint8).transpose(1,2,0))
            sub_patch = transforms.ToTensor()(sub_patch).float().unsqueeze(dim=0)
            sub_patch = transforms.Normalize( [0.4977, 0.5129, 0.5602], [1.1720, 1.1271, 1.2219])(sub_patch)
        else:
            raise
        return sub_patch

    def __len__(self):
        return len(self.clip_map)
    
    def __getitem__(self, index):
        self.station = index

        cm = self.clip_map[index]
        get_params = cm[:4]
        lpadx, rpadx, upady, bpady = cm[-4:]
        patch_arr = self.ds.ReadAsArray(*get_params)
        try:
            patch_arr = patch_arr[self.selected_band,:,:]
        except:
            pass
        w1, h1 = patch_arr.shape[-2:]
        
        if not all([lpadx, rpadx, upady, bpady]):
            if len(patch_arr.shape) == 2:
                patch_arr = np.pad(
                    patch_arr, 
                    ((upady, bpady), (lpadx,rpadx), ), 
                    'reflect')
            else:
                patch_arr = np.pad(
                    patch_arr, 
                    ((0,0), (upady, bpady), (lpadx,rpadx), ), 
                    'reflect')

        patch_arr = self.data_processor(patch_arr, approach=self.approach)

        return index, patch_arr
                
    def save_patch(self, saving_patch:np.ndarray, index=None):

        if self.ods is None:
            self.Create_out_file()

        index = self.station if index is None else index
        
        self.write_fun(saving_patch, index=index)
        
    
    def save_patch2d(self, save_patch:np.ndarray, index=None, ods=None, **kwargs):
        
        index = self.station if index is None else index

        cm = self.clip_map[index]
        oxoff, oyoff, oxsize, oysize = cm[4:8]
        lpadx, rpadx, upady, bpady = cm[-4:]
        
        if all([lpadx, rpadx, upady, bpady]):
            save_patch = save_patch[self.bs:-self.bs, self.bs:-self.bs]
        else:
            clipxs = lpadx if lpadx >= self.bs else self.bs
            clipys = upady if upady >= self.bs else self.bs

            clipxe = clipxs + oxsize
            clipye = clipys + oysize

            save_patch = save_patch[ clipys: clipye, clipxs: clipxe, ]
        
        if ods is None:
            self.ods.GetRasterBand(1).WriteArray(save_patch, oxoff, oyoff)
        else:
            ods.GetRasterBand(1).WriteArray(save_patch, oxoff, oyoff)
        
    def save_patch3d(self, save_patch:np.ndarray, index, ods=None, out_channels=None):

        index = self.station if index is None else index

        cm = self.clip_map[index]
        oxoff, oyoff, oxsize, oysize = cm[4:8]
        lpadx, rpadx, upady, bpady = cm[-4:]
        
        if all([lpadx, rpadx, upady, bpady]):
            save_patch = save_patch[:, self.bs:-self.bs, self.bs:-self.bs]
        else:
            clipxs = lpadx if lpadx >= self.bs else self.bs
            clipys = upady if upady >= self.bs else self.bs

            clipxe = clipxs + oxsize
            clipye = clipys + oysize

            save_patch = save_patch[:, clipys: clipye, clipxs: clipxe, ]
        
        if ods is None:
            for bi in range(self.out_channels):
                self.ods.GetRasterBand(bi+ 1).WriteArray(save_patch[bi,:,:], oxoff, oyoff)
        else:
            for bi in range(out_channels):
                ods.GetRasterBand(bi+ 1).WriteArray(save_patch[bi,:,:], oxoff, oyoff)

    def register_new_output(self, dst_dir, out_channels, out_dtype):
        
        '''
        In case you want to output multi files
        
        '''
        driver = gdal.GetDriverByName('GTiff')
        
        ods = driver.Create(
            dst_dir, self.rx, self.ry, out_channels, out_dtype
        )
        
        ods.SetProjection(self.ds.GetProjection())
        ods.SetGeoTransform(self.ds.GetGeoTransform())

        save_func = self.save_patch2d if out_channels == 1 else self.save_patch3d
        save_param = {'ods': ods, 'out_channels': out_channels}

        # self.another_outs[dst_dir] = [ods, save_func, save_param]
        self.another_outs[dst_dir] = [save_func, save_param]

    def save_to(self, dst_dir, patch, index):
        if dst_dir not in self.another_outs.keys():
            raise ValueError('Please use inference_loader.register_new_output'\
                             'first, or you can use inference_loader.save_patch')
        else:
            
            sf, sp = self.another_outs.get(dst_dir)
            sf(patch, index, **sp)

def loadGeoRasterToImage(image_dir, subset_param=None, out_dtype=None):

    ds = gdal.Open(image_dir, gdal.GA_ReadOnly)
    if ds is None:
        raise Exception(f'file not exist{image_dir}')
    im_width = ds.RasterXSize
    im_height= ds.RasterYSize
    nchannels = ds.RasterCount
    # print(nchannels)

    subset_param = [0, 0, im_width, im_height] if subset_param is None else subset_param
    
    im_data = ds.ReadAsArray(*subset_param)
    if out_dtype is not None:
        im_data = im_data.astype(out_dtype)

    if nchannels == 1:
        im_data = Image.fromarray(im_data)
    elif nchannels == 3 or nchannels == 4:
        im_data = Image.fromarray(im_data[:3,:,:].transpose(1,2,0)).convert('RGB') #TODO
    else:
        raise Exception(f'Parse image shape [{im_data.shape}] failed.')
    
    return im_data

def get_shifting_steps(l, s):
    return l // s if l % s == 1 else l // s + 1

def nagetive_pad_check(start, bs):
    if start < 0:
        neg_pad = -start
        start = 0
        out_start = 0
    else:
        neg_pad = 0
        out_start = start + bs
    return neg_pad, start, out_start

def positive_pad_check(start, ps , r, bs, stride, neg_pad):

    if start + ps < r:
        offset = ps
        out_offset = stride
        pos_pad = 0
    else:
        offset = r - start
        pos_pad = ps + start - r
        if offset < stride + bs:
            out_offset = offset - bs
        else:
            out_offset = stride
    
    if neg_pad > 0:
        offset -= neg_pad

    return pos_pad, offset, out_offset