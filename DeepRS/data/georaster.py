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



class multitask_georasterDataset():
    def __init__(self, cfg, split='train', repeat=1):
        
        # self.patch_size
        self.classes= cfg.DATASETS.CLASSES
        # self.labels = cfg.DATASETS.LABELS
        self.stations = cfg.DATASETS.STATIONS
        # self.auto_weight = cfg.LOSS.AUTO_WEIGHT
        self.patch_size = cfg.DATASETS.PATCH_SIZE
        # TODO this param indicates the size of the original patches
        self.original_size = 512
        self.max_anchor = self.original_size-self.patch_size
        # self.augment = augment(cfg)
        self.image_names =  cfg.DATASETS.IMAGE_FOLDER_NAMES
        self.label_names =  cfg.DATASETS.LABEL_FOLDER_NAMES
        self.image_idx =  cfg.DATASETS.IMAGE_FOLDER_INDEX
        self.label_idx =  cfg.DATASETS.LABEL_FOLDER_INDEX
        self.images = []
        self.masks = []
        self.stations_list = []
        
        self.norm_mean = cfg.DATASETS.MEAN
        self.norm_std = cfg.DATASETS.STD

        # TODO defectSimulate is None
        self.ds = None

        self._transpose_idx = [0,1,2,3,4,5,6]

        
        _split_file = os.path.join(self.root, '{}.csv'.format(self.split))
        

        self.jitter =  T.Compose({
            T.ColorJitter(brightness=[0.8,1.4], contrast=0.3, saturation=0.3),
            T.ToTensor(),
            })
        self.to_tensor = T.Compose({
            T.ToTensor()
        })

        # self.normalize = T.Normalize([0.2638, 0.3007, 0.2999],[0.1292, 0.1126, 0.1007])#google
        # self.normalize = T.Normalize([0.3722, 0.4498, 0.3964],[0.1220, 0.1139, 0.1064]) # gf2 zj
        self.normalize = T.Normalize(mean=self.norm_mean, std=self.norm_std) # gf2 zj
        
        self.PILToTensor = T.PILToTensor()
        
        if not os.path.isfile(_split_file):
            raise RuntimeError('Unkown dataset _split_file: {}'.format(_split_file))
        

        with open(os.path.join(_split_file), 'r') as lines:
            lines = list(lines)
            
            for line in tqdm(lines):
                
                relative_paths = line.rstrip('\n').split(',')
                
                relative_image_paths = [relative_paths[e] for e in self.image_idx]
                relative_label_paths = [relative_paths[e] for e in self.label_idx]

                

                self.update_stations(relative_image_paths[0])
                
                self.update_images(relative_image_paths)
                self.update_masks(relative_label_paths)

        
        if repeat > 1:
            self.images = self.images * repeat
            self.masks = self.masks * repeat

        self.samples_num = len(self.images)
        print('Number of samples: {}'.format(self.samples_num))

    def get_station(self, img_name=None):
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
    
    def update_images(self, impaths):
        _imgs = [os.path.join(self.root, e) for e in impaths]
        if all([os.path.isfile(i) for i in _imgs]):
            self.images.append(_imgs)
        else:
            raise

    def update_masks(self, lbpaths):
        _lbs = [os.path.join(self.root, e) for e in lbpaths]
        if all([os.path.isfile(i) for i in _lbs]):
            self.masks.append(_lbs)
        else:
            raise

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
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
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
        if self.max_anchor == 0:
            anchor_w = 0
            anchor_h = 0
        elif self.max_anchor > 0:
            anchor_w = np.random.randint(0, self.max_anchor)
            anchor_h = np.random.randint(0, self.max_anchor)
        else:
            raise
        
        fetch_param = [anchor_w, anchor_h, self.patch_size, self.patch_size]
        
        return fetch_param
        # self.patch_size

    # augmentation for Image instances
    def _transpose_single(self, *inputs):
        transpose_idx = np.random.choice(self._transpose_idx)
        output = []
        for input in inputs:
            output.append(input.transpose(transpose_idx))
        return output
    
    def _transpose(self, *inputs):
        transpose_idx = np.random.choice(self._transpose_idx)
        
        output = []
        for input in inputs:
            if isinstance(input, list):
                output.append([i.transpose(transpose_idx) for i in input])
            else:
                output.append(input.transpose(transpose_idx))

        return output
        

        

    def __getitem__(self, index):
        
        # TODO affine transform
        # image, im_ds = georaster_loader.loadGeoRaster(self.images[index], dt=gdal.GDT_Byte)
        # target, msk_ds = georaster_loader.loadGeoRaster(self.masks[index], dt=gdal.GDT_Byte)
        
        
        fetch_param = self._get_fetch_param()
        
        # print(index)
        # print(self.images[index])
        # print(self.masks[index])

        image = [loadGeoRasterToImage(e,fetch_param) for e in self.images[index]]
        target = [loadGeoRasterToImage(e,fetch_param, np.uint8) for e in self.masks[index]]

        # rotate and flip
        try:
            image, target = self._transpose(image, target) 
        except:
            print(
                f'image shape {self.images[index]}'
                f'label shape {self.masks[index]}'
                )
        

        image = [self.jitter(e) for e in image] 
        # image = self.to_tensor(image)
        image = [self.normalize(e) for e in image]


        # print('=='*10)
        # print(np.mean(image))
        # image, target = self._standardTsf(image, target, size=self.patch_size)

        target = [self.PILToTensor(e) for e in target]
        
        # # self.__check_out_init()
        # # self.__check_out(image, target, f'i_{index}.png')

        # image /= 255.
        # print(torch.mean(image), f'i_{index}.png')
        
        return image, [e.squeeze(0).long() for e in target]
    
    
    def __len__(self):
        return self.samples_num

        
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