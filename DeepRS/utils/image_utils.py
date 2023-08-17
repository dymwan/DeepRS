from osgeo import gdal
import numpy as np
'''
GDAL Data Type	    Code
GDT_Byte	        1
GDT_UInt16	        2
GDT_Int16	        3
GDT_UInt32	        4
GDT_Int32	        5
GDT_Float32	        6
GDT_Float64	        7
GDT_CInt16	        8
GDT_CInt32	        9
GDT_CFloat32	    10
GDT_CFloat64	    11

methods:    gdal.GetDataTypeName(code)
            gdal.GetDataTypeByName(code)

'''


class image_stretcher:
    def __init__(self, src_dir) -> None:
        
        self.ds:gdal.Dataset = gdal.Open(src_dir)
        self.xsize = self.ds.RasterXSize
        self.ysize = self.ds.RasterYSize
        self.nbands = self.ds.RasterCount
        
        # band:gdal.Band=  ds.GetRasterBand(1)
    
    def create_out(self, dst_dir, dtype=gdal.GDT_Byte):
        
        drv:gdal.Driver = gdal.GetDriverByName('GTiff')
        self.ods = drv.Create(dst_dir, self.xsize, self.ysize, self.nbands, dtype)
        
        self.ods.SetGeoTransform(self.ds.GetGeoTransform())
        self.ods.SetProjection(self.ds.GetProjection())
        
        
    
    def PERCENT_STRETCH(self, r=0.05, left_r=None, right_r=None, depth=None):
        
        
        def _is_ndarray_unsignedinterger(arr):
            # return np.issubdtype(arr.dtype, np.integer)
            return np.issubdtype(arr.dtype, np.unsignedinteger)
        
        def _get_bounding_thresholds(arr, left_r, right_r):
            ''' the input arr should represent a band but a whole raster'''
            if not _is_ndarray_unsignedinterger(arr):
                raise NotImplementedError
            
            N_pix = arr.shape[-2] * arr.shape[-1]
            
            left_count_max = int(N_pix * left_r)
            right_count_max = int(N_pix * right_r)
            
            count = np.bincount(arr.flatten())
             
            left_bound = 0
            right_bound = arr.max()
            
            left_sum = 0
            right_sum = 0
            
            right_step, left_step = 1, 1
            
            while right_step !=0 or left_step != 0:
                
                if left_sum < left_count_max:
                    left_sum += count[left_bound]
                    left_bound += left_step
                else:
                    left_step = 0
                
                if right_sum < right_count_max:
                    
                    right_sum += count[right_bound]
                    right_bound -= right_step
                else:
                    right_step = 0
            
            return left_bound, right_bound
        
        # arr = self.ds.ReadAsArray()
        
        left_r = left_r if left_r is not None else r
        right_r = right_r if right_r is not None else r
        
        nb = self.ds.RasterCount
        
        for bi in range(1, nb+1):
            
           
            bands = self.ds.GetRasterBand(bi)
            bandarr = bands.ReadAsArray()
            left_b, right_b= _get_bounding_thresholds(bandarr, left_r, right_r)
            
            print(left_b, right_b)
            # truncated_down = np.percentile(bandarr, left_r*100)
            # truncated_up = np.percentile(bandarr, 100- right_r *100)
            
            bands = self.ds.GetRasterBand(bi)
            bandarr = bands.ReadAsArray()


            bandarr = bandarr.astype(np.float32)
            max_out = 255
            min_out = 0
            bandarr = (bandarr - left_b) / (right_b - left_b) * (max_out - min_out) + min_out
   
            bandarr[bandarr<0] = 0
            bandarr[bandarr>255] = 255
            
            oband = self.ods.GetRasterBand(bi)
            oband.WriteArray(bandarr.astype(np.uint8))
    
    def MINMAX_STRETCH(self):
        
        for bi in range(1, self.nbands+1):
            
            bandi = self.ds.GetRasterBand(bi)           

            bandarr = bandi.ReadAsArray()
            
            bandarr = bandarr.astype(np.float32)
            bandarr -= bandarr.min()
            bandarr /= bandarr.max()
            
            bandarr *= 255
            
            oband = self.ods.GetRasterBand(bi)
            oband.WriteArray(bandarr.astype(np.uint8))
            
        

if __name__ == '__main__':
    
    src_dir = r'\\172.24.83.111\share1\dymwan\nongkeyuan\share\data\GF2_0611.tif'
    
    

    imst = image_stretcher(src_dir=src_dir)
    
    imst.create_out(r'\\172.24.83.111\share1\dymwan\nongkeyuan\share\data\GF2_0611_streeched_percet.tif')
    # imst.MINMAX_STRETCH()
    imst.PERCENT_STRETCH(0.02)