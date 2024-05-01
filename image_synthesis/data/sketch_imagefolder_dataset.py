
import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import sys
import glob
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from image_synthesis.data.base_dataset import BaseDataset
from image_synthesis.data.imagefolder_dataset import ImageFolderDataset
import image_synthesis.modeling.networks.third_party.eg3d.dnnlib as dnnlib
from image_synthesis.utils.pickle_utils import read_pickle
from image_synthesis.utils.ldmk_utils import draw_ldmk
from image_synthesis.utils.cv2_utils import contour_to_dist

try:
    import pyspng
except ImportError:
    pyspng = None



class TestPseduoSketchImageFolderDataset(ImageFolderDataset):
    def __init__(self,
        path,
        test_meta_path,
        w_distance_transform=False,
        resolution = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
        ):
        self._path = path
        self.w_distance_transform = w_distance_transform
        self._test_meta_path = test_meta_path

        self.image_fpaths = self.read_meta(test_meta_path)
        super().__init__(path=path,resolution=resolution,image_fpaths=self.image_fpaths,**super_kwargs)

    def __getitem__(self, idx):
        image_path = self.image_fpaths[idx]
        image = self._load_raw_image(image_path)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        data_dict = {}
        data_dict['image'] = image.copy()
        data_dict['image_path'] = image_path
        data_dict['camera'] = self.get_label(idx)
        return data_dict

    def __len__(self, ):
        return len(self.image_fpaths)


class PureImageFolderDataset(ImageFolderDataset):
    def __init__(self,
                 path,
                w_distance_transform=False,
                resolution = None, # Ensure specific resolution, None = highest available.                 
                 **super_kwargs):
        self._path = path
        self.w_distance_transform = w_distance_transform
        super().__init__(path=path, resolution=resolution, **super_kwargs)
        
    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        data_dict = {}
        data_dict['image'] = image.copy()
        if self.w_distance_transform is True:
            data_dict['sketch_dt'] = contour_to_dist(sketch[:,:,0].astype(np.uint8))
        data_dict['camera'] = self.get_label(idx)
        return data_dict

class SketchFolderDataset(Dataset):
    def __init__(self, path,
                       resolution = None, 
                       **super_kwargs):
        self._path = path
        # super().__init__(path=path, resolution=resolution, **super_kwargs)
        self.sketch_fpaths = glob.glob(os.path.join(path, '*.png')) + \
            glob.glob(os.path.join(path, '*.jpg'))

    def _load_raw_sketch(self, raw_idx):
        sketch_fpath = self.sketch_fpaths[raw_idx]
        self.sketch_path = sketch_fpath 
        sketch = np.array(PIL.Image.open(sketch_fpath))
        if sketch.ndim == 2:
            sketch = sketch[:, :, np.newaxis] # HW => HWC
        return sketch

    def __getitem__(self, idx):
        sketch = self._load_raw_sketch(idx)
        data_dict = {}
        data_dict['sketch'] = sketch.copy()
        data_dict['sketch_path'] = self.sketch_path
        return data_dict

    def __len__(self, ):
        return len(self.sketch_fpaths)

class SketchImageFolderDataset(ImageFolderDataset):
    def __init__(self,
        path,
        sketch_path,
        meta_sketch_path,
        test_meta_path=None, ## do not use it 
        w_distance_transform=False,
        resolution = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
        ):
        print(path, sketch_path)
        self._path = path
        self._sketch_path = sketch_path
        self.w_distance_transform = w_distance_transform
        self.sketch_fpaths = self.read_meta(meta_sketch_path)
        self.sketch_fpaths = [os.path.join(sketch_path, pth) for pth in self.sketch_fpaths]
        self.image_fpaths = [pth.replace(sketch_path, path) for pth in self.sketch_fpaths]
        # import pdb; pdb.set_trace();
        super().__init__(path=path,resolution=resolution,image_fpaths=self.image_fpaths,**super_kwargs)


    def _load_raw_sketch(self, raw_idx):
        # img_fname = self._image_fnames[raw_idx]
        # sketch_fname = img_fname.replace('.png', '_edges.jpg')
        # sketch_fpath = os.path.join(self._sketch_path, sketch_fname)
        sketch_fpath = self.sketch_fpaths[raw_idx]
        self.sketch_path = sketch_fpath 
        sketch = np.array(PIL.Image.open(sketch_fpath))
        if sketch.ndim == 2:
            sketch = sketch[:, :, np.newaxis] # HW => HWC
        return sketch

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        sketch = self._load_raw_sketch(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        data_dict = {}
        data_dict['image'] = image.copy()
        data_dict['sketch'] = sketch.copy()
        if self.w_distance_transform is True:
            data_dict['sketch_dt'] = contour_to_dist(sketch[:,:,0].astype(np.uint8))
        data_dict['camera'] = self.get_label(idx)
        data_dict['sketch_path'] = self.sketch_path
        return data_dict



# class TestSketchImageFolderDataset(ImageFolderDataset):
#     def __init__(self, 
#                 path,
#                 sketch_path,
#                 meta_sketch_path,
#                 w_distance_transform=False,
#                 resolution = None, # Ensure specific resolution, None = highest available.
#                 **super_kwargs,         # Additional arguments for the Dataset base class.
#                 ):
#         super().__init__(path=path,resolution=resolution,**super_kwargs)
#         self._path = path
#         self._meta_sketch_path = meta_sketch_path
#         self.w_distance_transform = w_distance_transform

#         self.sketch_fpaths = self.read_meta(meta_sketch_path)


#     def __getitem__(self, idx):
#         data_dict = {}
#         sketch_fpath = self.sketch_fpaths[idx]
#         sketch = self._load_raw(sketch_fpath)
#         data_dict['sketch'] = sketch.copy()
#         if self.w_distance_transform is True:
#             data_dict['sketch_dt'] = contour_to_dist(sketch[:,:,0].astype(np.uint8))
#         data_dict['sketch_path'] = sketch_fpath
#         try:
#             data_dict['camera'] = self.get_label(idx)
#         except Exception as ex:
#             print(str(ex))

#         return data_dict

#     def __len__(self, ):
#         return len(self.sketch_fpaths)


if __name__ == '__main__':
    data_root = '/root/dataset/FFHQ/train/CROP/'
    image_folder_dataset = ImageFolderDataset(data_root, use_labels=True)
    image_folder_dataset._load_raw_labels()
    import pdb; pdb.set_trace()