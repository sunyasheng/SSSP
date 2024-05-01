
import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import sys
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


class LDMKImageFolderDataset(ImageFolderDataset):
    def __init__(self,
        path,
        ldmk_path,
        w_distance_transform=False,
        resolution = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
        ):
        super().__init__(path=path,resolution=resolution,**super_kwargs)
        self._path = path
        self._ldmk_path = ldmk_path
        self.w_distance_transform = w_distance_transform
    
    def _load_raw_ldmk_edge(self, raw_idx):
        img_fname = self._image_fnames[raw_idx]
        ldmk_fname = img_fname.replace('.png','.pkl').replace('.jpg', '.pkl')
        ldmk_fpath = os.path.join(self._ldmk_path, ldmk_fname)
        ldmk_np = read_pickle(ldmk_fpath)['ldmk'].astype(np.int32)
        ldmk_edge = draw_ldmk(ldmk_np, image_shape=(self.resolution,self.resolution,3))
        return ldmk_edge

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        ldmk_edge = self._load_raw_ldmk_edge(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        data_dict = {}
        data_dict['image'] = image.copy()
        data_dict['ldmk_edge'] = ldmk_edge.copy()
        if self.w_distance_transform is True:
            data_dict['ldmk_dt'] = contour_to_dist(ldmk_edge[:,:,0].astype(np.uint8))
        data_dict['camera'] = self.get_label(idx)
        return data_dict


if __name__ == '__main__':
    data_root = '/root/dataset/FFHQ/train/CROP/'
    image_folder_dataset = ImageFolderDataset(data_root, use_labels=True)
    image_folder_dataset._load_raw_labels()
    import pdb; pdb.set_trace()