
import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from image_synthesis.data.base_dataset import BaseDataset
import image_synthesis.modeling.networks.third_party.eg3d.dnnlib as dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None


class ImageFolderDataset(BaseDataset):
    def __init__(self,
        path,
        resolution = None, # Ensure specific resolution, None = highest available.
        image_fpaths = None,
        **super_kwargs,         # Additional arguments for the Dataset base class.
        ):
        self._path = path
        self._zipfile = None
        
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')
        
        if image_fpaths is None:
            PIL.Image.init()
            self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
            if len(self._image_fnames) == 0:
                raise IOError('No image files found in the specified path')
        else:
            self._image_fnames = [os.path.basename(fp) for fp in image_fpaths]
            # self._type = 'dir'

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    @staticmethod
    def read_meta(meta_path):
        lines = open(meta_path, 'r').read().splitlines()
        paths = [l.strip() for l in lines]
        return paths

    def _load_raw(self, fpath):
        sketch = np.array(PIL.Image.open(fpath))
        if sketch.ndim == 2:
            sketch = sketch[:, :, np.newaxis] # HW => HWC
        return sketch

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        if isinstance(raw_idx, str):
            fname = raw_idx
        else:
            fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        # import pdb; pdb.set_trace();
        labels = [labels[fname.replace('\\', '/').replace('.png', '.jpg')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

if __name__ == '__main__':
    data_root = '/root/dataset/FFHQ/train/CROP/'
    image_folder_dataset = ImageFolderDataset(data_root, use_labels=True)
    image_folder_dataset._load_raw_labels()
    import pdb; pdb.set_trace()