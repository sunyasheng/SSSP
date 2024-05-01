import json
import logging
import os
import pickle
import sys
import ast
import torch
from ts.torch_handler.base_handler import BaseHandler
import io
import torchvision
from omegaconf import OmegaConf
import pandas as pd
import csv
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.io import (
    read_video_timestamps,
    read_video
)

import base64
from PIL import Image
import torch.nn.functional as F

from image_synthesis.utils.misc import get_model_parameters_info, copy_state_dict
from image_synthesis.utils.io import load_yaml_config
from image_synthesis.utils.misc import instantiate_from_config

logger = logging.getLogger(__name__)



class Sketch2NerfHandler(BaseHandler):
    def __init__(self):
        super(Sketch2NerfHandler, self).__init__()
        self.initialized = False
    
    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        # model_pt_path = 'OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000025e_89309iter.pth'

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # self.device = 'cuda'

        config = load_yaml_config('./configs/testing/sketch_128_quali.yaml')
        self.model = instantiate_from_config(config['model'])
        state_dict = torch.load(model_pt_path, map_location=self.device)
        copy_state_dict(state_dict['model'], self.model)

        self.model.to(self.device)
        self.model.eval()
        # import pdb; pdb.set_trace()

        self.initialized = True

    # based on
    # https://github.com/pytorch/serve/blob/6b6810f0aa12cf1a9ebc39aae771aa1c850eb29a/ts/torch_handler/vision_handler.py#L15
    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def inference(self, data, *args, **kwargs):
        marshalled_data = data.to(self.device)
        marshalled_data_dict = {'sketch': marshalled_data}
        batch = self.model.prepare_sketch(marshalled_data_dict)
        with torch.no_grad():
            sample = self.model.sample(batch, cat_dim=-1)

        i = 0
        cat = torch.cat([
                F.interpolate(marshalled_data[i:i+1],size=sample['pred_image'].shape[-2:],mode='bicubic'),
                sample['pred_image_left'][i:i+1]/255,
                sample['pred_image_frontal'][i:i+1]/255,
                sample['pred_image_right'][i:i+1]/255,], dim=-1)
        
        torchvision.utils.save_image(cat, 'cat.jpg')
        return cat

    def postprocess(self, inference_output):
        return [inference_output]


if __name__ == '__main__':
    handler = Sketch2NerfHandler()
    handler.initialize(ctx=None)
    data = torch.randn(1, 3, 256,256)
    handler.inference(data)

