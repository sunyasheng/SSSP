import sys
import os
import cv2
abspath = os.path.dirname(__file__)
sys.path.insert(0, abspath)

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_dir = os.path.abspath(root_dir)
# print(root_dir)
sys.path.append(os.path.join(root_dir, 'deploy'))
# print(os.path.join(root_dir, 'deploy'))
sys.path.append(os.path.join(root_dir, 'Sketch2Nerf'))

from server.server import Server

import io
import base64

import json
import logging
import os
import pickle
import sys
import ast
import torch
import io
import torchvision
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
import numpy as np

from PIL import Image
import torch.nn.functional as F

from image_synthesis.utils.misc import get_model_parameters_info, copy_state_dict
from image_synthesis.utils.io import load_yaml_config
from image_synthesis.utils.misc import instantiate_from_config


class Sketch2Nerf_Server(Server):
    def __int__(self):
        super().__init__()
        self.initialize()

    def initialize(self):
        sub_model_pt_path = 'OUTPUT/fs_ffhq_resnet_dualcontr_w_triplane_reverse_w_anchor_patchsampling_lab/checkpoint/000025e_89309iter.pth'
        model_pt_path = os.path.join(root_dir, 'Sketch2Nerf', sub_model_pt_path)
        self.device = 'cuda'
        config_path = os.path.join(root_dir, 'Sketch2Nerf', 'configs/testing/sketch_128_quali.yaml')
        config = load_yaml_config(config_path)
        self.model = instantiate_from_config(config['model'])
        state_dict = torch.load(model_pt_path, map_location=self.device)
        copy_state_dict(state_dict['model'], self.model)

        self.model.to(self.device)
        self.model.eval()

        self.initialized = True

    def preprocess(self, img_np):
        img_pil = Image.fromarray(img_np)
        img_ts = transforms.ToTensor()(img_pil)
        img_ts = img_ts.unsqueeze(0)
        img_ts = img_ts.to(self.device) * 255
        # import pdb; pdb.set_trace()
        return img_ts

    def inference(self, img_np):
        marshalled_data = self.preprocess(img_np)
        batch = {'sketch': marshalled_data}
        # batch = self.model.prepare_sketch(marshalled_data_dict)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            sample = self.model.sample(batch, cat_dim=-1)

        i = 0
        cat = torch.cat([
                F.interpolate(sample['input'][i:i+1]/255,size=sample['pred_image'].shape[-2:],mode='bicubic'),
                sample['pred_image_left'][i:i+1]/255,
                sample['pred_image_frontal'][i:i+1]/255,
                sample['pred_image_right'][i:i+1]/255,], dim=-1)
        
        torchvision.utils.save_image(cat, 'cat.jpg')
        cat_bytes = self.postprocess(cat)
        return cat_bytes

    def inference3d(self, img_np):
        marshalled_data = self.preprocess(img_np)
        batch = {'sketch': marshalled_data}
        
        import uuid
        mp4_save_path = os.path.join('results/temp_videos', str(uuid.uuid4())+'.mp4')
        os.makedirs(os.path.dirname(mp4_save_path), exist_ok=True)
        with torch.no_grad():
            sample = self.model.sample(batch, mp4_save_path=mp4_save_path, is_video_gen=True)        
        mp4_bytes = open(mp4_save_path, 'rb').read()
        # print('video save to {}.'.format(mp4_save_path))
        os.system('rm -rf {}'.format(mp4_save_path))
        return mp4_bytes

    ## torch tensor to bytes
    def postprocess(self, inference_output):
        inference_output = inference_output.permute(0,2,3,1)[0]*255
        inference_np = inference_output.cpu().numpy()
        inference_np = inference_np.astype(np.uint8)
        inference_np = cv2.cvtColor(inference_np, cv2.COLOR_RGB2BGR)
        # import pdb; pdb.set_trace()
        inference_bytes = cv2.imencode('.jpg', inference_np)[1].tobytes()

        return inference_bytes
    

serv = Sketch2Nerf_Server()


if __name__ == '__main__':
    image_path = '00009.png'
    image_np = cv2.imread(image_path)
    # serv.inference(image_np)
    serv.inference3d(image_np)
