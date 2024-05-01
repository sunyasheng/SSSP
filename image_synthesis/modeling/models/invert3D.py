# ------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import torch
import math
from torch import nn
from image_synthesis.utils.misc import instantiate_from_config
import time
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F
from torch.cuda.amp import autocast
import kornia
# from image_synthesis.utils.image_utils import SobelConv, prepare_simplified_sketch
# from image_synthesis.modeling.networks.third_party.sketch_simplification.model_gan import immean, imstd
# from image_synthesis.modeling.networks.third_party.sketch_simplification.model_gan import model as SktechSimplificationModel
from image_synthesis.utils.checkpoint_utils import copy_state_dict
from image_synthesis.utils.network_utils import freeze_network

class SketchInverter(nn.Module):
    def __init__(
        self,
        *,
        inverter_config,
        inversion_mode='sketch',
        is_train=False,
        is_triplane_reverse=False,
        is_flip_all=False
    ):
        super().__init__()
        self.is_train = is_train
        self.is_triplane_reverse = is_triplane_reverse
        self.is_flip_all = is_flip_all
        self.inverter = instantiate_from_config(inverter_config)
        # self.preprocess_sobel_conv = SobelConv()

        # model_ckpt_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        # model_ckpt_path = os.path.join(model_ckpt_dir, 'OUTPUT/pretrained/model_gan.pth')
        # self.sketch_simplification = self.prepare_sketch_simplication(model_ckpt_path)
        self.preprocess_sobel_conv = None
        self.sketch_simplification = None
        self.inversion_mode = inversion_mode

    @staticmethod
    def prepare_sketch_simplication(model_ckpt_path):
        model = SktechSimplificationModel
        state_dict = torch.load(model_ckpt_path)
        copy_state_dict(state_dict, model)
        freeze_network(model)
        model = model.eval()
        return model

    def parameters(self, recurse=True, name=None):
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            names = name.split('+')
            params = []
            for n in names:
                try: # the parameters() method is not overwritten for some classes
                    params += getattr(self, name).parameters(recurse=recurse, name=name)
                except:
                    params += getattr(self, name).parameters(recurse=recurse)
            return params

    @property
    def device(self):
        return self.inverter.device

    def get_ema_model(self):
        return self.inverter

    # @torch.no_grad()
    # def prepare_ldmk_edge(self, batch):
    #     ldmk_edge = batch['ldmk_edge'].float()/255.*2 - 1.
    #     ldmk_edge = ldmk_edge.permute(0,3,1,2)
    #     # import torchvision
    #     # torchvision.utils.save_image(ldmk_edge, 'ldmk_edge.jpg')
    #     # import pdb; pdb.set_trace()
    #     ldmk_edge = F.interpolate(ldmk_edge, size=(224,224), mode='area')
    #     # ldmk_edge = ldmk_edge.repeat(1,3,1,1)
    #     return {'ldmk_edge': ldmk_edge}

    # @torch.no_grad()
    # def prepare_ldmk_dt(self, batch):
    #     ldmk_dt = batch['ldmk_dt'].float()/255.*2 - 1
    #     ldmk_dt *= -1 # reverse
    #     ldmk_dt = ldmk_dt.unsqueeze(1)
    #     ldmk_dt = F.interpolate(ldmk_dt, size=(224,224), mode='area')
    #     return {'ldmk_dt': ldmk_dt}

    # @torch.no_grad()
    def prepare_sketch_dt(self, batch):
        sketch_dt = batch['sketch_dt'].float()/255.*2 -1
        sketch_dt *= -1
        # import pdb; pdb.set_trace();
        sketch_dt = sketch_dt.unsqueeze(1)
        sketch_dt = F.interpolate(sketch_dt, size=(224,224), mode='bicubic')
        return {'sketch_dt': sketch_dt}

    # @torch.no_grad()
    def prepare_sketch(self, batch):
        sketch = batch['sketch'].float()/255.*2 -1
        # import pdb; pdb.set_trace();
        sketch *= -1
        if sketch.shape[3] <=4: ## when sketch need to be permute
            sketch = sketch.permute(0,3,1,2)
        sketch = torch.mean(sketch, dim=1, keepdim=True)
        # sketch = F.max_pool2d(sketch, kernel_size=3, padding=1)
        sketch = F.max_pool2d(sketch, kernel_size=3, padding=1)
        sketch = F.interpolate(sketch, size=(224,224), mode='bicubic')
        # import torchvision
        # import pdb; pdb.set_trace();

        ## we further add binarization to enforce consistent real-world applications
        sketch[sketch>=0] = 1.0
        sketch[sketch<0] = -1.0

        return {'sketch': sketch}

    @staticmethod
    def flip_yaw(pose_matrix):
        flipped = pose_matrix.clone()
        flipped[:, 0, 1] *= -1
        flipped[:, 0, 2] *= -1
        flipped[:, 0, 3] *= -1
        flipped[:, 1, 0] *= -1
        flipped[:, 2, 0] *= -1

        # flipped[:, 1, 0] *= -1
        # flipped[:, 1, 2] *= -1
        # flipped[:, 1, 3] *= -1
        # flipped[:, 0, 1] *= -1
        # flipped[:, 2, 1] *= -1

        return flipped
        
    # in this function, we reverse the former half dimenstion to get the latter half dimension
    @staticmethod
    def reverse_batch(input, is_flip_all=False):
        res_input = {}
        batch_size = None
        for k, v in input.items():
            batch_size = v.shape[0] if batch_size is None else batch_size
            if k != 'camera':
                new_v = v.clone()
                if is_flip_all is True:
                    new_v = torch.flip(new_v, dims=[3,])
                else:
                    new_v[batch_size//2:] = torch.flip(new_v[:batch_size//2], dims=[3,])
                res_input[k] = new_v
            
            # # import pdb; pdb.set_trace()
            if k == 'camera':
                new_v = v.clone()
                pose, intrinsics = v[:, :16].reshape(batch_size,4,4), v[:,16:].reshape(batch_size,3,3)
                flipped_pose = SketchInverter.flip_yaw(pose)
                flip_v = torch.cat([flipped_pose.reshape(batch_size,-1), intrinsics.reshape(batch_size,-1)], dim=1)
                if is_flip_all is True:
                    new_v = flip_v
                else:
                    new_v[batch_size//2:] = flip_v[:batch_size//2]
                res_input[k] = new_v
        return res_input

    # @autocast(enabled=False)
    # @torch.no_grad()
    def prepare_input(self, batch):
        # import pdb; pdb.set_trace();
        input = {}
        input['camera'] = batch['camera'] if 'camera' in batch else None
        if 'image' in batch:
            input['image'] = batch['image'].float()/255.*2. - 1.
            # input['image'] = F.interpolate(input['image'], size=(224,224), mode='bicubic')

            if self.is_train is True:
                if 'sketch' not in batch:
                    with torch.no_grad():
                        batch_image = F.interpolate(batch['image'].float(), size=(224,224), mode='bicubic')
                        sobel_ts = self.preprocess_sobel_conv(batch_image/255.)
                        simplified_sketch = self.sketch_simplification(sobel_ts)
                        # import pdb; pdb.set_trace()
                        input['grad'] = prepare_simplified_sketch(simplified_sketch)                
                else:
                    input.update(self.prepare_sketch(batch))
                    input['grad'] = input['sketch']
                input['grad'] = input['grad'].detach().requires_grad_(True).repeat(1,3,1,1)

        # import pdb; pdb.set_trace();

        if 'ldmk_edge' in batch:
            input.update(self.prepare_ldmk_edge(batch))
            input['grad'] = input['ldmk_edge'] # we directly replace grad by ldmk_edge

        if 'sketch' in batch and self.is_train is False: ## at training time, only support on-the-fly sketch generation
            input.update(self.prepare_sketch(batch))
            input['grad'] = input['sketch']
            input['grad'] = input['grad'].detach().requires_grad_(True).repeat(1,3,1,1)

        ### distance transform might be noisy, delete it. 
        # if 'ldmk_dt' in batch:
        #     input.update(self.prepare_ldmk_dt(batch))
        #     input['grad'] = torch.cat([input['ldmk_dt'], input['grad']],dim=1)

        # if 'sketch_dt' in batch:
        #     input.update(self.prepare_sketch_dt(batch))
        #     input['grad'] = torch.cat([input['sketch_dt'], input['sketch']],dim=1)
        
        # import torchvision
        # import pdb; pdb.set_trace();
        if self.is_train is True and self.is_triplane_reverse is True:
            input = self.reverse_batch(input)
        
        if self.is_train is False and self.is_flip_all is True:
            input = self.reverse_batch(input, self.is_flip_all)
        return input

    @torch.no_grad()
    def sample(
        self,
        batch,
        mp4_save_path=None,
        plane_modes=None,
        is_video_gen=False,
        **kwargs,
    ):
        samples = {}
        self.eval()
        
        input = self.prepare_input(batch)
        # import pdb; pdb.set_trace()
        # forward inference
        if is_video_gen is True:
            output = self.inverter.sample_3d(input, self.preprocess_sobel_conv, self.sketch_simplification, mp4_save_path)
            return output
        else:
            output = self.inverter.sample(input, self.preprocess_sobel_conv, self.sketch_simplification, plane_modes=plane_modes, **kwargs)

        if 'pred_image' not in output:
            self.train()
            return {}
            
        h,w = output['pred_image'].shape[-2], output['pred_image'].shape[-1]
        pred = (output['pred_image']+1.)*0.5*255.
        gt = (input['image']+1.)*0.5 * 255. if 'image' in input else pred
        sketch = (input['grad']+1.)*0.5* 255.
        
        sketch = F.interpolate(sketch,(h,w),mode='bilinear')
        gt = F.interpolate(gt,(h,w),mode='bilinear')

        gt, pred, sketch = gt.cpu(), pred.cpu(), sketch.cpu()

        pred = torch.clamp(pred, min=0, max=255)
        sketch = torch.clamp(sketch, min=0, max=255)
        samples['sketch_gt_pred'] = torch.cat([sketch[:,:3], gt, pred], dim=-2)
        samples['pred_image'] = (output['pred_image']+1.)*0.5*255
        samples['image_raw'] = (output['image_raw']+1.)*0.5*255
        samples['input'] = (output['input']+1.)*0.5*255
        if 'image' in input: samples['image'] = (input['image']+1.)*0.5 * 255.

        samples['pred_image_overlay'] = self.overly_img_sketch(samples['pred_image'], samples['input'])
        # import torchvision
        # import pdb; pdb.set_trace();
        
        if self.is_train is False:
            if 'pred_image_left' in output:
                samples['pred_image_left'] = (output['pred_image_left']+1.)*0.5*225
            if 'pred_image_right' in output:
                samples['pred_image_right'] = (output['pred_image_right']+1.)*0.5*255
            if 'pred_image_frontal' in output:
                samples['pred_image_frontal'] = (output['pred_image_frontal']+1.)*0.5*255

        if 'pred_edge' in output:
            pred_edge = (output['pred_edge']+1)*0.5*255
            pred_edge = F.interpolate(pred_edge,(h,w),mode='bicubic')
            pred_edge = pred_edge.cpu()
            pred_edge = torch.clamp(pred_edge, min=0, max=255)
            samples['sketch_gt_pred'] = torch.cat([samples['sketch_gt_pred'],pred_edge], dim=-2)
            samples['pred_edge'] = pred_edge
        
        # import torchvision
        # import pdb; pdb.set_trace();

        self.train()
        return samples

    @staticmethod
    def overly_img_sketch(pred_img, input):
        pred_img = F.interpolate(pred_img, (input.shape[-2],input.shape[-1]), mode='bicubic')
        pred_img, input = torch.clamp(pred_img, min=0, max=255), torch.clamp(input, min=0, max=255)
        overlay = torch.maximum(pred_img, input)
        return overlay

    def forward(
        self,
        batch,
        name='none',
        **kwargs
    ):
        input = self.prepare_input(batch)
        # import pdb; pdb.set_trace()
        output = self.inverter(input, self.preprocess_sobel_conv, self.sketch_simplification, inversion_mode=self.inversion_mode, **kwargs)
        return output
