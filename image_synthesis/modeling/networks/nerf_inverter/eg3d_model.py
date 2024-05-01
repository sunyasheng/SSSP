import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import logging
import math
from collections import OrderedDict
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from third_party.eg3d.cam_utils import LookAtPoseSampler, FOV_to_intrinsics
from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.utils import checkpoint_utils
from image_synthesis.utils.network_utils import freeze_network

class EG3DBaseModel(nn.Module):
    def __init__(self, 
                generator_config,
                discriminator_config=None,
                w_anchor_w=True,
                is_train=False,
                fov_deg=18.837):
        super().__init__()
        self.generator = instantiate_from_config(generator_config)

        generator_ckpt_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        generator_ckpt_path = os.path.join(generator_ckpt_dir, generator_config['pretrain_path'])
        self.load_pretrained(generator_ckpt_path)
        self.generator = self.generator.eval()
        self.w_anchor_w = w_anchor_w
        freeze_network(self.generator)

        # self.w_disc = discriminator_config is not None
        # if self.w_disc:
        #     self.discriminator = instantiate_from_config(discriminator_config)
        #     self.load_pretrained_disc(discriminator_config['pretrain_path'])
        #     self.discriminator = self.discriminator.eval()
        #     freeze_network(self.discriminator)

        self.is_train = is_train
        self.fov_deg = fov_deg
        self.anchor_w = None
        self.frontal_conditioning_params, self.frontal_camera_params = self.set_cam_meta()
        self.left_conditioning_params, self.left_camera_params = self.set_cam_meta(angle_y=.4, angle_p=-0.2)
        self.right_conditioning_params, self.right_camera_params = self.set_cam_meta(angle_y=-.4, angle_p=-0.2)
        self.w_space_channel_num = 14

    def load_pretrained(self, pretrain_path):
        G_state_dict = torch.load(pretrain_path)
        checkpoint_utils.copy_state_dict(G_state_dict, self.generator)
        print('load eg3d pretrained model from {}'.format(pretrain_path))

    def load_pretrained_disc(self, disc_pretrain_path):
        D_state_dict = torch.load(disc_pretrain_path)
        checkpoint_utils.copy_state_dict(D_state_dict, self.discriminator, add_prefix='discriminator.')

    def get_anchor_wp(self, batch_size, device):
        if self.anchor_w is None:
            if self.eg3d_avg_w_path is not None:
                import pickle
                if os.path.exists(self.eg3d_avg_w_path):
                    anchor_w = pickle.load(open(self.eg3d_avg_w_path, 'rb'))
                else:
                    raise ValueError

                anchor_w = torch.from_numpy(anchor_w).float().to(device)
            else:
                c = self.frontal_conditioning_params.expand(batch_size, *self.frontal_conditioning_params.shape[1:]).to(device)
                with torch.no_grad():
                    anchor_w = self.generator.style_forward(torch.zeros(size=(batch_size,512)).to(c), c)
            
            anchor_w = anchor_w.detach().requires_grad_(True)
            self.anchor_w = anchor_w
            # import pdb; pdb.set_trace();
        return self.anchor_w

    def set_cam_meta(self, angle_p=-0.2, angle_y=0):
        G = self.generator
        fov_deg = self.fov_deg
        intrinsics = FOV_to_intrinsics(fov_deg, device=self.device)
        device = self.device

        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)

        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p,\
                 cam_pivot, radius=cam_radius, device=self.device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2,\
                 cam_pivot, radius=cam_radius, device=self.device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        return conditioning_params, camera_params


    ###### Operations before feeding it into triplane-generator
    def map_latent_code(self, delta, camera, space_mode='z'):
        if camera is None:
            camera = self.frontal_camera_params.expand(\
                            delta.shape[0], *self.frontal_camera_params.shape[1:]).to(delta)

        if space_mode == 'w':
            delta_w = delta
            # import pdb; pdb.set_trace();
            if self.is_finetune_z2w_mapper is True:
                ref_color_code = torch.randn(size=(delta.shape[0], self.generator.c_dim)).to(delta)
                delta_w = self.generator.mapping(delta_w, ref_color_code)[:,0]

            if self.w_anchor_w is True:
                updated_w = self.get_anchor_wp(delta_w.shape[0], delta_w.device) + delta_w
            else:
                updated_w = delta_w
            updated_wp = updated_w.unsqueeze(1).repeat(1, self.w_space_channel_num,1)

        elif space_mode == 'wp':
            delta_wp = delta
            if self.w_anchor_w is True:
                updated_wp = self.get_anchor_wp(delta_wp.shape[0], delta_wp.device).unsqueeze(1) + delta_wp
            else:
                updated_wp = delta_wp
            
        elif space_mode == 'z' or space_mode == 'zcam':
            if len(delta.shape) == 3:
                delta = delta.squeeze(1)
            # import pdb; pdb.set_trace();
            # delta = torch.randn(1,512).to(delta).repeat(delta.shape[0],1)
            updated_w = self.generator.mapping(delta, camera)
            updated_wp = updated_w
            # import pdb; pdb.set_trace()
        
        else:
            raise ValueError

        return updated_wp

    # def synthesize_image(self, sample_latent_code, camera_params=None):
    #     if camera_params is None:
    #         camera_params = self.frontal_camera_params.expand(\
    #                         sample_latent_code.shape[0], *self.frontal_camera_params.shape[1:]).to(sample_latent_code)
    #     # import pdb; pdb.set_trace();
    #     render_dict = self.generator.synthesis(sample_latent_code, camera_params, noise_mode='const')
    #     synthesized_img = render_dict['image']
    #     return synthesized_img

    def synthesize_image_dict(self, sample_latent_code, camera_params=None, neural_rendering_resolution=None, plane_modes=None,
                                         bbox=None, use_cached_backbone=False, cache_backbone=False):
        if camera_params is None:
            camera_params = self.frontal_camera_params.expand(\
                            sample_latent_code.shape[0], *self.frontal_camera_params.shape[1:]).to(sample_latent_code)
        # import pdb; pdb.set_trace();
        render_dict = self.generator.synthesis(sample_latent_code, camera_params, neural_rendering_resolution=neural_rendering_resolution,
                                noise_mode='const', plane_modes=plane_modes, bbox=bbox, use_cached_backbone=use_cached_backbone, cache_backbone=cache_backbone)
        return render_dict

    def modify_latent_code(self, latent_code_w_in, field, latent_code_w_plus=None):
        if len(latent_code_w_in.shape) == 3: 
            latent_code_w = latent_code_w_in[:,0]
        else:
            latent_code_w = latent_code_w_in
        return_dict = {}

        delta_w = field
        if latent_code_w_plus is None:
            edited_latent_code = latent_code_w.unsqueeze(1).repeat(
                1, self.w_space_channel_num, 1)
        else:
            edited_latent_code = latent_code_w_plus.clone()
 
        for layer_idx in range(self.replaced_layers):
            edited_latent_code[:, layer_idx, :] += delta_w

        return_dict['edited_latent_code'] = edited_latent_code
        return return_dict