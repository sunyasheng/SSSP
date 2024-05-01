import torch
import torch.nn as nn
from functools import partial
import sys
import os
import cv2
import numpy as np

from image_synthesis.utils.misc import instantiate_from_config
from .eg3d_model import EG3DBaseModel
from torch.nn import functional as F
from image_synthesis.modeling.losses.loss import VGGLoss, DiscLoss, Patch_Contrastive_Loss
from image_synthesis.modeling.networks.third_party.pixel2style2pixel.criteria.lpips.lpips import LPIPS
from image_synthesis.modeling.networks.third_party.pixel2style2pixel.criteria.w_norm import WNormLoss
import kornia.filters.sobel
from image_synthesis.modeling.losses.loss import Patch_Contrastive_Loss, PARTExtractor
from image_synthesis.modeling.losses.architecture import VGG19
from image_synthesis.modeling.losses.contr.patchnce import PatchSampleF
from image_synthesis.utils.image_utils import SobelConv, prepare_simplified_sketch
import torchvision
from image_synthesis.modeling.modules.arcface_operators import define_net_recog
from image_synthesis.modeling.losses.loss import GANLoss
from image_synthesis.utils.checkpoint_utils import *
from image_synthesis.utils.cam_utils import *
from image_synthesis.utils.io_utils import *
# from image_synthesis.utils.image_utils import rgb_to_lab
import kornia.color.lab
from image_synthesis.utils.network_utils import unfreeze_network


########### inherited from eg3d and further include an encoder ############
class Eg3dInverter(EG3DBaseModel):
    def __init__(self,
                encoder_config,
                generator_config,
                loss_config=None,
                net_recog_config=None,
                invert_space='z',
                w_anchor_w=True,
                vq_config=None,
                clip_loss_config=None,
                discriminator_config= None,
                varitex_encoder_config=None,
                is_train=True,
                is_unsupervised=False,
                w_triplet_loss=False,
                is_seperate_part=False,
                is_edge_loss=False,
                is_only_cam_pose=False,
                is_local_dense_sampling=False,
                is_lab_loss=False,
                is_finetune_z2w_mapper=False,
                finetune_color_head=False,
                eg3d_avg_w_path=None,
                mix_geo_tex_mode=None):
        super().__init__(generator_config=generator_config,
                        discriminator_config=discriminator_config,
                        w_anchor_w=w_anchor_w,
                        is_train=is_train)
        self.is_train = is_train
        self.is_local_dense_sampling = is_local_dense_sampling
        self.is_edge_loss = is_edge_loss
        self.is_lab_loss = is_lab_loss
        self.is_only_cam_pose = is_only_cam_pose
        self.is_finetune_z2w_mapper = is_finetune_z2w_mapper
        self.eg3d_avg_w_path = eg3d_avg_w_path
        # it will consume large amount of GPU memory
        if finetune_color_head is True:
            from image_synthesis.modeling.networks.decoder.eg3d_decoder import OSGDecoder
            new_decoder = OSGDecoder(32, {'decoder_lr_mul': self.generator.rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
            copy_state_dict(self.generator.decoder.net.state_dict(), new_decoder.net_color)
            copy_state_dict(self.generator.decoder.net.state_dict(), new_decoder.net)
            self.generator.decoder = new_decoder

            if self.is_train is True:
                unfreeze_network(self.generator.decoder.net_color)
            # import pdb; pdb.set_trace();

        self.w_triplet_loss = w_triplet_loss
        self.invert_space = invert_space
        self.mix_geo_tex_mode = mix_geo_tex_mode
        # self.D_reg_interval = None if not (discriminator_config is not None and 'D_reg_interval' in discriminator_config ) else None
        if discriminator_config is not None:
            from image_synthesis.modeling.modules.discriminator_operators import define_discriminator
            self.discriminator = define_discriminator(discriminator_config)
        else:
            self.discriminator = None
        # import pdb; pdb.set_trace();
        
        self.GAN_loss = GANLoss() if self.discriminator is not None else None
        self.encoder = instantiate_from_config(encoder_config)
        self.varitex_encoder = instantiate_from_config(varitex_encoder_config) if varitex_encoder_config is not None else None
        # self.vggloss = VGGLoss()
        # self.discloss = DiscLoss()
        # self.Sketch_CLIPLoss = instantiate_from_config(loss_config['sketch_cliploss']) if loss_config is not None and 'sketch_cliploss' in loss_config else None
        self.CLIPLoss = instantiate_from_config(clip_loss_config) if clip_loss_config is not None else None
        if self.CLIPLoss is not None:
            freeze_model(self.CLIPLoss)
        
        ## In this setting, relax mapper to cover different color patterns
        if self.is_train is True and self.is_finetune_z2w_mapper is True:
            unfreeze_network(self.generator.backbone.mapping)
            freeze_model(self.encoder)

        if loss_config is None or 'patch_contr_loss' not in loss_config:
            patch_encoder = None
        elif loss_config['patch_contr_loss'] == 'vgg':
            patch_encoder = VGG19().cuda().blocks
        else: # share encoder
            patch_encoder = self.encoder.blocks
        self.is_dual_contrastive = True if loss_config is not None and 'is_dual_contrastive' in loss_config else False
        self.is_edge_l1 = True if loss_config is not None and 'is_edge_l1' in loss_config else False
        self.net_recog = None if net_recog_config is None else define_net_recog(net_recog=net_recog_config['net_recog'], 
                                                                                pretrained_path=net_recog_config['pretrained_path'])
        if self.net_recog is not None:
            freeze_model(self.net_recog)

        self.netF = PatchSampleF() if patch_encoder is not None else None
        self.PatchContrastiveLoss = Patch_Contrastive_Loss(patch_encoder, self.netF) if patch_encoder is not None and self.netF is not None else None
        self.LPIPSLoss = LPIPS(net_type='alex').eval()
        if self.LPIPSLoss is not None:
            freeze_model(self.LPIPSLoss)

        self.PartExtractor = PARTExtractor(is_seperate_part=is_seperate_part)
        self.WNORMLoss = WNormLoss(start_from_latent_avg=False)
        self.is_unsupervised = is_unsupervised
        # import pdb; pdb.set_trace()
        # self.sobel_conv = SobelConv()
        # freeze_model(self.sobel_conv)

        self.neural_rendering_resolution = generator_config['params']['neural_rendering_resolution']
        self.local_patch_dict = None
        self.patch_full_resolution = 128
        self.local_patch_dict_gt512 = None
        if self.is_local_dense_sampling is True:
            multi_ratio = self.patch_full_resolution//128
            multiplier_fn = lambda xs: [x*multi_ratio for x in xs]
            self.eye1_bbox, self.eye2_bbox, self.nose_bbox, self.mouth_bbox = [46, 50, 32, 32], [81, 50, 32, 32],\
                                                                            [64, 68, 32, 32], [64, 94, 48, 48]
            self.local_patch_dict = {'eye1': {'bbox': multiplier_fn(self.eye1_bbox), 'neural_rendering_resolution': 128*multi_ratio},
                                    'eye2': {'bbox': multiplier_fn(self.eye2_bbox), 'neural_rendering_resolution': 128*multi_ratio},
                                    'nose': {'bbox': multiplier_fn(self.nose_bbox), 'neural_rendering_resolution': 128*multi_ratio},
                                    'mouth': {'bbox': multiplier_fn(self.mouth_bbox), 'neural_rendering_resolution': 128*multi_ratio}}
            ##### the patch bboxes in 512**512
            multi_ratio512 = 512//128
            multiplier_fn_512 = lambda xs: [x*multi_ratio512 for x in xs]
            # self.eye1_bbox_512, self.eye2_bbox_512, self.nose_bbox_512, self.mouth_bbox_512 = list(map(multiplier_fn_512, 
            #                                 [self.eye1_bbox, self.eye2_bbox, self.nose_bbox, self.mouth_bbox]))
            self.local_patch_dict_gt512 = {'eye1': {'bbox': multiplier_fn_512(self.eye1_bbox)},
                                            'eye2': {'bbox': multiplier_fn_512(self.eye2_bbox)},
                                            'nose': {'bbox': multiplier_fn_512(self.nose_bbox)},
                                            'mouth': {'bbox': multiplier_fn_512(self.mouth_bbox)}}
    
    @property
    def device(self):
        return self.generator.device

    def forward(self,
                input,
                preprocess_sobel_conv,
                sketch_simplification,
                return_loss=False,
                is_train=True,
                inversion_mode='sketch',
                **kwargs):
        out = {}
        
        camera = input['camera']
        image = input['image']
        grad = input['grad']
        # import pdb; pdb.set_trace();

        # cur_nimg = kwargs['step'] * image.shape[0]
        # if kwargs['step'] % 2 == 0:
        #     assert self.D_reg_interval %2 == 0, 'D_reg_interval must divide by 2'
        #     if kwargs['step'] % self.D_reg_interval == 0:
        #         loss_dict = self._train_D_loss('Dboth', self.D_reg_interval, image, grad, camera, cur_nimg)
        #     else:
        #         loss_dict = self._train_D_loss('Dmain', 1, image, grad, camera, cur_nimg)
        # elif kwargs['step'] %2 == 1:
        #     loss_dict = self._train_G_loss(image, grad, camera)
        if inversion_mode == 'sketch':
            encoder_in = grad
        elif inversion_mode == 'image':
            encoder_in = image
        else:
            raise ValueError
        
        if self.discriminator is not None:
            if kwargs['step'] % 2 == 0:
                loss_dict = self._train_D_loss(image, encoder_in, camera)
            elif kwargs['step'] %2 == 1:
                loss_dict = self._train_G_loss(image, encoder_in, camera, preprocess_sobel_conv, sketch_simplification)
            else:
                pass
        else:
            loss_dict = self._train_G_loss(image, encoder_in, camera, preprocess_sobel_conv, sketch_simplification)

        loss = sum(loss_dict.values())

        if return_loss:
            out['loss'] = loss
            out.update(loss_dict)
        return out

    # discriminator does not function a lot in this task
    # def _train_D_loss(self, phase, gain, image, grad, camera, cur_nimg):
    #     pred_image_dict, _ = self.forward_pred_img(grad, camera)
    #     loss_dict_D = self.discloss(self.discriminator, phase,
    #                             image, camera,
    #                             pred_image_dict, camera,
    #                             gain, cur_nimg=cur_nimg)
    #     return loss_dict_D
    
    ### directly borrowed from https://github.com/sicxu/Deep3DFaceRecon_pytorch
    @staticmethod
    def perceptual_loss(id_featureA, id_featureB):
        cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
            # assert torch.sum((cosine_d > 1).float()) == 0
        return torch.sum(1 - cosine_d) / cosine_d.shape[0]

    def _train_D_loss(self, image, encoder_in, camera):
        pred_image_dict, updated_wp = self.forward_pred_img(encoder_in, camera)
        image_resize = F.interpolate(image, size=(pred_image_dict['image'].shape[-2], pred_image_dict['image'].shape[-1]),
                                    mode='bilinear')

        pred_part_dict, gt_part_dict = self.PartExtractor(pred_image_dict['image'], image_resize)

        loss_dict = {}
        for part_name in gt_part_dict.keys():
            pred_part = pred_part_dict[part_name]
            gt_part = gt_part_dict[part_name]

            if self.net_recog is not None or self.CLIPLoss is not None or self.GAN_loss is not None:
                pred_part_resize = F.interpolate(pred_part, size=(224,224), mode='bilinear')
                gt_part_resize = F.interpolate(gt_part, size=(224,224), mode='bilinear')

            if self.GAN_loss is not None and part_name == 'full':
                fake_pred = self.discriminator(pred_part_resize)
                real_pred = self.discriminator(gt_part_resize)
                gan_loss = self.GAN_loss(fake_pred, target_is_real=False, for_discriminator=True) + \
                        self.GAN_loss(real_pred, target_is_real=True, for_discriminator=True)
                loss_dict['gan_D_loss'] = gan_loss
        return loss_dict

    @staticmethod
    def forward_gt_img(image, local_patch_dict_gt512):
        gt_image_dict = {}
        # gt_image_resize = F.interpolate(image, size=(self.patch_full_resolution,self.patch_full_resolution), mode='bilinear')
        for k,v in local_patch_dict_gt512.items():
            gt_image_dict[k+'_image'] = image[:, :, v['bbox'][1]-v['bbox'][3]//2:v['bbox'][1]+v['bbox'][3]//2,
                                                    v['bbox'][0]-v['bbox'][2]//2:v['bbox'][0]+v['bbox'][2]//2]
        gt_image_dict['image'] = image
        return gt_image_dict

    def _train_G_loss(self, image, encoder_in, camera, preprocess_sobel_conv, sketch_simplification):
        # import pdb; pdb.set_trace()
        pred_image_dict, updated_wp = self.forward_pred_img(encoder_in, camera, local_patch_dict=self.local_patch_dict, use_cached_backbone=False, cache_backbone=True)
        gt_image_dict = self.forward_gt_img(image, local_patch_dict_gt512=self.local_patch_dict_gt512) if self.local_patch_dict_gt512 is not None else None
        
        # import torchvision
        # import pdb; pdb.set_trace();

        # pred_image_dict['pred_edge'] = self.image2grad(pred_image_dict['image'], preprocess_sobel_conv, sketch_simplification)
        # image_resize = F.interpolate(image, size=(pred_image_dict['image'].shape[-2], pred_image_dict['image'].shape[-1]), mode='bilinear')

        loss_dict = {}
        ### triplet loss will force network add artfacts to cheat instead of enrich diversity 
        # if self.w_triplet_loss is True and self.w_anchor_w is True:
        #     avg = self.anchor_w
        #     loss_dict['triplet_loss'] = self.triplet_loss(updated_wp[:,0], avg - (updated_wp[:,0]-avg), avg)

        # pred_part_dict, gt_part_dict = self.PartExtractor(pred_image_dict['image'], image_resize)
        
        if self.invert_space == 'zcam':
            rotation = camera[:,:12].reshape(camera.shape[0],3,4)[:,:3,:3]
            trans = camera[:, :12].reshape(camera.shape[0],3,4)[:,:3,3]
            euler_angles = matrix_to_euler_angles(rotation, 'ZYX')
            gt_camera_angle_trans = torch.cat([euler_angles, trans], dim=1)
            loss_dict['loss_pose'] = nn.MSELoss()(pred_image_dict['pred_euler_angle_trans'], gt_camera_angle_trans) 
            if self.is_only_cam_pose is True:
                return loss_dict

        all_rgb_loss, all_lpips_loss, all_patch_contr_loss, all_edge_patch_contr_loss, all_edge_l1_loss, all_id_loss, all_clip_loss = 0,0,0,0,0,0,0
        # if self.is_dual_contrastive is True:
        #     pred_part_edge_dict, gt_part_edge_dict = self.PartExtractor(pred_image_dict['pred_edge'], encoder_in)
        patch_names = ['image']
        if self.local_patch_dict is not None:
            for k, v in self.local_patch_dict.items():
                patch_names.append(k+'_image')

        for part_name in patch_names:
            pred_part = pred_image_dict[part_name]
            gt_part = gt_image_dict[part_name]

            pred_part_resize = pred_part
            gt_part_resize = F.interpolate(gt_part, size=(pred_part.shape[-2],pred_part.shape[-1]), mode='bilinear')

            # import pdb; pdb.set_trace();
            all_rgb_loss += self.rgb_loss(pred_part_resize, gt_part_resize)
            all_lpips_loss += self.lpips_loss(pred_part_resize, gt_part_resize)

            if self.CLIPLoss is not None:
                clip_loss_dict = self.CLIPLoss((pred_part_resize+1)*0.5, (gt_part_resize+1)*0.5)
                all_clip_loss += sum(clip_loss_dict.values())
                # all_clip_loss += self.CLIPLoss((pred_part_resize+1)*0.5, (encoder_in))
                # import pdb; pdb.set_trace();

            # if part_name != 'image' and self.is_edge_loss is not None:
            #     edge_pred_part_resize = self.sobel_conv.binary_image2sketch(pred_part_resize)
            #     edge_gt_part_resize = self.sobel_conv.binary_image2sketch(gt_part_resize)
                # import torchvision
                # import pdb; pdb.set_trace();

            # if part_name == 'image':
            #     if self.net_recog is not None or self.CLIPLoss is not None or self.GAN_loss is not None:
            #         # pred_part_resize = F.interpolate(pred_part, size=(224,224), mode='bilinear')
            #         pred_part_resize = pred_part
            #         gt_part_resize = F.interpolate(gt_part, size=(pred_part.shape[-2],pred_part.shape[-1]), mode='bilinear')

            # if self.net_recog is not None and part_name == 'image':
            #     loss_dict['id_loss'] = self.perceptual_loss(self.net_recog((pred_part_resize+1.)*0.5), self.net_recog((gt_part_resize+1.)*0.5))
                # import torchvision
                # torchvision.utils.save_image(torch.cat([(pred_part_resize+1.)*0.5, (gt_part_resize+1.)*0.5], dim=-2), 'cat.jpg')
                # print(loss_dict['id_loss'].item())
                # import pdb; pdb.set_trace();

            # if self.CLIPLoss is not None and part_name == 'image':
            #     clip_loss = self.CLIPLoss((pred_part_resize+1)*0.5, (gt_part_resize+1)*0.5)
                # import torchvision
                # torchvision.utils.save_image(torch.cat([(pred_part_resize+1.)*0.5, (gt_part_resize+1.)*0.5], dim=-2), 'cat.jpg')
                # import pdb; pdb.set_trace();
                # print(clip_loss.keys())
                # loss_dict['clip_loss'] = sum(clip_loss.values())

            if self.GAN_loss is not None and part_name == 'image':
                fake_pred = self.discriminator(pred_part_resize)
                gan_loss = self.GAN_loss(fake_pred, target_is_real=True, for_discriminator=False)
                loss_dict['gan_G_loss'] = gan_loss

            # import pdb; pdb.set_trace();

            ###### very noisy, temporally without it
            # if self.is_edge_l1 is True: 
            #     pred_part_edge = pred_part_edge_dict[part_name]
            #     gt_part_edge = gt_part_edge_dict[part_name]
            #     all_edge_l1_loss += self.rgb_loss(pred_part_edge, gt_part_edge)
            
            # all_patch_contr_loss += self.img_patch_contr_loss(pred_part, gt_part)

            # if self.is_dual_contrastive is True:
            #     pred_part_edge = pred_part_edge_dict[part_name]
            #     gt_part_edge = gt_part_edge_dict[part_name]
            #     all_edge_patch_contr_loss += self.img_patch_contr_loss(pred_part_edge, gt_part_edge)

        loss_dict['rgb_loss'] = all_rgb_loss
        loss_dict['lpips_loss'] = all_lpips_loss

        if self.CLIPLoss is not None:
            loss_dict['clippsso_loss'] = all_clip_loss

        ###### very noisy, temporally without it
        if self.is_edge_l1 is True:
            loss_dict['edge_l1loss'] = all_edge_l1_loss

        # loss_dict['patch_contr_loss'] = all_patch_contr_loss
        # if self.is_dual_contrastive is True:
        #     loss_dict['edge_patch_contr_loss'] = all_edge_patch_contr_loss

        # rgb_loss_dict = self.rgb_loss(pred_image_dict['image'], image_resize)
        # loss_dict.update(rgb_loss_dict)

        # lpips_loss_dict = self.lpips_loss(pred_image_dict['image'], image_resize)
        # loss_dict.update(lpips_loss_dict)

        # patch_contr_loss_dict = self.img_patch_contr_loss(pred_image_dict['image'], image_resize)
        # loss_dict.update(patch_contr_loss_dict)

        # if self.is_dual_contrastive is True:
        #     sketch_patch_contr_loss_dict = self.img_patch_contr_loss(pred_image_dict['pred_edge'], encoder_in)
        #     loss_dict['edge_patch_contr_loss'] = sketch_patch_contr_loss_dict['patch_contr_loss']

        # wnorm_loss_dict = self.wnorm_loss(delta_wp)
        # loss_dict.update(wnorm_loss_dict)
        
        # vgg_loss_dict = self.vgg_loss(pred_image_dict['image'], image)
        # loss_dict.update(vgg_loss_dict)

        # if self.w_disc is True:
        #     disc_loss_dict = self.disc_loss(pred_image_dict, camera)
        #     loss_dict.update(disc_loss_dict)
        # sketch_cliploss_dict = self.sketch_clip_loss(pred_image_dict['pred_edge'], grad_resize)
        # loss_dict.update(sketch_cliploss_dict)
        
        return loss_dict

    @staticmethod
    def mix_geo_tex(geo_code, tex_code, mode='pure_geo'):
        assert mode in ['pure_geo', 'pure_tex', 'wo_mix']
        batch_size = geo_code.shape[0]
        if mode == 'pure_geo':
            geo_code = geo_code[0:1].repeat(batch_size,1,1)
        elif mode == 'pure_tex':
            tex_code = tex_code[0:1].repeat(batch_size,1,1)
        elif mode == 'wo_mix':
            pass
        else:
            raise ValueError
        return geo_code, tex_code

    def forward_pred_img(self, grad, camera, plane_modes=None, local_patch_dict=None, use_cached_backbone=False, cache_backbone=False):
        updated_wp = None
        if use_cached_backbone is False:
            # import pdb; pdb.set_trace();
            if self.varitex_encoder is not None:
                tex_dim = self.varitex_encoder.tex_dim
                tex_noise = torch.randn(size=(grad.shape[0], 1, tex_dim)).to(grad)
            if self.invert_space == 'z': ### do not use this one
                encoded_input = self.encoder(grad)
                if self.varitex_encoder is not None:
                    if self.mix_geo_tex_mode is not None and self.is_train is False:
                        encoded_input, tex_noise = self.mix_geo_tex(encoded_input, tex_noise, mode=self.mix_geo_tex_mode)
                    encoded_input = self.varitex_encoder(encoded_input, tex_noise)
                updated_wp = self.map_latent_code(encoded_input, camera)
            elif self.invert_space == 'zcam':
                ###### In this setting, encoder predicts z+cam
                # rotation = camera[:,:12].reshape(camera.shape[0],3,4)[:,:3,:3]
                # euler_angle = matrix_to_euler_angles(rotation, 'ZYX')
                # cycle_rotation = euler_angles_to_matrix(euler_angle, 'ZYX')
                # trans = camera[:, :12].reshape(camera.shape[0],3,4)[:,:3,3]
                # cycle_cam_params = angle_translation_to_camera(torch.cat([euler_angle,trans],dim=1))
                # import pdb; pdb.set_trace();

                encoded_input, encoded_angle_translation = self.encoder(grad)
                if self.is_only_cam_pose is False:
                    encoded_cam = angle_translation_to_camera(encoded_angle_translation)
                    updated_wp = self.map_latent_code(encoded_input, camera=encoded_cam, space_mode='z')
            elif self.invert_space == 'w':
                encoded_input = self.encoder(grad)
                if self.varitex_encoder is not None:
                    if self.mix_geo_tex_mode is not None and self.is_train is False:
                        encoded_input, tex_noise = self.mix_geo_tex(encoded_input, tex_noise, mode=self.mix_geo_tex_mode)
                    encoded_input = self.varitex_encoder(encoded_input, tex_noise)
                # updated_wp = encoded_input.repeat(1,14,1)
                updated_wp = self.map_latent_code(encoded_input.squeeze(1), camera=None, space_mode='w')
            elif self.invert_space == 'wp':
                encoded_input = self.encoder(grad)
                updated_wp = self.map_latent_code(encoded_input, camera=None, space_mode='wp')
            else:
                raise ValueError
        else:
            updated_wp = None
        # import pdb; pdb.set_trace();

        pred_image_dict = {}
        if self.is_only_cam_pose is False:
            pred_image_dict = self.synthesize_image_dict(updated_wp, camera, neural_rendering_resolution=self.neural_rendering_resolution, plane_modes=plane_modes, 
                                                                use_cached_backbone=use_cached_backbone, cache_backbone=cache_backbone)
            if local_patch_dict is not None:
                for k,v in local_patch_dict.items():
                    local_patch_i_dict = self.synthesize_image_dict(updated_wp, camera, plane_modes=plane_modes, 
                                                                use_cached_backbone=True, cache_backbone=True, **v)
                    for ki, vi in local_patch_i_dict.items():
                        pred_image_dict['{}_{}'.format(k, ki)] = vi
                # import pdb; pdb.set_trace();
            # pred_image_dict['pred_edge'] = kornia.filters.sobel(pred_image_dict['image'])
            # pred_image_dict['pred_edge'] = torch.norm(pred_image_dict['pred_edge'], dim=1,keepdim=True)
            # pred_image_dict['pred_edge'] = pred_image_dict['pred_edge'].repeat(1,3,1,1)
            # pred_image_dict['pred_edge'] = self.sobel_conv(pred_image_dict['image'])
            # import pdb; pdb.set_trace();
        
        if self.invert_space == 'zcam':
            pred_image_dict['pred_euler_angle_trans'] = encoded_angle_translation
        
        return pred_image_dict, updated_wp

    def img_patch_contr_loss(self, pred_img, gt_img, v_border=50, h_border=80):
        patch_contr_loss_dict = {}
        patch_contr_loss = self.PatchContrastiveLoss(pred_img, gt_img)
        # patch_contr_loss_dict['patch_contr_loss'] = patch_contr_loss
        # return patch_contr_loss_dict
        return patch_contr_loss

    @staticmethod
    def rgb2l(pred_img, gt_img):
        
        lab_pred_img = kornia.color.lab.rgb_to_lab(pred_img*0.5+0.5)
        lab_gt_img = kornia.color.lab.rgb_to_lab(gt_img*0.5+0.5)

        l_pred_img = lab_pred_img[:, :1].repeat(1,3,1,1)
        l_gt_img = lab_gt_img[:, :1].repeat(1,3,1,1)

        l_pred_img, l_gt_img = l_pred_img/100*2-1, l_gt_img/100*2-1
        # import pdb; pdb.set_trace();
        
        return l_pred_img, l_gt_img

    def lpips_loss(self, pred_img, gt_img, v_border=50, h_border=80):
        lpips_loss_dict = {}
        # pred_img, gt_img = pred_img[:,:,h_border:-h_border,v_border:-v_border], gt_img[:,:,h_border:-h_border,v_border:-v_border]
        if self.is_lab_loss is False:
            lpips_loss = self.LPIPSLoss(pred_img, gt_img)
        else:
            l_pred_img, l_gt_img = self.rgb2l(pred_img, gt_img)
            lpips_loss = self.LPIPSLoss(l_pred_img, l_gt_img)
        # lpips_loss_dict['lpips_loss'] = lpips_loss * 0.8
        # return lpips_loss_dict
        return lpips_loss

    def rgb_loss(self, pred_img, gt_img, v_border=50, h_border=80):
        rgb_loss_dict = {}
        # pred_img, gt_img = pred_img[:,:,h_border:-h_border,v_border:-v_border], gt_img[:,:,h_border:-h_border,v_border:-v_border]
        if self.is_lab_loss is False:
            l1_loss = nn.L1Loss()(pred_img, gt_img)
        else:
            l_pred_img, l_gt_img = self.rgb2l(pred_img, gt_img)
            l1_loss = nn.L1Loss()(l_pred_img, l_gt_img)
        # rgb_loss_dict['l1_loss'] = 0.1 * l1_loss
        # mse_loss = F.mse_loss(pred_img, gt_img)
        # rgb_loss_dict['mse_loss'] = mse_loss
        # return rgb_loss_dict
        return l1_loss

    @staticmethod
    def cosine_distance(a,b):
        a_norm = torch.nn.functional.normalize(a)
        b_norm = torch.nn.functional.normalize(b)
        distance = torch.sum(a_norm*b_norm, dim=1)
        return distance

    ### triplet loss is applied for diversity only added when delta setting, urge latents divergent from average codes
    def triplet_loss(self, pos, neg, avg):
        alpha = 1
        triplet_distance = self.cosine_distance(neg, avg)-self.cosine_distance(pos,avg)+alpha
        loss = torch.maximum(triplet_distance, torch.zeros_like(triplet_distance))
        loss = torch.mean(loss)
        # import pdb; pdb.set_trace();
        return loss

    # def part_loss(self, pred_img, gt_img):
    #     part_loss_dict = {}
    #     part_loss = self.PartLoss(pred_img, gt_img)
    #     part_loss_dict['part_loss'] = part_loss
    #     return part_loss_dict

    # def wnorm_loss(self, z_or_w):
    #     wnorm_loss_dict = {}
    #     wnorm_loss = self.WNORMLoss(z_or_w)
    #     wnorm_loss_dict['wnorm_loss'] = wnorm_loss * 0.005
    #     return wnorm_loss_dict

    # def sketch_clip_loss(self, sketch, target):
    #     sketch_cliploss_dict = {}
    #     sketch = F.interpolate(sketch,size=(224,224), mode='area')
    #     target = F.interpolate(target,size=(224,224), mode='area')
    #     res_cliploss_dict = self.Sketch_CLIPLoss(sketch, target)
    #     sketch_cliploss = sum(res_cliploss_dict.values())
    #     sketch_cliploss_dict['sketch_clip_loss'] = sketch_cliploss
    #     return sketch_cliploss_dict

    # def disc_loss(self, image_dict, conditional_params):
    #     disc_loss_dict = {}
    #     generated_pred = self.discriminator(image_dict, conditional_params)
    #     disc_loss = F.softplus(-generated_pred).mean()
    #     disc_loss_dict['disc_loss'] = disc_loss
    #     return disc_loss_dict

    def vgg_loss(self, pred_image, gt_img):
        vgg_loss_dict = {}
        vgg_loss = self.vggloss(pred_image, gt_img)
        # vgg_loss_dict['vgg_loss'] = vgg_loss
        # return vgg_loss_dict
        return vgg_loss

    def image2grad(self, image, preprocess_sobel_conv, sketch_simplification):
        # import pdb; pdb.set_trace();
        image_big = F.interpolate(image, size=(224,224), mode='bicubic')
        # image_big = image
        # image_blur = torchvision.transforms.GaussianBlur(kernel_size=5)(image_big)
        sobel_ts = preprocess_sobel_conv((image_big+1.0)*0.5)
        simplified_sketch = sketch_simplification(sobel_ts)
        # import pdb; pdb.set_trace()
        # simplified_sketch = F.interpolate(simplified_sketch, size=(224,224), mode='bicubic')
        pred_edge = prepare_simplified_sketch(simplified_sketch)
        return pred_edge.repeat(1,3,1,1)

    @torch.no_grad()
    def sample_3d(self,
                input,
                preprocess_sobel_conv,
                sketch_simplification,
                mp4_save_path,
                plane_modes=None,
                **kwargs):
        camera = input['camera']
        grad = input['grad']
        
        pred_image_dict, updated_wp = self.forward_pred_img(grad, camera, plane_modes=plane_modes, 
                        local_patch_dict=None, use_cached_backbone=False, cache_backbone=False)

        self.novel_view_synthesis(updated_wp, mp4_save_path)

        
    @torch.no_grad()
    def novel_view_synthesis(self, latent_code, mp4_save_path, angle_ys=None):
        if angle_ys is None: angle_ys = list(np.linspace(-0.3,0.3,60))
        img_nps = []
        for angle_y in angle_ys:
            _, camera_params = self.set_cam_meta(angle_y=angle_y, angle_p=-0.2)
            synthesized_image_dict = self.synthesize_image_dict(latent_code, camera_params)
            synthesized_image = synthesized_image_dict['image']
            synthesized_image = synthesized_image.permute(0,2,3,1)
            synthesized_image = torch.clamp(synthesized_image, -1, 1)
            syn_img_np = (synthesized_image.cpu().numpy()*0.5+0.5)*255
            syn_img_np = syn_img_np.astype(np.uint8)[0]
            img_nps.append(cv2.cvtColor(syn_img_np, cv2.COLOR_BGR2RGB))

        write2video(mp4_save_path, img_nps)


    @torch.no_grad()
    def sample(self,
               input,
               preprocess_sobel_conv,
               sketch_simplification,
               plane_modes=None,
               **kwargs):
        camera = input['camera']
        grad = input['grad']
        
        if camera.shape[1] == 0: camera = None
        # import pdb; pdb.set_trace()
        # frontal_camera = self.frontal_camera_params.expand(grad.shape[0], *self.frontal_camera_params.shape[1:]).to(grad)
        # camera = frontal_camera

        pred_image_dict, updated_wp = self.forward_pred_img(grad, camera, plane_modes=plane_modes, 
                        local_patch_dict=None, use_cached_backbone=False, cache_backbone=False)
        # pred_image_dict['pred_edge'] = self.image2grad(pred_image_dict['image'], preprocess_sobel_conv, sketch_simplification)

        out = {}
        if self.is_only_cam_pose is True: return out
        if self.is_train is False:
            left_camera = self.left_camera_params.expand(grad.shape[0],*self.left_camera_params.shape[1:]).to(grad)
            right_camera = self.right_camera_params.expand(grad.shape[0],*self.right_camera_params.shape[1:]).to(grad)
            
            pred_image_left_dict = self.synthesize_image_dict(updated_wp, left_camera, plane_modes=plane_modes)
            pred_image_right_dict = self.synthesize_image_dict(updated_wp, right_camera, plane_modes=plane_modes)
            pred_image_frontal_dict = self.synthesize_image_dict(updated_wp, None, plane_modes=plane_modes)

            out['pred_image_left'] = pred_image_left_dict['image']
            out['pred_image_right'] = pred_image_right_dict['image']
            out['pred_image_frontal'] = pred_image_frontal_dict['image']

        out['pred_image'] = pred_image_dict['image']
        out['image_raw'] = pred_image_dict['image_raw']
        out['input'] = grad
        if 'pred_edge' in pred_image_dict:
            out['pred_edge'] = pred_image_dict['pred_edge']
        return out
