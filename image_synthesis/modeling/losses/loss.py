import torch.nn as nn
from image_synthesis.modeling.losses.architecture import VGG19
from image_synthesis.modeling.losses.clip.cliploss import CLIPConvLoss
from image_synthesis.modeling.networks.third_party.eg3d.dual_discriminator import filtered_resizing
from image_synthesis.modeling.networks.third_party.eg3d.torch_utils.ops import conv2d_gradfix, upfirdn2d
from .contr.patchnce import *
import torch
import numpy as np
from easydict import EasyDict
import yaml
import torchvision
import torch.nn.functional as F


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class DiscLoss(nn.Module):
    def __init__(self, 
                r1_gamma=10, blur_init_sigma=0, blur_fade_kimg=0, augment_pipe=None, filter_mode='antialiased', dual_discrimination=True):
        super(DiscLoss, self).__init__()
        self.r1_gamma = r1_gamma
        self.blur_fade_kimg = blur_fade_kimg
        self.blur_init_sigma = blur_init_sigma
        self.augment_pipe = None
        self.filter_mode = filter_mode
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        self.blur_raw_target = True
        self.dual_discrimination = dual_discrimination

    def run_D(self, D, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = D(img, c, update_emas=update_emas)
        return logits
        
    
    def __call__(self, D, phase, real_img, real_c, gen_img, gen_c, gain, cur_nimg, neural_rendering_resolution=128):
        loss_dict = {}

        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                # gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                gen_logits = self.run_D(D, gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                # training_stats.report('Loss/scores/fake', gen_logits)
                # training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            # with torch.autograd.profiler.record_function('Dgen_backward'):
            #     loss_Dgen.mean().mul(gain).backward()
                loss_dict['loss_Dgen'] = loss_Dgen.mean() * gain

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(D, real_img_tmp, real_c, blur_sigma=blur_sigma)
                # training_stats.report('Loss/scores/real', real_logits)
                # training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    # training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                    loss_dict['loss_Dreal'] = loss_Dreal.mean() * gain

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    # training_stats.report('Loss/r1_penalty', r1_penalty)
                    # training_stats.report('Loss/D/reg', loss_Dr1)
                    loss_dict['loss_Dr1'] = loss_Dr1.mean() * gain

            # with torch.autograd.profiler.record_function(name + '_backward'):
            #     (loss_Dreal + loss_Dr1).mean().mul(gain).backward()
            
        return loss_dict


class Patch_Contrastive_Loss(nn.Module):
    # def __init__(self, encoder, from_rgb, netF, n_layers=[2,3,4,5], num_patches=128):
    def __init__(self, encoder, netF, n_layers=[2,3,4,5], num_patches=128):
        '''
            vgg_layer: cal loss on which layer
        '''
        super(Patch_Contrastive_Loss, self).__init__()
        self.encoder = encoder
        # self.from_rgb = from_rgb
        # self.netF = netF
        self.num_patches = num_patches
        self.n_layers = n_layers

        self.criterionNCE = []
        for i in range(len(n_layers)):
            self.criterionNCE.append(PatchNCELoss())
        # import pdb; pdb.set_trace()
        # self.netF = PatchSampleF(use_mlp=True)
        self.netF = netF

    def forward(self, x_fake, x_src):
        n_layers = self.n_layers
        # x_fake = self.from_rgb(x_fake)
        # x_src = self.from_rgb(x_src)

        feat_k, feat_q = [], []
        for i, block in enumerate(self.encoder):
            x_fake, x_src = block(x_fake), block(x_src)
            # print(x_fake.shape, x_src.shape)
            if i in n_layers:
                feat_k.append(x_fake)
                feat_q.append(x_src)

        feat_k_pool, sample_ids = self.netF(feat_k, self.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.num_patches, sample_ids)
        # import pdb;
        # pdb.set_trace();
        total_nce_loss = 0.0
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.criterionNCE):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()

        return total_nce_loss / len(n_layers)


class SketchCLIPLoss(CLIPConvLoss):
    def __init__(self, sketchclip_yaml_path):
        # super(SketchCLIPLoss, self).__init__()
        sketchclip_args = EasyDict(yaml.load(open(sketchclip_yaml_path,'r')))
        super().__init__(args=sketchclip_args)

    # def forward(self):
    #     return xx


class PARTExtractor():
    def __init__(self, img_size = 128, is_seperate_part=False):
        super().__init__()
        self.is_seperate_part = is_seperate_part
        part = {#'bg': [0, 0, 512],
                'eye1': [46, 50, 32],
                'eye2': [81, 50, 32],
                'nose': [64, 68, 32],
                'mouth': [64, 84, 46]}
        self.part = {}
        for k,v in part.items():
            self.part[k] = [int(vi*img_size*1.0/128) for vi in v]

        # self.l1_loss = nn.L1Loss()

    def __call__(self, pred, gt):
        loss_all = 0
        gt_part_dict, pred_part_dict = {}, {}
        
        gt_part_dict['full'] = gt
        pred_part_dict['full'] = pred
        
        if self.is_seperate_part is True:
            for part_i_name, part_i_rect in self.part.items():
                sz = part_i_rect[2]
                ll, lt = part_i_rect[0]-sz//2, part_i_rect[1]-sz//2

                gt_part_i = gt[:,:,lt:lt+sz, ll:ll+sz]
                pred_part_i = pred[:,:,lt:lt+sz, ll:ll+sz]

                # gt_pred_cat = torch.cat([gt_part_i, pred_part_i], dim=2)
                # torchvision.utils.save_image(gt_pred_cat*0.5+0.5, 'gt_pred_cat.jpg')
                # import pdb; pdb.set_trace();
                gt_part_dict[part_i_name] = gt_part_i
                pred_part_dict[part_i_name] = pred_part_i
            
        return pred_part_dict, gt_part_dict



class GANLoss(nn.Module):
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)
