import importlib
import random
import numpy as np
import torch
import warnings
import os
import torch.nn.functional as F
import torch



def copy_state_dict(state_dict, model, strip=None, replace=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and replace is None and name.startswith(strip):
            name = name[len(strip):]
        if strip is not None and replace is not None:
            name = name.replace(strip, replace)
        if name not in tgt_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)
        
class ImageMasker():
    def __init__(self, size=256, patch=16, offset=2, config=None):
        self.mask = np.ones(shape=(size, size))
        self.mask[128+patch*offset:, patch:-patch] = 0
        self.mask = torch.from_numpy(self.mask).float()
        self.config = config

    def __call__(self, image, ldmk_parsing_mask=None):
        """[summary]
        Args:
            image ([type]): image (0-1) mask (0.5)
        """
        if self.config is None:
            masked_image = (image*2-1) * self.mask.to(image.device)
            # masked_image = 0.5*(masked_image+1.0)
            return masked_image.float()
        else:
            mask_mode = self.config['params']['mask_mode']
            assert mask_mode == 'ldmk_parsing', 'mask mode {} must be lmdk_parsing'.format(mask_mode)
            # import torchvision
            # import pdb; pdb.set_trace();
            ldmk_parsing_mask[ldmk_parsing_mask>127] = 255
            ldmk_parsing_mask[ldmk_parsing_mask<=127] = 0
            ldmk_parsing_mask = 1 - ldmk_parsing_mask/255
            masked_image = (image*2-1) * ldmk_parsing_mask
            return masked_image.float()

class RefMasker():
    def __init__(self, size=256, ref_mask_mode='pcavs', patch=16):
        self.mask = np.ones(shape=(size, size))
        # self.mask[:128+patch*offset, patch:-patch] = 0
        self.mask = torch.from_numpy(self.mask).float()
        self.height, self.width = size//patch, size//patch
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)
        self.ref_mask_mode = ref_mask_mode

    def random_mouth_ts(self, data):
        bs = data.shape[0]
        mouth_radius = 4
        mask = torch.zeros(size=(bs,1,self.height,self.width))
        mask_rect = torch.zeros(size=(bs,1,3,6))
        mask_rect = self.binomial.sample(mask_rect.size())
        mask[:,:,self.height//2+2:self.height//2+3+mouth_radius//2,
             self.width//2-mouth_radius//2-1:self.width//2+mouth_radius//2+1] = mask_rect
        mask = 1 - mask
        mask = torch.nn.functional.interpolate(mask, size=(256,256), mode='nearest').to(data)
        return mask

    def random_mouth_zhanwang_ts(self, data):
        bs = data.shape[0]
        mouth_radius = 4
        mask = torch.zeros(size=(bs,1,self.height,self.width))
        mask_rect = torch.zeros(size=(bs,1,3,6))
        mask_rect = self.binomial.sample(mask_rect.size())
        mask[:,:,self.height//2+1:self.height//2+2+mouth_radius//2,
             self.width//2-mouth_radius//2-1:self.width//2+mouth_radius//2+1] = mask_rect
        mask = 1 - mask
        mask = torch.nn.functional.interpolate(mask, size=(256,256), mode='nearest').to(data)
        return mask

    ## put normalize for vqpp here, bad code
    def __call__(self, image, ref_mask_mode=None):
        """[summary]
        Args:
            image ([type]): image (0-1) mask (0.5)
        """
        ref_mask_mode = ref_mask_mode if ref_mask_mode is not None else self.ref_mask_mode
        # import pdb; pdb.set_trace();
        if ref_mask_mode == 'pcavs':
            masked_image = image * self.random_mouth_ts(image)
        elif ref_mask_mode == 'zhanwang':
            masked_image = (image*2-1)
            masked_image = masked_image * self.random_mouth_zhanwang_ts(image)
            masked_image = (masked_image+1)*0.5
        else:
            masked_image = (image*2-1) * self.mask.to(image.device)
        return masked_image.float()


def mask_token2image(masked_pos, img_size=256):
    float_masked_pos = masked_pos.float()
    patch_size = int(np.sqrt(float_masked_pos.size(-1)))
    float_masked_pos = float_masked_pos.reshape(float_masked_pos.shape[0], 1, patch_size, patch_size)
    float_masked_pos = F.interpolate(float_masked_pos, size=(img_size,img_size),mode='bilinear')
    float_masked_pos = float_masked_pos.repeat(1,3,1,1)
    return float_masked_pos


def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def merge_opts_to_config(config, opts):
    def modify_dict(c, nl, v):
        if len(nl) == 1:
            c[nl[0]] = type(c[nl[0]])(v)
        else:
            # print(nl)
            c[nl[0]] = modify_dict(c[nl[0]], nl[1:], v)
        return c

    if opts is not None and len(opts) > 0:
        assert len(opts) % 2 == 0, "each opts should be given by the name and values! The length shall be even number!"
        for i in range(len(opts) // 2):
            name = opts[2*i]
            value = opts[2*i+1]
            config = modify_dict(config, name.split('.'), value)
    return config 

def modify_config_for_debug(config):
    config['dataloader']['num_workers'] = 0
    config['dataloader']['batch_size'] = 1
    return config



def get_model_parameters_info(model):
    # for mn, m in model.named_modules():
    parameters = {'overall': {'trainable': 0, 'non_trainable': 0, 'total': 0}}
    for child_name, child_module in model.named_children():
        parameters[child_name] = {'trainable': 0, 'non_trainable': 0}
        for pn, p in child_module.named_parameters():
            if p.requires_grad:
                parameters[child_name]['trainable'] += p.numel()
            else:
                parameters[child_name]['non_trainable'] += p.numel()
        parameters[child_name]['total'] = parameters[child_name]['trainable'] + parameters[child_name]['non_trainable']
        
        parameters['overall']['trainable'] += parameters[child_name]['trainable']
        parameters['overall']['non_trainable'] += parameters[child_name]['non_trainable']
        parameters['overall']['total'] += parameters[child_name]['total']
    
    # format the numbers
    def format_number(num):
        K = 2**10
        M = 2**20
        G = 2**30
        if num > G: # K
            uint = 'G'
            num = round(float(num)/G, 2)
        elif num > M:
            uint = 'M'
            num = round(float(num)/M, 2)
        elif num > K:
            uint = 'K'
            num = round(float(num)/K, 2)
        else:
            uint = ''
        
        return '{}{}'.format(num, uint)
    
    def format_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                format_dict(v)
            else:
                d[k] = format_number(v)
    
    format_dict(parameters)
    return parameters


def format_seconds(seconds):
    h = int(seconds // 3600)
    m = int(seconds // 60 - h * 60)
    s = int(seconds % 60)

    d = int(h // 24)
    h = h - d * 24

    if d == 0:
        if h == 0:
            if m == 0:
                ft = '{:02d}s'.format(s)
            else:
                ft = '{:02d}m:{:02d}s'.format(m, s)
        else:
           ft = '{:02d}h:{:02d}m:{:02d}s'.format(h, m, s)
 
    else:
        ft = '{:d}d:{:02d}h:{:02d}m:{:02d}s'.format(d, h, m, s)

    return ft

def instantiate_from_config(config):
    if config is None or config == 'None':
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))

def class_from_string(class_name):
    module, cls = class_name.rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls

def get_all_file(dir, end_with='.h5'):
    if isinstance(end_with, str):
        end_with = [end_with]
    filenames = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            for ew in end_with:
                if f.endswith(ew):
                    filenames.append(os.path.join(root, f))
                    break
    return filenames


def get_sub_dirs(dir, abs=True):
    sub_dirs = os.listdir(dir)
    if abs:
        sub_dirs = [os.path.join(dir, s) for s in sub_dirs]
    return sub_dirs


def get_model_buffer(model):
    state_dict = model.state_dict()
    buffers_ = {}
    params_ = {n: p for n, p in model.named_parameters()}

    for k in state_dict:
        if k not in params_:
            buffers_[k] = state_dict[k]
    return buffers_
