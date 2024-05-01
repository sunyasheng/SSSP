import torch.nn.functional as F
import torch
from .arcface_torch.backbones import get_model
from kornia.geometry import warp_affine
import torch.nn as nn


def resize_n_crop(image, M=None, dsize=112):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    if M is None:
        rot = torch.eye(2).to(image)
        trans = torch.zeros(size=(2,1)).to(rot)
        M = torch.cat([rot, trans],dim=1).unsqueeze(0).repeat(image.shape[0],1,1)
    return warp_affine(image, M, dsize=(dsize, dsize))


def define_net_recog(net_recog, pretrained_path=None):
    net = RecogNetWrapper(net_recog=net_recog, pretrained_path=pretrained_path)
    net.eval()
    return net

class RecogNetWrapper(nn.Module):
    def __init__(self, net_recog, pretrained_path=None, input_size=112):
        super(RecogNetWrapper, self).__init__()
        net = get_model(name=net_recog, fp16=False)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            net.load_state_dict(state_dict)
            print("loading pretrained net_recog %s from %s" %(net_recog, pretrained_path))
        for param in net.parameters():
            param.requires_grad = False
        self.net = net
        self.preprocess = lambda x: 2 * x - 1
        self.input_size=input_size
        
    def forward(self, image, M=None):
        image = self.preprocess(resize_n_crop(image, M, self.input_size))
        # import pdb; pdb.set_trace()
        id_feature = F.normalize(self.net(image), dim=-1, p=2)
        return id_feature

