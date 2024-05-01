import torch
import torch.nn as nn
from omegaconf import OmegaConf
import sys

from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.modeling.networks.third_party.taming.models.vqgan import GumbelVQ, VQModel
from image_synthesis.modeling.networks.third_party.taming.models.cond_transformer import Net2NetTransformer
import os
import torchvision.transforms.functional as TF
import PIL
from image_synthesis.modeling.networks.encoder.base_codec import BaseCodec
from einops import rearrange
import math

class Encoder(nn.Module):
    def __init__(self, encoder, quant_conv, quantize):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.quantize = quantize

    @torch.no_grad()
    def forward(self, x):
        x = 2*x - 1
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, [_, _, indices] = self.quantize(h)
        return indices.view(x.shape[0], -1)

class Decoder(nn.Module):
    def __init__(self, decoder, post_quant_conv, quantize, w=16, h=16):
        super().__init__()
        self.decoder = decoder
        self.post_quant_conv = post_quant_conv
        self.quantize = quantize
        self.w = w
        self.h = h

    @torch.no_grad()
    def forward(self, indices):
        z = self.quantize.get_codebook_entry(indices.view(-1), shape=(indices.shape[0], self.h, self.w, -1))
        quant = self.post_quant_conv(z)
        dec = self.decoder(quant)
        x = torch.clamp(dec, -1., 1.)
        x = (x + 1.)/2.
        return x

    @torch.no_grad()
    def get_quant_feat(self, indices):
        z = self.quantize.get_codebook_entry(indices.view(-1), shape=(indices.shape[0], self.h, self.w, -1))
        quant = self.post_quant_conv(z)
        return quant


class TamingVQVAE(BaseCodec):
    def __init__(
            self, 
            trainable=False,
            token_shape=[16,16],
            config_path='OUTPUT/pretrained_model/taming_dvae/vqgan_imagenet_f16_16384.yaml',
            ckpt_path='OUTPUT/pretrained_model/taming_dvae/vqgan_imagenet_f16_16384.pth',
            num_tokens=16384,
            quantize_number=974,
            mapping_path='./help_folder/statistics/taming_vqvae_974.pt',
        ):
        super().__init__()
        
        model = self.LoadModel(config_path, ckpt_path)

        self.enc = Encoder(model.encoder, model.quant_conv, model.quantize)
        self.dec = Decoder(model.decoder, model.post_quant_conv, model.quantize, token_shape[0], token_shape[1])

        self.num_tokens = num_tokens
        self.quantize_number = quantize_number
        if self.quantize_number != 0 and mapping_path!=None:
            self.full_to_quantize = torch.load(mapping_path)
            self.quantize_to_full = torch.zeros(self.quantize_number)-1
            for idx, i in enumerate(self.full_to_quantize):
                if self.quantize_to_full[i] == -1:
                    self.quantize_to_full[i] = idx
            self.quantize_to_full = self.quantize_to_full.long()
    
        self.trainable = trainable
        self.token_shape = token_shape
        self._set_trainable()

    def LoadModel(self, config_path, ckpt_path):
        config = OmegaConf.load(config_path)
        model = VQModel(**config.model.params)
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
        return model

    @property
    def device(self):
        # import pdb; pdb.set_trace()
        return self.enc.quant_conv.weight.device

    def preprocess(self, imgs):
        """
        imgs: B x C x H x W, in the range 0-255
        """
        imgs = imgs.div(255) # map to 0 - 1
        return imgs
        # return map_pixels(imgs)   
    
    def postprocess(self, imgs):
        """
        imgs: B x C x H x W, in the range 0-1
        """
        imgs = imgs * 255
        return imgs

    def get_tokens(self, imgs, **kwargs):
        imgs = self.preprocess(imgs)
        code = self.enc(imgs)
        if self.quantize_number != 0:
            code = self.full_to_quantize[code]
        output = {'token': code}
        # output = {'token': rearrange(code, 'b h w -> b (h w)')}
        return output

    def decode(self, img_seq):
        if self.quantize_number != 0:
            img_seq=self.quantize_to_full[img_seq].type_as(img_seq)
        b, n = img_seq.shape
        img_seq = rearrange(img_seq, 'b (h w) -> b h w', h = int(math.sqrt(n)))

        x_rec = self.dec(img_seq)
        x_rec = self.postprocess(x_rec)
        return x_rec

    @torch.no_grad()
    def token2decoder_input(self, img_seq):
        # z = self.quantize.get_codebook_entry(indices.view(-1), shape=(indices.shape[0], self.h, self.w, -1))
        if self.quantize_number != 0:
            img_seq=self.quantize_to_full[img_seq].type_as(img_seq)
        b, n = img_seq.shape
        indices = rearrange(img_seq, 'b (h w) -> b h w', h = int(math.sqrt(n)))
        z = self.dec.quantize.get_codebook_entry(indices.view(-1), 
                shape=(indices.shape[0], self.dec.h, self.dec.w, -1))
        quant = self.dec.post_quant_conv(z)
        return quant

    @torch.no_grad()
    def get_quant_feat(self, img_seq):
        if self.quantize_number != 0:
            img_seq=self.quantize_to_full[img_seq].type_as(img_seq)
        b, n = img_seq.shape
        img_seq = rearrange(img_seq, 'b (h w) -> b h w', h = int(math.sqrt(n)))
        img_quant = self.dec.get_quant_feat(img_seq)
        return img_quant
