import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from image_synthesis.utils.misc import instantiate_from_config

from image_synthesis.modeling.networks.third_party.taming.modules.diffusionmodules.model import Encoder, Decoder
from image_synthesis.modeling.networks.third_party.taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from image_synthesis.modeling.networks.third_party.taming.modules.vqvae.quantize import GumbelQuantize
from image_synthesis.modeling.networks.third_party.taming.modules.vqvae.quantize import EMAVectorQuantizer

from image_synthesis.modeling.networks.third_party.sketch_simplification.model_gan import immean, imstd
from image_synthesis.modeling.networks.third_party.sketch_simplification.model_gan import model as SktechSimplificationModel

from image_synthesis.utils.checkpoint_utils import copy_state_dict
from image_synthesis.utils.network_utils import freeze_network
from image_synthesis.utils.image_utils import SobelConv, prepare_simplified_sketch


class VQSketchModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 mask_key="weight_mask",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 use_orthgonal=False
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # import pdb; pdb.set_trace();
        self.loss = instantiate_from_config(lossconfig)
        # self.mask_key = mask_key if 'mask' in lossconfig['target'].lower() else None

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape, use_orthgonal=use_orthgonal)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        self.preprocess_sobel_conv = SobelConv()
        model_ckpt_path = 'OUTPUT/pretrained/model_gan.pth'
        self.sketch_simplification = self.prepare_sketch_simplication(model_ckpt_path)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    @staticmethod
    def prepare_sketch_simplication(model_ckpt_path):
        model = SktechSimplificationModel
        state_dict = torch.load(model_ckpt_path)
        copy_state_dict(state_dict, model)
        freeze_network(model)
        model = model.eval()
        return model

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    @torch.no_grad()
    def encode_code(self, x):
        quant, _, _ = self.encode(x)
        quant = self.post_quant_conv(quant)
        return quant

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # import pdb; pdb.set_trace();
        if len(x.shape) == 5:
            x = x.reshape((-1, *x.shape[2:]))
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        batch_image = F.interpolate(x.float(), size=(224,224), mode='bicubic')
        sobel_ts = self.preprocess_sobel_conv(batch_image/255.)
        simplified_sketch = self.sketch_simplification(sobel_ts)
        # import pdb; pdb.set_trace()
        sketch = prepare_simplified_sketch(simplified_sketch)
        
        return sketch

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)

        xrec, qloss = self(x)

        # if self.mask_key is not None:
        #     weight_mask = self.get_input(batch, self.mask_key)
        #     weight_mask = weight_mask[0:1].repeat(x.size(0),1,1,1)

        if optimizer_idx == 0:
            # autoencode
            # if self.mask_key is not None:
            #     aeloss, log_dict_ae = self.loss(qloss, weight_mask, x, xrec, optimizer_idx, self.global_step,
            #                                     last_layer=self.get_last_layer(), split="train")
            # else:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            # if self.mask_key is not None:
            #     discloss, log_dict_disc = self.loss(qloss, weight_mask, x, xrec, optimizer_idx, self.global_step,
            #                                     last_layer=self.get_last_layer(), split="train")
            # else:
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        
        # if self.mask_key is not None:
        #     weight_mask = self.get_input(batch, self.mask_key)
        #     weight_mask = weight_mask[0:1].repeat(x.size(0),1,1,1)
        
        xrec, qloss = self(x)
        # import torchvision
        # torchvision.utils.save_image(weight_mask, 'weight_mask.png')
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # if self.mask_key is not None:
        #     aeloss, log_dict_ae = self.loss(qloss, weight_mask, x, xrec, 0, self.global_step,
        #                                         last_layer=self.get_last_layer(), split="val")

        #     discloss, log_dict_disc = self.loss(qloss, weight_mask, x, xrec, 1, self.global_step,
        #                                         last_layer=self.get_last_layer(), split="val")
        # else:
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log_dict(log_dict_ae)
        # self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
