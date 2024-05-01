import torch.nn as nn
import torch
from image_synthesis.utils.image_utils import SobelConv, prepare_simplified_sketch
from image_synthesis.modeling.networks.third_party.sketch_simplification.model_gan import model as SktechSimplificationModel
from image_synthesis.utils.checkpoint_utils import copy_state_dict
from image_synthesis.utils.network_utils import freeze_network


class Image2SimplifiedSketch(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.preprocess_sobel_conv = SobelConv()
        model_ckpt_path = 'OUTPUT/pretrained/model_gan.pth'
        self.sketch_simplification = self.prepare_sketch_simplication_model(model_ckpt_path)

    @staticmethod
    def prepare_sketch_simplication_model(model_ckpt_path):
        model = SktechSimplificationModel
        state_dict = torch.load(model_ckpt_path)
        copy_state_dict(state_dict, model)
        freeze_network(model)
        model = model.eval()
        return model

    def forward(self, img_ts):
        sobel_ts = self.preprocess_sobel_conv(img_ts)
        simplified_sketch = self.sketch_simplification(sobel_ts)
        binarization_sketch = prepare_simplified_sketch(simplified_sketch)
        return binarization_sketch
