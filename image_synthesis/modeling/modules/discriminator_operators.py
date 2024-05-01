from image_synthesis.modeling.networks.third_party.spade.models.networks.discriminator import MultiscaleDiscriminator
import argparse
from easydict import EasyDict as edict


def define_discriminator(discriminator_config):
    if discriminator_config['type'] == 'multi_scale':
        # parser = argparse.ArgumentParser(description='Dis')
        # parser = MultiscaleDiscriminator.modify_commandline_options(parser, is_train=True)
        # opt = parser.parse_args()
        opt = edict({'netD_subarch': 'n_layer',
                     'num_D': 2,
                     'n_layers_D': 4,
                     'no_ganFeat_loss': False,
                     'ndf': 64,
                     'label_nc': 0,
                     'output_nc': 3,
                     'contain_dontcare_label': False,
                     'no_instance': True,
                     'norm_D': 'spectralinstance'})
        mD = MultiscaleDiscriminator(opt)
    else:
        raise ValueError
    return mD
