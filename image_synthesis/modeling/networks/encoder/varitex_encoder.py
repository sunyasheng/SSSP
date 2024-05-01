import torch.nn as nn
import torch


class VariTexEncoder(nn.Module):
    def __init__(self, tex_dim, geo_dim, latent_dim=256, fused_dim=None):
        super().__init__()
        self.tex_dim = tex_dim
        fused_dim = geo_dim if fused_dim is None else tex_dim
        self.fuser = nn.Sequential(nn.Linear(tex_dim+geo_dim, latent_dim),
                                    nn.ReLU(), 
                                    nn.Linear(latent_dim, fused_dim))

    def forward(self, geo_code, tex_code):
        geo_tex_code = torch.cat([geo_code, tex_code], dim=2)
        mixed_geo_tex_code = self.fuser(geo_tex_code)
        return mixed_geo_tex_code
