from image_synthesis.modeling.networks.third_party.eg3d.triplane import TriPlaneGenerator, CSFTStyleGAN2Backbone


## Other inherited csft blocks are located in networks_stylegan2.py of eg3d project.
## heavily borrowed from https://github.com/TencentARC/GFPGAN/blob/9c3f2d62cb4e63a7ba7ce68648dd1667b2b2ef44/gfpgan/archs/gfpganv1_clean_arch.py
class CSFTTriplaneGenerator(TriPlaneGenerator):
    def __init__(self,
            z_dim,                      # Input latent (Z) dimensionality.
            c_dim,                      # Conditioning label (C) dimensionality.
            w_dim,                      # Intermediate latent (W) dimensionality.
            img_resolution,             # Output resolution.
            img_channels,               # Number of output color channels.
            neural_rendering_resolution,
            sr_num_fp16_res     = 0,
            mapping_kwargs      = {},   # Arguments for MappingNetwork.
            rendering_kwargs    = {},
            sr_kwargs = {},
            with_superresolution = True,
            sft_half = False,
            **synthesis_kwargs,         # Arguments for SynthesisNetwork.
            ):
        super(CSFTTriplaneGenerator, self).__init__(
            z_dim=z_dim,                      # Input latent (Z) dimensionality.
            c_dim=c_dim,                      # Conditioning label (C) dimensionality.
            w_dim=w_dim,                      # Intermediate latent (W) dimensionality.
            img_resolution=img_resolution,             # Output resolution.
            img_channels=img_channels,               # Number of output color channels.
            neural_rendering_resolution=neural_rendering_resolution,
            sr_num_fp16_res     = sr_num_fp16_res,
            mapping_kwargs      = mapping_kwargs,   # Arguments for MappingNetwork.
            rendering_kwargs    = rendering_kwargs,
            sr_kwargs = sr_kwargs,
            with_superresolution = with_superresolution,
            **synthesis_kwargs,         # Arguments for SynthesisNetwork.
        )
        self.sft_half = sft_half
        self.backbone = CSFTStyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)

    def synthesis(self, conditions, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(conditions, ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        # import pdb; pdb.set_trace();
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        if self.with_superresolution is True:
            sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        else:
            sr_image = rgb_image
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'feature_image': feature_image}

    def sample(self, conditions, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(conditions, ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_geo(self, conditions, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        planes = self.backbone.synthesis(conditions, ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, conditions, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(conditions, ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, conditions, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(conditions, ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
