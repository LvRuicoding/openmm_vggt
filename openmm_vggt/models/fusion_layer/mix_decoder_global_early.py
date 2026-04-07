import torch
import torch.nn as nn
from mmengine.registry import MODELS

from ..aggregator_window_attn_early import EarlyFusionAggregator
from .._mix_decoder_global_base import SimplePatchFusion, _MixDecoderGlobalBase


@MODELS.register_module()
class mix_decoder_global_early(_MixDecoderGlobalBase):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        cam_num=2,
        voxel_size=(0.4, 0.4, 0.8),
        point_cloud_range=(0.0, -40.0, -3.0, 80.0, 40.0, 3.0),
        voxel_encoder_filters=(128, 128),
        serializer_grid_size_2d=14.0,
        use_z_buffer_projection=True,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_camera=enable_camera,
            enable_point=enable_point,
            enable_depth=enable_depth,
            enable_track=enable_track,
            cam_num=cam_num,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            voxel_encoder_filters=voxel_encoder_filters,
            serializer_grid_size_2d=serializer_grid_size_2d,
            use_z_buffer_projection=use_z_buffer_projection,
        )
        self.aggregator = EarlyFusionAggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.early_voxel_feature_proj = nn.Linear(self.voxel_encoder.get_output_feature_dim(), embed_dim)
        self.early_patch_fusion = SimplePatchFusion(embed_dim)

    def _enable_early_patch_fusion(self) -> bool:
        return True

    def _project_early_voxel_features(self, voxel_features: torch.Tensor) -> torch.Tensor:
        return self.early_voxel_feature_proj(voxel_features)

    def _apply_early_patch_fusion(
        self,
        patch_tokens: torch.Tensor,
        fusion_inputs: dict,
        image_hw,
    ) -> torch.Tensor:
        dense_patch_voxel_tokens = fusion_inputs["dense_patch_voxel_tokens"]
        if patch_tokens.shape != dense_patch_voxel_tokens.shape:
            raise ValueError(
                f"Early fusion shape mismatch: patch tokens {tuple(patch_tokens.shape)} "
                f"vs voxel features {tuple(dense_patch_voxel_tokens.shape)}"
            )
        return self.early_patch_fusion(patch_tokens, dense_patch_voxel_tokens)
