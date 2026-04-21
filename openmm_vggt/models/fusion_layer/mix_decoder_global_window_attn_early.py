from typing import Tuple

import torch
import torch.nn as nn
from mmengine.registry import MODELS

from ..aggregator_window_attn_early import EarlyFusionAggregator
from .._mix_decoder_global_base import VoxelPositionEncoder3D, _MixDecoderGlobalBase
from ..window_attn_fusion import ShiftWindowPatchVoxelCrossFusion


@MODELS.register_module()
class mix_decoder_global_window_attn_early(_MixDecoderGlobalBase):
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
        use_top_k=True,
        top_k_per_patch=1,
        fusion_window_size: Tuple[int, int] = (10, 10),
        fusion_shift_size: Tuple[int, int] | None = None,
        fusion_window_stride: Tuple[int, int] | None = None,
        fusion_num_heads: int = 16,
        fusion_mlp_ratio: float = 4.0,
        fusion_attn_backend: str = "sdpa",
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
            use_top_k=use_top_k,
            top_k_per_patch=top_k_per_patch,
        )
        self.aggregator = EarlyFusionAggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.early_voxel_feature_proj = nn.Linear(self.voxel_encoder.get_output_feature_dim(), embed_dim)
        self.early_voxel_geometry_proj = VoxelPositionEncoder3D(
            embed_dim=embed_dim,
            point_cloud_range=self.fusion_point_cloud_range.cpu().tolist(),
        )
        if fusion_shift_size is None:
            fusion_shift_size = fusion_window_stride

        self.early_window_fusion = ShiftWindowPatchVoxelCrossFusion(
            embed_dim=embed_dim,
            window_size=tuple(fusion_window_size),
            shift_size=None if fusion_shift_size is None else tuple(fusion_shift_size),
            num_heads=fusion_num_heads,
            mlp_ratio=fusion_mlp_ratio,
            attn_backend=fusion_attn_backend,
        )

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
        patch_h = int(fusion_inputs["patch_h"])
        patch_w = int(fusion_inputs["patch_w"])
        if patch_h * patch_w != patch_tokens.shape[1]:
            raise ValueError(
                f"Patch count mismatch: expected {patch_h * patch_w} from patch grid "
                f"{patch_h}x{patch_w}, but got {patch_tokens.shape[1]}"
            )

        voxel_tokens = fusion_inputs["voxel_tokens"]
        if voxel_tokens.shape[0] == 0:
            return patch_tokens

        voxel_xyz = fusion_inputs["coord_xyz"].to(voxel_tokens.dtype)
        voxel_tokens = voxel_tokens + self.early_voxel_geometry_proj(voxel_xyz)

        flat_seq_ids = fusion_inputs["flat_seq_ids"]
        patch_y = fusion_inputs["patch_y"]
        patch_x = fusion_inputs["patch_x"]
        fused_patch_tokens = patch_tokens.clone()
        for seq_idx in range(patch_tokens.shape[0]):
            seq_mask = flat_seq_ids == seq_idx
            if seq_mask.any():
                fused_patch_tokens[seq_idx] = self.early_window_fusion(
                    fused_patch_tokens[seq_idx],
                    voxel_tokens[seq_mask],
                    patch_y[seq_mask],
                    patch_x[seq_mask],
                    patch_h,
                    patch_w,
                )
        return fused_patch_tokens
