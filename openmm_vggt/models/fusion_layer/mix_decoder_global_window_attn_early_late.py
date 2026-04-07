from typing import Tuple

import torch
import torch.nn as nn
from mmengine.registry import MODELS

from ..aggregator_window_attn_early import EarlyFusionAggregator
from .._mix_decoder_global_base import _MixDecoderGlobalBase
from ..window_attn_fusion import ShiftWindowPatchVoxelCrossFusion


@MODELS.register_module()
class mix_decoder_global_window_attn_early_late(_MixDecoderGlobalBase):
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
            use_z_buffer_projection=use_z_buffer_projection,
        )
        self.aggregator = EarlyFusionAggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        if fusion_shift_size is None:
            fusion_shift_size = fusion_window_stride

        self.early_voxel_feature_proj = nn.Linear(self.voxel_encoder.get_output_feature_dim(), embed_dim)
        self.early_voxel_geometry_proj = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.early_window_fusion = ShiftWindowPatchVoxelCrossFusion(
            embed_dim=embed_dim,
            window_size=tuple(fusion_window_size),
            shift_size=None if fusion_shift_size is None else tuple(fusion_shift_size),
            num_heads=fusion_num_heads,
            mlp_ratio=fusion_mlp_ratio,
            attn_backend=fusion_attn_backend,
        )

        late_embed_dim = embed_dim * 2
        self.final_layer_voxel_feature_proj = nn.Linear(self.voxel_encoder.get_output_feature_dim(), late_embed_dim)
        self.final_layer_voxel_geometry_proj = nn.Sequential(
            nn.Linear(3, late_embed_dim),
            nn.GELU(),
            nn.Linear(late_embed_dim, late_embed_dim),
        )
        self.final_layer_window_fusion = ShiftWindowPatchVoxelCrossFusion(
            embed_dim=late_embed_dim,
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
        return self._apply_window_patch_fusion(
            patch_tokens=patch_tokens,
            fusion_inputs=fusion_inputs,
            image_hw=image_hw,
            voxel_geometry_proj=self.early_voxel_geometry_proj,
            window_fusion=self.early_window_fusion,
        )

    def _enable_final_layer_patch_fusion(self) -> bool:
        return True

    def _project_final_layer_voxel_features(self, voxel_features: torch.Tensor) -> torch.Tensor:
        return self.final_layer_voxel_feature_proj(voxel_features)

    def _apply_final_layer_patch_fusion(
        self,
        patch_tokens: torch.Tensor,
        fusion_inputs: dict,
        image_hw,
    ) -> torch.Tensor:
        return self._apply_window_patch_fusion(
            patch_tokens=patch_tokens,
            fusion_inputs=fusion_inputs,
            image_hw=image_hw,
            voxel_geometry_proj=self.final_layer_voxel_geometry_proj,
            window_fusion=self.final_layer_window_fusion,
        )

    def _apply_window_patch_fusion(
        self,
        patch_tokens: torch.Tensor,
        fusion_inputs: dict,
        image_hw,
        voxel_geometry_proj: nn.Module,
        window_fusion: ShiftWindowPatchVoxelCrossFusion,
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

        image_h, image_w = image_hw
        depth_scale = torch.clamp(self.fusion_point_cloud_range[3] - self.fusion_point_cloud_range[0], min=1.0)
        voxel_geometry = torch.stack(
            [
                fusion_inputs["coord_x"] / max(float(image_w), 1.0),
                fusion_inputs["coord_y"] / max(float(image_h), 1.0),
                fusion_inputs["depth"] / depth_scale.to(fusion_inputs["depth"].dtype),
            ],
            dim=-1,
        ).to(voxel_tokens.dtype)
        voxel_tokens = voxel_tokens + voxel_geometry_proj(voxel_geometry)

        flat_seq_ids = fusion_inputs["flat_seq_ids"]
        patch_y = fusion_inputs["patch_y"]
        patch_x = fusion_inputs["patch_x"]
        fused_patch_tokens = patch_tokens.clone()
        for seq_idx in range(patch_tokens.shape[0]):
            seq_mask = flat_seq_ids == seq_idx
            if seq_mask.any():
                fused_patch_tokens[seq_idx] = window_fusion(
                    fused_patch_tokens[seq_idx],
                    voxel_tokens[seq_mask],
                    patch_y[seq_mask],
                    patch_x[seq_mask],
                    patch_h,
                    patch_w,
                )
        return fused_patch_tokens
