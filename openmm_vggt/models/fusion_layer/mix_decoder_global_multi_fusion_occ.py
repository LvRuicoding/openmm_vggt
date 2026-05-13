from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from mmengine.registry import MODELS

from openmm_vggt.heads.monoscene_occupancy_head import MonoSceneOccupancyHead
from openmm_vggt.heads.occupancy_head import OccupancyHead
from openmm_vggt.models.aggregator_multi_fusion import MultiFusionAggregator
from openmm_vggt.models.window_attn_fusion import ShiftWindowPatchVoxelCrossFusion

from .mix_decoder_global_window_attn_early_occ import mix_decoder_global_window_attn_early_occ


@MODELS.register_module()
class mix_decoder_global_multi_fusion_occ(mix_decoder_global_window_attn_early_occ):
    """Occupancy model with configurable multi-layer voxel-to-patch fusion."""

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
        occupancy_head=None,
        occupancy_view_indices: Sequence[int] | None = None,
        fusion_layers: Tuple[int, ...] = (0, 12, 24),
        fusion_methods: Mapping[int | str, str] | None = None,
        fusion_share_weights: bool = False,
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
            occupancy_head=None,
            occupancy_view_indices=occupancy_view_indices,
            fusion_window_size=fusion_window_size,
            fusion_shift_size=fusion_shift_size,
            fusion_window_stride=fusion_window_stride,
            fusion_num_heads=fusion_num_heads,
            fusion_mlp_ratio=fusion_mlp_ratio,
            fusion_attn_backend=fusion_attn_backend,
        )
        self.aggregator = MultiFusionAggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            fusion_layers=fusion_layers,
            fusion_methods=fusion_methods,
        )
        for unused_name in (
            "early_voxel_feature_proj",
            "early_voxel_geometry_proj",
            "early_window_fusion",
        ):
            if hasattr(self, unused_name):
                delattr(self, unused_name)

        self.fusion_layers = tuple(self.aggregator.fusion_layers)
        self.fusion_methods = dict(self.aggregator.fusion_methods)
        self.fusion_share_weights = bool(fusion_share_weights)

        unsupported = {
            method for method in self.fusion_methods.values() if method != "window_cross_attn"
        }
        if unsupported:
            raise ValueError(f"Unsupported fusion methods: {sorted(unsupported)}")

        if fusion_shift_size is None:
            fusion_shift_size = fusion_window_stride

        module_layers = (self.fusion_layers[:1] if self.fusion_share_weights else self.fusion_layers)
        self.multi_voxel_feature_proj = nn.ModuleDict()
        self.multi_voxel_geometry_proj = nn.ModuleDict()
        self.multi_window_fusion = nn.ModuleDict()
        for layer_id in module_layers:
            layer_key = str(layer_id)
            self.multi_voxel_feature_proj[layer_key] = nn.Linear(
                self.voxel_encoder.get_output_feature_dim(),
                embed_dim,
            )
            self.multi_voxel_geometry_proj[layer_key] = self._make_voxel_geometry_proj(embed_dim)
            self.multi_window_fusion[layer_key] = ShiftWindowPatchVoxelCrossFusion(
                embed_dim=embed_dim,
                window_size=tuple(fusion_window_size),
                shift_size=None if fusion_shift_size is None else tuple(fusion_shift_size),
                num_heads=fusion_num_heads,
                mlp_ratio=fusion_mlp_ratio,
                attn_backend=fusion_attn_backend,
            )

        self.occupancy_head = None
        if occupancy_head is not None:
            occupancy_head = dict(occupancy_head)
            occupancy_head_type = occupancy_head.pop("type", "OccupancyHead")
            occupancy_head_cls = {
                "OccupancyHead": OccupancyHead,
                "MonoSceneOccupancyHead": MonoSceneOccupancyHead,
            }.get(occupancy_head_type)
            if occupancy_head_cls is None:
                raise ValueError(f"Unsupported occupancy head type: {occupancy_head_type}")
            self.occupancy_head = occupancy_head_cls(
                token_dim=2 * embed_dim,
                patch_size=patch_size,
                **occupancy_head,
            )

    def _make_voxel_geometry_proj(self, embed_dim: int) -> nn.Module:
        from openmm_vggt.models._mix_decoder_global_base import VoxelPositionEncoder3D

        return VoxelPositionEncoder3D(
            embed_dim=embed_dim,
            point_cloud_range=self.fusion_point_cloud_range.cpu().tolist(),
        )

    def _module_key_for_layer(self, layer_id: int) -> str:
        return str(self.fusion_layers[0] if self.fusion_share_weights else int(layer_id))

    def _prepare_multi_fusion_base_inputs(self, others: dict, batch_size: int, frame_count: int, image_hw):
        return self._prepare_common_patch_fusion_inputs(
            others=others,
            batch_size=batch_size,
            frame_count=frame_count,
            image_hw=image_hw,
            project_voxel_features_fn=lambda voxel_features: voxel_features,
        )

    def _apply_multi_layer_patch_fusion(
        self,
        patch_tokens: torch.Tensor,
        fusion_inputs: dict,
        layer_id: int,
    ) -> torch.Tensor:
        if self.fusion_methods[int(layer_id)] != "window_cross_attn":
            raise ValueError(f"Unsupported fusion method at layer {layer_id}: {self.fusion_methods[int(layer_id)]}")

        patch_h = int(fusion_inputs["patch_h"])
        patch_w = int(fusion_inputs["patch_w"])
        if patch_h * patch_w != patch_tokens.shape[1]:
            raise ValueError(
                f"Patch count mismatch: expected {patch_h * patch_w} from patch grid "
                f"{patch_h}x{patch_w}, but got {patch_tokens.shape[1]}"
            )

        voxel_features = fusion_inputs["voxel_tokens"]
        if voxel_features.shape[0] == 0:
            return patch_tokens

        layer_key = self._module_key_for_layer(int(layer_id))
        voxel_tokens = self.multi_voxel_feature_proj[layer_key](voxel_features)
        voxel_xyz = fusion_inputs["coord_xyz"].to(voxel_tokens.dtype)
        voxel_tokens = voxel_tokens + self.multi_voxel_geometry_proj[layer_key](voxel_xyz)

        flat_seq_ids = fusion_inputs["flat_seq_ids"]
        patch_y = fusion_inputs["patch_y"]
        patch_x = fusion_inputs["patch_x"]
        window_fusion = self.multi_window_fusion[layer_key]

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

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, others=None):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        batch_size, sequence_length, _, image_h, image_w = images.size()
        frame_count = sequence_length // self.cam_num
        images_time_major = images

        images = images.reshape(batch_size, frame_count, self.cam_num, 3, image_h, image_w)
        images = images.permute(0, 2, 1, 3, 4, 5).contiguous().reshape(
            batch_size * self.cam_num,
            frame_count,
            3,
            image_h,
            image_w,
        )
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        patch_tokens = self.aggregator.extract_patch_tokens(images)
        fusion_inputs = self._prepare_multi_fusion_base_inputs(
            others,
            batch_size,
            frame_count,
            (image_h, image_w),
        )

        def patch_fusion_fn(tokens: torch.Tensor, layer_id: int) -> torch.Tensor:
            return self._apply_multi_layer_patch_fusion(tokens, fusion_inputs, layer_id)

        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images,
            patch_tokens=patch_tokens,
            patch_fusion_fn=patch_fusion_fn,
        )
        images = images.view(batch_size, sequence_length, 3, image_h, image_w)
        return self._forward_from_aggregator_outputs(
            aggregated_tokens_list=aggregated_tokens_list,
            patch_start_idx=patch_start_idx,
            images=images,
            images_time_major=images_time_major,
            batch_size=batch_size,
            sequence_length=sequence_length,
            frame_count=frame_count,
            image_h=image_h,
            image_w=image_w,
            query_points=query_points,
            others=others,
        )
