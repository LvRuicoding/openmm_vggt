import torch
import torch.nn as nn
from mmengine.registry import MODELS

from openmm_vggt.layers.attention import Attention

from ..aggregator_window_attn_early import EarlyFusionAggregator
from .._mix_decoder_global_base import _MixDecoderGlobalBase


class SerializerSelfAttentionResidual(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, qk_norm: bool = False, fused_attn: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(
            dim=embed_dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.attn(self.norm(tokens))


@MODELS.register_module()
class mix_decoder_global_serializer2d_early(_MixDecoderGlobalBase):
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
        fusion_num_heads: int = 16,
        fusion_qk_norm: bool = False,
        fusion_fused_attn: bool = True,
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
        self.serializer_grid_size_2d = float(serializer_grid_size_2d)
        self.aggregator = EarlyFusionAggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.early_voxel_feature_proj = nn.Linear(self.voxel_encoder.get_output_feature_dim(), embed_dim)
        self.serializer_fusion = SerializerSelfAttentionResidual(
            embed_dim=embed_dim,
            num_heads=fusion_num_heads,
            qk_norm=fusion_qk_norm,
            fused_attn=fusion_fused_attn,
        )

    def _enable_early_patch_fusion(self) -> bool:
        return True

    def _project_early_voxel_features(self, voxel_features: torch.Tensor) -> torch.Tensor:
        return self.early_voxel_feature_proj(voxel_features)

    def _build_patch_centers(
        self,
        patch_h: int,
        patch_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y_coords = torch.arange(patch_h, device=device, dtype=dtype) * self.patch_size + (self.patch_size / 2.0)
        x_coords = torch.arange(patch_w, device=device, dtype=dtype) * self.patch_size + (self.patch_size / 2.0)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
        return torch.stack([y_grid, x_grid], dim=-1).reshape(-1, 2)

    def _compute_snake_sort_keys(self, coords: torch.Tensor, token_kind: torch.Tensor) -> torch.Tensor:
        if coords.shape[0] == 0:
            return torch.zeros((0,), device=coords.device, dtype=torch.int64)

        grid_coords = torch.floor(coords / self.serializer_grid_size_2d).to(torch.int64)
        grid_coords = grid_coords - grid_coords.min(dim=0)[0]
        row_ids = grid_coords[:, 0]
        col_ids = grid_coords[:, 1]
        max_col = col_ids.max()
        snake_col_ids = torch.where(row_ids % 2 == 1, max_col - col_ids, col_ids)

        fine_coords = torch.floor(coords).to(torch.int64)
        fine_coords = fine_coords - fine_coords.min(dim=0)[0]
        fine_y = fine_coords[:, 0]
        fine_x = fine_coords[:, 1]

        col_span = int(max_col.item()) + 1
        fine_y_span = int(fine_y.max().item()) + 1
        fine_x_span = int(fine_x.max().item()) + 1
        kind_span = int(token_kind.max().item()) + 1 if token_kind.numel() > 0 else 1

        return ((((row_ids * col_span) + snake_col_ids) * fine_y_span + fine_y) * fine_x_span + fine_x) * kind_span + token_kind

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

        flat_seq_ids = fusion_inputs["flat_seq_ids"]
        coord_y = fusion_inputs["coord_y"]
        coord_x = fusion_inputs["coord_x"]
        patch_coords = self._build_patch_centers(
            patch_h,
            patch_w,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )

        fused_patch_tokens = patch_tokens.clone()
        for seq_idx in range(patch_tokens.shape[0]):
            seq_mask = flat_seq_ids == seq_idx
            if not seq_mask.any():
                continue

            seq_voxel_tokens = voxel_tokens[seq_mask]
            seq_voxel_coords = torch.stack([coord_y[seq_mask], coord_x[seq_mask]], dim=-1).to(patch_tokens.dtype)
            seq_patch_tokens = fused_patch_tokens[seq_idx]

            unified_tokens = torch.cat([seq_voxel_tokens, seq_patch_tokens], dim=0)
            unified_coords = torch.cat([seq_voxel_coords, patch_coords], dim=0)
            unified_kind = torch.cat(
                [
                    torch.zeros(seq_voxel_tokens.shape[0], device=patch_tokens.device, dtype=torch.int64),
                    torch.ones(seq_patch_tokens.shape[0], device=patch_tokens.device, dtype=torch.int64),
                ],
                dim=0,
            )

            sort_indices = torch.argsort(self._compute_snake_sort_keys(unified_coords, unified_kind), dim=0)
            inverse_indices = torch.empty_like(sort_indices)
            inverse_indices[sort_indices] = torch.arange(sort_indices.shape[0], device=sort_indices.device)

            serialized_tokens = unified_tokens[sort_indices].unsqueeze(0)
            serialized_tokens = self.serializer_fusion(serialized_tokens).squeeze(0)
            unified_tokens = serialized_tokens[inverse_indices]
            fused_patch_tokens[seq_idx] = unified_tokens[seq_voxel_tokens.shape[0]:]

        return fused_patch_tokens
