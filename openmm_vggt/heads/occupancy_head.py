from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OccupancyHead(nn.Module):
    def __init__(
        self,
        token_dim: int,
        patch_size: int,
        voxel_size: Tuple[float, float, float],
        point_cloud_range: Tuple[float, float, float, float, float, float],
        num_classes: int = 20,
        hidden_dim: int = 16,
        depth_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.num_classes = int(num_classes)
        self.depth_scale = float(depth_scale)

        voxel_size_tensor = torch.tensor(voxel_size, dtype=torch.float32)
        point_cloud_range_tensor = torch.tensor(point_cloud_range, dtype=torch.float32)
        grid_size = ((point_cloud_range_tensor[3:] - point_cloud_range_tensor[:3]) / voxel_size_tensor).round().to(torch.long)

        self.register_buffer("voxel_size", voxel_size_tensor, persistent=False)
        self.register_buffer("point_cloud_range", point_cloud_range_tensor, persistent=False)
        self.register_buffer("grid_size", grid_size, persistent=False)

        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(hidden_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, self.num_classes, kernel_size=1),
        )

    def _build_patch_center_grid(
        self,
        patch_h: int,
        patch_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ys = torch.arange(patch_h, device=device, dtype=dtype)
        xs = torch.arange(patch_w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        u = (grid_x + 0.5) * float(self.patch_size)
        v = (grid_y + 0.5) * float(self.patch_size)
        ones = torch.ones_like(u)
        return torch.stack([u, v, ones], dim=-1).reshape(-1, 3)

    def forward(
        self,
        last_frame_tokens: torch.Tensor,
        last_frame_depth: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        lidar_to_world: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, cam_num, patch_count, token_dim = last_frame_tokens.shape
        _, _, image_h, image_w = last_frame_depth.shape
        patch_h = image_h // self.patch_size
        patch_w = image_w // self.patch_size
        if patch_h * patch_w != patch_count:
            raise ValueError(
                f"Patch count mismatch: expected {patch_h * patch_w}, got {patch_count}"
            )

        depth_patch = F.interpolate(
            (last_frame_depth * self.depth_scale).reshape(batch_size * cam_num, 1, image_h, image_w),
            size=(patch_h, patch_w),
            mode="area",
        ).reshape(batch_size, cam_num, patch_h, patch_w)
        depth_patch_flat = depth_patch.reshape(batch_size, cam_num, patch_count)
        patch_uv1 = self._build_patch_center_grid(patch_h, patch_w, last_frame_tokens.device, last_frame_tokens.dtype)
        patch_uv1 = patch_uv1.view(1, 1, patch_count, 3).expand(batch_size, cam_num, -1, -1)

        inv_intrinsics = torch.inverse(intrinsics)
        rays = torch.matmul(inv_intrinsics.unsqueeze(2), patch_uv1.unsqueeze(-1)).squeeze(-1)
        cam_points = rays * depth_patch_flat.unsqueeze(-1).clamp_min(0.0)

        cam_points_h = torch.cat([cam_points, torch.ones_like(cam_points[..., :1])], dim=-1)
        world_points = torch.matmul(camera_to_world.unsqueeze(2), cam_points_h.unsqueeze(-1)).squeeze(-1)[..., :3]

        lidar_to_world_inv = torch.inverse(lidar_to_world)
        world_points_h = torch.cat([world_points, torch.ones_like(world_points[..., :1])], dim=-1)
        lidar_points = torch.matmul(lidar_to_world_inv.unsqueeze(1).unsqueeze(2), world_points_h.unsqueeze(-1)).squeeze(-1)[..., :3]

        voxel_coords = torch.floor(
            (lidar_points - self.point_cloud_range[:3].view(1, 1, 1, 3)) / self.voxel_size.view(1, 1, 1, 3)
        ).to(torch.long)
        valid = (
            (depth_patch_flat > 1e-3)
            & (voxel_coords[..., 0] >= 0)
            & (voxel_coords[..., 0] < self.grid_size[0])
            & (voxel_coords[..., 1] >= 0)
            & (voxel_coords[..., 1] < self.grid_size[1])
            & (voxel_coords[..., 2] >= 0)
            & (voxel_coords[..., 2] < self.grid_size[2])
        )

        projected_tokens = self.token_proj(last_frame_tokens)
        grid_x, grid_y, grid_z = [int(v.item()) for v in self.grid_size]
        volume = projected_tokens.new_zeros((batch_size, projected_tokens.shape[-1], grid_x, grid_y, grid_z))
        counts = projected_tokens.new_zeros((batch_size, 1, grid_x, grid_y, grid_z))

        for batch_idx in range(batch_size):
            batch_valid = valid[batch_idx]
            if not batch_valid.any():
                continue
            batch_coords = voxel_coords[batch_idx][batch_valid]
            flat_indices = (
                batch_coords[:, 0] * (grid_y * grid_z)
                + batch_coords[:, 1] * grid_z
                + batch_coords[:, 2]
            )
            batch_tokens = projected_tokens[batch_idx][batch_valid]
            volume_flat = volume[batch_idx].reshape(projected_tokens.shape[-1], -1)
            counts_flat = counts[batch_idx].reshape(1, -1)
            volume_flat.index_add_(1, flat_indices, batch_tokens.transpose(0, 1))
            counts_flat.index_add_(
                1,
                flat_indices,
                torch.ones((1, flat_indices.shape[0]), device=counts.device, dtype=counts.dtype),
            )

        volume = volume / counts.clamp_min(1.0)
        decoder_input = torch.cat([volume, counts.gt(0).to(volume.dtype)], dim=1)
        return self.decoder(decoder_input)
