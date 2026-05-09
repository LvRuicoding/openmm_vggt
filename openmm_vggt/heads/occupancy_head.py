from __future__ import annotations

from typing import List, Tuple

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
        voxel_chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.voxel_chunk_size = int(voxel_chunk_size)

        voxel_size_tensor = torch.tensor(voxel_size, dtype=torch.float32)
        point_cloud_range_tensor = torch.tensor(point_cloud_range, dtype=torch.float32)
        grid_size = ((point_cloud_range_tensor[3:] - point_cloud_range_tensor[:3]) / voxel_size_tensor).round().to(torch.long)

        self.register_buffer("voxel_size", voxel_size_tensor, persistent=False)
        self.register_buffer("point_cloud_range", point_cloud_range_tensor, persistent=False)
        self.register_buffer("grid_size", grid_size, persistent=False)

        self.layer_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(token_dim),
                    nn.Linear(token_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(4)
            ]
        )
        self.layer_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, self.num_classes, kernel_size=1),
        )

    def _voxel_centers(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid_x, grid_y, grid_z = [int(v.item()) for v in self.grid_size]
        xs = torch.arange(grid_x, device=device, dtype=dtype)
        ys = torch.arange(grid_y, device=device, dtype=dtype)
        zs = torch.arange(grid_z, device=device, dtype=dtype)
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
        centers = self.point_cloud_range[:3].to(device=device, dtype=dtype) + (grid + 0.5) * self.voxel_size.to(device=device, dtype=dtype)
        return centers.reshape(-1, 3)

    def _project_centers(
        self,
        centers_lidar: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        lidar_to_world: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, view_count = intrinsics.shape[:2]
        num_voxels = centers_lidar.shape[0]
        image_h, image_w = image_hw

        centers_h = torch.cat([centers_lidar, torch.ones_like(centers_lidar[..., :1])], dim=-1)  # [V, 4]
        centers_world = torch.matmul(lidar_to_world[:, None, None, :, :], centers_h.view(1, 1, num_voxels, 4, 1)).squeeze(-1)
        centers_world = centers_world.expand(batch_size, view_count, num_voxels, 4)

        world_to_camera = torch.inverse(camera_to_world)
        centers_cam = torch.matmul(world_to_camera[:, :, None, :, :], centers_world.unsqueeze(-1)).squeeze(-1)
        xyz_cam = centers_cam[..., :3]

        x = xyz_cam[..., 0]
        y = xyz_cam[..., 1]
        z = xyz_cam[..., 2].clamp(min=1e-5)

        fx = intrinsics[..., 0, 0][..., None]
        fy = intrinsics[..., 1, 1][..., None]
        cx = intrinsics[..., 0, 2][..., None]
        cy = intrinsics[..., 1, 2][..., None]

        u = fx * x / z + cx
        v = fy * y / z + cy
        valid = (
            (z > 0)
            & (u >= 0)
            & (u <= image_w - 1)
            & (v >= 0)
            & (v <= image_h - 1)
        )

        u_norm = (u / max(image_w - 1, 1)) * 2 - 1
        v_norm = (v / max(image_h - 1, 1)) * 2 - 1
        grid = torch.stack([u_norm, v_norm], dim=-1)
        return grid, valid

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        lidar_to_world: torch.Tensor,
    ) -> torch.Tensor:
        if not aggregated_tokens_list:
            raise ValueError("aggregated_tokens_list must not be empty")

        if len(aggregated_tokens_list) != len(self.layer_proj):
            raise ValueError(
                f"Expected {len(self.layer_proj)} feature layers, got {len(aggregated_tokens_list)}"
            )

        tokens = aggregated_tokens_list[-1]
        if tokens.ndim != 4:
            raise ValueError(f"Expected tokens with shape [B, S, N, D], got {tuple(tokens.shape)}")

        batch_size, view_count, patch_count, token_dim = tokens.shape
        image_h, image_w = images.shape[-2:]
        patch_h = image_h // self.patch_size
        patch_w = image_w // self.patch_size
        if patch_h * patch_w != patch_count:
            raise ValueError(f"Patch count mismatch: expected {patch_h * patch_w}, got {patch_count}")

        feat_maps = []
        for layer_tokens in aggregated_tokens_list:
            if layer_tokens.shape != tokens.shape:
                raise ValueError(
                    "All occupancy feature layers must share shape "
                    f"{tuple(tokens.shape)}, got {tuple(layer_tokens.shape)}"
                )
            feat_maps.append(
                layer_tokens.reshape(batch_size * view_count, patch_h, patch_w, token_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        centers = self._voxel_centers(device=tokens.device, dtype=tokens.dtype)
        num_voxels = centers.shape[0]
        grid_x, grid_y, grid_z = [int(v.item()) for v in self.grid_size]

        center_norm = (centers - self.point_cloud_range[:3].to(device=tokens.device, dtype=tokens.dtype))
        center_norm = center_norm / torch.clamp(
            self.point_cloud_range[3:] - self.point_cloud_range[:3], min=1e-6
        ).to(device=tokens.device, dtype=tokens.dtype)
        center_norm = center_norm * 2.0 - 1.0
        pos_feat = self.pos_proj(center_norm)

        volume = tokens.new_zeros((batch_size, self.hidden_dim, num_voxels))
        for start in range(0, num_voxels, self.voxel_chunk_size):
            end = min(start + self.voxel_chunk_size, num_voxels)
            centers_chunk = centers[start:end]
            chunk_voxels = end - start

            grid, valid = self._project_centers(
                centers_chunk,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                lidar_to_world=lidar_to_world,
                image_hw=(image_h, image_w),
            )
            grid = grid.reshape(batch_size * view_count, chunk_voxels, 1, 2)
            valid = valid.to(dtype=tokens.dtype).unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1.0)

            layer_feats = []
            for feat_map, layer_proj in zip(feat_maps, self.layer_proj):
                sampled = F.grid_sample(
                    feat_map,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                sampled = sampled.squeeze(-1).permute(0, 2, 1).reshape(batch_size, view_count, chunk_voxels, token_dim)
                sampled = sampled * valid
                layer_feats.append(layer_proj(sampled.sum(dim=1) / denom))

            voxel_feat = self.layer_fuse(torch.cat(layer_feats, dim=-1))
            fused = self.fuse(torch.cat([voxel_feat, pos_feat[start:end].view(1, chunk_voxels, -1).expand(batch_size, -1, -1)], dim=-1))
            volume[:, :, start:end] = fused.transpose(1, 2)

        volume = volume.reshape(batch_size, self.hidden_dim, grid_x, grid_y, grid_z)
        return self.decoder(volume)
