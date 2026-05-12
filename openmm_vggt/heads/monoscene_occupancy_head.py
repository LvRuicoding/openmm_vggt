from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class _Bottleneck3D(nn.Module):
    """DDR-style 3D residual bottleneck used by MonoScene."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        norm_layer,
        stride: int = 1,
        dilation: Tuple[int, int, int] = (1, 1, 1),
        expansion: int = 4,
        downsample: nn.Module | None = None,
        bn_momentum: float = 0.0003,
    ) -> None:
        super().__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 1, 3),
            stride=(1, 1, stride),
            dilation=(1, 1, dilation[0]),
            padding=(0, 0, dilation[0]),
            bias=False,
        )
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 3, 1),
            stride=(1, stride, 1),
            dilation=(1, dilation[1], 1),
            padding=(0, dilation[1], 0),
            bias=False,
        )
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            dilation=(dilation[2], 1, 1),
            padding=(dilation[2], 0, 0),
            bias=False,
        )
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out5 = self.bn5(self.conv5(self.relu(out4)))
        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out5 + residual)


class _Process(nn.Module):
    def __init__(self, feature: int, norm_layer, bn_momentum: float, dilations=(1, 2, 3)) -> None:
        super().__init__()
        self.main = nn.Sequential(
            *[
                _Bottleneck3D(
                    feature,
                    feature // 4,
                    bn_momentum=bn_momentum,
                    norm_layer=norm_layer,
                    dilation=(dil, dil, dil),
                )
                for dil in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class _Downsample(nn.Module):
    def __init__(self, feature: int, norm_layer, bn_momentum: float, expansion: int = 8) -> None:
        super().__init__()
        out_channels = int(feature * expansion / 4)
        self.main = _Bottleneck3D(
            feature,
            feature // 4,
            bn_momentum=bn_momentum,
            expansion=expansion,
            stride=2,
            downsample=nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, out_channels, kernel_size=1, stride=1, bias=False),
                norm_layer(out_channels, momentum=bn_momentum),
            ),
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class _Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer, bn_momentum: float) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
            norm_layer(out_channels, momentum=bn_momentum),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class _SegmentationHead(nn.Module):
    def __init__(self, inplanes: int, planes: int, num_classes: int, dilations=(1, 2, 3)) -> None:
        super().__init__()
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False)
                for dil in dilations
            ]
        )
        self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for _ in dilations])
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False)
                for dil in dilations
            ]
        )
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for _ in dilations])
        self.relu = nn.ReLU()
        self.conv_classes = nn.Conv3d(planes, num_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv0(x))
        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x)))))
        for idx in range(1, len(self.conv1)):
            y = y + self.bn2[idx](self.conv2[idx](self.relu(self.bn1[idx](self.conv1[idx](x)))))
        return self.conv_classes(self.relu(y + x))


class _ASPP3D(nn.Module):
    def __init__(self, planes: int, norm_layer, dilations=(1, 2, 3)) -> None:
        super().__init__()
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False)
                for dil in dilations
            ]
        )
        self.bn1 = nn.ModuleList([norm_layer(planes) for _ in dilations])
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False)
                for dil in dilations
            ]
        )
        self.bn2 = nn.ModuleList([norm_layer(planes) for _ in dilations])
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x)))))
        for idx in range(1, len(self.conv1)):
            y = y + self.bn2[idx](self.conv2[idx](self.relu(self.bn1[idx](self.conv1[idx](x)))))
        return self.relu(y + x)


class _CPMegaVoxels(nn.Module):
    def __init__(
        self,
        feature: int,
        size: Tuple[int, int, int],
        n_relations: int = 4,
        bn_momentum: float = 0.0003,
    ) -> None:
        super().__init__()
        self.size = tuple(int(v) for v in size)
        self.n_relations = int(n_relations)
        self.flatten_size = self.size[0] * self.size[1] * self.size[2]
        self.feature = int(feature)
        self.context_feature = self.feature * 2
        self.flatten_context_size = (self.size[0] // 2) * (self.size[1] // 2) * (self.size[2] // 2)
        padding = tuple((size_dim + 1) % 2 for size_dim in self.size)

        self.mega_context = nn.Conv3d(
            self.feature,
            self.context_feature,
            stride=2,
            padding=padding,
            kernel_size=3,
        )
        self.context_prior_logits = nn.ModuleList(
            [
                nn.Conv3d(
                    self.feature,
                    self.flatten_context_size,
                    padding=0,
                    kernel_size=1,
                )
                for _ in range(self.n_relations)
            ]
        )
        self.aspp = _ASPP3D(self.feature, nn.BatchNorm3d, dilations=(1, 2, 3))
        self.resize = nn.Sequential(
            nn.Conv3d(
                self.context_feature * self.n_relations + self.feature,
                self.feature,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            _Process(self.feature, nn.BatchNorm3d, bn_momentum, dilations=(1,)),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        x_agg = self.aspp(x)

        x_mega_context = self.mega_context(x_agg).reshape(batch_size, self.context_feature, -1)
        x_mega_context = x_mega_context.permute(0, 2, 1)

        context_prior_logits = []
        context_relations = []
        for rel_idx in range(self.n_relations):
            context_prior_logit = self.context_prior_logits[rel_idx](x_agg)
            context_prior_logit = context_prior_logit.reshape(
                batch_size,
                self.flatten_context_size,
                self.flatten_size,
            )
            context_prior_logits.append(context_prior_logit.unsqueeze(1))

            context_prior = torch.sigmoid(context_prior_logit.permute(0, 2, 1))
            context_rel = torch.bmm(context_prior, x_mega_context)
            context_relations.append(context_rel)

        x_context = torch.cat(context_relations, dim=2)
        x_context = x_context.permute(0, 2, 1).reshape(
            batch_size,
            -1,
            self.size[0],
            self.size[1],
            self.size[2],
        )
        x = self.resize(torch.cat([x, x_context], dim=1))
        return {
            "x": x,
            "P_logits": torch.cat(context_prior_logits, dim=1),
        }


class _MonoSceneUNet3DKitti(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature: int,
        full_scene_size: Tuple[int, int, int],
        project_scale: int,
        context_prior: bool = False,
        n_relations: int = 4,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.project_scale = int(project_scale)
        self.full_scene_size = tuple(int(v) for v in full_scene_size)
        self.feature = int(feature)
        self.context_prior = bool(context_prior)

        size_l1 = tuple(int(v / self.project_scale) for v in self.full_scene_size)
        size_l2 = tuple(v // 2 for v in size_l1)
        size_l3 = tuple(v // 2 for v in size_l2)

        norm_layer = nn.BatchNorm3d
        self.process_l1 = nn.Sequential(
            _Process(self.feature, norm_layer, bn_momentum, dilations=(1, 2, 3)),
            _Downsample(self.feature, norm_layer, bn_momentum),
        )
        self.process_l2 = nn.Sequential(
            _Process(self.feature * 2, norm_layer, bn_momentum, dilations=(1, 2, 3)),
            _Downsample(self.feature * 2, norm_layer, bn_momentum),
        )
        self.up_13_l2 = _Upsample(self.feature * 4, self.feature * 2, norm_layer, bn_momentum)
        self.up_12_l1 = _Upsample(self.feature * 2, self.feature, norm_layer, bn_momentum)
        self.up_l1_lfull = _Upsample(self.feature, self.feature // 2, norm_layer, bn_momentum)
        self.ssc_head = _SegmentationHead(self.feature // 2, self.feature // 2, num_classes, dilations=(1, 2, 3))
        self.cp_mega_voxels = (
            _CPMegaVoxels(self.feature * 4, size_l3, n_relations=n_relations, bn_momentum=bn_momentum)
            if self.context_prior
            else None
        )

    def forward(self, x3d_l1: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        x3d_l2 = self.process_l1(x3d_l1)
        x3d_l3 = self.process_l2(x3d_l2)
        if self.cp_mega_voxels is not None:
            cp_outputs = self.cp_mega_voxels(x3d_l3)
            x3d_l3 = cp_outputs["x"]
            outputs["P_logits"] = cp_outputs["P_logits"]
        x3d_up_l2 = self.up_13_l2(x3d_l3) + x3d_l2
        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)
        outputs["ssc_logit"] = self.ssc_head(x3d_up_lfull)
        return outputs


class MonoSceneOccupancyHead(nn.Module):
    """MonoScene-style occupancy head adapted to VGGT token features.

    This keeps the existing model contract: token features are projected into a
    coarse voxel volume using camera geometry, then a MonoScene KITTI 3D UNet
    upsamples to the target occupancy grid and returns logits shaped
    [B, num_classes, X, Y, Z].
    """

    def __init__(
        self,
        token_dim: int,
        patch_size: int,
        voxel_size: Tuple[float, float, float],
        point_cloud_range: Tuple[float, float, float, float, float, float],
        num_classes: int = 20,
        feature: int = 64,
        project_scale: int = 2,
        context_prior: bool = False,
        n_relations: int = 4,
        voxel_chunk_size: int = 65536,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.num_classes = int(num_classes)
        self.feature = int(feature)
        self.project_scale = int(project_scale)
        self.voxel_chunk_size = int(voxel_chunk_size)

        if self.project_scale < 1:
            raise ValueError("project_scale must be >= 1")

        voxel_size_tensor = torch.tensor(voxel_size, dtype=torch.float32)
        point_cloud_range_tensor = torch.tensor(point_cloud_range, dtype=torch.float32)
        grid_size = ((point_cloud_range_tensor[3:] - point_cloud_range_tensor[:3]) / voxel_size_tensor).round().to(torch.long)
        if torch.any(grid_size % self.project_scale != 0):
            raise ValueError(
                f"grid_size {tuple(int(v) for v in grid_size)} must be divisible by project_scale={self.project_scale}"
            )

        self.register_buffer("voxel_size", voxel_size_tensor, persistent=False)
        self.register_buffer("point_cloud_range", point_cloud_range_tensor, persistent=False)
        self.register_buffer("grid_size", grid_size, persistent=False)
        self.register_buffer("coarse_grid_size", grid_size // self.project_scale, persistent=False)

        self.layer_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(token_dim),
                    nn.Linear(token_dim, self.feature),
                    nn.GELU(),
                    nn.Linear(self.feature, self.feature),
                )
                for _ in range(4)
            ]
        )
        self.layer_fuse = nn.Sequential(
            nn.LayerNorm(self.feature * 4),
            nn.Linear(self.feature * 4, self.feature),
            nn.GELU(),
            nn.Linear(self.feature, self.feature),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(3, self.feature),
            nn.GELU(),
            nn.Linear(self.feature, self.feature),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.feature * 2),
            nn.Linear(self.feature * 2, self.feature),
            nn.GELU(),
            nn.Linear(self.feature, self.feature),
            nn.GELU(),
        )
        self.decoder = _MonoSceneUNet3DKitti(
            num_classes=self.num_classes,
            feature=self.feature,
            full_scene_size=tuple(int(v.item()) for v in self.grid_size),
            project_scale=self.project_scale,
            context_prior=context_prior,
            n_relations=n_relations,
            bn_momentum=bn_momentum,
        )

    def _coarse_voxel_centers(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid_x, grid_y, grid_z = [int(v.item()) for v in self.coarse_grid_size]
        xs = torch.arange(grid_x, device=device, dtype=dtype)
        ys = torch.arange(grid_y, device=device, dtype=dtype)
        zs = torch.arange(grid_z, device=device, dtype=dtype)
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
        coarse_voxel_size = self.voxel_size.to(device=device, dtype=dtype) * self.project_scale
        centers = self.point_cloud_range[:3].to(device=device, dtype=dtype) + (grid + 0.5) * coarse_voxel_size
        return centers.reshape(-1, 3)

    def _project_centers_to_patches(
        self,
        centers_lidar: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        lidar_to_world: torch.Tensor,
        image_hw: Tuple[int, int],
        patch_hw: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, view_count = intrinsics.shape[:2]
        num_voxels = centers_lidar.shape[0]
        image_h, image_w = image_hw
        patch_h, patch_w = patch_hw

        centers_h = torch.cat([centers_lidar, torch.ones_like(centers_lidar[..., :1])], dim=-1)
        centers_world = torch.matmul(lidar_to_world[:, None, None, :, :], centers_h.view(1, 1, num_voxels, 4, 1)).squeeze(-1)
        centers_world = centers_world.expand(batch_size, view_count, num_voxels, 4)

        world_to_camera = torch.inverse(camera_to_world)
        centers_cam = torch.matmul(world_to_camera[:, :, None, :, :], centers_world.unsqueeze(-1)).squeeze(-1)
        xyz_cam = centers_cam[..., :3]

        x = xyz_cam[..., 0]
        y = xyz_cam[..., 1]
        z_raw = xyz_cam[..., 2]
        z = z_raw.clamp(min=1e-5)

        fx = intrinsics[..., 0, 0][..., None]
        fy = intrinsics[..., 1, 1][..., None]
        cx = intrinsics[..., 0, 2][..., None]
        cy = intrinsics[..., 1, 2][..., None]

        u = fx * x / z + cx
        v = fy * y / z + cy
        valid = (z_raw > 0) & (u >= 0) & (u <= image_w - 1) & (v >= 0) & (v <= image_h - 1)

        patch_x = torch.div(u, self.patch_size, rounding_mode="floor").to(torch.long)
        patch_y = torch.div(v, self.patch_size, rounding_mode="floor").to(torch.long)
        valid = valid & (patch_x >= 0) & (patch_x < patch_w) & (patch_y >= 0) & (patch_y < patch_h)
        patch_index = patch_y * patch_w + patch_x
        return patch_index, valid

    @staticmethod
    def _flosp_gather(
        feat_map: torch.Tensor,
        patch_index: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_views, channels, patch_h, patch_w = feat_map.shape
        patch_count = patch_h * patch_w
        src = feat_map.reshape(batch_views, channels, patch_count)
        zeros = src.new_zeros((batch_views, channels, 1))
        src = torch.cat([src, zeros], dim=2)

        flat_index = patch_index.reshape(batch_views, -1).clone()
        flat_valid = valid.reshape(batch_views, -1)
        flat_index[~flat_valid] = patch_count
        gather_index = flat_index[:, None, :].expand(-1, channels, -1)
        return torch.gather(src, 2, gather_index)

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        lidar_to_world: torch.Tensor,
    ) -> torch.Tensor:
        if len(aggregated_tokens_list) != len(self.layer_proj):
            raise ValueError(f"Expected {len(self.layer_proj)} feature layers, got {len(aggregated_tokens_list)}")

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

        centers = self._coarse_voxel_centers(device=tokens.device, dtype=tokens.dtype)
        num_voxels = centers.shape[0]
        coarse_x, coarse_y, coarse_z = [int(v.item()) for v in self.coarse_grid_size]

        center_norm = centers - self.point_cloud_range[:3].to(device=tokens.device, dtype=tokens.dtype)
        center_norm = center_norm / torch.clamp(
            self.point_cloud_range[3:] - self.point_cloud_range[:3],
            min=1e-6,
        ).to(device=tokens.device, dtype=tokens.dtype)
        center_norm = center_norm * 2.0 - 1.0
        pos_feat = self.pos_proj(center_norm)

        volume = tokens.new_zeros((batch_size, self.feature, num_voxels))
        for start in range(0, num_voxels, self.voxel_chunk_size):
            end = min(start + self.voxel_chunk_size, num_voxels)
            centers_chunk = centers[start:end]
            chunk_voxels = end - start

            patch_index, valid = self._project_centers_to_patches(
                centers_chunk,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                lidar_to_world=lidar_to_world,
                image_hw=(image_h, image_w),
                patch_hw=(patch_h, patch_w),
            )

            layer_feats = []
            for feat_map, layer_proj in zip(feat_maps, self.layer_proj):
                sampled = self._flosp_gather(
                    feat_map,
                    patch_index=patch_index,
                    valid=valid,
                )
                sampled = sampled.permute(0, 2, 1).reshape(batch_size, view_count, chunk_voxels, token_dim)
                view_weights = valid.to(dtype=sampled.dtype).unsqueeze(-1)
                sampled_sum = (sampled * view_weights).sum(dim=1)
                view_count_valid = view_weights.sum(dim=1).clamp_min(1.0)
                layer_feats.append(layer_proj(sampled_sum / view_count_valid))

            voxel_feat = self.layer_fuse(torch.cat(layer_feats, dim=-1))
            fused = self.fuse(
                torch.cat(
                    [
                        voxel_feat,
                        pos_feat[start:end].view(1, chunk_voxels, -1).expand(batch_size, -1, -1),
                    ],
                    dim=-1,
                )
            )
            volume[:, :, start:end] = fused.transpose(1, 2)

        volume = volume.reshape(batch_size, self.feature, coarse_x, coarse_y, coarse_z)
        return self.decoder(volume)
