import torch
import torch.nn as nn
import torch.nn.functional as F

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size, src.shape[1]))
    counts = src.new_zeros((dim_size, 1))
    out.index_add_(0, index, src)
    counts.index_add_(0, index, torch.ones((src.shape[0], 1), device=src.device, dtype=src.dtype))
    return out / counts.clamp_min(1.0)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.full((dim_size, src.shape[1]), torch.finfo(src.dtype).min, device=src.device, dtype=src.dtype)
    scatter_index = index.unsqueeze(-1).expand(-1, src.shape[1])
    out.scatter_reduce_(0, scatter_index, src, reduce="amax", include_self=True)
    return out


class PFNLayerV2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_norm=True,
        last_layer=False,
        pre_norm=False,
        first_layer=False,
    ):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        self.pre_norm = pre_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            norm_channels = in_channels if self.pre_norm else out_channels
            affine = not (self.pre_norm and first_layer)
            self.norm = nn.BatchNorm1d(norm_channels, eps=1e-3, momentum=0.01, affine=affine)
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.norm = None
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.relu = nn.ReLU()

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is None:
            return x
        if self.training and x.shape[0] <= 1:
            return F.batch_norm(
                x,
                self.norm.running_mean,
                self.norm.running_var,
                self.norm.weight,
                self.norm.bias,
                training=False,
                momentum=0.0,
                eps=self.norm.eps,
            )
        return self.norm(x)

    def forward(self, inputs, unq_inv):
        x = inputs
        if self.use_norm and self.pre_norm:
            x = self._apply_norm(x)
            x = self.linear(x)
        else:
            x = self.linear(x)
            if self.use_norm:
                x = self._apply_norm(x)
        x = self.relu(x)
        x_max = scatter_max(x, unq_inv, int(unq_inv.max().item()) + 1)
        if self.last_vfe:
            return x_max
        return torch.cat([x, x_max[unq_inv, :]], dim=1)


class PCDetDynamicVoxelVFE(nn.Module):
    def __init__(
        self,
        num_point_features,
        voxel_size,
        grid_size,
        point_cloud_range,
        use_norm=True,
        with_distance=False,
        use_absolute_xyz=True,
        num_filters=(128, 128),
        pre_norm=False,
    ):
        super().__init__()
        self.use_norm = use_norm
        self.pre_norm = pre_norm
        self.with_distance = with_distance
        self.use_absolute_xyz = use_absolute_xyz

        feature_dim = num_point_features
        feature_dim += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            feature_dim += 1

        num_filters = [feature_dim] + list(num_filters)
        self.pfn_layers = nn.ModuleList(
            [
                PFNLayerV2(
                    num_filters[i],
                    num_filters[i + 1],
                    self.use_norm,
                    last_layer=(i >= len(num_filters) - 2),
                    pre_norm=self.pre_norm,
                    first_layer=(i == 0),
                )
                for i in range(len(num_filters) - 1)
            ]
        )

        self.register_buffer("grid_size", torch.tensor(grid_size, dtype=torch.int32), persistent=False)
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32), persistent=False)
        self.register_buffer("point_cloud_range", torch.tensor(point_cloud_range, dtype=torch.float32), persistent=False)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.scale_xyz = int(grid_size[0] * grid_size[1] * grid_size[2])
        self.scale_yz = int(grid_size[1] * grid_size[2])
        self.scale_z = int(grid_size[2])
        self.out_dim = num_filters[-1]

    def get_output_feature_dim(self):
        return self.out_dim

    def forward(self, points: torch.Tensor):
        points_xyz = points[:, [1, 2, 3]].contiguous()
        points_coords = torch.floor(
            (points_xyz - self.point_cloud_range[[0, 1, 2]]) / self.voxel_size[[0, 1, 2]]
        ).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1, 2]])).all(dim=1)
        points = points[mask]
        points_xyz = points_xyz[mask]
        points_coords = points_coords[mask]
        if points.shape[0] == 0:
            return points.new_zeros((0, self.out_dim)), points.new_zeros((0, 4), dtype=torch.long)

        merge_coords = (
            points[:, 0].int() * self.scale_xyz
            + points_coords[:, 0] * self.scale_yz
            + points_coords[:, 1] * self.scale_z
            + points_coords[:, 2]
        )
        unq_coords, unq_inv, _ = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = scatter_mean(points_xyz, unq_inv, int(unq_inv.max().item()) + 1)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [torch.cat([points_xyz, points[:, 4:]], dim=-1), f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]
        if self.with_distance:
            points_dist = torch.norm(points_xyz, 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack(
            (
                unq_coords // self.scale_xyz,
                (unq_coords % self.scale_xyz) // self.scale_yz,
                (unq_coords % self.scale_yz) // self.scale_z,
                unq_coords % self.scale_z,
            ),
            dim=1,
        )
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        return features, voxel_coords
