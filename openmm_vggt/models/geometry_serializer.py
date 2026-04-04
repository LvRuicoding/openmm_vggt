import torch
import torch.nn as nn


class GeometrySerializer(nn.Module):
    def __init__(self, grid_size_2d=16.0, default_main_view=0):
        super().__init__()
        self.grid_size_2d = grid_size_2d
        self.default_main_view = default_main_view

    def choose_main_views(self, batch_size, num_views, device):
        if self.training:
            return torch.randint(0, num_views, (batch_size,), device=device)
        view_id = min(self.default_main_view, num_views - 1)
        return torch.full((batch_size,), view_id, device=device, dtype=torch.long)

    def compute_snake_sort_keys(self, coords, padding_mask, grid_size):
        q_coords = torch.floor(coords / grid_size).to(torch.int64)
        valid_mask = ~padding_mask
        if valid_mask.any():
            min_coords = q_coords[valid_mask].min(dim=0)[0]
            q_coords = q_coords - min_coords
            d0, d1, d2 = q_coords[:, 0], q_coords[:, 1], q_coords[:, 2]
            max_d1 = d1[valid_mask].max()
            max_d2 = d2[valid_mask].max()
        else:
            d0, d1, d2 = q_coords[:, 0], q_coords[:, 1], q_coords[:, 2]
            max_d1 = torch.zeros((), dtype=q_coords.dtype, device=q_coords.device)
            max_d2 = torch.zeros((), dtype=q_coords.dtype, device=q_coords.device)

        snake_d1 = torch.where(d0 % 2 == 1, max_d1 - d1, d1)
        snake_d2 = torch.where((d0 + d1) % 2 == 1, max_d2 - d2, d2)
        multiplier = 100000
        sort_keys = d0 * (multiplier ** 2) + snake_d1 * multiplier + snake_d2
        pad_key = torch.full_like(sort_keys, torch.iinfo(sort_keys.dtype).max)
        return torch.where(padding_mask, pad_key, sort_keys)

    def project_lidar_to_selected_views(self, lidar_coords, K, T_c2w, view_indices):
        batch_size, num_points, _ = lidar_coords.shape
        batch_indices = torch.arange(batch_size, device=lidar_coords.device)
        selected_K = K[batch_indices, view_indices]
        selected_T_c2w = T_c2w[batch_indices, view_indices]

        xyz1 = torch.cat([lidar_coords, torch.ones_like(lidar_coords[..., :1])], dim=-1)
        cam_coords_homo = torch.matmul(
            torch.inverse(selected_T_c2w).unsqueeze(1),
            xyz1.unsqueeze(-1),
        ).squeeze(-1)
        cam_coords = cam_coords_homo[..., :3]
        img_coords_homo = torch.matmul(selected_K.unsqueeze(1), cam_coords.unsqueeze(-1)).squeeze(-1)
        depth = img_coords_homo[..., 2:3]
        safe_depth = torch.clamp(depth, min=1e-5)
        uv = img_coords_homo[..., :2] / safe_depth
        camera_ids = view_indices.view(batch_size, 1, 1).expand(-1, num_points, -1).to(dtype=uv.dtype)
        projected = torch.cat([camera_ids, uv], dim=-1)
        valid = depth.squeeze(-1) > 1e-5
        return projected, valid

    def forward(self, lidar_tokens, lidar_coords, img_tokens, img_coords, K, T_c2w, lidar_padding_mask=None, img_padding_mask=None):
        batch_size, num_lidar, channels = lidar_tokens.shape
        _, num_img, _ = img_tokens.shape
        if lidar_padding_mask is None:
            lidar_padding_mask = torch.zeros((batch_size, num_lidar), dtype=torch.bool, device=lidar_tokens.device)
        if img_padding_mask is None:
            img_padding_mask = torch.zeros((batch_size, num_img), dtype=torch.bool, device=img_tokens.device)

        num_views = K.shape[1]
        main_view_indices = self.choose_main_views(batch_size, num_views, lidar_tokens.device)
        projected_lidar, lidar_visible = self.project_lidar_to_selected_views(lidar_coords, K, T_c2w, main_view_indices)
        lidar_padding_mask = lidar_padding_mask | (~lidar_visible)

        unified_coords = torch.cat([projected_lidar, img_coords], dim=1)
        unified_tokens = torch.cat([lidar_tokens, img_tokens], dim=1)
        unified_padding_mask = torch.cat([lidar_padding_mask, img_padding_mask], dim=1)

        lidar_indices = torch.arange(num_lidar, device=lidar_tokens.device, dtype=torch.long)
        sorted_indices_list = []
        for batch_idx in range(batch_size):
            cam_ids = img_coords[batch_idx, :, 0].long()
            main_view = int(main_view_indices[batch_idx].item())
            ordered_parts = []

            for view_idx in range(num_views):
                view_img_rel = torch.nonzero(cam_ids == view_idx, as_tuple=False).squeeze(1)
                view_img_abs = view_img_rel + num_lidar
                if view_idx != main_view:
                    ordered_parts.append(view_img_abs)
                    continue

                main_coords = torch.cat(
                    [projected_lidar[batch_idx], img_coords[batch_idx, view_img_rel]],
                    dim=0,
                )
                main_padding = torch.cat(
                    [lidar_padding_mask[batch_idx], img_padding_mask[batch_idx, view_img_rel]],
                    dim=0,
                )
                main_indices = torch.cat([lidar_indices, view_img_abs], dim=0)
                main_sort_keys = self.compute_snake_sort_keys(main_coords, main_padding, self.grid_size_2d)
                main_order = torch.argsort(main_sort_keys, dim=0)
                ordered_parts.append(main_indices[main_order])

            sorted_indices_list.append(torch.cat(ordered_parts, dim=0))

        sorted_indices = torch.stack(sorted_indices_list, dim=0)
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, channels)
        sorted_tokens = torch.gather(unified_tokens, 1, expanded_indices)
        sorted_padding_mask = torch.gather(unified_padding_mask, 1, sorted_indices)
        return sorted_tokens, sorted_indices, sorted_padding_mask, num_lidar
