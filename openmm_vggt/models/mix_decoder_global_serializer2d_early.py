from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmengine.registry import MODELS

from openmm_vggt.layers.attention import Attention
from openmm_vggt.utils.pose_enc import extri_to_pose_encoding, pose_encoding_to_extri_intri

from .aggregator_window_attn_early import EarlyFusionAggregator
from .mix_decoder_global import mix_decoder_global


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
class mix_decoder_global_serializer2d_early(mix_decoder_global):
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

    def _project_voxels_to_image_coords(
        self,
        voxel_coords: torch.Tensor,
        voxel_batch_ids: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz1 = torch.cat([voxel_coords, torch.ones_like(voxel_coords[..., :1])], dim=-1)
        world_to_camera = torch.inverse(camera_to_world)[voxel_batch_ids]
        cam_coords = torch.matmul(world_to_camera, xyz1[:, None, :, None]).squeeze(-1)[..., :3]
        img_coords = torch.matmul(intrinsics[voxel_batch_ids], cam_coords.unsqueeze(-1)).squeeze(-1)
        depth = img_coords[..., 2]
        safe_depth = torch.clamp(depth, min=1e-5)
        u = img_coords[..., 0] / safe_depth
        v = img_coords[..., 1] / safe_depth
        return u, v, depth

    def _prepare_early_fusion_data(
        self,
        others: dict,
        batch_size: int,
        frame_count: int,
        image_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor | int]:
        points = others["points"].reshape(
            batch_size * frame_count,
            others["points"].shape[2],
            others["points"].shape[3],
        )
        point_mask = others["point_mask"].reshape(batch_size * frame_count, others["point_mask"].shape[2])
        intrinsics = others["intrinsics"].reshape(batch_size, frame_count, self.cam_num, 3, 3)
        intrinsics = intrinsics.reshape(batch_size * frame_count, self.cam_num, 3, 3)
        camera_to_world = others["camera_to_world"].reshape(batch_size, frame_count, self.cam_num, 4, 4)
        camera_to_world = camera_to_world.reshape(batch_size * frame_count, self.cam_num, 4, 4)

        lidar_to_world = None
        if "lidar_to_world" in others:
            lidar_to_world = others["lidar_to_world"].reshape(batch_size * frame_count, 4, 4)

        voxel_features, voxel_coords = self._encode_voxels(points, point_mask, lidar_to_world=lidar_to_world)
        patch_h = image_hw[0] // self.patch_size
        patch_w = image_hw[1] // self.patch_size
        if voxel_features.shape[0] == 0:
            empty_long = torch.zeros((0,), dtype=torch.long, device=points.device)
            empty_float = torch.zeros((0,), dtype=points.dtype, device=points.device)
            return {
                "voxel_tokens": points.new_zeros((0, self.early_voxel_feature_proj.out_features)),
                "flat_seq_ids": empty_long,
                "coord_y": empty_float,
                "coord_x": empty_float,
                "patch_h": patch_h,
                "patch_w": patch_w,
            }

        voxel_tokens = self.early_voxel_feature_proj(voxel_features)
        voxel_centers = self._voxel_coords_to_centers(voxel_coords)
        voxel_batch_ids = voxel_coords[:, 0].long()
        if lidar_to_world is not None:
            voxel_centers = self._local_voxel_centers_to_world(voxel_centers, voxel_batch_ids, lidar_to_world)
        u, v, depth = self._project_voxels_to_image_coords(
            voxel_centers,
            voxel_batch_ids,
            intrinsics,
            camera_to_world,
        )
        patch_x = torch.floor(u / self.patch_size).to(torch.long)
        patch_y = torch.floor(v / self.patch_size).to(torch.long)
        visible = (
            (depth > 1e-5)
            & (patch_x >= 0)
            & (patch_x < patch_w)
            & (patch_y >= 0)
            & (patch_y < patch_h)
        )

        cam_ids = torch.arange(self.cam_num, device=voxel_tokens.device, dtype=torch.long).unsqueeze(0).expand_as(patch_y)
        frame_ids = (voxel_batch_ids % frame_count).unsqueeze(1).expand_as(patch_y)
        batch_ids = (voxel_batch_ids // frame_count).unsqueeze(1).expand_as(patch_y)
        camera_major_batch_ids = batch_ids * self.cam_num + cam_ids
        flat_seq_ids = camera_major_batch_ids * frame_count + frame_ids

        flat_visible = visible.reshape(-1)
        flat_voxel_tokens = voxel_tokens.unsqueeze(1).expand(-1, self.cam_num, -1).reshape(-1, voxel_tokens.shape[-1])
        return {
            "voxel_tokens": flat_voxel_tokens[flat_visible],
            "flat_seq_ids": flat_seq_ids.reshape(-1)[flat_visible],
            "coord_y": v.reshape(-1)[flat_visible],
            "coord_x": u.reshape(-1)[flat_visible],
            "patch_h": patch_h,
            "patch_w": patch_w,
        }

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

    def _apply_early_serializer_fusion(
        self,
        patch_tokens: torch.Tensor,
        fusion_data: Dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        patch_h = int(fusion_data["patch_h"])
        patch_w = int(fusion_data["patch_w"])
        if patch_h * patch_w != patch_tokens.shape[1]:
            raise ValueError(
                f"Patch count mismatch: expected {patch_h * patch_w} from patch grid "
                f"{patch_h}x{patch_w}, but got {patch_tokens.shape[1]}"
            )

        voxel_tokens = fusion_data["voxel_tokens"]
        if voxel_tokens.shape[0] == 0:
            return patch_tokens

        flat_seq_ids = fusion_data["flat_seq_ids"]
        coord_y = fusion_data["coord_y"]
        coord_x = fusion_data["coord_x"]
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

        early_fusion_data = self._prepare_early_fusion_data(others, batch_size, frame_count, (image_h, image_w))
        patch_tokens = self.aggregator.extract_patch_tokens(images)
        patch_tokens = self._apply_early_serializer_fusion(patch_tokens, early_fusion_data)
        aggregated_tokens_list, patch_start_idx = self.aggregator(images, patch_tokens=patch_tokens)
        images = images.view(batch_size, sequence_length, 3, image_h, image_w)

        real_world_extrinsics = others["extrinsics"]
        real_world_extrinsics_enc = extri_to_pose_encoding(real_world_extrinsics)
        real_world_extrinsics_enc = real_world_extrinsics_enc.reshape(batch_size, frame_count, self.cam_num, 7).permute(
            0, 2, 1, 3
        )
        pose_tokens = real_world_extrinsics_enc
        relative_pose_enc = pose_tokens[:, :, 0, :] if self.cam_num > 1 else pose_tokens
        relative_pose_enc_t, relative_pose_enc_r = relative_pose_enc[..., :3], relative_pose_enc[..., 3:]
        mean_t = torch.mean(relative_pose_enc_t, dim=-1, keepdim=True)
        std_t = torch.std(relative_pose_enc_t, dim=-1, keepdim=True).clamp(1e-5, 1e2)
        norm_relative_pose_enc_t = (relative_pose_enc_t - mean_t) / std_t
        relative_pose_enc = torch.cat([norm_relative_pose_enc_t / 10, relative_pose_enc_r], dim=-1)

        ext_enc = self.rel_pose_embed(relative_pose_enc)
        intrinsics = others["intrinsics"]
        k_cam = intrinsics.reshape(batch_size, frame_count, self.cam_num, 3, 3)[:, 0, :, :, :]
        intri_vec = torch.stack(
            [
                k_cam[..., 0, 0] / image_w,
                k_cam[..., 1, 1] / image_h,
                k_cam[..., 0, 2] / image_w,
                k_cam[..., 1, 2] / image_h,
            ],
            dim=-1,
        ).to(ext_enc.dtype)
        int_enc = self.intri_embed(intri_vec)
        relative_pose_enc = self.layer_norm(self.pose_intri_fuse(torch.cat([ext_enc, int_enc], dim=-1)))

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                batch_size * sequence_length,
                image_h // self.patch_size,
                image_w // self.patch_size,
                device=images.device,
            )
            pos = pos + 1
            pos_special = torch.zeros(batch_size * sequence_length, 2, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        last_pose_enc = aggregated_tokens_list[-1][..., 0, :]
        selected_layers = [aggregated_tokens_list[i] for i in self.selected_list]

        agg_layers_depths_tokens_list: List[torch.Tensor] = []
        agg_layers_camera_frame_tokens_list: List[torch.Tensor] = []
        agg_layers_camera_relative_tokens_list: List[torch.Tensor] = []

        for aggregated_tokens in selected_layers:
            n_cam_seq, n_frames, n_tokens, channels = aggregated_tokens.size()
            aggregated_tokens = self.batch_norm(aggregated_tokens.reshape(-1, n_tokens, channels).permute(0, 2, 1))
            aggregated_tokens = aggregated_tokens.permute(0, 2, 1).reshape(n_cam_seq, n_frames, n_tokens, channels)
            _, _, token_count, channels = aggregated_tokens.size()
            frame_tokens = aggregated_tokens.view(batch_size, -1, token_count, channels)
            frame_depth_tokens = frame_tokens[:, :, patch_start_idx:, :]

            frame_pose_tokens = frame_tokens[:, :, 0, :].unsqueeze(2)
            frame_relative_tokens = relative_pose_enc.unsqueeze(2).expand(batch_size, self.cam_num, frame_count, channels)
            frame_relative_tokens = frame_relative_tokens.reshape(batch_size, -1, 1, channels)
            last_frame_pose_enc = last_pose_enc.reshape(batch_size, -1, 1, channels)
            frame_tokens = torch.cat(
                [frame_relative_tokens, last_frame_pose_enc + frame_pose_tokens, frame_depth_tokens],
                dim=2,
            )

            atten_idx = 0
            merged_token_count = frame_tokens.shape[2]
            for _ in range(self.depth):
                frame_tokens, atten_idx = self._process_mv_attention(
                    frame_tokens, batch_size, sequence_length, merged_token_count, channels, atten_idx, pos
                )

            agg_frame_tokens = frame_tokens.view(batch_size, sequence_length, merged_token_count, channels)
            agg_layers_depths_tokens_list.append(agg_frame_tokens[..., 2:, :])
            agg_camera_frame_tokens = self.camera_tokens_agg(agg_frame_tokens[..., 1, :], "frame")
            agg_camera_relative_tokens = self.camera_tokens_agg(agg_frame_tokens[..., 0, :], "multiview")
            agg_layers_camera_frame_tokens_list.append(agg_camera_frame_tokens)
            agg_layers_camera_relative_tokens_list.append(agg_camera_relative_tokens)

        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                predictions["seq_enc_list"] = self.camera_head(agg_layers_camera_frame_tokens_list)
                predictions["mv_enc_list"] = self.camera_relative_head(agg_layers_camera_relative_tokens_list)
                predictions["mv_env"] = predictions["mv_enc_list"][-1]
                predictions["seq_enc"] = predictions["seq_enc_list"][-1]
                frame_extrinsic, _ = pose_encoding_to_extri_intri(predictions["seq_enc"], (image_h, image_w))
                relative_extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["mv_env"], (image_h, image_w))
                zeros = torch.zeros(
                    (batch_size, frame_count * self.cam_num, 1, 3),
                    device=images.device,
                    dtype=frame_extrinsic.dtype,
                )
                ones = torch.ones(
                    (batch_size, frame_count * self.cam_num, 1, 1),
                    device=images.device,
                    dtype=frame_extrinsic.dtype,
                )
                homo_tail = torch.cat([zeros, ones], dim=-1)
                relative_pose = relative_extrinsic.unsqueeze(1).expand(batch_size, frame_count, self.cam_num, 3, 4)
                relative_pose = relative_pose.reshape(batch_size, -1, 3, 4)
                relative_pose = torch.cat([relative_pose, homo_tail], dim=-2)
                frame_pose = frame_extrinsic.unsqueeze(2).expand(batch_size, frame_count, self.cam_num, 3, 4)
                frame_pose = frame_pose.reshape(batch_size, -1, 3, 4)
                frame_pose = torch.cat([frame_pose, homo_tail], dim=-2)
                predictions["extrinsic"] = relative_pose.matmul(frame_pose)[..., :3, :]
                predictions["intrinsic"] = intrinsic.unsqueeze(1).expand(batch_size, frame_count, self.cam_num, 3, 3)
                predictions["intrinsic"] = predictions["intrinsic"].reshape(batch_size, -1, 3, 3)

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(agg_layers_depths_tokens_list, images=images, patch_start_idx=0)
                predictions["depth"] = self._camera_major_to_time_major(depth, batch_size, frame_count)
                predictions["depth_conf"] = self._camera_major_to_time_major(depth_conf, batch_size, frame_count)

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(selected_layers, images=images, patch_start_idx=patch_start_idx)
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                selected_layers,
                images=images,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images_time_major
        return predictions
