from typing import List

import torch
import torch.nn as nn
from mmengine.registry import MODELS

from openmm_vggt.utils.pose_enc import extri_to_pose_encoding, pose_encoding_to_extri_intri

from .aggregator_window_attn_early import EarlyFusionAggregator
from ._mix_decoder_global_base import SimplePatchFusion, _MixDecoderGlobalBase


@MODELS.register_module()
class mix_decoder_global_early(_MixDecoderGlobalBase):
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
        self.aggregator = EarlyFusionAggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.early_voxel_feature_proj = nn.Linear(self.voxel_encoder.get_output_feature_dim(), embed_dim)
        self.early_patch_fusion = SimplePatchFusion(embed_dim)

    def _prepare_early_patch_voxel_features(self, others: dict, batch_size: int, frame_count: int, image_hw):
        points = others["points"].reshape(batch_size * frame_count, others["points"].shape[2], others["points"].shape[3])
        point_mask = others["point_mask"].reshape(batch_size * frame_count, others["point_mask"].shape[2])
        intrinsics = others["intrinsics"].reshape(batch_size, frame_count, self.cam_num, 3, 3).reshape(batch_size * frame_count, self.cam_num, 3, 3)
        camera_to_world = others["camera_to_world"].reshape(batch_size, frame_count, self.cam_num, 4, 4).reshape(batch_size * frame_count, self.cam_num, 4, 4)
        lidar_to_world = None
        if "lidar_to_world" in others:
            lidar_to_world = others["lidar_to_world"].reshape(batch_size * frame_count, 4, 4)

        voxel_features, voxel_coords = self._encode_voxels(points, point_mask, lidar_to_world=lidar_to_world)
        voxel_tokens = self.early_voxel_feature_proj(voxel_features)
        voxel_centers = self._voxel_coords_to_centers(voxel_coords)
        patch_h = image_hw[0] // self.patch_size
        patch_w = image_hw[1] // self.patch_size
        patch_count = patch_h * patch_w
        channels = voxel_tokens.shape[-1]
        if voxel_tokens.shape[0] == 0:
            return voxel_tokens.new_zeros((batch_size * frame_count, self.cam_num, patch_count, channels))

        voxel_batch_ids = voxel_coords[:, 0].long()
        if lidar_to_world is not None:
            voxel_centers = self._local_voxel_centers_to_world(voxel_centers, voxel_batch_ids, lidar_to_world)
        patch_y, patch_x, visible = self._project_voxels_to_patches(
            voxel_centers,
            voxel_batch_ids,
            intrinsics,
            camera_to_world,
            patch_h,
            patch_w,
        )

        cam_ids = torch.arange(self.cam_num, device=voxel_tokens.device, dtype=torch.long).unsqueeze(0).expand_as(patch_y)
        flat_visible = visible.reshape(-1)
        flat_patch_index = (patch_y * patch_w + patch_x).reshape(-1)
        flat_voxel_batch_ids = voxel_batch_ids.unsqueeze(1).expand(-1, self.cam_num).reshape(-1)
        flat_cam_ids = cam_ids.reshape(-1)
        flat_features = voxel_tokens.unsqueeze(1).expand(-1, self.cam_num, -1).reshape(-1, channels)

        flat_output_size = batch_size * frame_count * self.cam_num * patch_count
        flat_output_index = ((flat_voxel_batch_ids * self.cam_num + flat_cam_ids) * patch_count) + flat_patch_index

        flat_patch_voxel_features = voxel_tokens.new_zeros((flat_output_size, channels))
        flat_patch_voxel_counts = voxel_tokens.new_zeros((flat_output_size, 1))
        valid_output_index = flat_output_index[flat_visible]
        flat_patch_voxel_features.index_add_(0, valid_output_index, flat_features[flat_visible])
        flat_patch_voxel_counts.index_add_(
            0,
            valid_output_index,
            torch.ones((valid_output_index.shape[0], 1), device=voxel_tokens.device, dtype=voxel_tokens.dtype),
        )

        flat_patch_voxel_features = flat_patch_voxel_features / flat_patch_voxel_counts.clamp_min(1.0)
        return flat_patch_voxel_features.reshape(batch_size * frame_count, self.cam_num, patch_count, channels)

    def _apply_early_patch_fusion(
        self,
        patch_tokens: torch.Tensor,
        patch_voxel_features: torch.Tensor,
        batch_size: int,
        frame_count: int,
    ) -> torch.Tensor:
        _, cam_num, patch_count, channels = patch_voxel_features.shape
        flat_patch_voxel_features = patch_voxel_features.reshape(batch_size, frame_count, cam_num, patch_count, channels)
        flat_patch_voxel_features = flat_patch_voxel_features.permute(0, 2, 1, 3, 4).reshape(-1, patch_count, channels)
        if patch_tokens.shape != flat_patch_voxel_features.shape:
            raise ValueError(
                f"Early fusion shape mismatch: patch tokens {tuple(patch_tokens.shape)} "
                f"vs voxel features {tuple(flat_patch_voxel_features.shape)}"
            )
        return self.early_patch_fusion(patch_tokens, flat_patch_voxel_features)

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
        early_patch_voxel_features = self._prepare_early_patch_voxel_features(others, batch_size, frame_count, (image_h, image_w))
        patch_tokens = self._apply_early_patch_fusion(patch_tokens, early_patch_voxel_features, batch_size, frame_count)
        aggregated_tokens_list, patch_start_idx = self.aggregator(images, patch_tokens=patch_tokens)
        images = images.view(batch_size, sequence_length, 3, image_h, image_w)

        real_world_extrinsics = others["extrinsics"]
        real_world_extrinsics_enc = extri_to_pose_encoding(real_world_extrinsics)
        real_world_extrinsics_enc = real_world_extrinsics_enc.reshape(batch_size, frame_count, self.cam_num, 7).permute(0, 2, 1, 3)
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
