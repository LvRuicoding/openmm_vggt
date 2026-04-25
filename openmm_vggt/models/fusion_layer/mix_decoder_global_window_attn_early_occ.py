from __future__ import annotations

from typing import List, Tuple

import torch
from mmengine.registry import MODELS

from openmm_vggt.heads.occupancy_head import OccupancyHead
from openmm_vggt.utils.pose_enc import extri_to_pose_encoding, pose_encoding_to_extri_intri

from .mix_decoder_global_window_attn_early import mix_decoder_global_window_attn_early


@MODELS.register_module()
class mix_decoder_global_window_attn_early_occ(mix_decoder_global_window_attn_early):
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
            fusion_window_size=fusion_window_size,
            fusion_shift_size=fusion_shift_size,
            fusion_window_stride=fusion_window_stride,
            fusion_num_heads=fusion_num_heads,
            fusion_mlp_ratio=fusion_mlp_ratio,
            fusion_attn_backend=fusion_attn_backend,
        )
        self.occupancy_head = None if occupancy_head is None else OccupancyHead(
            token_dim=2 * embed_dim,
            patch_size=patch_size,
            **occupancy_head,
        )
        if self.occupancy_head is not None and self.depth_head is None:
            raise ValueError("occupancy_head requires enable_depth=True")

    def _forward_from_aggregator_outputs(
        self,
        aggregated_tokens_list,
        patch_start_idx: int,
        images: torch.Tensor,
        images_time_major: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        frame_count: int,
        image_h: int,
        image_w: int,
        query_points: torch.Tensor = None,
        others=None,
    ):
        real_world_extrinsics = others["extrinsics"]
        real_world_extrinsics_enc = extri_to_pose_encoding(real_world_extrinsics)
        real_world_extrinsics_enc = real_world_extrinsics_enc.reshape(batch_size, frame_count, self.cam_num, 7).permute(0, 2, 1, 3)
        pose_tokens = real_world_extrinsics_enc
        relative_pose_enc = pose_tokens[:, :, 0, :] if self.cam_num > 1 else pose_tokens
        relative_pose_enc_T, relative_pose_enc_R = relative_pose_enc[..., :3], relative_pose_enc[..., 3:]
        mean_t = torch.mean(relative_pose_enc_T, dim=-1, keepdim=True)
        std_t = torch.std(relative_pose_enc_T, dim=-1, keepdim=True).clamp(1e-5, 1e2)
        norm_relative_pose_enc_T = (relative_pose_enc_T - mean_t) / std_t
        relative_pose_enc = torch.cat([norm_relative_pose_enc_T / 10, relative_pose_enc_R], dim=-1)

        ext_enc = self.rel_pose_embed(relative_pose_enc)
        intrinsics = others["intrinsics"]
        k_cam = intrinsics.reshape(batch_size, frame_count, self.cam_num, 3, 3)[:, 0, :, :, :]
        intri_vec = torch.stack([
            k_cam[..., 0, 0] / image_w,
            k_cam[..., 1, 1] / image_h,
            k_cam[..., 0, 2] / image_w,
            k_cam[..., 1, 2] / image_h,
        ], dim=-1).to(ext_enc.dtype)
        int_enc = self.intri_embed(intri_vec)
        relative_pose_enc = self.layer_norm(self.pose_intri_fuse(torch.cat([ext_enc, int_enc], dim=-1)))

        pos = None
        if self.rope is not None:
            pos = self.position_getter(batch_size * sequence_length, image_h // self.patch_size, image_w // self.patch_size, device=images.device)
            pos = pos + 1
            pos_special = torch.zeros(batch_size * sequence_length, 2, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        final_layer_fusion_inputs = self._prepare_final_layer_fusion_inputs(
            others=others,
            batch_size=batch_size,
            frame_count=frame_count,
            image_hw=(image_h, image_w),
        )
        aggregated_tokens_list = self._fuse_final_aggregator_layer(
            aggregated_tokens_list=aggregated_tokens_list,
            patch_start_idx=patch_start_idx,
            fusion_inputs=final_layer_fusion_inputs,
            image_hw=(image_h, image_w),
        )

        last_pose_enc = aggregated_tokens_list[-1][..., 0, :]
        selected_layers = [aggregated_tokens_list[i] for i in self.selected_list]

        agg_layers_depths_tokens_list: List[torch.Tensor] = []
        agg_layers_camera_frame_tokens_list: List[torch.Tensor] = []
        agg_layers_camera_relative_tokens_list: List[torch.Tensor] = []

        for aggregated_tokens in selected_layers:
            n_cam_seq, n_frames, n_tokens, channels = aggregated_tokens.size()
            aggregated_tokens = self.batch_norm(
                aggregated_tokens.reshape(-1, n_tokens, channels).permute(0, 2, 1)
            ).permute(0, 2, 1).reshape(n_cam_seq, n_frames, n_tokens, channels)
            _, _, token_count, channels = aggregated_tokens.size()
            frame_tokens = aggregated_tokens.view(batch_size, -1, token_count, channels)
            frame_depth_tokens = frame_tokens[:, :, patch_start_idx:, :]

            frame_pose_tokens = frame_tokens[:, :, 0, :].unsqueeze(2)
            frame_relative_tokens = relative_pose_enc.unsqueeze(2).expand(batch_size, self.cam_num, frame_count, channels).reshape(batch_size, -1, 1, channels)
            last_frame_pose_enc = last_pose_enc.reshape(batch_size, -1, 1, channels)
            frame_tokens = torch.cat([frame_relative_tokens, last_frame_pose_enc + frame_pose_tokens, frame_depth_tokens], dim=2)

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
                zeros = torch.zeros((batch_size, frame_count * self.cam_num, 1, 3), device=images.device, dtype=frame_extrinsic.dtype)
                ones = torch.ones((batch_size, frame_count * self.cam_num, 1, 1), device=images.device, dtype=frame_extrinsic.dtype)
                homo_tail = torch.cat([zeros, ones], dim=-1)
                relative_pose = relative_extrinsic.unsqueeze(1).expand(batch_size, frame_count, self.cam_num, 3, 4).reshape(batch_size, -1, 3, 4)
                relative_pose = torch.cat([relative_pose, homo_tail], dim=-2)
                frame_pose = frame_extrinsic.unsqueeze(2).expand(batch_size, frame_count, self.cam_num, 3, 4).reshape(batch_size, -1, 3, 4)
                frame_pose = torch.cat([frame_pose, homo_tail], dim=-2)
                predictions["extrinsic"] = relative_pose.matmul(frame_pose)[..., :3, :]
                predictions["intrinsic"] = intrinsic.unsqueeze(1).expand(batch_size, frame_count, self.cam_num, 3, 3).reshape(batch_size, -1, 3, 3)

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(agg_layers_depths_tokens_list, images=images, patch_start_idx=0)
                predictions["depth"] = self._camera_major_to_time_major(depth, batch_size, frame_count)
                predictions["depth_conf"] = self._camera_major_to_time_major(depth_conf, batch_size, frame_count)
                if self.occupancy_head is not None and others is not None:
                    last_frame_tokens = agg_layers_depths_tokens_list[-1][:, -self.cam_num :, :, :]
                    last_frame_depth = predictions["depth"][:, -self.cam_num :, ...]
                    if last_frame_depth.ndim == 5 and last_frame_depth.shape[2] == 1:
                        last_frame_depth = last_frame_depth.squeeze(2)
                    elif last_frame_depth.ndim == 5 and last_frame_depth.shape[-1] == 1:
                        last_frame_depth = last_frame_depth.squeeze(-1)
                    predictions["occupancy_logits"] = self.occupancy_head(
                        last_frame_tokens=last_frame_tokens,
                        last_frame_depth=last_frame_depth,
                        intrinsics=others["intrinsics"][:, -self.cam_num :, :, :],
                        camera_to_world=others["camera_to_world"][:, -self.cam_num :, :, :],
                        lidar_to_world=others["lidar_to_world"][:, -1, :, :],
                    )

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(selected_layers, images=images, patch_start_idx=patch_start_idx)
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(selected_layers, images=images, patch_start_idx=patch_start_idx, query_points=query_points)
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images_time_major
        return predictions
