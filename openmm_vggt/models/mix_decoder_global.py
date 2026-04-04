from typing import List

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from mmengine.registry import MODELS
from torch.utils.checkpoint import checkpoint

from .aggregator import Aggregator
from .pcdet_dynamic_voxel_vfe import PCDetDynamicVoxelVFE
from openmm_vggt.heads.camera_head import CameraHead
from openmm_vggt.heads.dpt_head import DPTHead
from openmm_vggt.heads.track_head import TrackHead
from openmm_vggt.layers.block import Block
from openmm_vggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from openmm_vggt.utils.pose_enc import extri_to_pose_encoding, pose_encoding_to_extri_intri


class SimplePatchFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.voxel_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, patch_tokens: torch.Tensor, voxel_features: torch.Tensor):
        voxel_features = self.voxel_proj(voxel_features)
        gate = self.gate(torch.cat([patch_tokens, voxel_features], dim=-1))
        return patch_tokens + gate * voxel_features


@MODELS.register_module()
class mix_decoder_global(nn.Module, PyTorchModelHubMixin):
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
        super().__init__()
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1", intermediate_layer_idx=[0, 1, 2, 3]) if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.camera_relative_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None

        self.cam_num = cam_num
        self.patch_size = patch_size
        self.rel_pose_embed = nn.Linear(7, 2048)
        self.intri_embed = nn.Linear(4, 2048)
        self.pose_intri_fuse = nn.Linear(4096, 2048)
        self.layer_norm = nn.LayerNorm([2048], eps=1e-05, elementwise_affine=True)
        self.rope = RotaryPositionEmbedding2D(frequency=100)
        self.position_getter = PositionGetter()
        self.batch_norm = nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True)
        self.depth = 1
        self.use_reentrant = False
        self.mv_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim * 2,
                    num_heads=16,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=0.01,
                    qk_norm=True,
                    rope=self.rope,
                )
                for _ in range(self.depth)
            ]
        )

        self.selected_list = [4, 11, 17, 23]
        grid_size = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
        ]
        self.voxel_encoder = PCDetDynamicVoxelVFE(
            num_point_features=4,
            voxel_size=voxel_size,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            use_norm=True,
            with_distance=False,
            use_absolute_xyz=True,
            num_filters=voxel_encoder_filters,
        )
        self.voxel_feature_proj = nn.Linear(self.voxel_encoder.get_output_feature_dim(), embed_dim * 2)
        self.register_buffer("fusion_voxel_size", torch.tensor(voxel_size, dtype=torch.float32), persistent=False)
        self.register_buffer("fusion_point_cloud_range", torch.tensor(point_cloud_range, dtype=torch.float32), persistent=False)
        self.patch_fusions = nn.ModuleList([SimplePatchFusion(embed_dim * 2) for _ in self.selected_list])

    def camera_tokens_agg(self, camera_tokens, token_type):
        if camera_tokens.dim() == 2:
            camera_tokens = camera_tokens.unsqueeze(-1)
        batch_size, steps, channels = camera_tokens.size()
        frame_count = steps // self.cam_num
        camera_tokens = camera_tokens.reshape(batch_size, self.cam_num, frame_count, channels)
        if token_type == "multiview":
            camera_tokens = torch.mean(camera_tokens, dim=2)
        elif token_type == "frame":
            camera_tokens = torch.mean(camera_tokens, dim=1)
        return camera_tokens.unsqueeze(2)

    def _process_mv_attention(self, tokens, batch_size, sequence_length, token_count, channels, mv_idx, pos=None):
        if tokens.shape != (batch_size, sequence_length * token_count, channels):
            tokens = tokens.view(batch_size, sequence_length, token_count, channels).view((batch_size, sequence_length * token_count, channels))
        if pos is not None and pos.shape != (batch_size, sequence_length * token_count, 2):
            pos = pos.view(batch_size, sequence_length, token_count, 2).view((batch_size, sequence_length * token_count, 2))
        if self.training:
            tokens = checkpoint(self.mv_blocks[mv_idx], tokens, pos, use_reentrant=self.use_reentrant)
        else:
            tokens = self.mv_blocks[mv_idx](tokens, pos=pos)
        return tokens, mv_idx + 1

    def _camera_major_to_time_major(self, tensor, batch_size, frame_count):
        shape = tensor.shape
        tensor = tensor.reshape(batch_size, self.cam_num, frame_count, *shape[2:])
        tensor = tensor.permute(0, 2, 1, *range(3, len(shape) + 1))
        return tensor.reshape(batch_size, frame_count * self.cam_num, *shape[2:])

    def _time_major_to_camera_major(self, tensor, batch_size, frame_count):
        shape = tensor.shape
        tensor = tensor.reshape(batch_size, frame_count, self.cam_num, *shape[2:])
        tensor = tensor.permute(0, 2, 1, *range(3, len(shape) + 1))
        return tensor.reshape(batch_size, frame_count * self.cam_num, *shape[2:])

    def _project_voxels_to_patches(self, voxel_coords, voxel_batch_ids, intrinsics, camera_to_world, patch_h, patch_w):
        xyz1 = torch.cat([voxel_coords, torch.ones_like(voxel_coords[..., :1])], dim=-1)
        world_to_camera = torch.inverse(camera_to_world)[voxel_batch_ids]
        cam_coords = torch.matmul(world_to_camera, xyz1[:, None, :, None]).squeeze(-1)[..., :3]
        img_coords = torch.matmul(intrinsics[voxel_batch_ids], cam_coords.unsqueeze(-1)).squeeze(-1)
        depth = img_coords[..., 2]
        safe_depth = torch.clamp(depth, min=1e-5)
        u = img_coords[..., 0] / safe_depth
        v = img_coords[..., 1] / safe_depth
        patch_x = torch.floor(u / self.patch_size).to(torch.long)
        patch_y = torch.floor(v / self.patch_size).to(torch.long)
        visible = (
            (depth > 1e-5)
            & (patch_x >= 0)
            & (patch_x < patch_w)
            & (patch_y >= 0)
            & (patch_y < patch_h)
        )
        return patch_y, patch_x, visible

    def _encode_voxels(self, points: torch.Tensor, point_mask: torch.Tensor):
        device = points.device
        batch_frames, _, _ = points.shape
        valid_indices = torch.nonzero(point_mask, as_tuple=False)
        if valid_indices.shape[0] == 0:
            empty_feat = points.new_zeros((0, self.voxel_feature_proj.in_features))
            empty_coord = torch.zeros((0, 4), dtype=torch.long, device=device)
            return empty_feat, empty_coord

        batch_ids = valid_indices[:, 0].to(points.dtype).unsqueeze(-1)
        selected_points = points[valid_indices[:, 0], valid_indices[:, 1]]
        flat_points = torch.cat([batch_ids, selected_points], dim=-1)
        voxel_features, voxel_coords = self.voxel_encoder(flat_points)
        return voxel_features, voxel_coords

    def _voxel_coords_to_centers(self, voxel_coords: torch.Tensor):
        xyz_index = voxel_coords[:, [3, 2, 1]].to(self.fusion_voxel_size.dtype)
        return (
            self.fusion_point_cloud_range[:3]
            + (xyz_index + 0.5) * self.fusion_voxel_size
        )

    def _prepare_patch_voxel_features(self, others: dict, batch_size: int, frame_count: int, image_hw):
        points = others["points"].reshape(batch_size * frame_count, others["points"].shape[2], others["points"].shape[3])
        point_mask = others["point_mask"].reshape(batch_size * frame_count, others["point_mask"].shape[2])
        intrinsics = others["intrinsics"].reshape(batch_size, frame_count, self.cam_num, 3, 3).reshape(batch_size * frame_count, self.cam_num, 3, 3)
        camera_to_world = others["camera_to_world"].reshape(batch_size, frame_count, self.cam_num, 4, 4).reshape(batch_size * frame_count, self.cam_num, 4, 4)

        voxel_features, voxel_coords = self._encode_voxels(points, point_mask)
        voxel_tokens = self.voxel_feature_proj(voxel_features)
        voxel_centers = self._voxel_coords_to_centers(voxel_coords)
        patch_h = image_hw[0] // self.patch_size
        patch_w = image_hw[1] // self.patch_size
        patch_count = patch_h * patch_w
        channels = voxel_tokens.shape[-1]
        if voxel_tokens.shape[0] == 0:
            return voxel_tokens.new_zeros((batch_size * frame_count, self.cam_num, patch_count, channels))

        voxel_batch_ids = voxel_coords[:, 0].long()
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
        flat_output_index = (
            ((flat_voxel_batch_ids * self.cam_num + flat_cam_ids) * patch_count) + flat_patch_index
        )

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

    def _mix_patch_tokens(self, patch_tokens: torch.Tensor, patch_voxel_features: torch.Tensor, layer_idx: int):
        batch_size, sequence_length, patch_count, channels = patch_tokens.shape
        frame_count = sequence_length // self.cam_num
        patch_tokens = patch_tokens.reshape(batch_size, frame_count, self.cam_num, patch_count, channels)
        flat_patch_tokens = patch_tokens.reshape(batch_size * frame_count, self.cam_num * patch_count, channels)
        flat_patch_voxel_features = patch_voxel_features.reshape(batch_size * frame_count, self.cam_num * patch_count, channels)
        fused_tokens = self.patch_fusions[layer_idx](flat_patch_tokens, flat_patch_voxel_features)
        fused_tokens = fused_tokens.reshape(batch_size, frame_count, self.cam_num, patch_count, channels)
        return fused_tokens.reshape(batch_size, sequence_length, patch_count, channels)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, others=None):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        batch_size, sequence_length, _, image_h, image_w = images.size()
        frame_count = sequence_length // self.cam_num
        images_time_major = images

        images = images.reshape(batch_size, frame_count, self.cam_num, 3, image_h, image_w)
        images = images.permute(0, 2, 1, 3, 4, 5).contiguous().reshape(batch_size * self.cam_num, frame_count, 3, image_h, image_w)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        images = images.view(batch_size, sequence_length, 3, image_h, image_w)

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
        K_cam = intrinsics.reshape(batch_size, frame_count, self.cam_num, 3, 3)[:, 0, :, :, :]
        intri_vec = torch.stack([
            K_cam[..., 0, 0] / image_w,
            K_cam[..., 1, 1] / image_h,
            K_cam[..., 0, 2] / image_w,
            K_cam[..., 1, 2] / image_h,
        ], dim=-1).to(ext_enc.dtype)
        int_enc = self.intri_embed(intri_vec)
        relative_pose_enc = self.layer_norm(self.pose_intri_fuse(torch.cat([ext_enc, int_enc], dim=-1)))

        pos = None
        if self.rope is not None:
            pos = self.position_getter(batch_size * sequence_length, image_h // self.patch_size, image_w // self.patch_size, device=images.device)
            pos = pos + 1
            pos_special = torch.zeros(batch_size * sequence_length, 2, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        last_pose_enc = aggregated_tokens_list[-1][..., 0, :]
        selected_layers = [aggregated_tokens_list[i] for i in self.selected_list]
        patch_voxel_features = self._prepare_patch_voxel_features(others, batch_size, frame_count, (image_h, image_w))

        agg_layers_depths_tokens_list: List[torch.Tensor] = []
        agg_layers_camera_frame_tokens_list: List[torch.Tensor] = []
        agg_layers_camera_relative_tokens_list: List[torch.Tensor] = []

        for layer_idx, aggregated_tokens in enumerate(selected_layers):
            n_cam_seq, n_frames, n_tokens, channels = aggregated_tokens.size()
            aggregated_tokens = self.batch_norm(
                aggregated_tokens.reshape(-1, n_tokens, channels).permute(0, 2, 1)
            ).permute(0, 2, 1).reshape(n_cam_seq, n_frames, n_tokens, channels)
            _, _, token_count, channels = aggregated_tokens.size()
            frame_tokens = aggregated_tokens.view(batch_size, -1, token_count, channels)
            frame_depth_tokens = frame_tokens[:, :, patch_start_idx:, :]
            frame_depth_tokens = self._camera_major_to_time_major(frame_depth_tokens, batch_size, frame_count)
            frame_depth_tokens = self._mix_patch_tokens(frame_depth_tokens, patch_voxel_features, layer_idx)
            frame_depth_tokens = self._time_major_to_camera_major(frame_depth_tokens, batch_size, frame_count)

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
