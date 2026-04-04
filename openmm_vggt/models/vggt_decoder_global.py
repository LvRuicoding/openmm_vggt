# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from .aggregator import Aggregator

from openmm_vggt.heads.camera_head import CameraHead,CameraHead_m,CameraHead_trans,CameraHead_decoder
from openmm_vggt.heads.dpt_head import DPTHead,DPTHead_m
from openmm_vggt.heads.track_head import TrackHead
from openmm_vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri,extri_to_pose_encoding
from openmm_vggt.layers.block import Block
from openmm_vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter

from openmm_vggt.layers.block import Block
from .utils.seperate_camera_head import CameraHead_seperate
from torch.utils.checkpoint import checkpoint
from mmengine.registry import MODELS


@MODELS.register_module()
class VGGT_decoder_global(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,cam_num=6):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        # self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1",intermediate_layer_idx=[0,1,2,3]) if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        # add seq_multiview atten
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.camera_relative_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.cam_num = cam_num
        self.rel_pose_embed = nn.Linear(7,2048)
        self.intri_embed = nn.Linear(4, 2048)  # fx/W, fy/H, cx/W, cy/H
        self.pose_intri_fuse = nn.Linear(4096, 2048)  # concat(extri_enc, intri_enc) -> 2048
        self.layer_norm = nn.LayerNorm([2048], eps=1e-05, elementwise_affine=True)
        self.rope = RotaryPositionEmbedding2D(frequency=100) 
        self.position_getter = PositionGetter() if self.rope is not None else None
        self.patch_size = 14
        self.batch_norm = nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True)
        depth = 1
        self.depth = depth
        self.use_reentrant = False
        self.mv_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim*2,
                    num_heads=16,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=0.01,
                    qk_norm=True,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )
        
    def camera_tokens_agg(self, camera_tokens, token_type, relative_pose_enc=None):
        """
        Aggregate camera tokens based on token type.
        
        Args:
            camera_tokens: Input tokens with shape [B, S, C] where S = f_num * cam_num
            token_type: "multiview" (aggregate across cameras) or "frame" (aggregate across time)
            relative_pose_enc: Optional pose encoding for multiview aggregation
            
        Returns:
            Aggregated tokens with shape [B, cam_num, 1, C] or [B, f_num, 1, C]
        """
        # Handle case where input is 2D (B, S) after indexing
        if camera_tokens.dim() == 2:
            B, S = camera_tokens.size()
            C = 1
            camera_tokens = camera_tokens.unsqueeze(-1)
        else:
            B, S, C = camera_tokens.size()
        
        f_num = S // self.cam_num

        # agg_frame_tokens is camera-major: [B, cam_num*f_num, C]
        # Reshape to (B, cam_num, f_num, C)
        camera_tokens = camera_tokens.reshape(B, self.cam_num, f_num, C)

        if token_type == "multiview":
            # Average across time steps: (B, cam_num, f_num, C) -> (B, cam_num, C)
            camera_tokens = torch.mean(camera_tokens, dim=2)
        elif token_type == "frame":
            # Average across cameras: (B, cam_num, f_num, C) -> (B, f_num, C)
            camera_tokens = torch.mean(camera_tokens, dim=1)

        return camera_tokens.unsqueeze(2)  # Add token dimension: (B, *, 1, C)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None,others = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length (n_time_steps * cam_num), 3: RGB channels, H: height, W: width
                For stereo temporal: S = n_time_steps * 2 (3 frames × 2 cameras)
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, S, _, H, W = images.size()
        
        # Reshape: S = n_time_steps * cam_num
        # For stereo: cam_num=2, so f_num = S // 2 (number of time steps)
        f_num = S // self.cam_num
        
        images_time_major = images  # keep time-major for output alignment

        # Data comes in time-major order: [t0_02, t0_03, t1_02, t1_03, ...]
        # Convert to camera-major order: [t0_02, t1_02, ..., t0_03, t1_03, ...]
        # for the aggregator's reshape(B*cam_num, f_num, ...) expectation
        images = images.reshape(B, f_num, self.cam_num, 3, H, W)  # [B, f_num, cam_num, 3, H, W]
        images = images.permute(0, 2, 1, 3, 4, 5)  # [B, cam_num, f_num, 3, H, W]
        images = images.reshape(B * self.cam_num, f_num, 3, H, W)  # [B*cam_num, f_num, 3, H, W]
        
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        images = images.view(B, S, 3, H, W)
        
        real_world_extrinics = others["extrinsics"]  # B S 3 4
        real_world_extrinics_enc = extri_to_pose_encoding(real_world_extrinics)  # B S 7(4+3)

        # Convert extrinsics from time-major to camera-major order
        # Input: [B, S, 7] where S = f_num * cam_num in time-major order
        # Output: [B, cam_num, f_num, 7] in camera-major order
        real_world_extrinics_enc = real_world_extrinics_enc.reshape(B, f_num, self.cam_num, 7)  # [B, f_num, cam_num, 7]
        real_world_extrinics_enc = real_world_extrinics_enc.permute(0, 2, 1, 3)  # [B, cam_num, f_num, 7]
        
        pose_tokens = real_world_extrinics_enc  # [B, cam_num, f_num, 7]
        
        if self.cam_num == 1:
            # For monocular sequences, each frame should keep its own pose token.
            relative_pose_enc = pose_tokens
        else:
            # For stereo rigs, camera-relative pose is assumed fixed across time.
            # Extract the first time step's pose for each camera: B cam_num 7
            relative_pose_enc = pose_tokens[:, :, 0, :]

        relative_pose_enc_T, relative_pose_enc_R = relative_pose_enc[..., :3], relative_pose_enc[..., 3:]

        # Input normalization
        mean_t = torch.mean(relative_pose_enc_T, dim=-1).unsqueeze(-1)
        std_t = torch.std(relative_pose_enc_T, dim=-1).unsqueeze(-1)
        std_t = torch.clamp(std_t, 1e-5, 1e2)
        norm_relative_pose_enc_T = (relative_pose_enc_T - mean_t) / std_t
        relative_pose_enc = torch.cat([norm_relative_pose_enc_T / 10, relative_pose_enc_R], dim=-1)
        
        ext_enc = self.rel_pose_embed(relative_pose_enc)  # B cam_num 2048

        # Fuse camera intrinsics: take first time step's intrinsics per camera
        K = others["intrinsics"]  # B S 3 3
        K_cam = K.reshape(B, f_num, self.cam_num, 3, 3)[:, 0, :, :, :]  # B cam_num 3 3
        intri_vec = torch.stack([
            K_cam[..., 0, 0] / W, K_cam[..., 1, 1] / H,
            K_cam[..., 0, 2] / W, K_cam[..., 1, 2] / H,
        ], dim=-1).to(ext_enc.dtype)  # B cam_num 4
        int_enc = self.intri_embed(intri_vec)  # B cam_num 2048
        relative_pose_enc = self.pose_intri_fuse(torch.cat([ext_enc, int_enc], dim=-1))  # B cam_num 2048

        relative_pose_enc = self.layer_norm(relative_pose_enc)

        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
            pos = pos + 1
            pos_special = torch.zeros(B * S, 2, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)  # B*S 742 2
            
        last_pose_enc = aggregated_tokens_list[-1][..., 0, :]  # B*cam_num f_num 2048
        selected_list = [4, 11, 17, 23]
        aggregated_tokens_list = [aggregated_tokens_list[i] for i in selected_list]
        layer_nums = len(aggregated_tokens_list)
        
        agg_layers_depths_tokens_list = []
        agg_layers_camera_frame_tokens_list = []
        agg_layers_camera_relative_tokens_list = []
        
        for layer in range(layer_nums):
            aggregated_tokens = aggregated_tokens_list[layer]
            n_cam_seq, n_frames, n_tokens, dim = aggregated_tokens.size()
            aggregated_tokens = self.batch_norm(
                aggregated_tokens.reshape(-1, n_tokens, dim).permute(0, 2, 1)
            ).permute(0, 2, 1).reshape(n_cam_seq, n_frames, n_tokens, dim)
            
            _, _, N, C = aggregated_tokens.size()

            # Keep camera-major order: aggregated_tokens is (B*cam_num, f_num, N, C)
            frame_tokens = aggregated_tokens.view(B, -1, N, C)  # B S N C (camera-major)

            frame_depth_tokens = frame_tokens[:, :, patch_start_idx:, :]  # B S (N-2) C
            frame_pose_tokens = frame_tokens[:, :, 0, :].unsqueeze(2)  # B S 1 C

            # Expand relative_pose_enc (B, cam_num, C) -> (B, S, 1, C) in camera-major order
            frame_relative_tokens = relative_pose_enc.unsqueeze(2).expand(B, self.cam_num, f_num, C)
            frame_relative_tokens = frame_relative_tokens.reshape(B, -1, 1, C)  # B S 1 C

            # last_pose_enc is (B*cam_num, f_num, C) -> (B, S, 1, C) camera-major
            last_frame_pose_enc = last_pose_enc.reshape(B, -1, 1, C)  # B S 1 C
            
            frame_tokens = torch.cat([frame_relative_tokens, last_frame_pose_enc + frame_pose_tokens, frame_depth_tokens], dim=2)

            atten_idx = 0
            frame_token_count = frame_tokens.shape[2]
            for _ in range(self.depth):
                frame_tokens, atten_idx = self._process_mv_attention(
                    frame_tokens, B, S, frame_token_count, C, atten_idx, pos
                )
            
            agg_frame_tokens = frame_tokens.view(B, S, frame_token_count, C)  # B S N C

            agg_layers_depths_tokens_list.append(agg_frame_tokens[..., 2:, :])  # 4x B S (N-2) C

            agg_camera_frame_tokens = self.camera_tokens_agg(agg_frame_tokens[..., 1, :], 'frame')  # B f_num 1 C
            agg_camera_relative_tokens = self.camera_tokens_agg(agg_frame_tokens[..., 0, :], 'multiview', relative_pose_enc)  # B cam_num 1 C

            agg_layers_camera_frame_tokens_list.append(agg_camera_frame_tokens)
            agg_layers_camera_relative_tokens_list.append(agg_camera_relative_tokens)

        # Convert outputs from camera-major back to time-major order
        # This ensures outputs match the input data order
        def convert_camera_major_to_time_major(tensor):
            """Convert tensor from camera-major [B, S, ...] to time-major order."""
            if tensor.shape[1] != S:
                return tensor
            # Reshape from [B, S, ...] to [B, cam_num, f_num, ...]
            shape = tensor.shape
            tensor = tensor.reshape(B, self.cam_num, f_num, *shape[2:])
            # Permute from [B, cam_num, f_num, ...] to [B, f_num, cam_num, ...]
            n_extra = len(shape) - 2
            tensor = tensor.permute(0, 2, 1, *range(3, 3 + n_extra))
            # Reshape back to [B, S, ...]
            tensor = tensor.reshape(B, S, *shape[2:])
            return tensor

        # Predictions
        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                predictions["seq_enc_list"] = self.camera_head(agg_layers_camera_frame_tokens_list)
                predictions["mv_enc_list"] = self.camera_relative_head(agg_layers_camera_relative_tokens_list)
                predictions["mv_env"] = predictions["mv_enc_list"][-1]
                predictions["seq_enc"] = predictions["seq_enc_list"][-1]
                
                if self.cam_num == 1:
                    predictions["pose_enc"] = predictions["seq_enc"]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["seq_enc"], (H, W))
                    predictions["extrinsic"] = extrinsic
                    predictions["intrinsic"] = intrinsic
                else:
                    frame_extrinsic, _ = pose_encoding_to_extri_intri(predictions["seq_enc"], (H, W))
                    relative_extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["mv_env"], (H, W))

                    zeros = torch.zeros((B, f_num * self.cam_num, 1, 3), device=images.device, dtype=frame_extrinsic.dtype)
                    ones = torch.ones((B, f_num * self.cam_num, 1, 1), device=images.device, dtype=frame_extrinsic.dtype)
                    homo_tail = torch.cat([zeros, ones], dim=-1)

                    relative_pose = relative_extrinsic.unsqueeze(1).expand(B, f_num, self.cam_num, 3, 4).reshape(B, -1, 3, 4)
                    relative_pose = torch.cat([relative_pose, homo_tail], dim=-2)
                    frame_pose = frame_extrinsic.unsqueeze(2).expand(B, f_num, self.cam_num, 3, 4).reshape(B, -1, 3, 4)
                    frame_pose = torch.cat([frame_pose, homo_tail], dim=-2)

                    extrinsic = relative_pose.matmul(frame_pose)[..., :3, :]
                    intrinsic = intrinsic.unsqueeze(1).expand(B, f_num, self.cam_num, 3, 3).reshape(B, -1, 3, 3)
                    
                    predictions["extrinsic"] = extrinsic
                    predictions["intrinsic"] = intrinsic

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    agg_layers_depths_tokens_list, images=images, patch_start_idx=0
                )
                # Convert from camera-major to time-major order
                depth = convert_camera_major_to_time_major(depth)
                depth_conf = convert_camera_major_to_time_major(depth_conf)
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images_time_major
        return predictions
    def _process_mv_attention(self, tokens, B, S, P, C, mv_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view((B, S * P, C))

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view((B, S * P, 2))
        
                # by default, self.aa_block_size=1, which processes one block at a time
 
        if self.training:
            tokens = checkpoint(self.mv_blocks[mv_idx], tokens, pos, use_reentrant=self.use_reentrant)
        else:
            tokens = self.mv_blocks[mv_idx](tokens, pos=pos)
        # print(tokens.size())
        mv_idx += 1

        return tokens, mv_idx

@MODELS.register_module(name="VGGT_decoder_raw_global")
class VGGT_decoder_raw(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None,others = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        # aggregated_tokens_list: 24 *torchsize(1 36 3 280 518)
        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                # seq_enc_list,multiview_enc_list = self.camera_head(aggregated_tokens_list)
                # predictions["pose_enc"] = {"seq":seq_enc_list[-1],"multiview":multiview_enc_list[-1]}  # pose encoding of the last iteration
                # predictions["pose_enc_list"] = {"seq":seq_enc_list,"multiview":multiview_enc_list}
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        # if not self.training:
        predictions["images"] = images  # store the images for visualization during inference
        # import sys
        # sys.path.append("/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/vggt")
        # print("Converting pose encoding to extrinsic and intrinsic matrices...")
        # extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        # predictions["extrinsic"] = extrinsic
        # predictions["intrinsic"] = intrinsic

        # print("Processing model outputs...")
        # for key in predictions.keys():
        #     if isinstance(predictions[key], torch.Tensor):
        #         predictions[key] = predictions[key].detach().cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

        # from demo_viser import viser_wrapper
        # viser_wrapper(predictions,use_point_map=False)
        # # viser_wrapper(pred_dict,use_point_map=False)
        # assert 1==0
        return predictions
