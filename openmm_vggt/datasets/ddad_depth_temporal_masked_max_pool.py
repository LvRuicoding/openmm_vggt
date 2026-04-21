#!/usr/bin/env python3
"""DDAD temporal dataset variant using masked max-pooling depth downsampling."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mmengine.registry import DATASETS

from .ddad_depth_temporal import (
    DDADDepthTemporalDataset,
    _preprocess_pil_rgb,
    DEFAULT_CAMERA_NAMES,
)
from .kitti_local_utils import resize_intrinsics


def _resize_depth_map_masked_max_pool(depth_map: np.ndarray, out_hw: Tuple[int, int]) -> torch.Tensor:
    out_h, out_w = out_hw
    depth_tensor = torch.from_numpy(depth_map.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    valid_mask = depth_tensor > 0.0
    masked_depth = torch.where(valid_mask, depth_tensor, depth_tensor.new_full((), float("-inf")))
    pooled = F.adaptive_max_pool2d(masked_depth, output_size=(out_h, out_w)).squeeze(0).squeeze(0)
    pooled[~torch.isfinite(pooled)] = -1.0
    return pooled


@DATASETS.register_module()
class DDADDepthTemporalMaskedMaxPoolDataset(DDADDepthTemporalDataset):
    """DDAD temporal dataset using masked max-pooling for depth downsampling.

    Sparse depth is downsampled from raw resolution to ``image_size`` by taking
    the maximum valid depth in each adaptive output bin. Empty bins are marked
    as ``-1`` to preserve the original invalid-depth convention.
    """

    def __init__(self, *args, camera_names=DEFAULT_CAMERA_NAMES, **kwargs) -> None:
        super().__init__(*args, camera_names=camera_names, **kwargs)

    def __getitem__(self, index: int):
        scene_idx, last_local_pos = self.samples[index]
        scene_entries = self.scene_to_global[scene_idx]

        raw_positions = [
            last_local_pos - (self.n_time_steps - 1 - t) * self.stride
            for t in range(self.n_time_steps)
        ]
        clamped_positions = [max(0, pos) for pos in raw_positions]

        images_list: List[torch.Tensor] = []
        depths_list: List[torch.Tensor] = []
        extrinsics_list: List[torch.Tensor] = []
        intrinsics_list: List[torch.Tensor] = []
        camera_to_world_list: List[torch.Tensor] = []
        lidar_to_world_list: List[torch.Tensor] = []
        lidar_points_list: List[torch.Tensor] = []
        lidar_mask_list: List[torch.Tensor] = []

        first_sample_idx = scene_entries[clamped_positions[0]][0]
        last_sample_idx = scene_entries[clamped_positions[-1]][0]
        sequence_name = f"{self.scene_names[scene_idx]}_{first_sample_idx:06d}_{last_sample_idx:06d}"

        for t_idx, local_pos in enumerate(clamped_positions):
            sample_idx_in_scene, global_idx = scene_entries[local_pos]
            sample = self.ddad[global_idx][0]
            datum_lookup = {datum["datum_name"]: datum for datum in sample}

            is_last = t_idx == self.n_time_steps - 1
            for cam_name in self.camera_names:
                cam_datum = datum_lookup[cam_name]
                rgb = cam_datum["rgb"]
                orig_hw = (rgb.height, rgb.width)

                images_list.append(_preprocess_pil_rgb(rgb, self.image_size))

                if is_last and "depth" in cam_datum:
                    depths_list.append(_resize_depth_map_masked_max_pool(cam_datum["depth"], self.image_size))
                else:
                    depths_list.append(torch.full(self.image_size, -1.0, dtype=torch.float32))

                pose_wc = cam_datum["pose"]
                world_to_camera = pose_wc.inverse().matrix.astype(np.float32)
                camera_to_world = pose_wc.matrix.astype(np.float32)
                extrinsics_list.append(torch.from_numpy(world_to_camera[:3, :4]))
                camera_to_world_list.append(torch.from_numpy(camera_to_world))

                intrinsics = np.asarray(cam_datum["intrinsics"], dtype=np.float32)
                intrinsics = resize_intrinsics(intrinsics, orig_hw=orig_hw, out_hw=self.image_size)
                intrinsics_list.append(torch.from_numpy(intrinsics))

            if self.return_lidar:
                lidar_datum = datum_lookup[self.lidar_name]
                lidar_to_world_list.append(torch.from_numpy(lidar_datum["pose"].matrix.astype(np.float32)))
                lidar_points, lidar_mask = self._load_world_lidar_points(lidar_datum)
                lidar_points_list.append(lidar_points)
                lidar_mask_list.append(lidar_mask)

        output = {
            "images": torch.stack(images_list, dim=0),
            "depths": torch.stack(depths_list, dim=0),
            "extrinsics": torch.stack(extrinsics_list, dim=0),
            "intrinsics": torch.stack(intrinsics_list, dim=0),
            "sequence_name": sequence_name,
        }
        if self.return_lidar:
            output["camera_to_world"] = torch.stack(camera_to_world_list, dim=0)
            output["lidar_to_world"] = torch.stack(lidar_to_world_list, dim=0)
            output["points"] = torch.stack(lidar_points_list, dim=0)
            output["point_mask"] = torch.stack(lidar_mask_list, dim=0)
        return output
