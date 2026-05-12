#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from mmengine.registry import DATASETS

from .kitti_local_utils import (
    load_camera_transform_imu_to_rectified,
    load_oxts_poses,
    load_rectified_intrinsics,
    load_transform_imu_to_velodyne,
    preprocess_depth_png,
    preprocess_rgb_like_demo,
    resize_intrinsics,
    resolve_kitti_depth_root,
)
from .kitti_semantic_occ import (
    DEFAULT_SPLITS,
    SEMANTIC_KITTI_RAW_MAPPING,
    KITTISemanticOccupancyDataset,
    _estimate_contiguous_offset,
    _load_semantic_poses,
)


@dataclass
class _DepthSequenceRecord:
    sequence_id: str
    raw_drive_name: str
    raw_root: Path
    semantic_root: Path
    raw_hw: Tuple[int, int]
    frame_ids: List[str]
    raw_frame_ids: Dict[str, str]
    image_paths_02: Dict[str, Path]
    image_paths_03: Dict[str, Path]
    extrinsics_02: Dict[str, np.ndarray]
    extrinsics_03: Dict[str, np.ndarray]
    intrinsics_02: np.ndarray
    intrinsics_03: np.ndarray
    world_velodyne_poses: Dict[str, np.ndarray]


@DATASETS.register_module()
class KITTISemanticOccupancyDepthDataset(KITTISemanticOccupancyDataset):
    """SemanticKITTI occupancy samples with KITTI depth-completion supervision.

    Sampling stays identical to ``KITTISemanticOccupancyDataset``: the split,
    sequence list, and dense voxel target checks are all driven by
    SemanticKITTI. KITTI depth-completion PNGs are only used as an additional
    target for the selected samples.
    """

    def __init__(
        self,
        *args,
        depth_root: str,
        require_depth_gt: bool = True,
        **kwargs,
    ) -> None:
        self.depth_root = resolve_kitti_depth_root(depth_root)
        self.require_depth_gt = bool(require_depth_gt)
        self.depth_gt_paths = self._index_depth_gt_paths()
        super().__init__(*args, **kwargs)

    def _load_sequences(self) -> List[_DepthSequenceRecord]:
        records: List[_DepthSequenceRecord] = []
        for sequence_id in self.sequences:
            semantic_seq_root = self.semantic_root / sequence_id
            raw_drive_rel = SEMANTIC_KITTI_RAW_MAPPING[sequence_id]
            raw_drive_root = self.raw_root / raw_drive_rel
            calib_root = raw_drive_root.parent
            if not semantic_seq_root.is_dir() or not raw_drive_root.is_dir():
                if self.strict:
                    raise FileNotFoundError(f"Missing sequence pair: {semantic_seq_root} | {raw_drive_root}")
                continue

            oxts_poses = load_oxts_poses(raw_drive_root)
            t_imu_velo = load_transform_imu_to_velodyne(calib_root)
            intrinsics_02 = load_rectified_intrinsics(calib_root / "calib_cam_to_cam.txt", "image_02").astype(np.float32)
            intrinsics_03 = load_rectified_intrinsics(calib_root / "calib_cam_to_cam.txt", "image_03").astype(np.float32)
            t_cam_imu_02 = load_camera_transform_imu_to_rectified(calib_root, "image_02")
            t_cam_imu_03 = load_camera_transform_imu_to_rectified(calib_root, "image_03")

            image_paths_02 = {path.stem: path for path in sorted((raw_drive_root / "image_02" / "data").glob("*.png"))}
            image_paths_03 = {path.stem: path for path in sorted((raw_drive_root / "image_03" / "data").glob("*.png"))}
            semantic_velodyne = {path.stem: path for path in sorted((semantic_seq_root / "velodyne").glob("*.bin"))}
            semantic_labels = {path.stem: path for path in sorted((semantic_seq_root / "labels").glob("*.label"))}

            semantic_frame_ids = sorted(semantic_velodyne)
            if sequence_id not in DEFAULT_SPLITS["test"]:
                semantic_frame_ids = [frame_id for frame_id in semantic_frame_ids if frame_id in semantic_labels]
            raw_frame_ids = sorted(set(image_paths_02) & set(image_paths_03) & set(oxts_poses))
            if not semantic_frame_ids or not raw_frame_ids:
                continue

            raw_positions = np.stack(
                [(oxts_poses[raw_frame_id] @ t_imu_velo)[:3, 3] for raw_frame_id in raw_frame_ids],
                axis=0,
            )
            semantic_poses = _load_semantic_poses(semantic_seq_root / "poses.txt")
            semantic_pose_count = min(semantic_poses.shape[0], len(semantic_frame_ids))
            semantic_frame_ids = semantic_frame_ids[:semantic_pose_count]
            semantic_poses = semantic_poses[:semantic_pose_count]
            if len(raw_frame_ids) < len(semantic_frame_ids):
                continue
            offset = _estimate_contiguous_offset(semantic_poses, raw_positions)
            mapped_raw_ids = raw_frame_ids[offset : offset + len(semantic_frame_ids)]
            if len(mapped_raw_ids) != len(semantic_frame_ids):
                continue

            extrinsics_02: Dict[str, np.ndarray] = {}
            extrinsics_03: Dict[str, np.ndarray] = {}
            world_velodyne_poses: Dict[str, np.ndarray] = {}
            image_paths_02_sem: Dict[str, Path] = {}
            image_paths_03_sem: Dict[str, Path] = {}
            raw_frame_ids_sem: Dict[str, str] = {}
            for semantic_frame_id, raw_frame_id in zip(semantic_frame_ids, mapped_raw_ids):
                t_world_imu = oxts_poses[raw_frame_id]
                t_imu_world = np.linalg.inv(t_world_imu)
                extrinsics_02[semantic_frame_id] = (t_cam_imu_02 @ t_imu_world)[:3, :4].astype(np.float32)
                extrinsics_03[semantic_frame_id] = (t_cam_imu_03 @ t_imu_world)[:3, :4].astype(np.float32)
                world_velodyne_poses[semantic_frame_id] = (t_world_imu @ t_imu_velo).astype(np.float32)
                image_paths_02_sem[semantic_frame_id] = image_paths_02[raw_frame_id]
                image_paths_03_sem[semantic_frame_id] = image_paths_03[raw_frame_id]
                raw_frame_ids_sem[semantic_frame_id] = raw_frame_id

            first_path = image_paths_02_sem[semantic_frame_ids[0]]
            raw_w, raw_h = Image.open(first_path).size

            records.append(
                _DepthSequenceRecord(
                    sequence_id=sequence_id,
                    raw_drive_name=raw_drive_rel.split("/")[-1],
                    raw_root=raw_drive_root,
                    semantic_root=semantic_seq_root,
                    raw_hw=(raw_h, raw_w),
                    frame_ids=semantic_frame_ids,
                    raw_frame_ids=raw_frame_ids_sem,
                    image_paths_02=image_paths_02_sem,
                    image_paths_03=image_paths_03_sem,
                    extrinsics_02=extrinsics_02,
                    extrinsics_03=extrinsics_03,
                    intrinsics_02=intrinsics_02,
                    intrinsics_03=intrinsics_03,
                    world_velodyne_poses=world_velodyne_poses,
                )
            )

        records = sorted(records, key=lambda item: item.sequence_id)
        if self.max_sequences is not None:
            records = records[: self.max_sequences]
        return records

    def _index_depth_gt_paths(self) -> Dict[Tuple[str, str, str], Path]:
        paths: Dict[Tuple[str, str, str], Path] = {}
        for split_name in ("train", "val"):
            split_root = self.depth_root / split_name
            if not split_root.is_dir():
                continue
            for drive_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
                for camera_name in ("image_02", "image_03"):
                    gt_dir = drive_dir / "proj_depth" / "groundtruth" / camera_name
                    if not gt_dir.is_dir():
                        continue
                    for gt_path in sorted(gt_dir.glob("*.png")):
                        paths[(drive_dir.name, camera_name, gt_path.stem)] = gt_path
        if not paths:
            raise FileNotFoundError(f"No KITTI depth GT pngs found under {self.depth_root}")
        return paths

    def _get_depth_gt_path(self, record: _DepthSequenceRecord, semantic_frame_id: str, camera_name: str) -> Optional[Path]:
        raw_frame_id = record.raw_frame_ids[semantic_frame_id]
        return self.depth_gt_paths.get((record.raw_drive_name, camera_name, raw_frame_id))

    def _format_missing_depth_message(self, record: _DepthSequenceRecord, semantic_frame_id: str) -> str:
        raw_frame_id = record.raw_frame_ids[semantic_frame_id]
        return (
            "Missing KITTI depth GT for SemanticKITTI sample "
            f"sequence={record.sequence_id}, semantic_frame={semantic_frame_id}, "
            f"raw_drive={record.raw_drive_name}, raw_frame={raw_frame_id}, depth_root={self.depth_root}"
        )

    def _build_samples(self) -> List[Tuple[int, int]]:
        samples = super()._build_samples()
        if self.require_depth_gt:
            kept_samples: List[Tuple[int, int]] = []
            missing_samples: List[dict[str, object]] = []
            for record_idx, last_idx in samples:
                record = self.records[record_idx]
                last_frame_id = record.frame_ids[last_idx]
                missing_cameras = [
                    camera_name
                    for camera_name in ("image_02", "image_03")
                    if self._get_depth_gt_path(record, last_frame_id, camera_name) is None
                ]
                if missing_cameras:
                    missing_samples.append(
                        {
                            "sequence_id": record.sequence_id,
                            "semantic_frame_id": last_frame_id,
                            "raw_drive": record.raw_drive_name,
                            "raw_frame_id": record.raw_frame_ids[last_frame_id],
                            "missing_cameras": tuple(missing_cameras),
                        }
                    )
                    continue
                kept_samples.append((record_idx, last_idx))
            self.depth_missing_samples = missing_samples
            self.depth_kept_samples = len(kept_samples)
            return kept_samples
        self.depth_missing_samples = []
        self.depth_kept_samples = len(samples)
        return samples

    def _load_depth_or_invalid(
        self,
        record: _DepthSequenceRecord,
        semantic_frame_id: str,
        camera_name: str,
        use_gt: bool,
    ) -> torch.Tensor:
        if not use_gt:
            return torch.full(self.image_size, -1.0, dtype=torch.float32)
        gt_path = self._get_depth_gt_path(record, semantic_frame_id, camera_name)
        if gt_path is None:
            return torch.full(self.image_size, -1.0, dtype=torch.float32)
        return preprocess_depth_png(gt_path, self.image_size)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record_idx, last_idx = self.samples[index]
        record = self.records[record_idx]

        raw_indices = [
            last_idx - (self.n_time_steps - 1 - t) * self.stride
            for t in range(self.n_time_steps)
        ]
        clamped_indices = [max(0, idx) for idx in raw_indices]
        time_frame_ids = [record.frame_ids[idx] for idx in clamped_indices]

        images_list: List[torch.Tensor] = []
        depths_list: List[torch.Tensor] = []
        extrinsics_list: List[torch.Tensor] = []
        intrinsics_list: List[torch.Tensor] = []
        camera_to_world_list: List[torch.Tensor] = []
        lidar_to_world_list: List[torch.Tensor] = []
        lidar_points_list: List[torch.Tensor] = []
        lidar_mask_list: List[torch.Tensor] = []

        orig_hw = record.raw_hw
        color_jitter_factors = self._sample_color_jitter()

        for time_idx, frame_id in enumerate(time_frame_ids):
            use_depth_gt = time_idx == self.n_time_steps - 1
            for camera_name in ("image_02", "image_03"):
                if camera_name == "image_02":
                    image_path = record.image_paths_02[frame_id]
                    intrinsics_raw = record.intrinsics_02
                    extrinsics = record.extrinsics_02[frame_id]
                else:
                    image_path = record.image_paths_03[frame_id]
                    intrinsics_raw = record.intrinsics_03
                    extrinsics = record.extrinsics_03[frame_id]

                image, _ = preprocess_rgb_like_demo(image_path, self.image_size)
                image = self._apply_color_jitter(image, color_jitter_factors)
                images_list.append(image)
                depths_list.append(self._load_depth_or_invalid(record, frame_id, camera_name, use_depth_gt))

                extrinsics_list.append(torch.from_numpy(extrinsics.copy()))
                extrinsics_h = np.eye(4, dtype=np.float32)
                extrinsics_h[:3, :] = extrinsics
                camera_to_world_list.append(torch.from_numpy(np.linalg.inv(extrinsics_h).astype(np.float32)))
                intrinsics_list.append(torch.from_numpy(resize_intrinsics(intrinsics_raw, orig_hw=orig_hw, out_hw=self.image_size)))

            lidar_points, lidar_mask = self._load_world_lidar_points(record, frame_id)
            lidar_points_list.append(lidar_points)
            lidar_mask_list.append(lidar_mask)
            lidar_to_world_list.append(torch.from_numpy(record.world_velodyne_poses[frame_id].astype(np.float32)))

        target_frame_id = time_frame_ids[-1]
        occupancy_target, occupancy_valid_mask = self._build_occupancy_target(record, target_frame_id)

        sample = {
            "images": torch.stack(images_list, dim=0),
            "depths": torch.stack(depths_list, dim=0),
            "extrinsics": torch.stack(extrinsics_list, dim=0),
            "intrinsics": torch.stack(intrinsics_list, dim=0),
            "camera_to_world": torch.stack(camera_to_world_list, dim=0),
            "lidar_to_world": torch.stack(lidar_to_world_list, dim=0),
            "points": torch.stack(lidar_points_list, dim=0),
            "point_mask": torch.stack(lidar_mask_list, dim=0),
            "occupancy_target": occupancy_target,
            "occupancy_valid_mask": occupancy_valid_mask,
        }
        if self.frustum_size > 0:
            frustums_masks, frustums_class_dists = self._build_frustum_targets(
                record,
                target_frame_id,
                occupancy_target=occupancy_target,
                occupancy_valid_mask=occupancy_valid_mask,
            )
            sample["frustums_masks"] = frustums_masks
            sample["frustums_class_dists"] = frustums_class_dists
        return sample
