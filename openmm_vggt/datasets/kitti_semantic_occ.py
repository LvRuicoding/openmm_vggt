#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mmengine.registry import DATASETS

from .kitti_local_utils import (
    load_camera_transform_imu_to_rectified,
    load_oxts_poses,
    load_rectified_intrinsics,
    load_transform_imu_to_velodyne,
    preprocess_rgb_like_demo,
    resize_intrinsics,
)


SEMANTIC_KITTI_RAW_MAPPING: Dict[str, str] = {
    "00": "2011_10_03/2011_10_03_drive_0027_sync",
    "01": "2011_10_03/2011_10_03_drive_0042_sync",
    "02": "2011_10_03/2011_10_03_drive_0034_sync",
    "03": "2011_09_26/2011_09_26_drive_0067_sync",
    "04": "2011_09_30/2011_09_30_drive_0016_sync",
    "05": "2011_09_30/2011_09_30_drive_0018_sync",
    "06": "2011_09_30/2011_09_30_drive_0020_sync",
    "07": "2011_09_30/2011_09_30_drive_0027_sync",
    "08": "2011_09_30/2011_09_30_drive_0028_sync",
    "09": "2011_09_30/2011_09_30_drive_0033_sync",
    "10": "2011_09_30/2011_09_30_drive_0034_sync",
    "11": "2011_09_30/2011_09_30_drive_0012_sync",
    "12": "2011_09_30/2011_09_30_drive_0013_sync",
    "13": "2011_09_30/2011_09_30_drive_0014_sync",
    "14": "2011_09_30/2011_09_30_drive_0015_sync",
    "15": "2011_09_30/2011_09_30_drive_0016_sync",
    "16": "2011_09_30/2011_09_30_drive_0018_sync",
    "17": "2011_09_30/2011_09_30_drive_0019_sync",
    "18": "2011_09_30/2011_09_30_drive_0027_sync",
    "19": "2011_09_30/2011_09_30_drive_0028_sync",
    "20": "2011_09_30/2011_09_30_drive_0033_sync",
    "21": "2011_09_30/2011_09_30_drive_0034_sync",
}

DEFAULT_SPLITS: Dict[str, Tuple[str, ...]] = {
    "train": ("00", "01", "02", "03", "04", "05", "06", "07", "09", "10"),
    "val": ("08",),
    "test": ("11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"),
}

RAW_TO_LEARNING = {
    0: 0,
    1: 0,
    10: 1,
    11: 2,
    13: 5,
    15: 3,
    16: 5,
    18: 4,
    20: 5,
    30: 6,
    31: 7,
    32: 8,
    40: 9,
    44: 10,
    48: 11,
    49: 12,
    50: 13,
    51: 14,
    52: 0,
    60: 9,
    70: 15,
    71: 16,
    72: 17,
    80: 18,
    81: 19,
    99: 0,
    252: 1,
    253: 7,
    254: 6,
    255: 8,
    256: 5,
    257: 5,
    258: 4,
    259: 5,
}


@dataclass
class _SequenceRecord:
    sequence_id: str
    raw_drive_name: str
    raw_root: Path
    semantic_root: Path
    raw_hw: Tuple[int, int]
    frame_ids: List[str]
    image_paths_02: Dict[str, Path]
    image_paths_03: Dict[str, Path]
    extrinsics_02: Dict[str, np.ndarray]
    extrinsics_03: Dict[str, np.ndarray]
    intrinsics_02: np.ndarray
    intrinsics_03: np.ndarray
    world_velodyne_poses: Dict[str, np.ndarray]


def _raw_label_to_learning(raw_label: np.ndarray) -> np.ndarray:
    semantic_label = raw_label & 0xFFFF
    mapped = np.zeros_like(semantic_label, dtype=np.uint8)
    for raw_id, learning_id in RAW_TO_LEARNING.items():
        mapped[semantic_label == raw_id] = np.uint8(learning_id)
    return mapped


def _load_semantic_poses(poses_path: Path) -> np.ndarray:
    poses: List[np.ndarray] = []
    with poses_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            values = np.fromstring(line.strip(), sep=" ", dtype=np.float64)
            if values.size != 12:
                continue
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :] = values.reshape(3, 4)
            poses.append(pose)
    if not poses:
        raise RuntimeError(f"No valid semantic poses found in {poses_path}")
    return np.stack(poses, axis=0)


def _estimate_contiguous_offset(semantic_poses: np.ndarray, raw_positions: np.ndarray) -> int:
    semantic_positions = semantic_poses[:, :3, 3]
    semantic_steps = np.linalg.norm(np.diff(semantic_positions, axis=0), axis=1)
    raw_steps = np.linalg.norm(np.diff(raw_positions, axis=0), axis=1)
    max_offset = raw_positions.shape[0] - semantic_positions.shape[0]
    if max_offset < 0:
        raise RuntimeError("Raw drive is shorter than semantic sequence")
    if semantic_steps.size == 0:
        return 0

    compare_len = min(200, semantic_steps.shape[0], raw_steps.shape[0])
    semantic_ref = semantic_steps[:compare_len]
    best_offset = 0
    best_score = float("inf")
    for offset in range(max_offset + 1):
        raw_ref = raw_steps[offset : offset + compare_len]
        if raw_ref.shape[0] != compare_len:
            break
        score = float(np.mean(np.abs(raw_ref - semantic_ref)))
        if score < best_score:
            best_score = score
            best_offset = offset
    return best_offset


def _majority_assign(flat_indices: np.ndarray, labels: np.ndarray, target: np.ndarray) -> None:
    if flat_indices.size == 0:
        return
    order = np.argsort(flat_indices, kind="mergesort")
    flat_indices = flat_indices[order]
    labels = labels[order]
    start = 0
    while start < flat_indices.shape[0]:
        end = start + 1
        current = flat_indices[start]
        while end < flat_indices.shape[0] and flat_indices[end] == current:
            end += 1
        counts = np.bincount(labels[start:end], minlength=20)
        counts[0] = 0
        best = int(np.argmax(counts))
        if best > 0:
            target[current] = best
        start = end


def _raycast_free_voxels(
    occupied_mask: np.ndarray,
    valid_mask: np.ndarray,
    point_voxels: np.ndarray,
    grid_size: np.ndarray,
) -> None:
    unique_points = np.unique(point_voxels, axis=0)
    origin = np.zeros(3, dtype=np.float32)
    for endpoint in unique_points:
        endpoint_f = endpoint.astype(np.float32) + 0.5
        steps = int(np.max(np.abs(endpoint_f - origin)))
        if steps <= 0:
            continue
        line = np.linspace(origin, endpoint_f, num=steps + 1, endpoint=True)
        traversed = np.floor(line).astype(np.int32)
        traversed = np.clip(traversed, 0, grid_size.reshape(1, 3) - 1)
        if traversed.shape[0] > 1:
            traversed = traversed[:-1]
        if traversed.size == 0:
            continue
        flat = (
            traversed[:, 0] * grid_size[1] * grid_size[2]
            + traversed[:, 1] * grid_size[2]
            + traversed[:, 2]
        )
        valid_mask.reshape(-1)[flat] = True
        free_only = ~occupied_mask.reshape(-1)[flat]
        if free_only.any():
            valid_mask.reshape(-1)[flat[free_only]] = True


@DATASETS.register_module()
class KITTISemanticOccupancyDataset(Dataset):
    def __init__(
        self,
        semantic_root: str,
        raw_root: str,
        split: str = "train",
        sequences: Optional[List[str]] = None,
        n_time_steps: int = 3,
        stride: int = 1,
        image_size: Tuple[int, int] = (280, 518),
        strict: bool = False,
        max_sequences: Optional[int] = None,
        max_samples: Optional[int] = None,
        max_lidar_points: int = 32768,
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (0.0, -25.6, -2.0, 51.2, 25.6, 4.4),
        occupancy_cache_dir: Optional[str] = None,
    ) -> None:
        assert n_time_steps >= 1
        assert image_size[0] % 14 == 0 and image_size[1] % 14 == 0

        self.semantic_root = Path(semantic_root)
        self.raw_root = Path(raw_root)
        self.split = split
        self.sequences = list(sequences) if sequences is not None else list(DEFAULT_SPLITS[split])
        self.n_time_steps = int(n_time_steps)
        self.stride = int(stride)
        self.image_size = image_size
        self.strict = strict
        self.max_sequences = max_sequences
        self.max_samples = max_samples
        self.max_lidar_points = max_lidar_points
        self.seq_len = self.n_time_steps * 2

        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.grid_size = np.round((self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int32)
        self.num_classes = 20
        self.ignore_index = 255

        if occupancy_cache_dir is None:
            occupancy_cache_dir = str(self.semantic_root / "_occ_cache")
        self.occupancy_cache_dir = Path(occupancy_cache_dir)
        self.occupancy_cache_dir.mkdir(parents=True, exist_ok=True)

        self.records = self._load_sequences()
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"No valid occupancy samples found for split={self.split}")

    def _load_sequences(self) -> List[_SequenceRecord]:
        records: List[_SequenceRecord] = []
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
            for semantic_frame_id, raw_frame_id in zip(semantic_frame_ids, mapped_raw_ids):
                t_world_imu = oxts_poses[raw_frame_id]
                t_imu_world = np.linalg.inv(t_world_imu)
                extrinsics_02[semantic_frame_id] = (t_cam_imu_02 @ t_imu_world)[:3, :4].astype(np.float32)
                extrinsics_03[semantic_frame_id] = (t_cam_imu_03 @ t_imu_world)[:3, :4].astype(np.float32)
                world_velodyne_poses[semantic_frame_id] = (t_world_imu @ t_imu_velo).astype(np.float32)
                image_paths_02_sem[semantic_frame_id] = image_paths_02[raw_frame_id]
                image_paths_03_sem[semantic_frame_id] = image_paths_03[raw_frame_id]

            first_path = image_paths_02_sem[semantic_frame_ids[0]]
            raw_w, raw_h = Image.open(first_path).size

            records.append(
                _SequenceRecord(
                    sequence_id=sequence_id,
                    raw_drive_name=raw_drive_rel.split("/")[-1],
                    raw_root=raw_drive_root,
                    semantic_root=semantic_seq_root,
                    raw_hw=(raw_h, raw_w),
                    frame_ids=semantic_frame_ids,
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

    def _build_samples(self) -> List[Tuple[int, int]]:
        samples: List[Tuple[int, int]] = []
        for record_idx, record in enumerate(self.records):
            for last_idx in range(len(record.frame_ids)):
                samples.append((record_idx, last_idx))
        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_world_lidar_points(self, record: _SequenceRecord, frame_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        points = np.fromfile(record.semantic_root / "velodyne" / f"{frame_id}.bin", dtype=np.float32).reshape(-1, 4)
        xyz1 = np.concatenate([points[:, :3], np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        world_xyz = (record.world_velodyne_poses[frame_id] @ xyz1.T).T[:, :3].astype(np.float32)
        world_points = np.concatenate([world_xyz, points[:, 3:4]], axis=1)

        if world_points.shape[0] > self.max_lidar_points:
            indices = np.linspace(0, world_points.shape[0] - 1, self.max_lidar_points, dtype=np.int64)
            world_points = world_points[indices]

        point_tensor = torch.zeros((self.max_lidar_points, 4), dtype=torch.float32)
        point_mask = torch.zeros((self.max_lidar_points,), dtype=torch.bool)
        if world_points.shape[0] > 0:
            count = world_points.shape[0]
            point_tensor[:count] = torch.from_numpy(world_points)
            point_mask[:count] = True
        return point_tensor, point_mask

    def _occupancy_cache_path(self, sequence_id: str, frame_id: str) -> Path:
        grid_tag = f"{self.grid_size[0]}_{self.grid_size[1]}_{self.grid_size[2]}"
        return self.occupancy_cache_dir / f"{sequence_id}_{frame_id}_{grid_tag}.npz"

    def _build_occupancy_target(self, record: _SequenceRecord, frame_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_path = self._occupancy_cache_path(record.sequence_id, frame_id)
        if cache_path.is_file():
            cached = np.load(cache_path)
            return (
                torch.from_numpy(cached["target"].astype(np.int64)),
                torch.from_numpy(cached["valid_mask"].astype(bool)),
            )

        points = np.fromfile(record.semantic_root / "velodyne" / f"{frame_id}.bin", dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(record.semantic_root / "labels" / f"{frame_id}.label", dtype=np.uint32)
        learning_labels = _raw_label_to_learning(labels)

        coords = np.floor((points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int32)
        inside = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < self.grid_size[0])
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < self.grid_size[1])
            & (coords[:, 2] >= 0)
            & (coords[:, 2] < self.grid_size[2])
        )
        coords = coords[inside]
        learning_labels = learning_labels[inside]

        occupied_target = np.full(int(np.prod(self.grid_size)), self.ignore_index, dtype=np.uint8)
        valid_mask = np.zeros(int(np.prod(self.grid_size)), dtype=bool)
        occupied_mask = np.zeros(int(np.prod(self.grid_size)), dtype=bool)

        occupied_points = learning_labels > 0
        occupied_coords = coords[occupied_points]
        occupied_labels = learning_labels[occupied_points]
        if occupied_coords.shape[0] > 0:
            occupied_flat = (
                occupied_coords[:, 0] * self.grid_size[1] * self.grid_size[2]
                + occupied_coords[:, 1] * self.grid_size[2]
                + occupied_coords[:, 2]
            )
            _majority_assign(occupied_flat, occupied_labels.astype(np.int64), occupied_target)
            occupied_mask[occupied_flat] = True
            valid_mask[occupied_flat] = True

        if coords.shape[0] > 0:
            _raycast_free_voxels(occupied_mask.reshape(*self.grid_size), valid_mask.reshape(*self.grid_size), coords, self.grid_size)

        target = occupied_target.reshape(*self.grid_size)
        target[(valid_mask & ~occupied_mask).reshape(*self.grid_size)] = 0
        valid_mask = valid_mask.reshape(*self.grid_size)

        np.savez_compressed(cache_path, target=target, valid_mask=valid_mask)
        return torch.from_numpy(target.astype(np.int64)), torch.from_numpy(valid_mask.astype(bool))

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
        extrinsics_list: List[torch.Tensor] = []
        intrinsics_list: List[torch.Tensor] = []
        camera_to_world_list: List[torch.Tensor] = []
        lidar_to_world_list: List[torch.Tensor] = []
        lidar_points_list: List[torch.Tensor] = []
        lidar_mask_list: List[torch.Tensor] = []

        orig_hw = record.raw_hw

        for frame_id in time_frame_ids:
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
                images_list.append(image)

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

        return {
            "images": torch.stack(images_list, dim=0),
            "extrinsics": torch.stack(extrinsics_list, dim=0),
            "intrinsics": torch.stack(intrinsics_list, dim=0),
            "camera_to_world": torch.stack(camera_to_world_list, dim=0),
            "lidar_to_world": torch.stack(lidar_to_world_list, dim=0),
            "points": torch.stack(lidar_points_list, dim=0),
            "point_mask": torch.stack(lidar_mask_list, dim=0),
            "occupancy_target": occupancy_target,
            "occupancy_valid_mask": occupancy_valid_mask,
        }
