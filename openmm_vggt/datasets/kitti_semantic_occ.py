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
        best = int(np.argmax(counts))
        target[current] = np.uint8(best)
        start = end


def _mark_free_voxels(
    points_xyz: np.ndarray,
    grid_size: np.ndarray,
    voxel_size: np.ndarray,
    point_cloud_range: np.ndarray,
    occupied_flat_indices: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    if points_xyz.size == 0:
        return

    origin = np.zeros(3, dtype=np.float32)
    occupied = set(int(index) for index in np.unique(occupied_flat_indices))
    grid_y = int(grid_size[1])
    grid_z = int(grid_size[2])

    for point in points_xyz:
        ray = point.astype(np.float32) - origin
        ray_len = float(np.linalg.norm(ray))
        if ray_len <= 1e-6:
            continue

        step_count = max(int(np.ceil(ray_len / (float(np.min(voxel_size)) * 0.5))), 1)
        # Exclude the endpoint so occupied voxels keep their semantic label.
        ts = np.linspace(0.0, 1.0, step_count, endpoint=False, dtype=np.float32)
        coords = np.floor((origin[None, :] + ts[:, None] * ray[None, :] - point_cloud_range[:3]) / voxel_size).astype(np.int32)
        inside = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < grid_size[0])
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < grid_size[1])
            & (coords[:, 2] >= 0)
            & (coords[:, 2] < grid_size[2])
        )
        if not inside.any():
            continue

        coords = coords[inside]
        flat = coords[:, 0] * grid_y * grid_z + coords[:, 1] * grid_z + coords[:, 2]
        for flat_index in np.unique(flat):
            flat_index = int(flat_index)
            if flat_index in occupied:
                continue
            target[flat_index] = 0
            valid_mask[flat_index] = True


def _downsample_dense_labels_object_priority(
    target: np.ndarray,
    valid_mask: np.ndarray,
    output_grid_size: np.ndarray,
    ignore_index: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    output_grid_size = np.asarray(output_grid_size, dtype=np.int32)
    if np.array_equal(np.asarray(target.shape, dtype=np.int32), output_grid_size):
        return target, valid_mask

    input_grid_size = np.asarray(target.shape, dtype=np.int32)
    if np.any(input_grid_size % output_grid_size != 0):
        raise RuntimeError(
            f"Dense voxel grid {tuple(input_grid_size.tolist())} cannot be downsampled "
            f"exactly to {tuple(output_grid_size.tolist())}"
        )

    factors = (input_grid_size // output_grid_size).astype(np.int32)
    reshaped_target = target.reshape(
        output_grid_size[0],
        factors[0],
        output_grid_size[1],
        factors[1],
        output_grid_size[2],
        factors[2],
    )
    reshaped_valid = valid_mask.reshape(
        output_grid_size[0],
        factors[0],
        output_grid_size[1],
        factors[1],
        output_grid_size[2],
        factors[2],
    )
    blocks_target = reshaped_target.transpose(0, 2, 4, 1, 3, 5).reshape(-1, int(np.prod(factors)))
    blocks_valid = reshaped_valid.transpose(0, 2, 4, 1, 3, 5).reshape(-1, int(np.prod(factors)))

    output = np.full(blocks_target.shape[0], ignore_index, dtype=np.int64)
    output_valid = blocks_valid.any(axis=1)
    for block_idx in np.nonzero(output_valid)[0]:
        labels = blocks_target[block_idx, blocks_valid[block_idx]]
        object_labels = labels[(labels > 0) & (labels < num_classes)]
        if object_labels.size > 0:
            counts = np.bincount(object_labels, minlength=num_classes)
            output[block_idx] = int(np.argmax(counts[1:]) + 1)
        else:
            output[block_idx] = 0

    return output.reshape(*output_grid_size), output_valid.reshape(*output_grid_size)


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
        dense_voxel_root: Optional[str] = None,
        require_dense_voxel_target: bool = False,
        occupancy_cache_dir: Optional[str] = None,
        frustum_size: int = 0,
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
        self.dense_voxel_root = Path(dense_voxel_root) if dense_voxel_root is not None else None
        self.require_dense_voxel_target = bool(require_dense_voxel_target)
        self.frustum_size = int(frustum_size)
        if self.frustum_size < 0:
            raise ValueError(f"frustum_size must be >= 0, got {self.frustum_size}")
        if self.require_dense_voxel_target and self.dense_voxel_root is None:
            raise ValueError("require_dense_voxel_target=True requires dense_voxel_root to be set.")

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
                if self.require_dense_voxel_target:
                    min_last_idx = (self.n_time_steps - 1) * self.stride
                    if last_idx < min_last_idx:
                        continue
                    last_frame_id = record.frame_ids[last_idx]
                    dense_label_path = (
                        self.dense_voxel_root
                        / "sequences"
                        / record.sequence_id
                        / "voxels"
                        / f"{last_frame_id}.label"
                    )
                    if not dense_label_path.is_file():
                        continue
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
        source_tag = "dense_v1" if self.dense_voxel_root is not None else "ssc_free_v1"
        return self.occupancy_cache_dir / f"{sequence_id}_{frame_id}_{grid_tag}_{source_tag}.npz"

    def _frustum_cache_path(self, sequence_id: str, frame_id: str) -> Path:
        grid_tag = f"{self.grid_size[0]}_{self.grid_size[1]}_{self.grid_size[2]}"
        image_tag = f"{self.image_size[0]}_{self.image_size[1]}"
        source_tag = "dense_v1" if self.dense_voxel_root is not None else "ssc_free_v1"
        return self.occupancy_cache_dir / (
            f"{sequence_id}_{frame_id}_{grid_tag}_{image_tag}_{source_tag}_frustum_{self.frustum_size}.npz"
        )

    @staticmethod
    def _unpack_voxel_mask(path: Path, expected_size: int) -> np.ndarray:
        packed = np.fromfile(path, dtype=np.uint8)
        mask = np.unpackbits(packed, bitorder="little")[:expected_size]
        if mask.shape[0] != expected_size:
            raise RuntimeError(f"Unexpected packed voxel mask size for {path}: {mask.shape[0]} != {expected_size}")
        return mask.astype(bool)

    def _load_dense_voxel_target(self, sequence_id: str, frame_id: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.dense_voxel_root is None:
            return None

        voxel_dir = self.dense_voxel_root / "sequences" / sequence_id / "voxels"
        label_path = voxel_dir / f"{frame_id}.label"
        invalid_path = voxel_dir / f"{frame_id}.invalid"
        if not label_path.is_file():
            return None

        expected_size = int(np.prod(self.grid_size))
        raw_label = np.fromfile(label_path, dtype=np.uint16)
        source_size = raw_label.shape[0]
        if source_size == expected_size:
            source_grid_size = self.grid_size
        elif source_size == 256 * 256 * 32:
            source_grid_size = np.asarray((256, 256, 32), dtype=np.int32)
        else:
            raise RuntimeError(f"Unexpected dense voxel label size for {label_path}: {source_size}")

        target = _raw_label_to_learning(raw_label.astype(np.uint32)).astype(np.int64)
        valid_mask = np.ones(source_size, dtype=bool)
        if invalid_path.is_file():
            valid_mask &= ~self._unpack_voxel_mask(invalid_path, source_size)

        target = target.reshape(*source_grid_size)
        valid_mask = valid_mask.reshape(*source_grid_size)
        target, valid_mask = _downsample_dense_labels_object_priority(
            target=target,
            valid_mask=valid_mask,
            output_grid_size=self.grid_size,
            ignore_index=self.ignore_index,
            num_classes=self.num_classes,
        )
        return torch.from_numpy(target), torch.from_numpy(valid_mask)

    def _build_occupancy_target(self, record: _SequenceRecord, frame_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_path = self._occupancy_cache_path(record.sequence_id, frame_id)
        if cache_path.is_file():
            cached = np.load(cache_path)
            return (
                torch.from_numpy(cached["target"].astype(np.int64)),
                torch.from_numpy(cached["valid_mask"].astype(bool)),
            )

        dense_target = self._load_dense_voxel_target(record.sequence_id, frame_id)
        if dense_target is not None:
            target, valid_mask = dense_target
            np.savez_compressed(cache_path, target=target.numpy().astype(np.int64), valid_mask=valid_mask.numpy().astype(bool))
            return target, valid_mask

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
        points_inside = points[:, :3][inside]
        coords = coords[inside]
        learning_labels = learning_labels[inside]
        semantic_known = learning_labels != 0
        occupied_coords = coords[semantic_known]
        occupied_labels = learning_labels[semantic_known]

        target = np.full(int(np.prod(self.grid_size)), self.ignore_index, dtype=np.uint8)
        valid_mask = np.zeros(int(np.prod(self.grid_size)), dtype=bool)

        if points_inside.shape[0] > 0:
            flat_indices = (
                occupied_coords[:, 0] * self.grid_size[1] * self.grid_size[2]
                + occupied_coords[:, 1] * self.grid_size[2]
                + occupied_coords[:, 2]
            )
            if flat_indices.size > 0:
                valid_mask[np.unique(flat_indices)] = True
            _mark_free_voxels(
                points_inside,
                self.grid_size,
                self.voxel_size,
                self.point_cloud_range,
                flat_indices,
                target,
                valid_mask,
            )
            _majority_assign(flat_indices, occupied_labels.astype(np.int64), target)

        target = target.reshape(*self.grid_size)
        valid_mask = valid_mask.reshape(*self.grid_size)

        np.savez_compressed(cache_path, target=target, valid_mask=valid_mask)
        return torch.from_numpy(target.astype(np.int64)), torch.from_numpy(valid_mask.astype(bool))

    def _voxel_centers_lidar(self) -> np.ndarray:
        grid_x, grid_y, grid_z = self.grid_size.tolist()
        xs, ys, zs = np.meshgrid(
            np.arange(grid_x, dtype=np.float32),
            np.arange(grid_y, dtype=np.float32),
            np.arange(grid_z, dtype=np.float32),
            indexing="ij",
        )
        coords = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
        return self.point_cloud_range[:3] + (coords + 0.5) * self.voxel_size

    @staticmethod
    def _project_lidar_centers_to_image(
        centers_lidar: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics_world_to_camera: np.ndarray,
        lidar_to_world: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers_h = np.concatenate(
            [centers_lidar.astype(np.float32), np.ones((centers_lidar.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        centers_world = (lidar_to_world.astype(np.float32) @ centers_h.T).T
        xyz_camera = (extrinsics_world_to_camera.astype(np.float32) @ centers_world.T).T

        z = xyz_camera[:, 2]
        safe_z = np.maximum(z, 1e-5)
        u = intrinsics[0, 0] * xyz_camera[:, 0] / safe_z + intrinsics[0, 2]
        v = intrinsics[1, 1] * xyz_camera[:, 1] / safe_z + intrinsics[1, 2]
        return u, v, z

    def _build_frustum_targets(
        self,
        record: _SequenceRecord,
        frame_id: str,
        occupancy_target: torch.Tensor,
        occupancy_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_path = self._frustum_cache_path(record.sequence_id, frame_id)
        if cache_path.is_file():
            cached = np.load(cache_path)
            return (
                torch.from_numpy(cached["frustums_masks"].astype(bool)),
                torch.from_numpy(cached["frustums_class_dists"].astype(np.float32)),
            )

        target = occupancy_target.numpy()
        valid_target = occupancy_valid_mask.numpy().astype(bool) & (target != self.ignore_index)
        flat_target = target.reshape(-1)
        flat_valid_target = valid_target.reshape(-1)
        centers_lidar = self._voxel_centers_lidar()
        image_h, image_w = self.image_size
        num_frustums = self.frustum_size * self.frustum_size

        frustum_masks = np.zeros((2, num_frustums, *self.grid_size.tolist()), dtype=bool)
        frustum_class_dists = np.zeros((2, num_frustums, self.num_classes), dtype=np.float32)
        lidar_to_world = record.world_velodyne_poses[frame_id]
        camera_specs = (
            (record.intrinsics_02, record.extrinsics_02[frame_id]),
            (record.intrinsics_03, record.extrinsics_03[frame_id]),
        )

        for cam_idx, (intrinsics_raw, extrinsics) in enumerate(camera_specs):
            intrinsics = resize_intrinsics(intrinsics_raw, orig_hw=record.raw_hw, out_hw=self.image_size)
            pix_x, pix_y, pix_z = self._project_lidar_centers_to_image(
                centers_lidar,
                intrinsics=intrinsics,
                extrinsics_world_to_camera=extrinsics,
                lidar_to_world=lidar_to_world,
            )
            in_image = (
                (pix_z > 0)
                & (pix_x >= 0)
                & (pix_x < image_w)
                & (pix_y >= 0)
                & (pix_y < image_h)
                & flat_valid_target
            )
            frustum_idx = 0
            for y_idx in range(self.frustum_size):
                y0 = y_idx * image_h / self.frustum_size
                y1 = (y_idx + 1) * image_h / self.frustum_size
                for x_idx in range(self.frustum_size):
                    x0 = x_idx * image_w / self.frustum_size
                    x1 = (x_idx + 1) * image_w / self.frustum_size
                    mask_flat = in_image & (pix_x >= x0) & (pix_x < x1) & (pix_y >= y0) & (pix_y < y1)
                    frustum_masks[cam_idx, frustum_idx] = mask_flat.reshape(*self.grid_size.tolist())
                    if mask_flat.any():
                        counts = np.bincount(flat_target[mask_flat].astype(np.int64), minlength=self.num_classes)
                        frustum_class_dists[cam_idx, frustum_idx] = counts[: self.num_classes].astype(np.float32)
                    frustum_idx += 1

        np.savez_compressed(
            cache_path,
            frustums_masks=frustum_masks,
            frustums_class_dists=frustum_class_dists,
        )
        return torch.from_numpy(frustum_masks), torch.from_numpy(frustum_class_dists)

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

        sample = {
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
