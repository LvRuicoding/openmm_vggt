#!/usr/bin/env python3
"""KITTI Depth Completion dataset with stereo + temporal context.

Each sample consists of ``n_time_steps`` consecutive frames from a drive,
loaded for BOTH image_02 and image_03.  The images are organized as::

    Time step t-1: [image_02, image_03]
    Time step t0:  [image_02, image_03]
    Time step t+1: [image_02, image_03]

The model receives S = n_time_steps * 2 images with cam_num = 2 (stereo pair)
and f_num = n_time_steps (temporal frames). The last time step's image_02 and
image_03 both carry ground-truth depth annotations (where available); all other
slots receive a depth map filled with -1.

Data is returned in time-major order [t0_02, t0_03, t1_02, t1_03, ...].
The model's forward() reshapes this as (B, f_num, cam_num, ...) and permutes
to camera-major order internally before feeding the aggregator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    preprocess_depth_png,
    preprocess_rgb_like_demo,
    resize_intrinsics,
    resolve_kitti_depth_root,
)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _StereoDrive:
    """Metadata for one drive that has both image_02 and image_03."""
    name: str
    date: str
    raw_hw: Tuple[int, int]
    # Per-camera image paths and GT depth paths
    image_paths_02: Dict[str, Path]
    image_paths_03: Dict[str, Path]
    gt_paths_02: Dict[str, Path]       # GT for image_02
    gt_paths_03: Dict[str, Path]       # GT for image_03
    lidar_paths: Dict[str, Path]       # raw velodyne .bin per frame
    frame_ids: List[str]               # sorted intersection of valid frames
    extrinsics_02: Dict[str, np.ndarray]  # 3x4 cam02-world
    extrinsics_03: Dict[str, np.ndarray]  # 3x4 cam03-world
    world_velodyne_poses: Dict[str, np.ndarray]  # 4x4 world<-velodyne per frame
    intrinsics_02: np.ndarray          # 3x3
    intrinsics_03: np.ndarray          # 3x3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@DATASETS.register_module()
class KITTIDepthCompletionStereoDataset(Dataset):
    """KITTI Depth Completion dataset with stereo + temporal context.

    Parameters
    ----------
    depth_root:
        Root of the KITTI depth completion annotation tree
        (``train/`` and ``val/`` sub-directories).
    raw_root:
        Root of the raw KITTI data (contains date-level directories with
        ``calib_*.txt`` and drive-level directories with ``image_02/``,
        ``image_03/``, ``oxts/``).
    split:
        ``"train"`` or ``"val"``.
    n_time_steps:
        Number of temporal frames to use (default 3 → centre ±1).
        The total sequence length fed to the model is ``n_time_steps * 2``
        (stereo pair per time step).
    stride:
        Temporal stride between consecutive selected frames (default 1).
    image_size:
        ``(H, W)`` to resize images to.  Both dimensions **must** be
        multiples of 14.  Defaults to (378, 1246) which is closest to the
        native KITTI resolution (375, 1242) while satisfying the constraint.
    strict:
        If ``True``, raise on any missing file; otherwise skip silently.
    max_sequences:
        Cap on number of drives loaded (useful for smoke tests).
    max_samples:
        Cap on total number of samples (useful for smoke tests).
    """

    def __init__(
        self,
        depth_root: str,
        raw_root: str,
        split: str = "train",
        n_time_steps: int = 3,
        stride: int = 1,
        image_size: Tuple[int, int] = (280, 518),
        strict: bool = False,
        max_sequences: Optional[int] = None,
        max_samples: Optional[int] = None,
        return_lidar: bool = False,
        max_lidar_points: int = 32768,
    ) -> None:
        assert n_time_steps >= 1, "n_time_steps must be >= 1"
        assert image_size[0] % 14 == 0 and image_size[1] % 14 == 0, (
            f"image_size {image_size} must have both dims as multiples of 14"
        )

        self.depth_root = resolve_kitti_depth_root(depth_root)
        self.raw_root = Path(raw_root)
        self.split = split
        self.n_time_steps = n_time_steps
        self.stride = stride
        self.image_size = image_size
        self.strict = strict
        self.max_sequences = max_sequences
        self.max_samples = max_samples
        self.return_lidar = return_lidar
        self.max_lidar_points = max_lidar_points

        # Total model sequence length = n_time_steps stereo pairs
        self.seq_len = n_time_steps * 2  # cam_num for the model

        self.drives = self._load_drives()
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(
                f"No valid stereo samples found for split={split}, "
                f"n_time_steps={n_time_steps}, stride={stride}."
            )

    # ------------------------------------------------------------------
    # Drive loading
    # ------------------------------------------------------------------

    def _load_drives(self) -> List[_StereoDrive]:
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"Raw KITTI root not found: {self.raw_root}")

        split_root = self.depth_root / self.split
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Depth-completion split root not found: {split_root}"
            )

        drives: List[_StereoDrive] = []
        skipped = 0

        for drive_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            date = drive_dir.name[:10]
            raw_drive = self.raw_root / date / drive_dir.name
            calib_root = self.raw_root / date

            if not raw_drive.is_dir():
                if self.strict:
                    raise FileNotFoundError(
                        f"Missing raw drive: {raw_drive}"
                    )
                skipped += 1
                continue

            # ---- pose (shared for both cameras) ----
            try:
                oxts_poses = load_oxts_poses(raw_drive)
            except Exception:
                if self.strict:
                    raise
                skipped += 1
                continue

            # ---- camera-specific setup ----
            cam_data = {}
            ok = True
            lidar_dir = raw_drive / "velodyne_points" / "data"
            if self.return_lidar and not lidar_dir.is_dir():
                if self.strict:
                    raise FileNotFoundError(f"Missing raw lidar dir {lidar_dir}")
                skipped += 1
                continue
            lidar_paths = {p.stem: p for p in sorted(lidar_dir.glob("*.bin"))} if lidar_dir.is_dir() else {}
            try:
                T_imu_velo = load_transform_imu_to_velodyne(calib_root)
            except Exception:
                if self.return_lidar and self.strict:
                    raise
                T_imu_velo = None

            for cam in ("image_02", "image_03"):
                gt_dir = drive_dir / "proj_depth" / "groundtruth" / cam
                rgb_dir = raw_drive / cam / "data"

                # GT depth is required for image_02; image_03 GT is used when available.
                if cam == "image_02" and not gt_dir.is_dir():
                    ok = False
                    break
                if not rgb_dir.is_dir():
                    if self.strict:
                        raise FileNotFoundError(
                            f"Missing raw image dir {rgb_dir}"
                        )
                    ok = False
                    break

                try:
                    K = load_rectified_intrinsics(
                        calib_root / "calib_cam_to_cam.txt", cam
                    )
                    T_cam_imu = load_camera_transform_imu_to_rectified(
                        calib_root, cam
                    )
                except Exception:
                    if self.strict:
                        raise
                    ok = False
                    break

                image_paths = {
                    p.stem: p for p in sorted(rgb_dir.glob("*.png"))
                }
                gt_paths = (
                    {p.stem: p for p in sorted(gt_dir.glob("*.png"))}
                    if gt_dir.is_dir()
                    else {}
                )

                extrinsics: Dict[str, np.ndarray] = {}
                for fid, T_world_imu in oxts_poses.items():
                    T_imu_world = np.linalg.inv(T_world_imu)
                    T_cam_world = T_cam_imu @ T_imu_world
                    extrinsics[fid] = T_cam_world[:3, :4].astype(np.float32)

                cam_data[cam] = dict(
                    K=K.astype(np.float32),
                    image_paths=image_paths,
                    gt_paths=gt_paths,
                    extrinsics=extrinsics,
                )

            if not ok or "image_02" not in cam_data or "image_03" not in cam_data:
                skipped += 1
                continue

            # ---- build joint frame_ids ----
            # Frames must have: rgb_02, rgb_03, gt_02, oxts
            valid_ids = (
                set(cam_data["image_02"]["image_paths"])
                & set(cam_data["image_03"]["image_paths"])
                & set(cam_data["image_02"]["gt_paths"])
                & set(oxts_poses)
            )
            if self.return_lidar:
                if T_imu_velo is None:
                    skipped += 1
                    continue
                valid_ids = valid_ids & set(lidar_paths)
            frame_ids = sorted(valid_ids)
            span = (self.n_time_steps - 1) * self.stride + 1
            if len(frame_ids) < span:
                continue

            # Peek image size
            first_path = cam_data["image_02"]["image_paths"][frame_ids[0]]
            raw_w, raw_h = Image.open(first_path).size

            world_velodyne_poses: Dict[str, np.ndarray] = {}
            if self.return_lidar:
                for fid in frame_ids:
                    world_velodyne_poses[fid] = (oxts_poses[fid] @ T_imu_velo).astype(np.float32)

            drives.append(
                _StereoDrive(
                    name=drive_dir.name,
                    date=date,
                    raw_hw=(raw_h, raw_w),
                    image_paths_02=cam_data["image_02"]["image_paths"],
                    image_paths_03=cam_data["image_03"]["image_paths"],
                    gt_paths_02=cam_data["image_02"]["gt_paths"],
                    gt_paths_03=cam_data["image_03"]["gt_paths"],
                    lidar_paths=lidar_paths,
                    frame_ids=frame_ids,
                    extrinsics_02=cam_data["image_02"]["extrinsics"],
                    extrinsics_03=cam_data["image_03"]["extrinsics"],
                    world_velodyne_poses=world_velodyne_poses,
                    intrinsics_02=cam_data["image_02"]["K"],
                    intrinsics_03=cam_data["image_03"]["K"],
                )
            )

        drives = sorted(drives, key=lambda d: d.name)
        if self.max_sequences is not None:
            drives = drives[: self.max_sequences]

        if not drives:
            raise RuntimeError(
                f"No KITTI stereo drives available for split={self.split}. "
                f"Checked: {self.depth_root / self.split}"
            )

        print(
            f"[KITTIDepthCompletionStereoDataset] split={self.split}: "
            f"{len(drives)} drives loaded"
            + (f", {skipped} skipped." if skipped else "."),
            flush=True,
        )
        return drives

    # ------------------------------------------------------------------
    # Sample index building
    # ------------------------------------------------------------------

    def _build_samples(self) -> List[Tuple[int, int]]:
        """Return list of (drive_idx, last_frame_idx) pairs.

        The window covers n_time_steps frames ending at last_frame_idx.
        When last_frame_idx < (n_time_steps-1)*stride (i.e. not enough history),
        the earliest available frame is repeated to pad the window (scheme A).
        GT depth supervision is applied only to the last time step (cam02).
        """
        samples: List[Tuple[int, int]] = []
        for drive_idx, drive in enumerate(self.drives):
            n = len(drive.frame_ids)
            for last_idx in range(n):
                # The last frame must have a GT depth annotation
                last_fid = drive.frame_ids[last_idx]
                if last_fid not in drive.gt_paths_02:
                    continue
                samples.append((drive_idx, last_idx))

        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        return samples

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def _load_world_lidar_points(self, drive: _StereoDrive, frame_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        points = np.fromfile(drive.lidar_paths[frame_id], dtype=np.float32).reshape(-1, 4)
        xyz1 = np.concatenate(
            [points[:, :3], np.ones((points.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        world_xyz = (drive.world_velodyne_poses[frame_id] @ xyz1.T).T[:, :3]
        world_points = np.concatenate([world_xyz.astype(np.float32), points[:, 3:4]], axis=1)

        if world_points.shape[0] > self.max_lidar_points:
            indices = np.linspace(0, world_points.shape[0] - 1, self.max_lidar_points, dtype=np.int64)
            world_points = world_points[indices]

        point_tensor = torch.zeros((self.max_lidar_points, 4), dtype=torch.float32)
        point_mask = torch.zeros((self.max_lidar_points,), dtype=torch.bool)
        if world_points.shape[0] > 0:
            valid = torch.from_numpy(world_points)
            count = valid.shape[0]
            point_tensor[:count] = valid
            point_mask[:count] = True
        return point_tensor, point_mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        drive_idx, last_idx = self.samples[index]
        drive = self.drives[drive_idx]

        # Collect the n_time_steps frame indices ending at last_idx.
        # When history is insufficient, clamp to 0 (repeat the first frame).
        raw_indices = [
            last_idx - (self.n_time_steps - 1 - t) * self.stride
            for t in range(self.n_time_steps)
        ]
        # Clamp negative indices to 0 (pad with the earliest available frame)
        clamped_indices = [max(0, idx) for idx in raw_indices]
        time_frame_ids = [drive.frame_ids[idx] for idx in clamped_indices]

        images_list: List[torch.Tensor] = []
        depths_list: List[torch.Tensor] = []
        extrinsics_list: List[torch.Tensor] = []
        intrinsics_list: List[torch.Tensor] = []
        camera_to_world_list: List[torch.Tensor] = []
        lidar_to_world_list: List[torch.Tensor] = []
        lidar_points_list: List[torch.Tensor] = []
        lidar_mask_list: List[torch.Tensor] = []

        orig_hw = drive.raw_hw  # (H, W) of raw image

        # Build data in time-major order: [t0_02, t0_03, t1_02, t1_03, ...]
        # The model's forward() expects time-major and reshapes as
        # (B, f_num, cam_num, ...) before permuting to camera-major internally.
        for t_idx, fid in enumerate(time_frame_ids):
            is_last = (t_idx == self.n_time_steps - 1)  # only last time step has GT
            for cam_idx, cam in enumerate(("image_02", "image_03")):
                if cam == "image_02":
                    img_path = drive.image_paths_02[fid]
                    K_raw = drive.intrinsics_02
                    ext = drive.extrinsics_02[fid]
                    gt_path = drive.gt_paths_02.get(fid, None)
                else:
                    img_path = drive.image_paths_03[fid]
                    K_raw = drive.intrinsics_03
                    ext = drive.extrinsics_03[fid]
                    gt_path = drive.gt_paths_03.get(fid, None)

                # RGB
                image, _ = preprocess_rgb_like_demo(img_path, self.image_size)
                images_list.append(image)

                # Depth: GT for the last time step's image_02 and image_03 (if available)
                if is_last and gt_path is not None:
                    depth = preprocess_depth_png(gt_path, self.image_size)
                else:
                    depth = torch.full(
                        self.image_size, -1.0, dtype=torch.float32
                    )
                depths_list.append(depth)

                # Extrinsics (3x4)
                extrinsics_list.append(
                    torch.from_numpy(ext.copy())
                )
                ext_h = np.eye(4, dtype=np.float32)
                ext_h[:3, :] = ext
                camera_to_world_list.append(torch.from_numpy(np.linalg.inv(ext_h).astype(np.float32)))

                # Intrinsics (3x3): resize from raw resolution
                K = resize_intrinsics(
                    K_raw, orig_hw=orig_hw, out_hw=self.image_size
                )
                intrinsics_list.append(torch.from_numpy(K))

            if self.return_lidar:
                lidar_points, lidar_mask = self._load_world_lidar_points(drive, fid)
                lidar_to_world_list.append(
                    torch.from_numpy(drive.world_velodyne_poses[fid].astype(np.float32))
                )
                lidar_points_list.append(lidar_points)
                lidar_mask_list.append(lidar_mask)

        # Stack: images  [S, 3, H, W]  with S = n_time_steps * 2
        #        depths  [S, H, W]
        #        extr    [S, 3, 4]
        #        intr    [S, 3, 3]
        output = {
            "images": torch.stack(images_list, dim=0),
            "depths": torch.stack(depths_list, dim=0),
            "extrinsics": torch.stack(extrinsics_list, dim=0),
            "intrinsics": torch.stack(intrinsics_list, dim=0),
            "sequence_name": (
                f"{drive.name}_{time_frame_ids[0]}_{time_frame_ids[-1]}"
            ),
        }
        if self.return_lidar:
            output["camera_to_world"] = torch.stack(camera_to_world_list, dim=0)
            output["lidar_to_world"] = torch.stack(lidar_to_world_list, dim=0)
            output["points"] = torch.stack(lidar_points_list, dim=0)
            output["point_mask"] = torch.stack(lidar_mask_list, dim=0)
        return output
        
