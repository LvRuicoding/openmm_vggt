#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mmengine.registry import DATASETS

from .kitti_local_utils import (
    DEFAULT_IMAGE_HW,
    crop_intrinsics,
    infer_crop_box,
    load_camera_transform_imu_to_rectified,
    load_oxts_poses,
    load_rectified_intrinsics,
    load_selection_intrinsics,
    preprocess_depth_png,
    preprocess_rgb_like_demo,
    resize_intrinsics,
    resolve_kitti_depth_root,
)


@dataclass
class DriveInfo:
    name: str
    date: str
    camera: str
    raw_hw: Tuple[int, int]
    image_paths: Dict[str, Path]
    gt_paths: Dict[str, Path]
    frame_ids: List[str]
    extrinsics: Dict[str, np.ndarray]
    intrinsics: np.ndarray


@dataclass
class ValSelectionSample:
    sequence_name: str
    drive_name: str
    date: str
    frame_ids: List[str]
    center_index: int
    crop_box: Tuple[int, int, int, int]
    center_image_path: Path
    center_depth_path: Path
    center_intrinsics: np.ndarray
    raw_intrinsics: np.ndarray
    image_paths: Dict[str, Path]
    extrinsics: Dict[str, np.ndarray]
@DATASETS.register_module()
class KITTIDepthCompletionSequenceDataset(Dataset):
    def __init__(
        self,
        depth_root: str,
        raw_root: str,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_HW,
        camera: str = "image_02",
        cameras: Tuple[str, ...] | List[str] | None = None,
        split: str = "train",
        num_frames: int = 6,
        stride: int = 1,
        strict: bool = False,
        max_sequences: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.depth_root = resolve_kitti_depth_root(depth_root)
        self.raw_root = Path(raw_root)
        self.image_size = image_size
        self.cameras = tuple(cameras) if cameras is not None else (camera,)
        self.split = split
        self.num_frames = num_frames
        self.stride = stride
        self.strict = strict
        self.max_sequences = max_sequences
        self.max_samples = max_samples

        self.drives = self._load_drives()
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(
                f"No valid KITTI samples found for split={self.split}, cameras={self.cameras}, "
                f"num_frames={self.num_frames}, stride={self.stride}."
            )

    def _load_drives(self) -> List[DriveInfo]:
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"Raw KITTI root not found: {self.raw_root}")

        if self.split == "all":
            split_names = ["train", "val"]
        elif self.split in {"train", "val"}:
            split_names = [self.split]
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        drives: List[DriveInfo] = []
        skipped = 0
        for split_name in split_names:
            split_root = self.depth_root / split_name
            if not split_root.is_dir():
                raise FileNotFoundError(f"Annotated KITTI split root not found: {split_root}")

            for drive_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
                date = drive_dir.name[:10]
                raw_drive_root = self.raw_root / date / drive_dir.name
                calib_root = self.raw_root / date

                if not raw_drive_root.is_dir():
                    if self.strict:
                        raise FileNotFoundError(f"Missing raw drive directory for {drive_dir.name}: {raw_drive_root}")
                    skipped += 1
                    continue

                try:
                    oxts_poses = load_oxts_poses(raw_drive_root)
                except Exception:
                    if self.strict:
                        raise
                    skipped += 1
                    continue

                for camera_name in self.cameras:
                    gt_dir = drive_dir / "proj_depth" / "groundtruth" / camera_name
                    rgb_dir = raw_drive_root / camera_name / "data"
                    if not gt_dir.is_dir():
                        continue
                    if not rgb_dir.is_dir():
                        if self.strict:
                            raise FileNotFoundError(f"Missing raw image directory for {drive_dir.name} {camera_name}: {rgb_dir}")
                        skipped += 1
                        continue

                    try:
                        intrinsics = load_rectified_intrinsics(calib_root / "calib_cam_to_cam.txt", camera_name)
                        t_cam_imu = load_camera_transform_imu_to_rectified(calib_root, camera_name)
                    except Exception:
                        if self.strict:
                            raise
                        skipped += 1
                        continue

                    image_paths = {path.stem: path for path in sorted(rgb_dir.glob("*.png"))}
                    gt_paths = {path.stem: path for path in sorted(gt_dir.glob("*.png"))}
                    frame_ids = sorted(set(image_paths) & set(gt_paths) & set(oxts_poses))
                    if len(frame_ids) < self.num_frames:
                        continue

                    sample_image = Image.open(image_paths[frame_ids[0]]).convert("RGB")
                    raw_w, raw_h = sample_image.size
                    raw_hw = (raw_h, raw_w)

                    extrinsics: Dict[str, np.ndarray] = {}
                    for frame_id in frame_ids:
                        t_world_imu = oxts_poses[frame_id]
                        t_imu_world = np.linalg.inv(t_world_imu)
                        t_cam_world = t_cam_imu @ t_imu_world
                        extrinsics[frame_id] = t_cam_world[:3, :4].astype(np.float32)

                    drives.append(
                        DriveInfo(
                            name=drive_dir.name,
                            date=date,
                            camera=camera_name,
                            raw_hw=raw_hw,
                            image_paths=image_paths,
                            gt_paths=gt_paths,
                            frame_ids=frame_ids,
                            extrinsics=extrinsics,
                            intrinsics=intrinsics.astype(np.float32),
                        )
                    )
        drives = sorted(drives, key=lambda x: (x.name, x.camera))
        if self.max_sequences is not None:
            drives = drives[: self.max_sequences]
        if not drives:
            raise RuntimeError(
                f"No KITTI drives available for split={self.split}. "
                f"Checked annotated root: {self.depth_root}"
            )
        print(
            f"Collected {len(drives)} KITTI drives for split={self.split}, cameras={self.cameras}"
            + (f", skipped {skipped} invalid drives." if skipped else "."),
            flush=True,
        )
        return drives

    def _build_samples(self) -> List[Tuple[int, int]]:
        samples: List[Tuple[int, int]] = []
        span = (self.num_frames - 1) * self.stride + 1
        for drive_idx, drive in enumerate(self.drives):
            max_start = len(drive.frame_ids) - span
            for start in range(max_start + 1):
                window = drive.frame_ids[start : start + span : self.stride]
                if len(window) != self.num_frames:
                    continue
                numeric_ids = [int(frame_id) for frame_id in window]
                expected = list(range(numeric_ids[0], numeric_ids[0] + span, self.stride))
                if numeric_ids != expected:
                    continue
                samples.append((drive_idx, start))

        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        drive_idx, start = self.samples[index]
        drive = self.drives[drive_idx]
        span = (self.num_frames - 1) * self.stride + 1
        frame_ids = drive.frame_ids[start : start + span : self.stride]

        images = []
        depths = []
        extrinsics = []
        intrinsics = []

        for frame_id in frame_ids:
            image, orig_hw = preprocess_rgb_like_demo(drive.image_paths[frame_id], self.image_size)
            depth = preprocess_depth_png(drive.gt_paths[frame_id], self.image_size)
            intrinsic = resize_intrinsics(drive.intrinsics, orig_hw=orig_hw, out_hw=self.image_size)

            images.append(image)
            depths.append(depth)
            extrinsics.append(torch.from_numpy(drive.extrinsics[frame_id].copy()))
            intrinsics.append(torch.from_numpy(intrinsic))

        return {
            "images": torch.stack(images, dim=0),
            "depths": torch.stack(depths, dim=0),
            "extrinsics": torch.stack(extrinsics, dim=0),
            "intrinsics": torch.stack(intrinsics, dim=0),
            "sequence_name": f"{drive.name}_{drive.camera}_{frame_ids[0]}_{frame_ids[-1]}",
        }


@DATASETS.register_module()
class KITTIDepthCompletionTrainDataset(KITTIDepthCompletionSequenceDataset):
    pass


@DATASETS.register_module()
class KITTIDepthSelectionValDataset(Dataset):
    def __init__(
        self,
        depth_root: str,
        raw_root: str,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_HW,
        camera: str = "image_02",
        num_frames: int = 6,
        stride: int = 1,
        strict: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.depth_root = resolve_kitti_depth_root(depth_root)
        self.raw_root = Path(raw_root)
        self.image_size = image_size
        self.camera = camera
        self.num_frames = num_frames
        self.stride = stride
        self.strict = strict
        self.max_samples = max_samples

        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(
                f"No official KITTI validation samples found for camera={self.camera}, "
                f"num_frames={self.num_frames}, stride={self.stride}."
            )

    def _window_frame_ids(self, frame_id: str) -> Tuple[List[str], int]:
        center_value = int(frame_id)
        left_count = self.num_frames // 2
        right_count = self.num_frames - left_count - 1
        frame_values = [
            center_value + offset * self.stride
            for offset in range(-left_count, right_count + 1)
        ]
        frame_ids = [f"{value:010d}" for value in frame_values]
        return frame_ids, left_count

    def _load_samples(self) -> List[ValSelectionSample]:
        val_root = self.depth_root / "depth_selection" / "val_selection_cropped"
        image_root = val_root / "image"
        depth_root = val_root / "groundtruth_depth"
        intrinsics_root = val_root / "intrinsics"
        if not image_root.is_dir() or not depth_root.is_dir() or not intrinsics_root.is_dir():
            raise FileNotFoundError(f"Official KITTI val_selection_cropped not found under {val_root}")

        samples: List[ValSelectionSample] = []
        skipped = 0
        for center_image_path in sorted(image_root.glob("*.png")):
            stem = center_image_path.stem
            parts = stem.split("_")
            if len(parts) != 10:
                if self.strict:
                    raise ValueError(f"Unexpected val image naming: {center_image_path.name}")
                skipped += 1
                continue

            date = "_".join(parts[:3])
            drive_name = "_".join(parts[:6])
            if parts[6] != "image" or parts[8] != "image":
                if self.strict:
                    raise ValueError(f"Unexpected val image naming: {center_image_path.name}")
                skipped += 1
                continue

            frame_id = parts[7]
            camera = "_".join(parts[8:10])
            if camera != self.camera:
                continue

            center_depth_path = depth_root / f"{drive_name}_groundtruth_depth_{frame_id}_{camera}.png"
            center_intr_path = intrinsics_root / f"{drive_name}_image_{frame_id}_{camera}.txt"
            raw_drive_root = self.raw_root / date / drive_name
            raw_image_root = raw_drive_root / camera / "data"
            calib_root = self.raw_root / date

            if not center_depth_path.is_file() or not center_intr_path.is_file():
                if self.strict:
                    raise FileNotFoundError(f"Missing official val pair for {center_image_path.name}")
                skipped += 1
                continue
            if not raw_image_root.is_dir():
                if self.strict:
                    raise FileNotFoundError(f"Missing raw image root for {center_image_path.name}: {raw_image_root}")
                skipped += 1
                continue

            try:
                raw_intrinsics = load_rectified_intrinsics(calib_root / "calib_cam_to_cam.txt", camera)
                center_intrinsics = load_selection_intrinsics(center_intr_path)
                t_cam_imu = load_camera_transform_imu_to_rectified(calib_root, camera)
                oxts_poses = load_oxts_poses(raw_drive_root)
            except Exception:
                if self.strict:
                    raise
                skipped += 1
                continue

            raw_center_path = raw_image_root / f"{frame_id}.png"
            if not raw_center_path.is_file():
                if self.strict:
                    raise FileNotFoundError(f"Missing center raw frame: {raw_center_path}")
                skipped += 1
                continue

            center_hw = Image.open(center_image_path).convert("RGB").size
            raw_hw = Image.open(raw_center_path).convert("RGB").size
            crop_box = infer_crop_box(
                raw_intrinsics=raw_intrinsics,
                cropped_intrinsics=center_intrinsics,
                raw_hw=(raw_hw[1], raw_hw[0]),
                cropped_hw=(center_hw[1], center_hw[0]),
            )

            frame_ids, center_index = self._window_frame_ids(frame_id)
            image_paths: Dict[str, Path] = {}
            extrinsics: Dict[str, np.ndarray] = {}
            valid = True
            for window_frame_id in frame_ids:
                image_path = raw_image_root / f"{window_frame_id}.png"
                if not image_path.is_file() or window_frame_id not in oxts_poses:
                    valid = False
                    break
                t_world_imu = oxts_poses[window_frame_id]
                t_imu_world = np.linalg.inv(t_world_imu)
                t_cam_world = t_cam_imu @ t_imu_world
                image_paths[window_frame_id] = image_path
                extrinsics[window_frame_id] = t_cam_world[:3, :4].astype(np.float32)

            if not valid:
                if self.strict:
                    raise FileNotFoundError(f"Incomplete raw context for {center_image_path.name}")
                skipped += 1
                continue

            samples.append(
                ValSelectionSample(
                    sequence_name=f"{drive_name}_{frame_ids[0]}_{frame_ids[-1]}",
                    drive_name=drive_name,
                    date=date,
                    frame_ids=frame_ids,
                    center_index=center_index,
                    crop_box=crop_box,
                    center_image_path=center_image_path,
                    center_depth_path=center_depth_path,
                    center_intrinsics=center_intrinsics.astype(np.float32),
                    raw_intrinsics=raw_intrinsics.astype(np.float32),
                    image_paths=image_paths,
                    extrinsics=extrinsics,
                )
            )

        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        print(
            f"Collected {len(samples)} official KITTI val samples for camera={self.camera}"
            + (f", skipped {skipped} invalid samples." if skipped else "."),
            flush=True,
        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        images = []
        depths = []
        extrinsics = []
        intrinsics = []

        for frame_idx, frame_id in enumerate(sample.frame_ids):
            if frame_idx == sample.center_index:
                image, orig_hw = preprocess_rgb_like_demo(sample.center_image_path, self.image_size)
                depth = preprocess_depth_png(sample.center_depth_path, self.image_size)
                intrinsic = resize_intrinsics(sample.center_intrinsics, orig_hw=orig_hw, out_hw=self.image_size)
            else:
                image, orig_hw = preprocess_rgb_like_demo(
                    sample.image_paths[frame_id],
                    self.image_size,
                    crop_box=sample.crop_box,
                )
                intrinsic = crop_intrinsics(sample.raw_intrinsics, sample.crop_box[0], sample.crop_box[1])
                intrinsic = resize_intrinsics(intrinsic, orig_hw=orig_hw, out_hw=self.image_size)
                depth = torch.full(self.image_size, -1.0, dtype=torch.float32)

            images.append(image)
            depths.append(depth)
            extrinsics.append(torch.from_numpy(sample.extrinsics[frame_id].copy()))
            intrinsics.append(torch.from_numpy(intrinsic))

        return {
            "images": torch.stack(images, dim=0),
            "depths": torch.stack(depths, dim=0),
            "extrinsics": torch.stack(extrinsics, dim=0),
            "intrinsics": torch.stack(intrinsics, dim=0),
            "sequence_name": sample.sequence_name,
        }
