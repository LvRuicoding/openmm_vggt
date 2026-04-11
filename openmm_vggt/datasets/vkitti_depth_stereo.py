#!/usr/bin/env python3
"""vKITTI depth dataset with stereo + temporal context."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mmengine.registry import DATASETS

from .kitti_local_utils import preprocess_rgb_like_demo, resize_intrinsics


DEFAULT_SPLIT_SCENES = {
    "train": ("Scene01", "Scene02", "Scene06", "Scene18"),
    "val": ("Scene20",),
}


@dataclass
class _VKITTITrack:
    """Metadata for one vKITTI scene variant."""

    name: str
    scene: str
    variant: str
    raw_hw: Tuple[int, int]
    image_paths_0: Dict[str, Path]
    image_paths_1: Dict[str, Path]
    depth_paths_0: Dict[str, Path]
    depth_paths_1: Dict[str, Path]
    extrinsics_0: Dict[str, np.ndarray]
    extrinsics_1: Dict[str, np.ndarray]
    intrinsics_0: Dict[str, np.ndarray]
    intrinsics_1: Dict[str, np.ndarray]
    frame_ids: List[str]


def _frame_id_from_name(path: Path) -> str:
    stem = path.stem
    return stem.rsplit("_", 1)[-1]


def _load_vkitti_depth_png(depth_path: Path, out_hw: Tuple[int, int]) -> torch.Tensor:
    out_h, out_w = out_hw
    depth_png = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_png is None:
        raise FileNotFoundError(depth_path)
    depth_png = cv2.resize(depth_png, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    # vKITTI depth PNGs store metric depth in centimeters, with 65535 used as invalid.
    depth = depth_png.astype(np.float32) / 100.0
    invalid_mask = (depth_png == 0) | (depth_png == np.iinfo(depth_png.dtype).max)
    depth[invalid_mask] = -1.0
    return torch.from_numpy(depth)


def _load_intrinsics_table(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    rows = np.loadtxt(path, skiprows=1, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]

    intrinsics: Dict[int, Dict[str, np.ndarray]] = {0: {}, 1: {}}
    for row in rows:
        frame_id = f"{int(row[0]):05d}"
        camera_id = int(row[1])
        fx, fy, cx, cy = row[2:6]
        intrinsics[camera_id][frame_id] = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    return intrinsics


def _load_extrinsics_table(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    rows = np.loadtxt(path, skiprows=1, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]

    extrinsics: Dict[int, Dict[str, np.ndarray]] = {0: {}, 1: {}}
    for row in rows:
        frame_id = f"{int(row[0]):05d}"
        camera_id = int(row[1])
        matrix = row[2:].reshape(4, 4).astype(np.float32)
        extrinsics[camera_id][frame_id] = matrix[:3, :4]
    return extrinsics


@DATASETS.register_module()
class VKITTIDepthStereoDataset(Dataset):
    """vKITTI stereo depth dataset with temporal context.

    The model receives ``n_time_steps`` consecutive stereo pairs in time-major
    order ``[t0_cam0, t0_cam1, t1_cam0, t1_cam1, ...]``. Supervision matches
    the KITTI setup: only the last time step carries GT depth.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        scene_names: Optional[Sequence[str]] = None,
        variants: Optional[Sequence[str]] = None,
        n_time_steps: int = 3,
        stride: int = 1,
        image_size: Tuple[int, int] = (280, 518),
        strict: bool = False,
        max_sequences: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        assert n_time_steps >= 1, "n_time_steps must be >= 1"
        assert image_size[0] % 14 == 0 and image_size[1] % 14 == 0, (
            f"image_size {image_size} must have both dims as multiples of 14"
        )

        self.root = Path(root)
        self.split = split
        self.scene_names = tuple(scene_names or DEFAULT_SPLIT_SCENES.get(split, ()))
        self.variants = tuple(variants) if variants is not None else None
        self.n_time_steps = n_time_steps
        self.stride = stride
        self.image_size = image_size
        self.strict = strict
        self.max_sequences = max_sequences
        self.max_samples = max_samples

        self.seq_len = n_time_steps * 2

        self.tracks = self._load_tracks()
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(
                f"No valid vKITTI stereo samples found for split={split}, "
                f"scenes={self.scene_names}, n_time_steps={n_time_steps}, stride={stride}."
            )

    def _load_tracks(self) -> List[_VKITTITrack]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"vKITTI root not found: {self.root}")
        if not self.scene_names:
            raise ValueError(
                f"No scene_names provided for split={self.split}. "
                "Pass scene_names explicitly or use a known split."
            )

        tracks: List[_VKITTITrack] = []
        skipped = 0
        span = (self.n_time_steps - 1) * self.stride + 1

        for scene_name in self.scene_names:
            scene_dir = self.root / scene_name
            if not scene_dir.is_dir():
                if self.strict:
                    raise FileNotFoundError(f"Missing scene directory: {scene_dir}")
                skipped += 1
                continue

            variant_dirs = sorted(path for path in scene_dir.iterdir() if path.is_dir())
            if self.variants is not None:
                allowed = set(self.variants)
                variant_dirs = [path for path in variant_dirs if path.name in allowed]

            for variant_dir in variant_dirs:
                try:
                    track = self._load_track(scene_name, variant_dir)
                except Exception:
                    if self.strict:
                        raise
                    skipped += 1
                    continue

                if len(track.frame_ids) < span:
                    skipped += 1
                    continue
                tracks.append(track)

        tracks = sorted(tracks, key=lambda track: track.name)
        if self.max_sequences is not None:
            tracks = tracks[: self.max_sequences]

        if not tracks:
            raise RuntimeError(
                f"No vKITTI tracks available for split={self.split}, scenes={self.scene_names}. "
                f"Checked: {self.root}"
            )

        print(
            f"[VKITTIDepthStereoDataset] split={self.split}: "
            f"{len(tracks)} tracks loaded"
            + (f", {skipped} skipped." if skipped else "."),
            flush=True,
        )
        return tracks

    def _load_track(self, scene_name: str, variant_dir: Path) -> _VKITTITrack:
        frames_root = variant_dir / "frames"
        rgb_dir_0 = frames_root / "rgb" / "Camera_0"
        rgb_dir_1 = frames_root / "rgb" / "Camera_1"
        depth_dir_0 = frames_root / "depth" / "Camera_0"
        depth_dir_1 = frames_root / "depth" / "Camera_1"

        for required_dir in (rgb_dir_0, rgb_dir_1, depth_dir_0, depth_dir_1):
            if not required_dir.is_dir():
                raise FileNotFoundError(f"Missing vKITTI frames directory: {required_dir}")

        intrinsics = _load_intrinsics_table(variant_dir / "intrinsic.txt")
        extrinsics = _load_extrinsics_table(variant_dir / "extrinsic.txt")

        image_paths_0 = {_frame_id_from_name(path): path for path in sorted(rgb_dir_0.glob("*.jpg"))}
        image_paths_1 = {_frame_id_from_name(path): path for path in sorted(rgb_dir_1.glob("*.jpg"))}
        depth_paths_0 = {_frame_id_from_name(path): path for path in sorted(depth_dir_0.glob("*.png"))}
        depth_paths_1 = {_frame_id_from_name(path): path for path in sorted(depth_dir_1.glob("*.png"))}

        valid_ids = (
            set(image_paths_0)
            & set(image_paths_1)
            & set(depth_paths_0)
            & set(depth_paths_1)
            & set(intrinsics[0])
            & set(intrinsics[1])
            & set(extrinsics[0])
            & set(extrinsics[1])
        )
        frame_ids = sorted(valid_ids)
        if not frame_ids:
            raise RuntimeError(f"No valid stereo frames found in {variant_dir}")

        first_path = image_paths_0[frame_ids[0]]
        raw_w, raw_h = Image.open(first_path).size

        return _VKITTITrack(
            name=f"{scene_name}_{variant_dir.name}",
            scene=scene_name,
            variant=variant_dir.name,
            raw_hw=(raw_h, raw_w),
            image_paths_0=image_paths_0,
            image_paths_1=image_paths_1,
            depth_paths_0=depth_paths_0,
            depth_paths_1=depth_paths_1,
            extrinsics_0=extrinsics[0],
            extrinsics_1=extrinsics[1],
            intrinsics_0=intrinsics[0],
            intrinsics_1=intrinsics[1],
            frame_ids=frame_ids,
        )

    def _build_samples(self) -> List[Tuple[int, int]]:
        samples: List[Tuple[int, int]] = []
        for track_idx, track in enumerate(self.tracks):
            for last_idx in range(len(track.frame_ids)):
                samples.append((track_idx, last_idx))
        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        track_idx, last_idx = self.samples[index]
        track = self.tracks[track_idx]

        raw_indices = [
            last_idx - (self.n_time_steps - 1 - t) * self.stride
            for t in range(self.n_time_steps)
        ]
        clamped_indices = [max(0, idx) for idx in raw_indices]
        time_frame_ids = [track.frame_ids[idx] for idx in clamped_indices]

        images_list: List[torch.Tensor] = []
        depths_list: List[torch.Tensor] = []
        extrinsics_list: List[torch.Tensor] = []
        intrinsics_list: List[torch.Tensor] = []

        orig_hw = track.raw_hw

        for t_idx, frame_id in enumerate(time_frame_ids):
            is_last = t_idx == self.n_time_steps - 1
            for camera_idx in (0, 1):
                if camera_idx == 0:
                    image_path = track.image_paths_0[frame_id]
                    depth_path = track.depth_paths_0[frame_id]
                    extrinsics = track.extrinsics_0[frame_id]
                    intrinsics = track.intrinsics_0[frame_id]
                else:
                    image_path = track.image_paths_1[frame_id]
                    depth_path = track.depth_paths_1[frame_id]
                    extrinsics = track.extrinsics_1[frame_id]
                    intrinsics = track.intrinsics_1[frame_id]

                image, _ = preprocess_rgb_like_demo(image_path, self.image_size)
                images_list.append(image)

                if is_last:
                    depth = _load_vkitti_depth_png(depth_path, self.image_size)
                else:
                    depth = torch.full(self.image_size, -1.0, dtype=torch.float32)
                depths_list.append(depth)

                extrinsics_list.append(torch.from_numpy(extrinsics.copy()))
                intrinsics_list.append(
                    torch.from_numpy(
                        resize_intrinsics(intrinsics, orig_hw=orig_hw, out_hw=self.image_size)
                    )
                )

        return {
            "images": torch.stack(images_list, dim=0),
            "depths": torch.stack(depths_list, dim=0),
            "extrinsics": torch.stack(extrinsics_list, dim=0),
            "intrinsics": torch.stack(intrinsics_list, dim=0),
            "sequence_name": f"{track.name}_{time_frame_ids[0]}_{time_frame_ids[-1]}",
        }
