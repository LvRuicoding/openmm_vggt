#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mmengine.registry import DATASETS


@dataclass
class SequenceInfo:
    name: str
    image_paths: List[Path]
    depth_paths: List[Path]
    extrinsics: np.ndarray  # [N, 3, 4]
    intrinsics: np.ndarray  # [N, 3, 3]


def _camera_name_to_id(camera_name: str) -> int:
    try:
        return int(camera_name.split("_")[-1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unsupported camera name: {camera_name}") from exc


def _frame_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


@DATASETS.register_module()
class VKITTI2SequenceDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        num_frames: int,
        stride: int,
        image_size: Tuple[int, int],
        variations: Sequence[str] | None = None,
        worlds: Sequence[str] | None = None,
        cameras: Sequence[str] | None = None,
        max_sequences: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.stride = stride
        self.image_size = image_size  # (H, W)
        self.variations = set(variations) if variations else None
        self.worlds = set(worlds) if worlds else None
        self.cameras = tuple(cameras) if cameras else ("Camera_0",)
        self.max_sequences = max_sequences
        self.max_samples = max_samples

        self.sequences = self._load_sequences()
        self.samples = self._build_samples()

        if not self.samples:
            raise RuntimeError(f"No samples found for split={split} under {root}")

    def _read_extrinsics(self, path: Path) -> Dict[Tuple[int, int], np.ndarray]:
        raw_rows = np.loadtxt(path, skiprows=1, dtype=np.float32)
        if raw_rows.ndim == 1:
            raw_rows = raw_rows[None]

        extrinsics: Dict[Tuple[int, int], np.ndarray] = {}
        for row in raw_rows:
            frame_id = int(row[0])
            camera_id = int(row[1])
            values = row[2:].reshape(4, 4)
            extrinsics[(frame_id, camera_id)] = values[:3, :4]
        return extrinsics

    def _read_intrinsics(self, path: Path) -> Dict[Tuple[int, int], np.ndarray]:
        raw_rows = np.loadtxt(path, skiprows=1, dtype=np.float32)
        if raw_rows.ndim == 1:
            raw_rows = raw_rows[None]

        intrinsics: Dict[Tuple[int, int], np.ndarray] = {}
        for row in raw_rows:
            frame_id = int(row[0])
            camera_id = int(row[1])
            fx, fy, cx, cy = row[2:].tolist()
            intrinsics[(frame_id, camera_id)] = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
        return intrinsics

    def _split_sequences(self, sequences: List[SequenceInfo]) -> List[SequenceInfo]:
        sequences = sorted(sequences, key=lambda x: x.name)
        if self.max_sequences is not None:
            sequences = sequences[: self.max_sequences]

        if len(sequences) < 2:
            return sequences

        split_idx = max(1, int(math.floor(len(sequences) * 0.8)))
        if self.split == "train":
            return sequences[:split_idx]
        if self.split == "val":
            return sequences[split_idx:]
        if self.split == "all":
            return sequences
        raise ValueError(f"Unsupported split: {self.split}")

    def _load_sequences(self) -> List[SequenceInfo]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        sequences: List[SequenceInfo] = []
        for world_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            world = world_dir.name
            if self.worlds is not None and world not in self.worlds:
                continue

            for variation_dir in sorted(p for p in world_dir.iterdir() if p.is_dir()):
                variation = variation_dir.name
                if self.variations is not None and variation not in self.variations:
                    continue

                rgb_root = variation_dir / "frames" / "rgb"
                depth_root = variation_dir / "frames" / "depth"
                extr_path = variation_dir / "extrinsic.txt"
                intr_path = variation_dir / "intrinsic.txt"
                if not rgb_root.is_dir() or not depth_root.is_dir():
                    continue
                if not extr_path.is_file() or not intr_path.is_file():
                    continue

                extrinsics_by_key = self._read_extrinsics(extr_path)
                intrinsics_by_key = self._read_intrinsics(intr_path)

                for camera_name in self.cameras:
                    camera_id = _camera_name_to_id(camera_name)
                    rgb_dir = rgb_root / camera_name
                    depth_dir = depth_root / camera_name
                    if not rgb_dir.is_dir() or not depth_dir.is_dir():
                        continue

                    image_paths: List[Path] = []
                    depth_paths: List[Path] = []
                    extrinsics: List[np.ndarray] = []
                    intrinsics: List[np.ndarray] = []
                    for image_path in sorted(rgb_dir.glob("rgb_*.jpg")):
                        frame_id = _frame_id_from_path(image_path)
                        depth_path = depth_dir / f"depth_{frame_id:05d}.png"
                        key = (frame_id, camera_id)
                        if key not in extrinsics_by_key or key not in intrinsics_by_key:
                            continue
                        if not depth_path.is_file():
                            continue
                        image_paths.append(image_path)
                        depth_paths.append(depth_path)
                        extrinsics.append(extrinsics_by_key[key])
                        intrinsics.append(intrinsics_by_key[key])

                    if len(image_paths) < self.num_frames:
                        continue

                    sequences.append(
                        SequenceInfo(
                            name=f"{world}_{variation}_{camera_name}",
                            image_paths=image_paths,
                            depth_paths=depth_paths,
                            extrinsics=np.stack(extrinsics, axis=0),
                            intrinsics=np.stack(intrinsics, axis=0),
                        )
                    )

        return self._split_sequences(sequences)

    def _build_samples(self) -> List[Tuple[int, int]]:
        samples: List[Tuple[int, int]] = []
        span = (self.num_frames - 1) * self.stride + 1
        for seq_idx, seq in enumerate(self.sequences):
            max_start = len(seq.image_paths) - span
            for start in range(max_start + 1):
                samples.append((seq_idx, start))

        if self.max_samples is not None:
            samples = samples[: self.max_samples]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_intrinsics(self, intrinsic: np.ndarray, orig_hw: Tuple[int, int]) -> np.ndarray:
        orig_h, orig_w = orig_hw
        out_h, out_w = self.image_size
        sx = out_w / orig_w
        sy = out_h / orig_h
        k = intrinsic.copy()
        k[0, 0] *= sx
        k[0, 2] *= sx
        k[1, 1] *= sy
        k[1, 2] *= sy
        return k

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seq_idx, start = self.samples[index]
        seq = self.sequences[seq_idx]
        frame_ids = [start + i * self.stride for i in range(self.num_frames)]

        images = []
        depths = []
        extrinsics = []
        intrinsics = []

        for frame_id in frame_ids:
            image = Image.open(seq.image_paths[frame_id]).convert("RGB")
            orig_w, orig_h = image.size
            image = image.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BICUBIC)
            image = np.asarray(image, dtype=np.float32) / 255.0

            depth = cv2.imread(str(seq.depth_paths[frame_id]), cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise FileNotFoundError(seq.depth_paths[frame_id])
            depth = cv2.resize(
                depth,
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)
            depth = depth / 100.0  # VKITTI2 stores depth in centimeters

            intrinsic = self._resize_intrinsics(seq.intrinsics[frame_id], (orig_h, orig_w))

            images.append(torch.from_numpy(image).permute(2, 0, 1))
            depths.append(torch.from_numpy(depth))
            extrinsics.append(torch.from_numpy(seq.extrinsics[frame_id].copy()))
            intrinsics.append(torch.from_numpy(intrinsic))

        return {
            "images": torch.stack(images, dim=0),  # [S, 3, H, W]
            "depths": torch.stack(depths, dim=0),  # [S, H, W]
            "extrinsics": torch.stack(extrinsics, dim=0),  # [S, 3, 4]
            "intrinsics": torch.stack(intrinsics, dim=0),  # [S, 3, 3]
            "sequence_name": seq.name,
        }
