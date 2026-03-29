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


VKITTI_INTRINSICS = np.array(
    [[725.0, 0.0, 620.5], [0.0, 725.0, 187.0], [0.0, 0.0, 1.0]], dtype=np.float32
)


@dataclass
class SequenceInfo:
    name: str
    image_paths: List[Path]
    depth_paths: List[Path]
    extrinsics: np.ndarray  # [N, 3, 4]


@DATASETS.register_module()
class VKITTI1SequenceDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        num_frames: int,
        stride: int,
        image_size: Tuple[int, int],
        variations: Sequence[str] | None = None,
        worlds: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.stride = stride
        self.image_size = image_size  # (H, W)
        self.variations = set(variations) if variations else None
        self.worlds = set(worlds) if worlds else None

        self.sequences = self._load_sequences()
        self.samples = self._build_samples()

        if not self.samples:
            raise RuntimeError(f"No samples found for split={split} under {root}")

    def _load_sequences(self) -> List[SequenceInfo]:
        rgb_root = self.root / "vkitti_1.3.1_rgb"
        depth_root = self.root / "vkitti_1.3.1_depthgt"
        extr_root = self.root / "vkitti_1.3.1_extrinsicsgt"

        for required_dir in (rgb_root, depth_root, extr_root):
            if not required_dir.is_dir():
                raise FileNotFoundError(f"Required dataset directory not found: {required_dir}")

        sequences: List[SequenceInfo] = []
        extrinsic_files = sorted(extr_root.glob("*.txt"))
        if not extrinsic_files:
            raise FileNotFoundError(f"No extrinsics files found under {extr_root}")

        for extr_path in extrinsic_files:
            if extr_path.name.startswith("README"):
                continue

            stem = extr_path.stem
            world, variation = stem.split("_", 1)
            if self.worlds is not None and world not in self.worlds:
                continue
            if self.variations is not None and variation not in self.variations:
                continue

            rgb_dir = rgb_root / world / variation
            depth_dir = depth_root / world / variation
            if not rgb_dir.is_dir():
                raise FileNotFoundError(f"Missing RGB directory for sequence {stem}: {rgb_dir}")
            if not depth_dir.is_dir():
                raise FileNotFoundError(f"Missing depth directory for sequence {stem}: {depth_dir}")

            raw_rows = np.loadtxt(extr_path, skiprows=1, dtype=np.float32)
            if raw_rows.ndim == 1:
                raw_rows = raw_rows[None]
            extrinsics_by_frame: Dict[int, np.ndarray] = {}
            for row in raw_rows:
                frame_id = int(row[0])
                values = row[1:].reshape(4, 4)
                extrinsics_by_frame[frame_id] = values[:3, :4]

            image_paths: List[Path] = []
            depth_paths: List[Path] = []
            extrinsics: List[np.ndarray] = []
            for image_path in sorted(rgb_dir.glob("*.png")):
                frame_id = int(image_path.stem)
                depth_path = depth_dir / image_path.name
                if frame_id not in extrinsics_by_frame:
                    raise FileNotFoundError(f"Missing extrinsics for frame {frame_id} in sequence {stem}")
                if not depth_path.is_file():
                    raise FileNotFoundError(f"Missing depth file for frame {frame_id} in sequence {stem}: {depth_path}")
                image_paths.append(image_path)
                depth_paths.append(depth_path)
                extrinsics.append(extrinsics_by_frame[frame_id])

            if len(image_paths) < self.num_frames:
                continue

            sequences.append(
                SequenceInfo(
                    name=stem,
                    image_paths=image_paths,
                    depth_paths=depth_paths,
                    extrinsics=np.stack(extrinsics, axis=0),
                )
            )

        sequences = sorted(sequences, key=lambda x: x.name)
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

    def _build_samples(self) -> List[Tuple[int, int]]:
        samples: List[Tuple[int, int]] = []
        span = (self.num_frames - 1) * self.stride + 1
        for seq_idx, seq in enumerate(self.sequences):
            max_start = len(seq.image_paths) - span
            for start in range(max_start + 1):
                samples.append((seq_idx, start))
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
            depth = depth / 100.0  # VKITTI stores depth in centimeters

            intrinsic = self._resize_intrinsics(VKITTI_INTRINSICS, (orig_h, orig_w))

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
