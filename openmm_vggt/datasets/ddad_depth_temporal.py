#!/usr/bin/env python3
"""DDAD multi-view depth dataset with optional temporal context."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mmengine.registry import DATASETS

# DGP resolves its cache directory at import time from DGP_PATH / HOME.
# Defaulting to /tmp keeps the dataset usable in restricted environments while
# still allowing callers to override DGP_PATH explicitly.
os.environ.setdefault("DGP_PATH", str(Path(tempfile.gettempdir()) / "openmm_vggt_dgp"))

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset

from .kitti_local_utils import resize_intrinsics


DEFAULT_CAMERA_NAMES = (
    "CAMERA_01",
    "CAMERA_05",
    "CAMERA_06",
    "CAMERA_07",
    "CAMERA_08",
    "CAMERA_09",
)

_SPLIT_TO_KEY = {
    "train": "0",
    "val": "1",
    "test": "2",
    "train_overfit": "3",
}


def _find_scene_json(scene_dir: Path) -> Path:
    matches = sorted(scene_dir.glob("scene_*.json"))
    if not matches:
        raise FileNotFoundError(f"No scene_*.json found in {scene_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected exactly one scene_*.json in {scene_dir}, found {len(matches)}")
    return matches[0]


def _is_scene_dataset_json(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    return isinstance(payload, dict) and "scene_splits" in payload


def _discover_scene_jsons(
    root: Path,
    scene_ids: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> List[Path]:
    if root.is_file():
        return [root]

    if (root / "scene.json").is_file():
        return [root / "scene.json"]

    direct_matches = sorted(root.glob("scene_*.json"))
    if direct_matches:
        return direct_matches

    if scene_ids is None:
        candidate_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    else:
        candidate_dirs = [root / scene_id for scene_id in scene_ids]

    scene_jsons: List[Path] = []
    for scene_dir in candidate_dirs:
        if not scene_dir.is_dir():
            if strict:
                raise FileNotFoundError(f"Missing DDAD scene directory: {scene_dir}")
            continue
        try:
            scene_jsons.append(_find_scene_json(scene_dir))
        except Exception:
            if strict:
                raise
            continue
    return scene_jsons


def _build_scene_dataset_json(scene_jsons: Sequence[Path], split: str) -> Path:
    split_key = _SPLIT_TO_KEY.get(split, "0")
    digest = hashlib.sha1(
        "\n".join(str(path.resolve()) for path in scene_jsons).encode("utf-8")
    ).hexdigest()[:12]
    wrapper_dir = Path(tempfile.gettempdir()) / "openmm_vggt_ddad_scene_datasets"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = wrapper_dir / f"ddad_{split}_{digest}.json"
    if wrapper_path.is_file():
        return wrapper_path

    payload = {
        "metadata": {
            "name": f"ddad_{split}_{digest}",
            "version": "1.0",
            "creation_date": "2026-04-14T00:00:00Z",
            "creator": "openmm_vggt",
            "origin": "INTERNAL",
            "available_annotation_types": [],
            "metadata": {},
            "description": "Auto-generated single-use DDAD scene dataset wrapper",
            "frame_per_second": 0.0,
        },
        "scene_splits": {
            split_key: {
                "filenames": [str(path.resolve()) for path in scene_jsons],
            }
        },
    }
    wrapper_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return wrapper_path


def _preprocess_pil_rgb(image: Image.Image, out_hw: Tuple[int, int]) -> torch.Tensor:
    out_h, out_w = out_hw
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    image = image.convert("RGB")
    image = image.resize((out_w, out_h), Image.Resampling.BICUBIC)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1)


def _resize_depth_map(depth_map: np.ndarray, out_hw: Tuple[int, int]) -> torch.Tensor:
    out_h, out_w = out_hw
    resized = cv2.resize(depth_map.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    resized[resized <= 0.0] = -1.0
    return torch.from_numpy(resized)


@DATASETS.register_module()
class DDADDepthTemporalDataset(Dataset):
    """DDAD multi-view depth dataset with temporal context.

    Images are returned in time-major order:
    ``[t0_cam0, t0_cam1, ..., t0_camN, t1_cam0, ...]``.
    Only the last time step carries depth supervision; earlier steps receive
    depth maps filled with ``-1`` to match the KITTI training setup.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        scene_ids: Optional[Sequence[str]] = None,
        camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
        lidar_name: str = "LIDAR",
        n_time_steps: int = 1,
        stride: int = 1,
        image_size: Tuple[int, int] = (280, 518),
        strict: bool = False,
        max_sequences: Optional[int] = None,
        max_samples: Optional[int] = None,
        return_lidar: bool = True,
        max_lidar_points: int = 32768,
    ) -> None:
        assert n_time_steps >= 1, "n_time_steps must be >= 1"
        assert image_size[0] % 14 == 0 and image_size[1] % 14 == 0, (
            f"image_size {image_size} must have both dims as multiples of 14"
        )

        self.root = Path(root)
        self.split = split
        self.scene_ids = tuple(scene_ids) if scene_ids is not None else None
        self.camera_names = tuple(camera_names)
        self.lidar_name = lidar_name
        self.n_time_steps = n_time_steps
        self.stride = stride
        self.image_size = image_size
        self.strict = strict
        self.max_sequences = max_sequences
        self.max_samples = max_samples
        self.return_lidar = return_lidar
        self.max_lidar_points = max_lidar_points
        self.seq_len = n_time_steps * len(self.camera_names)

        dataset_root: Optional[str] = None
        if _is_scene_dataset_json(self.root):
            wrapper_json = self.root.resolve()
            dataset_root = str(wrapper_json.parent)
            self.scene_jsons = []
            self.scene_names = []
        else:
            scene_jsons = _discover_scene_jsons(self.root, scene_ids=self.scene_ids, strict=self.strict)
            if not scene_jsons:
                raise RuntimeError(
                    f"No DDAD scene JSONs found under root={self.root} scene_ids={self.scene_ids}"
                )
            if self.max_sequences is not None:
                scene_jsons = scene_jsons[: self.max_sequences]
            self.scene_jsons = [path.resolve() for path in scene_jsons]
            self.scene_names = [path.parent.name for path in self.scene_jsons]
            wrapper_json = _build_scene_dataset_json(self.scene_jsons, split=self.split)

        datum_names = list(self.camera_names)
        if self.return_lidar:
            datum_names.append(self.lidar_name)

        self.ddad = SynchronizedSceneDataset(
            str(wrapper_json),
            split=self.split,
            datum_names=datum_names,
            generate_depth_from_datum=self.lidar_name if self.return_lidar else None,
            dataset_root=dataset_root,
        )

        if not self.scene_names:
            split_key = _SPLIT_TO_KEY.get(self.split, "0")
            with wrapper_json.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            scene_files = payload["scene_splits"][split_key]["filenames"]
            base_root = wrapper_json.parent if dataset_root is None else Path(dataset_root)
            self.scene_names = [Path(base_root / scene_file).parent.name for scene_file in scene_files]

        self.scene_to_global: Dict[int, List[Tuple[int, int]]] = {}
        for global_idx, (scene_idx, sample_idx_in_scene, _) in enumerate(self.ddad.dataset_item_index):
            self.scene_to_global.setdefault(scene_idx, []).append((sample_idx_in_scene, global_idx))
        for scene_idx in list(self.scene_to_global.keys()):
            self.scene_to_global[scene_idx] = sorted(self.scene_to_global[scene_idx], key=lambda pair: pair[0])

        self.samples: List[Tuple[int, int]] = []
        for scene_idx, entries in sorted(self.scene_to_global.items()):
            for local_pos in range(len(entries)):
                self.samples.append((scene_idx, local_pos))
        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]
        if not self.samples:
            raise RuntimeError(
                f"No DDAD samples found for split={self.split} under root={self.root}"
            )

        print(
            f"[DDADDepthTemporalDataset] split={self.split}: "
            f"{len(self.scene_names)} scenes, {len(self.samples)} samples loaded.",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_world_lidar_points(self, lidar_datum) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = np.asarray(lidar_datum["point_cloud"], dtype=np.float32)
        extras = lidar_datum.get("extra_channels")
        if extras is not None:
            extras = np.asarray(extras, dtype=np.float32)
            intensity = extras[:, :1] if extras.ndim == 2 and extras.shape[1] > 0 else np.zeros((xyz.shape[0], 1), dtype=np.float32)
        else:
            intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)

        world_xyz = lidar_datum["pose"] * xyz
        world_points = np.concatenate([world_xyz.astype(np.float32), intensity.astype(np.float32)], axis=1)
        if world_points.shape[0] > self.max_lidar_points:
            indices = np.linspace(0, world_points.shape[0] - 1, self.max_lidar_points, dtype=np.int64)
            world_points = world_points[indices]

        point_tensor = torch.zeros((self.max_lidar_points, 4), dtype=torch.float32)
        point_mask = torch.zeros((self.max_lidar_points,), dtype=torch.bool)
        if world_points.shape[0] > 0:
            count = min(world_points.shape[0], self.max_lidar_points)
            valid = torch.from_numpy(world_points[:count])
            point_tensor[:count] = valid
            point_mask[:count] = True
        return point_tensor, point_mask

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
                rgb: Image.Image = cam_datum["rgb"]
                orig_hw = (rgb.height, rgb.width)

                images_list.append(_preprocess_pil_rgb(rgb, self.image_size))

                if is_last and "depth" in cam_datum:
                    depths_list.append(_resize_depth_map(cam_datum["depth"], self.image_size))
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
