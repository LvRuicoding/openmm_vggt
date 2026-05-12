#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from mmengine.config import Config
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openmm_vggt.datasets.kitti_semantic_occ_depth import KITTISemanticOccupancyDepthDataset


def _dataset_cfg(cfg: Config, split: str) -> dict[str, Any]:
    source = cfg.train_dataset if split == "train" else cfg.val_dataset
    dataset_cfg = dict(source)
    dataset_cfg.pop("type", None)
    return dataset_cfg


def _project_world_to_image(
    world_xyz: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(camera_to_world)
    xyz1 = np.concatenate([world_xyz, np.ones((world_xyz.shape[0], 1), dtype=np.float32)], axis=1)
    cam_xyz = (world_to_camera @ xyz1.T).T[:, :3]
    z = cam_xyz[:, 2]
    safe_z = np.maximum(z, 1e-6)
    uvw = (intrinsics @ cam_xyz.T).T
    u = uvw[:, 0] / safe_z
    v = uvw[:, 1] / safe_z
    h, w = image_hw
    valid = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, valid


def _check_sample(dataset: KITTISemanticOccupancyDepthDataset, sample_index: int) -> dict[str, Any]:
    record_idx, last_idx = dataset.samples[sample_index]
    record = dataset.records[record_idx]
    semantic_frame_id = record.frame_ids[last_idx]
    raw_frame_id = record.raw_frame_ids[semantic_frame_id]
    sample = dataset[sample_index]
    image_h, image_w = tuple(int(v) for v in sample["images"].shape[-2:])

    frame_count = int(sample["lidar_to_world"].shape[0])
    cam_num = int(sample["intrinsics"].shape[0] // frame_count)
    intrinsics = sample["intrinsics"].reshape(frame_count, cam_num, 3, 3).numpy()
    camera_to_world = sample["camera_to_world"].reshape(frame_count, cam_num, 4, 4).numpy()

    points_world = sample["points"][-1][sample["point_mask"][-1]].numpy()[:, :3]
    max_points = min(points_world.shape[0], 20000)
    if points_world.shape[0] > max_points:
        points_world = points_world[:max_points]

    camera_reports: list[dict[str, Any]] = []
    for cam_idx, camera_name in enumerate(("image_02", "image_03")):
        image_path = record.image_paths_02[semantic_frame_id] if camera_name == "image_02" else record.image_paths_03[semantic_frame_id]
        depth_path = dataset._get_depth_gt_path(record, semantic_frame_id, camera_name)
        if depth_path is None:
            depth_raw_shape = None
            depth_valid_raw = 0
        else:
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth_raw_shape = list(depth_raw.shape[:2]) if depth_raw is not None else None
            depth_valid_raw = int((depth_raw > 0).sum()) if depth_raw is not None else 0

        raw_w, raw_h = Image.open(image_path).size
        resized_k = intrinsics[-1, cam_idx]
        fx_scale = resized_k[0, 0] / (record.intrinsics_02[0, 0] if camera_name == "image_02" else record.intrinsics_03[0, 0])
        fy_scale = resized_k[1, 1] / (record.intrinsics_02[1, 1] if camera_name == "image_02" else record.intrinsics_03[1, 1])

        u, v, valid = _project_world_to_image(
            points_world,
            camera_to_world[-1, cam_idx],
            resized_k,
            image_hw=(image_h, image_w),
        )
        projected_valid = int(valid.sum())
        projected_ratio = projected_valid / max(int(points_world.shape[0]), 1)
        if projected_valid:
            u_valid = u[valid]
            v_valid = v[valid]
            uv_bounds = [
                float(u_valid.min()),
                float(v_valid.min()),
                float(u_valid.max()),
                float(v_valid.max()),
            ]
        else:
            uv_bounds = []

        depth_tensor = sample["depths"].reshape(frame_count, cam_num, image_h, image_w)[-1, cam_idx]
        camera_reports.append(
            {
                "camera": camera_name,
                "image_path": str(image_path),
                "depth_path": str(depth_path) if depth_path is not None else None,
                "raw_image_hw": [raw_h, raw_w],
                "raw_depth_hw": depth_raw_shape,
                "output_image_hw": [image_h, image_w],
                "depth_valid_raw_pixels": depth_valid_raw,
                "depth_valid_resized_pixels": int((depth_tensor > 0).sum().item()),
                "fx_scale": float(fx_scale),
                "fy_scale": float(fy_scale),
                "expected_x_scale": image_w / float(raw_w),
                "expected_y_scale": image_h / float(raw_h),
                "projected_lidar_points": projected_valid,
                "projected_lidar_ratio": projected_ratio,
                "projected_uv_bounds": uv_bounds,
            }
        )

    return {
        "sample_index": sample_index,
        "sequence_id": record.sequence_id,
        "semantic_frame_id": semantic_frame_id,
        "raw_drive": record.raw_drive_name,
        "raw_frame_id": raw_frame_id,
        "points_checked": int(points_world.shape[0]),
        "cameras": camera_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check SemanticKITTI occupancy depth/camera alignment.")
    parser.add_argument("config", type=str)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--indices", type=str, default="0,-1", help="Comma-separated dataset sample indices.")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = KITTISemanticOccupancyDepthDataset(**_dataset_cfg(cfg, args.split))
    indices = []
    for raw_item in args.indices.split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        index = int(raw_item)
        if index < 0:
            index = len(dataset) + index
        indices.append(index)

    reports = [_check_sample(dataset, index) for index in indices]
    print(json.dumps(reports, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
