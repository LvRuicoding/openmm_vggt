#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from mmengine.config import Config
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openmm_vggt.datasets.kitti_semantic_occ import KITTISemanticOccupancyDataset


DEFAULT_CONFIG = REPO_ROOT / "configs" / "occupancy" / "kitti_semantic_occ_mix_window_attn_early_ft_monoscene_head_cp_364x1218.py"


def _dataset_cfg(cfg: Config, split: str) -> dict[str, Any]:
    source = cfg.train_dataset if split == "train" else cfg.val_dataset
    dataset_cfg = dict(source)
    dataset_cfg.pop("type", None)
    return dataset_cfg


def _parse_indices(raw_indices: str, dataset_len: int) -> list[int]:
    indices: list[int] = []
    for raw_item in raw_indices.split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        index = int(raw_item)
        if index < 0:
            index = dataset_len + index
        if index < 0 or index >= dataset_len:
            raise IndexError(f"Sample index out of range: {raw_item} -> {index}, dataset_len={dataset_len}")
        indices.append(index)
    return indices


def _project_world_points(
    world_xyz: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(camera_to_world)
    xyz1 = np.concatenate(
        [world_xyz.astype(np.float32), np.ones((world_xyz.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    camera_xyz = (world_to_camera.astype(np.float32) @ xyz1.T).T[:, :3]
    z = camera_xyz[:, 2]
    safe_z = np.maximum(z, 1e-6)
    u = intrinsics[0, 0] * camera_xyz[:, 0] / safe_z + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera_xyz[:, 1] / safe_z + intrinsics[1, 2]
    image_h, image_w = image_hw
    valid = (z > 0) & (u >= 0) & (u < image_w) & (v >= 0) & (v < image_h)
    return u, v, z, valid


def _overlay_points(
    image_path: Path,
    out_hw: tuple[int, int],
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    output_path: Path,
    max_points: int,
) -> None:
    out_h, out_w = out_hw
    image = Image.open(image_path).convert("RGB").resize((out_w, out_h), Image.Resampling.BICUBIC)
    canvas = np.asarray(image).copy()

    valid_indices = np.nonzero(valid)[0]
    if valid_indices.size > max_points:
        valid_indices = valid_indices[np.linspace(0, valid_indices.size - 1, max_points, dtype=np.int64)]

    if valid_indices.size > 0:
        uu = np.rint(u[valid_indices]).astype(np.int32)
        vv = np.rint(v[valid_indices]).astype(np.int32)
        zz = np.clip(z[valid_indices], 0.0, 80.0) / 80.0
        colors = np.stack(
            [
                (255.0 * (1.0 - zz)).astype(np.uint8),
                (255.0 * zz).astype(np.uint8),
                np.full_like(zz, 64.0, dtype=np.float32).astype(np.uint8),
            ],
            axis=1,
        )
        for du, dv in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)):
            x = np.clip(uu + du, 0, out_w - 1)
            y = np.clip(vv + dv, 0, out_h - 1)
            canvas[y, x] = colors

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output_path)


def _path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _frame_mapping(record: Any, semantic_frame_id: str) -> dict[str, Any]:
    image_02_path = record.image_paths_02[semantic_frame_id]
    image_03_path = record.image_paths_03[semantic_frame_id]
    raw_frame_02 = image_02_path.stem
    raw_frame_03 = image_03_path.stem
    try:
        offset_02 = int(raw_frame_02) - int(semantic_frame_id)
        offset_03 = int(raw_frame_03) - int(semantic_frame_id)
    except ValueError:
        offset_02 = None
        offset_03 = None
    return {
        "semantic_frame_id": semantic_frame_id,
        "raw_frame_id_image_02": raw_frame_02,
        "raw_frame_id_image_03": raw_frame_03,
        "raw_minus_semantic_offset_image_02": offset_02,
        "raw_minus_semantic_offset_image_03": offset_03,
        "image_02_path": str(image_02_path),
        "image_03_path": str(image_03_path),
    }


def _class_histogram(target: np.ndarray, valid_mask: np.ndarray, ignore_index: int) -> dict[str, int]:
    valid = valid_mask.astype(bool) & (target != ignore_index)
    values = target[valid]
    if values.size == 0:
        return {}
    labels, counts = np.unique(values.astype(np.int64), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _check_sample(
    dataset: KITTISemanticOccupancyDataset,
    sample_index: int,
    output_dir: Path | None,
    max_overlay_points: int,
) -> dict[str, Any]:
    record_idx, last_idx = dataset.samples[sample_index]
    record = dataset.records[record_idx]

    raw_indices = [
        last_idx - (dataset.n_time_steps - 1 - t) * dataset.stride
        for t in range(dataset.n_time_steps)
    ]
    clamped_indices = [max(0, idx) for idx in raw_indices]
    time_frame_ids = [record.frame_ids[idx] for idx in clamped_indices]
    target_frame_id = time_frame_ids[-1]

    semantic_label_path = record.semantic_root / "labels" / f"{target_frame_id}.label"
    dense_label_path = None
    if dataset.dense_voxel_root is not None:
        dense_label_path = dataset.dense_voxel_root / "sequences" / record.sequence_id / "voxels" / f"{target_frame_id}.label"

    sample = dataset[sample_index]
    images = sample["images"]
    image_h, image_w = int(images.shape[-2]), int(images.shape[-1])
    camera_count = 2
    target_camera_start = (dataset.n_time_steps - 1) * camera_count

    points_world = sample["points"][-1][sample["point_mask"][-1]].numpy()[:, :3]
    if points_world.shape[0] > 0:
        point_indices = np.linspace(
            0,
            points_world.shape[0] - 1,
            min(points_world.shape[0], max_overlay_points * 2),
            dtype=np.int64,
        )
        points_for_projection = points_world[point_indices]
    else:
        points_for_projection = points_world

    camera_reports: list[dict[str, Any]] = []
    for cam_offset, camera_name in enumerate(("image_02", "image_03")):
        camera_slot = target_camera_start + cam_offset
        image_path = record.image_paths_02[target_frame_id] if camera_name == "image_02" else record.image_paths_03[target_frame_id]
        camera_to_world = sample["camera_to_world"][camera_slot].numpy()
        intrinsics = sample["intrinsics"][camera_slot].numpy()
        u, v, z, valid = _project_world_points(
            points_for_projection,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            image_hw=(image_h, image_w),
        )

        overlay_path = None
        if output_dir is not None:
            overlay_path = (
                output_dir
                / f"sample_{sample_index:06d}_seq_{record.sequence_id}_sem_{target_frame_id}_{camera_name}_raw_{image_path.stem}_lidar_overlay.png"
            )
            _overlay_points(
                image_path=image_path,
                out_hw=(image_h, image_w),
                u=u,
                v=v,
                z=z,
                valid=valid,
                output_path=overlay_path,
                max_points=max_overlay_points,
            )

        valid_count = int(valid.sum())
        camera_report = {
            "camera": camera_name,
            "target_camera_slot_in_images": camera_slot,
            "image_path": str(image_path),
            "raw_frame_id": image_path.stem,
            "projected_lidar_points": valid_count,
            "projected_lidar_ratio": float(valid_count / max(points_for_projection.shape[0], 1)),
            "overlay_path": _path_or_none(overlay_path),
        }
        if valid_count > 0:
            camera_report["projected_uv_bounds"] = [
                float(u[valid].min()),
                float(v[valid].min()),
                float(u[valid].max()),
                float(v[valid].max()),
            ]
        else:
            camera_report["projected_uv_bounds"] = []
        camera_reports.append(camera_report)

    occupancy_target = sample["occupancy_target"].numpy()
    occupancy_valid_mask = sample["occupancy_valid_mask"].numpy()

    return {
        "sample_index": sample_index,
        "sequence_id": record.sequence_id,
        "raw_drive_name": record.raw_drive_name,
        "n_time_steps": dataset.n_time_steps,
        "stride": dataset.stride,
        "image_tensor_order": "for each time frame: image_02 then image_03; occupancy_target matches the last time frame",
        "time_frame_mappings": [_frame_mapping(record, frame_id) for frame_id in time_frame_ids],
        "target_semantic_frame_id": target_frame_id,
        "target_image_slots": {
            "image_02": target_camera_start,
            "image_03": target_camera_start + 1,
        },
        "semantic_point_label_path": str(semantic_label_path),
        "semantic_point_label_exists": semantic_label_path.is_file(),
        "dense_voxel_label_path": _path_or_none(dense_label_path),
        "dense_voxel_label_exists": dense_label_path.is_file() if dense_label_path is not None else None,
        "occupancy_target_shape": list(occupancy_target.shape),
        "occupancy_valid_voxels": int(occupancy_valid_mask.sum()),
        "occupancy_class_histogram": _class_histogram(
            occupancy_target,
            occupancy_valid_mask,
            ignore_index=dataset.ignore_index,
        ),
        "target_cameras": camera_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check how KITTISemanticOccupancyDataset pairs SemanticKITTI voxel labels with KITTI raw images."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="MMEngine config path. Defaults to the 364x1218 SemanticKITTI occupancy config.",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--indices", default="0,1,2,-1", help="Comma-separated dataset sample indices. Negative indices are supported.")
    parser.add_argument("--output-json", default="", help="Optional path to save the JSON report.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional directory for target-frame LiDAR projection overlay PNGs.",
    )
    parser.add_argument("--max-overlay-points", type=int, default=20000)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = KITTISemanticOccupancyDataset(**_dataset_cfg(cfg, args.split))
    indices = _parse_indices(args.indices, len(dataset))
    output_dir = Path(args.output_dir) if args.output_dir else None

    reports = [
        _check_sample(
            dataset=dataset,
            sample_index=index,
            output_dir=output_dir,
            max_overlay_points=args.max_overlay_points,
        )
        for index in indices
    ]
    payload = {
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "dataset_len": len(dataset),
        "reports": reports,
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
