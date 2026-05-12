#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from mmengine.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openmm_vggt.datasets.kitti_semantic_occ import KITTISemanticOccupancyDataset


def _build_dataset(cfg: Config, split: str, max_samples: int | None) -> KITTISemanticOccupancyDataset:
    dataset_cfg = dict(cfg.train_dataset if split == "train" else cfg.val_dataset)
    dataset_cfg.pop("type", None)
    if max_samples is not None:
        dataset_cfg["max_samples"] = max_samples
    return KITTISemanticOccupancyDataset(**dataset_cfg)


def _coarse_voxel_centers(
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    project_scale: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid_size = ((point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size).round().to(torch.long)
    coarse_grid_size = grid_size // int(project_scale)
    grid_x, grid_y, grid_z = [int(v.item()) for v in coarse_grid_size]
    xs = torch.arange(grid_x, device=device, dtype=dtype)
    ys = torch.arange(grid_y, device=device, dtype=dtype)
    zs = torch.arange(grid_z, device=device, dtype=dtype)
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
    coarse_voxel_size = voxel_size.to(device=device, dtype=dtype) * int(project_scale)
    centers = point_cloud_range[:3].to(device=device, dtype=dtype) + (grid + 0.5) * coarse_voxel_size
    return centers.reshape(-1, 3), grid_size, coarse_grid_size


def _project_valid_mask(
    centers_lidar: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    lidar_to_world: torch.Tensor,
    image_hw: tuple[int, int],
    chunk_size: int,
) -> torch.Tensor:
    image_h, image_w = image_hw
    view_count = int(intrinsics.shape[0])
    any_valid_chunks = []
    world_to_camera = torch.inverse(camera_to_world)

    for start in range(0, centers_lidar.shape[0], chunk_size):
        end = min(start + chunk_size, centers_lidar.shape[0])
        centers = centers_lidar[start:end]
        centers_h = torch.cat([centers, torch.ones_like(centers[..., :1])], dim=-1)
        centers_world = torch.matmul(lidar_to_world.view(1, 4, 4), centers_h.view(1, -1, 4, 1)).squeeze(-1)
        centers_world = centers_world.expand(view_count, -1, -1)
        centers_cam = torch.matmul(world_to_camera[:, None, :, :], centers_world.unsqueeze(-1)).squeeze(-1)
        xyz_cam = centers_cam[..., :3]

        x = xyz_cam[..., 0]
        y = xyz_cam[..., 1]
        z_raw = xyz_cam[..., 2]
        z = z_raw.clamp(min=1e-5)

        fx = intrinsics[..., 0, 0].unsqueeze(-1)
        fy = intrinsics[..., 1, 1].unsqueeze(-1)
        cx = intrinsics[..., 0, 2].unsqueeze(-1)
        cy = intrinsics[..., 1, 2].unsqueeze(-1)
        u = fx * x / z + cx
        v = fy * y / z + cy
        valid = (z_raw > 0) & (u >= 0) & (u <= image_w - 1) & (v >= 0) & (v <= image_h - 1)
        any_valid_chunks.append(valid.any(dim=0))

    return torch.cat(any_valid_chunks, dim=0)


def _sample_identity(sample: dict[str, Any]) -> dict[str, Any]:
    meta = sample.get("meta")
    if isinstance(meta, dict):
        return {k: str(v) for k, v in meta.items()}
    return {}


def _worker(args: tuple[str, str, list[int], int, str, int, str]) -> list[dict[str, Any]]:
    config_path, split, sample_indices, worker_rank, device_name, chunk_size, view_mode = args
    cfg = Config.fromfile(config_path)
    dataset = _build_dataset(cfg, split=split, max_samples=None)

    head_cfg = dict(cfg.model["occupancy_head"])
    voxel_size = torch.tensor(head_cfg["voxel_size"], dtype=torch.float32)
    point_cloud_range = torch.tensor(head_cfg["point_cloud_range"], dtype=torch.float32)
    project_scale = int(head_cfg.get("project_scale", 1))

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    centers, grid_size, coarse_grid_size = _coarse_voxel_centers(
        voxel_size=voxel_size.to(device),
        point_cloud_range=point_cloud_range.to(device),
        project_scale=project_scale,
        device=device,
        dtype=torch.float32,
    )
    coarse_total = int(centers.shape[0])
    full_block = int(project_scale) ** 3

    results: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        sample = dataset[int(sample_index)]
        images = sample["images"]
        image_h, image_w = int(images.shape[-2]), int(images.shape[-1])

        frame_count = int(sample["lidar_to_world"].shape[0])
        cam_num = int(sample["intrinsics"].shape[0] // frame_count)
        intrinsics = sample["intrinsics"].reshape(frame_count, cam_num, 3, 3)
        camera_to_world = sample["camera_to_world"].reshape(frame_count, cam_num, 4, 4)
        if view_mode == "last-frame":
            intrinsics = intrinsics[-1].reshape(cam_num, 3, 3).to(device)
            camera_to_world = camera_to_world[-1].reshape(cam_num, 4, 4).to(device)
        elif view_mode == "all":
            intrinsics = intrinsics.permute(1, 0, 2, 3).reshape(frame_count * cam_num, 3, 3).to(device)
            camera_to_world = camera_to_world.permute(1, 0, 2, 3).reshape(frame_count * cam_num, 4, 4).to(device)
        else:
            raise ValueError(f"Unsupported view_mode: {view_mode}")
        lidar_to_world = sample["lidar_to_world"][-1].to(device)

        with torch.no_grad():
            any_valid = _project_valid_mask(
                centers_lidar=centers,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                lidar_to_world=lidar_to_world,
                image_hw=(image_h, image_w),
                chunk_size=chunk_size,
            )
        covered = int(any_valid.sum().item())
        missing = coarse_total - covered
        results.append(
            {
                "sample_index": int(sample_index),
                "worker_rank": int(worker_rank),
                "device": str(device),
                "image_hw": [image_h, image_w],
                "view_mode": view_mode,
                "views": int(intrinsics.shape[0]),
                "grid_size": [int(v) for v in grid_size.cpu().tolist()],
                "coarse_grid_size": [int(v) for v in coarse_grid_size.cpu().tolist()],
                "project_scale": int(project_scale),
                "coarse_total": coarse_total,
                "coarse_covered": covered,
                "coarse_missing": missing,
                "coarse_missing_ratio": missing / coarse_total,
                "full_grid_equiv_total": coarse_total * full_block,
                "full_grid_equiv_missing": missing * full_block,
                "identity": _sample_identity(sample),
            }
        )
    return results


def _split_evenly(items: list[int], parts: int) -> list[list[int]]:
    return [items[i::parts] for i in range(parts)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Count KITTI occupancy lifting voxels without any projected 2D feature.")
    parser.add_argument("config", type=str, help="Path to the occupancy config.")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument(
        "--view-mode",
        choices=("last-frame", "all"),
        default="last-frame",
        help="Use only the last frame stereo pair, or all temporal stereo views passed by the current code.",
    )
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = _build_dataset(cfg, split=args.split, max_samples=None)
    rng = random.Random(args.seed)
    sample_count = min(int(args.num_samples), len(dataset))
    sample_indices = rng.sample(range(len(dataset)), sample_count)
    del dataset

    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    if not devices:
        devices = ["cpu"]
    worker_count = min(len(devices), sample_count) if sample_count > 0 else 1
    shards = _split_evenly(sample_indices, worker_count)
    worker_args = [
        (args.config, args.split, shard, worker_rank, devices[worker_rank], int(args.chunk_size), args.view_mode)
        for worker_rank, shard in enumerate(shards)
        if shard
    ]

    all_results: list[dict[str, Any]] = []
    if len(worker_args) == 1:
        all_results.extend(_worker(worker_args[0]))
    else:
        with ProcessPoolExecutor(max_workers=len(worker_args)) as executor:
            futures = [executor.submit(_worker, item) for item in worker_args]
            for future in as_completed(futures):
                all_results.extend(future.result())

    all_results.sort(key=lambda item: item["sample_index"])
    total_missing = sum(item["coarse_missing"] for item in all_results)
    total_voxels = sum(item["coarse_total"] for item in all_results)
    summary = {
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "seed": int(args.seed),
        "sample_indices": sample_indices,
        "view_mode": args.view_mode,
        "num_samples": len(all_results),
        "coarse_missing_total": total_missing,
        "coarse_voxel_total": total_voxels,
        "coarse_missing_ratio": total_missing / total_voxels if total_voxels else math.nan,
        "full_grid_equiv_missing_total": sum(item["full_grid_equiv_missing"] for item in all_results),
        "full_grid_equiv_total": sum(item["full_grid_equiv_total"] for item in all_results),
        "samples": all_results,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
