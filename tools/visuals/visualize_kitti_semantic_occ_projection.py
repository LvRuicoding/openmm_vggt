#!/usr/bin/env python3
"""Visualize the final 3D projection used by the KITTI semantic occupancy model.

The script mirrors the geometry inside ``OccupancyHead``:
1. Load one sample from ``KITTISemanticOccupancyDataset``.
2. Run ``mix_decoder_global_window_attn_early_occ`` on that sample.
3. Take the last-timestep stereo depth prediction.
4. Downsample depth to patch resolution and back-project patch centers to 3D.
5. Transform those 3D points into the last LiDAR frame.
6. Save 3D/BEV/side-view plots together with RGB overlay plots.

Example:
  python tools/visuals/visualize_kitti_semantic_occ_projection.py \
      0 \
      /tmp/kitti_occ_projection
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openmm-vggt")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS

import openmm_vggt  # noqa: F401


DEFAULT_CONFIG = REPO_ROOT / "configs" / "occupancy" / "kitti_semantic_occ_mix_window_attn_early_ft.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize how the KITTI semantic occupancy model projects the last stereo pair into 3D space."
    )
    parser.add_argument("sample_index", type=int, help="Dataset sample index.")
    parser.add_argument("output_dir", type=str, help="Directory to save visualization results.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to the occupancy config.",
    )
    parser.add_argument(
        "--split",
        choices=("auto", "train", "val"),
        default="auto",
        help=(
            "Dataset split to visualize. "
            "'auto' treats sample_index as a global index over train first, then val."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to cfg.checkpoint.",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default=None,
        help=(
            "Model weights path. This is an alias with higher priority than --checkpoint. "
            "Weights are loaded non-strictly: matching tensors are reused, while missing "
            "or shape-mismatched tensors are skipped."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device, e.g. cpu, cuda, cuda:0. Default uses cuda when available.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution. Equivalent to --device cpu.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for point subsampling.")
    parser.add_argument("--max-lidar-points", type=int, default=12000, help="Maximum raw LiDAR points to draw.")
    parser.add_argument(
        "--max-projected-patches",
        type=int,
        default=2500,
        help="Maximum projected patch centers to draw per camera.",
    )
    parser.add_argument(
        "--max-voxel-cuboids",
        type=int,
        default=600,
        help="Maximum voxel-feature cuboids to draw.",
    )
    parser.add_argument(
        "--voxel-cuboid-alpha",
        type=float,
        default=0.08,
        help="Transparency for voxel-feature cuboids.",
    )
    parser.add_argument(
        "--max-patch-hit-voxels",
        type=int,
        default=2000,
        help="Maximum patch-hit voxels to draw in the patch-occupancy heat visualization.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def sample_indices(total: int, limit: int, rng: random.Random) -> np.ndarray:
    if total <= limit:
        return np.arange(total, dtype=np.int64)
    return np.asarray(sorted(rng.sample(range(total), limit)), dtype=np.int64)


def normalize_extrinsics_to_first_frame(extrinsics: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, _, _ = extrinsics.shape
    extrinsics_h = torch.zeros((batch_size, sequence_length, 4, 4), dtype=extrinsics.dtype, device=extrinsics.device)
    extrinsics_h[..., :3, :] = extrinsics
    extrinsics_h[..., 3, 3] = 1.0
    first_inv = torch.inverse(extrinsics_h[:, 0])
    normalized = torch.matmul(extrinsics_h, first_inv.unsqueeze(1))
    return normalized[..., :3, :]


def build_model_others(batch: dict[str, torch.Tensor], normalized_extrinsics: torch.Tensor) -> dict[str, torch.Tensor]:
    others = {
        "extrinsics": normalized_extrinsics,
        "intrinsics": batch["intrinsics"],
    }
    for key in ("camera_to_world", "lidar_to_world", "points", "point_mask"):
        if key in batch:
            others[key] = batch[key]
    return others


def load_partial_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:]: value for key, value in state_dict.items()}

    model_state = model.state_dict()
    loaded = {}
    missing = []
    mismatched = []
    for key, target in model_state.items():
        if key not in state_dict:
            missing.append(key)
            continue
        source = state_dict[key]
        if tuple(source.shape) != tuple(target.shape):
            mismatched.append((key, tuple(target.shape), tuple(source.shape)))
            continue
        loaded[key] = source

    model.load_state_dict(loaded, strict=False)
    return {
        "loaded_tensors": len(loaded),
        "missing_tensors": len(missing),
        "mismatched_tensors": len(mismatched),
        "missing_examples": missing[:10],
        "mismatched_examples": mismatched[:5],
    }


def build_dataset(cfg: Config, split: str):
    dataset_key = f"{split}_dataset"
    if dataset_key not in cfg:
        raise KeyError(f"Config does not contain {dataset_key}")
    dataset_cfg = copy.deepcopy(cfg[dataset_key])
    return DATASETS.build(dataset_cfg)


def resolve_dataset(cfg: Config, split: str, sample_index: int):
    if split in ("train", "val"):
        dataset = build_dataset(cfg, split)
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError(f"sample_index {sample_index} out of range for split={split}: [0, {len(dataset) - 1}]")
        return dataset, split, sample_index, {
            "mode": "explicit",
            "requested_split": split,
            "requested_index": sample_index,
            "resolved_index": sample_index,
            "train_len": None,
            "val_len": None,
        }

    train_dataset = build_dataset(cfg, "train")
    val_dataset = build_dataset(cfg, "val")
    train_len = len(train_dataset)
    val_len = len(val_dataset)
    if sample_index < 0 or sample_index >= train_len + val_len:
        raise IndexError(
            f"sample_index {sample_index} out of combined range [0, {train_len + val_len - 1}] "
            f"for split=auto (train={train_len}, val={val_len})"
        )
    if sample_index < train_len:
        return train_dataset, "train", sample_index, {
            "mode": "auto",
            "requested_split": "auto",
            "requested_index": sample_index,
            "resolved_index": sample_index,
            "train_len": train_len,
            "val_len": val_len,
        }
    resolved_index = sample_index - train_len
    return val_dataset, "val", resolved_index, {
        "mode": "auto",
        "requested_split": "auto",
        "requested_index": sample_index,
        "resolved_index": resolved_index,
        "train_len": train_len,
        "val_len": val_len,
    }


def extract_sample_meta(dataset, sample_index: int) -> dict[str, Any]:
    meta: dict[str, Any] = {"sample_index": sample_index}
    if not hasattr(dataset, "samples") or not hasattr(dataset, "records"):
        return meta

    record_idx, last_idx = dataset.samples[sample_index]
    record = dataset.records[record_idx]
    raw_indices = [
        last_idx - (dataset.n_time_steps - 1 - t) * dataset.stride
        for t in range(dataset.n_time_steps)
    ]
    clamped_indices = [max(0, idx) for idx in raw_indices]
    time_frame_ids = [record.frame_ids[idx] for idx in clamped_indices]
    meta.update(
        {
            "sequence_id": record.sequence_id,
            "raw_drive_name": record.raw_drive_name,
            "time_frame_ids": time_frame_ids,
            "target_frame_id": time_frame_ids[-1],
        }
    )
    return meta


def collate_single_sample(sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
    return batch


def tensor_image_to_rgb(image: torch.Tensor) -> np.ndarray:
    return np.clip(image.detach().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)


def build_patch_center_grid(
    patch_h: int,
    patch_w: int,
    patch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys = torch.arange(patch_h, device=device, dtype=dtype)
    xs = torch.arange(patch_w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    u = (grid_x + 0.5) * float(patch_size)
    v = (grid_y + 0.5) * float(patch_size)
    ones = torch.ones_like(u)
    return torch.stack([u, v, ones], dim=-1).reshape(-1, 3)


def project_last_frame_to_lidar(
    last_frame_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    lidar_to_world: torch.Tensor,
    patch_size: int,
    depth_scale: float,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
) -> dict[str, torch.Tensor]:
    cam_num, image_h, image_w = last_frame_depth.shape
    patch_h = image_h // patch_size
    patch_w = image_w // patch_size
    patch_count = patch_h * patch_w

    depth_patch = F.interpolate(
        (last_frame_depth * depth_scale).unsqueeze(1),
        size=(patch_h, patch_w),
        mode="area",
    ).squeeze(1)
    depth_patch_flat = depth_patch.reshape(cam_num, patch_count)
    patch_uv1 = build_patch_center_grid(patch_h, patch_w, patch_size, last_frame_depth.device, last_frame_depth.dtype)
    patch_uv1 = patch_uv1.view(1, patch_count, 3).expand(cam_num, -1, -1)

    inv_intrinsics = torch.inverse(intrinsics)
    rays = torch.matmul(inv_intrinsics.unsqueeze(1), patch_uv1.unsqueeze(-1)).squeeze(-1)
    cam_points = rays * depth_patch_flat.unsqueeze(-1).clamp_min(0.0)

    cam_points_h = torch.cat([cam_points, torch.ones_like(cam_points[..., :1])], dim=-1)
    world_points = torch.matmul(camera_to_world.unsqueeze(1), cam_points_h.unsqueeze(-1)).squeeze(-1)[..., :3]

    lidar_to_world_inv = torch.inverse(lidar_to_world)
    world_points_h = torch.cat([world_points, torch.ones_like(world_points[..., :1])], dim=-1)
    lidar_points = torch.matmul(
        lidar_to_world_inv.view(1, 1, 4, 4),
        world_points_h.unsqueeze(-1),
    ).squeeze(-1)[..., :3]

    voxel_coords = torch.floor(
        (lidar_points - point_cloud_range[:3].view(1, 1, 3)) / voxel_size.view(1, 1, 3)
    ).to(torch.long)
    grid_size = ((point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size).round().to(torch.long)
    valid = (
        (depth_patch_flat > 1e-3)
        & (voxel_coords[..., 0] >= 0)
        & (voxel_coords[..., 0] < grid_size[0])
        & (voxel_coords[..., 1] >= 0)
        & (voxel_coords[..., 1] < grid_size[1])
        & (voxel_coords[..., 2] >= 0)
        & (voxel_coords[..., 2] < grid_size[2])
    )

    return {
        "depth_patch": depth_patch,
        "patch_uv": patch_uv1[..., :2],
        "cam_points": cam_points,
        "world_points": world_points,
        "lidar_points": lidar_points,
        "voxel_coords": voxel_coords,
        "valid": valid,
        "patch_hw": torch.tensor([patch_h, patch_w], device=last_frame_depth.device),
    }


def world_points_to_lidar(points: torch.Tensor, point_mask: torch.Tensor, lidar_to_world: torch.Tensor) -> torch.Tensor:
    valid_points = points[point_mask]
    if valid_points.numel() == 0:
        return valid_points.new_zeros((0, 3))
    xyz1 = torch.cat([valid_points[:, :3], torch.ones_like(valid_points[:, :1])], dim=-1)
    world_to_lidar = torch.inverse(lidar_to_world)
    return torch.matmul(world_to_lidar, xyz1.unsqueeze(-1)).squeeze(-1)[..., :3]


def encode_last_frame_voxels(
    model: torch.nn.Module,
    last_frame_world_points: torch.Tensor,
    last_frame_point_mask: torch.Tensor,
    last_frame_lidar_to_world: torch.Tensor,
) -> torch.Tensor:
    points = last_frame_world_points.unsqueeze(0)
    point_mask = last_frame_point_mask.unsqueeze(0)
    lidar_to_world = last_frame_lidar_to_world.unsqueeze(0)

    if not hasattr(model, "_encode_voxels") or not hasattr(model, "_voxel_coords_to_centers"):
        raise AttributeError("Model does not expose voxel encoding helpers required by this visualization.")

    with torch.no_grad():
        _, voxel_coords = model._encode_voxels(points, point_mask, lidar_to_world=lidar_to_world)
        voxel_centers = model._voxel_coords_to_centers(voxel_coords)
    return voxel_centers


def aggregate_patch_hit_voxels(
    projection: dict[str, torch.Tensor],
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_valid = projection["valid"].reshape(-1)
    flat_coords = projection["voxel_coords"].reshape(-1, 3)
    hit_coords = flat_coords[flat_valid]
    if hit_coords.numel() == 0:
        empty_centers = point_cloud_range.new_zeros((0, 3))
        empty_counts = torch.zeros((0,), dtype=torch.long, device=point_cloud_range.device)
        return empty_centers, empty_counts

    unique_coords, counts = torch.unique(hit_coords, dim=0, return_counts=True)
    centers = point_cloud_range[:3].view(1, 3) + (unique_coords.to(torch.float32) + 0.5) * voxel_size.view(1, 3)
    return centers, counts.to(torch.long)


def draw_point_set_3d(ax, points: np.ndarray, color: str, label: str, size: float, alpha: float) -> None:
    if points.shape[0] == 0:
        return
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, c=color, alpha=alpha, linewidths=0.0, label=label)


def draw_point_set_2d(ax, points: np.ndarray, a: int, b: int, color: str, label: str, size: float, alpha: float) -> None:
    if points.shape[0] == 0:
        return
    ax.scatter(points[:, a], points[:, b], s=size, c=color, alpha=alpha, linewidths=0.0, label=label)


def draw_voxel_cuboids_3d(
    ax,
    centers: np.ndarray,
    voxel_size: np.ndarray,
    color,
    alpha: float,
    label: str,
) -> None:
    if centers.shape[0] == 0:
        return
    lower = centers - (voxel_size.reshape(1, 3) / 2.0)
    ax.bar3d(
        lower[:, 0],
        lower[:, 1],
        lower[:, 2],
        np.full((centers.shape[0],), voxel_size[0], dtype=np.float32),
        np.full((centers.shape[0],), voxel_size[1], dtype=np.float32),
        np.full((centers.shape[0],), voxel_size[2], dtype=np.float32),
        color=color,
        alpha=alpha,
        shade=False,
        zsort="average",
        label=label,
    )


def draw_voxel_rectangles_2d(
    ax,
    centers: np.ndarray,
    voxel_size: np.ndarray,
    dims: tuple[int, int],
    color,
    alpha: float,
) -> None:
    if centers.shape[0] == 0:
        return
    dim_a, dim_b = dims
    width = float(voxel_size[dim_a])
    height = float(voxel_size[dim_b])
    patches = [
        Rectangle(
            (float(center[dim_a] - width / 2.0), float(center[dim_b] - height / 2.0)),
            width,
            height,
        )
        for center in centers
    ]
    collection = PatchCollection(
        patches,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
    )
    ax.add_collection(collection)


def save_patch_hit_voxel_figure(
    output_path: Path,
    point_cloud_range: np.ndarray,
    voxel_size: np.ndarray,
    hit_centers: np.ndarray,
    hit_counts: np.ndarray,
    title: str,
) -> None:
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_bev = fig.add_subplot(grid[0, 1])
    ax_xz = fig.add_subplot(grid[1, 0])
    ax_yz = fig.add_subplot(grid[1, 1])
    fig.suptitle(f"{title} | patch-hit voxel heat", fontsize=13)

    if hit_centers.shape[0] == 0:
        for ax in (ax_3d, ax_bev, ax_xz, ax_yz):
            if ax is ax_3d:
                set_axes_limits(ax, point_cloud_range)
            ax.text(0.5, 0.5, "No patch-hit voxels", transform=ax.transAxes, ha="center", va="center")
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    count_min = float(hit_counts.min())
    count_max = float(hit_counts.max())
    if count_max <= count_min:
        count_max = count_min + 1.0
    norm = plt.Normalize(vmin=count_min, vmax=count_max)
    cmap = plt.get_cmap("Blues")
    colors = cmap(norm(hit_counts))

    draw_voxel_cuboids_3d(ax_3d, hit_centers, voxel_size, colors, 0.28, "patch-hit voxels")
    set_axes_limits(ax_3d, point_cloud_range)
    ax_3d.view_init(elev=28, azim=-62)
    ax_3d.set_xlabel("x (m)")
    ax_3d.set_ylabel("y (m)")
    ax_3d.set_zlabel("z (m)")

    draw_voxel_rectangles_2d(ax_bev, hit_centers, voxel_size, (0, 1), colors, 0.8)
    ax_bev.set_xlim(point_cloud_range[0], point_cloud_range[3])
    ax_bev.set_ylim(point_cloud_range[1], point_cloud_range[4])
    ax_bev.set_title("BEV (x-y)")
    ax_bev.set_xlabel("x (m)")
    ax_bev.set_ylabel("y (m)")
    ax_bev.grid(alpha=0.15)

    draw_voxel_rectangles_2d(ax_xz, hit_centers, voxel_size, (0, 2), colors, 0.8)
    ax_xz.set_xlim(point_cloud_range[0], point_cloud_range[3])
    ax_xz.set_ylim(point_cloud_range[2], point_cloud_range[5])
    ax_xz.set_title("Front / side (x-z)")
    ax_xz.set_xlabel("x (m)")
    ax_xz.set_ylabel("z (m)")
    ax_xz.grid(alpha=0.15)

    draw_voxel_rectangles_2d(ax_yz, hit_centers, voxel_size, (1, 2), colors, 0.8)
    ax_yz.set_xlim(point_cloud_range[1], point_cloud_range[4])
    ax_yz.set_ylim(point_cloud_range[2], point_cloud_range[5])
    ax_yz.set_title("Side (y-z)")
    ax_yz.set_xlabel("y (m)")
    ax_yz.set_ylabel("z (m)")
    ax_yz.grid(alpha=0.15)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(hit_counts)
    fig.colorbar(scalar_mappable, ax=[ax_3d, ax_bev, ax_xz, ax_yz], fraction=0.02, pad=0.02, label="patch hits per voxel")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def set_axes_limits(ax, point_cloud_range: np.ndarray, dims: tuple[int, int, int] | None = None) -> None:
    if dims is None:
        ax.set_xlim(point_cloud_range[0], point_cloud_range[3])
        ax.set_ylim(point_cloud_range[1], point_cloud_range[4])
        ax.set_zlim(point_cloud_range[2], point_cloud_range[5])
        return
    mins = [point_cloud_range[0], point_cloud_range[1], point_cloud_range[2]]
    maxs = [point_cloud_range[3], point_cloud_range[4], point_cloud_range[5]]
    getters = [ax.set_xlim, ax.set_ylim, ax.set_zlim]
    for axis_idx in dims:
        getters[axis_idx](mins[axis_idx], maxs[axis_idx])


def save_projection_figure(
    output_path: Path,
    point_cloud_range: np.ndarray,
    lidar_points: np.ndarray,
    voxel_centers: np.ndarray,
    voxel_size: np.ndarray,
    projected_cam0: np.ndarray,
    projected_cam1: np.ndarray,
    projected_all: np.ndarray,
    title: str,
    voxel_cuboid_alpha: float,
) -> None:
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_bev = fig.add_subplot(grid[0, 1])
    ax_xz = fig.add_subplot(grid[1, 0])
    ax_yz = fig.add_subplot(grid[1, 1])
    fig.suptitle(title, fontsize=13)

    draw_point_set_3d(ax_3d, lidar_points, "#7a7a7a", "last lidar", 1.0, 0.08)
    draw_voxel_cuboids_3d(ax_3d, voxel_centers, voxel_size, "#8fb996", voxel_cuboid_alpha, "voxel features")
    draw_point_set_3d(ax_3d, projected_cam0, "#ff8c42", "cam02 projected patches", 8.0, 0.65)
    draw_point_set_3d(ax_3d, projected_cam1, "#00b4d8", "cam03 projected patches", 8.0, 0.65)
    set_axes_limits(ax_3d, point_cloud_range)
    ax_3d.view_init(elev=28, azim=-62)
    ax_3d.set_xlabel("x (m)")
    ax_3d.set_ylabel("y (m)")
    ax_3d.set_zlabel("z (m)")
    ax_3d.legend(loc="upper right", fontsize=8)

    draw_point_set_2d(ax_bev, lidar_points, 0, 1, "#7a7a7a", "last lidar", 1.0, 0.08)
    draw_voxel_rectangles_2d(ax_bev, voxel_centers, voxel_size, (0, 1), "#8fb996", voxel_cuboid_alpha)
    draw_point_set_2d(ax_bev, projected_cam0, 0, 1, "#ff8c42", "cam02 projected patches", 8.0, 0.65)
    draw_point_set_2d(ax_bev, projected_cam1, 0, 1, "#00b4d8", "cam03 projected patches", 8.0, 0.65)
    ax_bev.set_xlim(point_cloud_range[0], point_cloud_range[3])
    ax_bev.set_ylim(point_cloud_range[1], point_cloud_range[4])
    ax_bev.set_title("BEV (x-y)")
    ax_bev.set_xlabel("x (m)")
    ax_bev.set_ylabel("y (m)")
    ax_bev.grid(alpha=0.15)

    draw_point_set_2d(ax_xz, lidar_points, 0, 2, "#7a7a7a", "last lidar", 1.0, 0.08)
    draw_voxel_rectangles_2d(ax_xz, voxel_centers, voxel_size, (0, 2), "#8fb996", voxel_cuboid_alpha)
    draw_point_set_2d(ax_xz, projected_all, 0, 2, "#ff8c42", "projected patch features", 7.0, 0.6)
    ax_xz.set_xlim(point_cloud_range[0], point_cloud_range[3])
    ax_xz.set_ylim(point_cloud_range[2], point_cloud_range[5])
    ax_xz.set_title("Front / side (x-z)")
    ax_xz.set_xlabel("x (m)")
    ax_xz.set_ylabel("z (m)")
    ax_xz.grid(alpha=0.15)

    draw_point_set_2d(ax_yz, lidar_points, 1, 2, "#7a7a7a", "last lidar", 1.0, 0.08)
    draw_voxel_rectangles_2d(ax_yz, voxel_centers, voxel_size, (1, 2), "#8fb996", voxel_cuboid_alpha)
    draw_point_set_2d(ax_yz, projected_all, 1, 2, "#00b4d8", "projected patch features", 7.0, 0.6)
    ax_yz.set_xlim(point_cloud_range[1], point_cloud_range[4])
    ax_yz.set_ylim(point_cloud_range[2], point_cloud_range[5])
    ax_yz.set_title("Side (y-z)")
    ax_yz.set_xlabel("y (m)")
    ax_yz.set_ylabel("z (m)")
    ax_yz.grid(alpha=0.15)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_image_overlay_figure(
    output_path: Path,
    images: torch.Tensor,
    patch_uv: torch.Tensor,
    valid: torch.Tensor,
    depth_patch: torch.Tensor,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    for cam_idx, ax in enumerate(axes):
        rgb = tensor_image_to_rgb(images[cam_idx])
        ax.imshow(rgb)
        if valid[cam_idx].any():
            uv = patch_uv[cam_idx][valid[cam_idx]].detach().cpu().numpy()
            depth = depth_patch[cam_idx].reshape(-1)[valid[cam_idx]].detach().cpu().numpy()
            scatter = ax.scatter(
                uv[:, 0],
                uv[:, 1],
                c=depth,
                cmap="turbo",
                s=16,
                alpha=0.85,
                linewidths=0.0,
            )
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="projected patch depth (m)")
        ax.set_title(f"cam0{cam_idx + 2} overlay")
        ax.axis("off")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    cfg = Config.fromfile(args.config)
    dataset, resolved_split, resolved_index, dataset_resolution = resolve_dataset(cfg, args.split, args.sample_index)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu") if args.cpu else resolve_device(args.device)
    sample = dataset[resolved_index]
    sample_meta = extract_sample_meta(dataset, resolved_index)
    batch = collate_single_sample(sample, device=device)

    checkpoint_path = (
        args.model_weights
        if args.model_weights is not None
        else args.checkpoint if args.checkpoint is not None else cfg.get("checkpoint", None)
    )
    if not checkpoint_path:
        raise ValueError("A checkpoint is required. Pass --checkpoint or set cfg.checkpoint.")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = MODELS.build(copy.deepcopy(cfg.model))
    load_report = load_partial_checkpoint(model, checkpoint_path)
    if hasattr(model, "occupancy_head"):
        model.occupancy_head = None
    model.to(device)
    model.eval()

    normalized_extrinsics = normalize_extrinsics_to_first_frame(batch["extrinsics"])
    others = build_model_others(batch, normalized_extrinsics)

    with torch.no_grad():
        predictions = model(batch["images"], others=others)

    if "depth" not in predictions:
        raise RuntimeError("Model output does not contain 'depth'.")
    pred_depth = predictions["depth"]
    if pred_depth.ndim == 5 and pred_depth.shape[2] == 1:
        pred_depth = pred_depth[:, :, 0]
    elif pred_depth.ndim == 5 and pred_depth.shape[-1] == 1:
        pred_depth = pred_depth[..., 0]
    pred_depth = pred_depth[0]

    cam_num = int(cfg.model.get("cam_num", 2))
    patch_size = int(cfg.model.get("patch_size", 14))
    depth_scale = float(cfg.model.get("occupancy_head", {}).get("depth_scale", cfg.get("depth_pred_scale", 1.0)))
    voxel_size = torch.tensor(cfg.model["occupancy_head"]["voxel_size"], dtype=torch.float32, device=device)
    point_cloud_range = torch.tensor(cfg.model["occupancy_head"]["point_cloud_range"], dtype=torch.float32, device=device)

    last_frame_depth = pred_depth[-cam_num:]
    last_frame_images = batch["images"][0, -cam_num:]
    last_frame_intrinsics = batch["intrinsics"][0, -cam_num:]
    last_frame_camera_to_world = batch["camera_to_world"][0, -cam_num:]
    last_frame_lidar_to_world = batch["lidar_to_world"][0, -1]
    last_frame_world_points = batch["points"][0, -1]
    last_frame_point_mask = batch["point_mask"][0, -1]

    projection = project_last_frame_to_lidar(
        last_frame_depth=last_frame_depth,
        intrinsics=last_frame_intrinsics,
        camera_to_world=last_frame_camera_to_world,
        lidar_to_world=last_frame_lidar_to_world,
        patch_size=patch_size,
        depth_scale=depth_scale,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
    )

    lidar_points_local = world_points_to_lidar(last_frame_world_points, last_frame_point_mask, last_frame_lidar_to_world)
    voxel_centers = encode_last_frame_voxels(model, last_frame_world_points, last_frame_point_mask, last_frame_lidar_to_world)
    patch_hit_centers, patch_hit_counts = aggregate_patch_hit_voxels(projection, voxel_size, point_cloud_range)

    lidar_points_np = lidar_points_local.detach().cpu().numpy()
    voxel_centers_np = voxel_centers.detach().cpu().numpy()
    patch_hit_centers_np = patch_hit_centers.detach().cpu().numpy()
    patch_hit_counts_np = patch_hit_counts.detach().cpu().numpy()

    cam0_points = projection["lidar_points"][0][projection["valid"][0]].detach().cpu().numpy()
    cam1_points = projection["lidar_points"][1][projection["valid"][1]].detach().cpu().numpy()
    projected_all_np = projection["lidar_points"][projection["valid"]].detach().cpu().numpy()

    lidar_points_np = lidar_points_np[sample_indices(lidar_points_np.shape[0], args.max_lidar_points, rng)]
    voxel_centers_np = voxel_centers_np[sample_indices(voxel_centers_np.shape[0], args.max_voxel_cuboids, rng)]
    cam0_points = cam0_points[sample_indices(cam0_points.shape[0], args.max_projected_patches, rng)]
    cam1_points = cam1_points[sample_indices(cam1_points.shape[0], args.max_projected_patches, rng)]
    projected_all_np = projected_all_np[sample_indices(projected_all_np.shape[0], args.max_projected_patches * cam_num, rng)]
    patch_hit_indices = sample_indices(patch_hit_centers_np.shape[0], args.max_patch_hit_voxels, rng)
    patch_hit_centers_np = patch_hit_centers_np[patch_hit_indices]
    patch_hit_counts_np = patch_hit_counts_np[patch_hit_indices]

    safe_id = sanitize_name(
        "_".join(
            str(sample_meta.get(key, "na"))
            for key in ("sequence_id", "target_frame_id")
        )
    )
    if safe_id == "na_na":
        safe_id = f"sample_{args.sample_index:06d}"

    projection_path = output_dir / f"{args.sample_index:06d}_{safe_id}_projection.png"
    overlay_path = output_dir / f"{args.sample_index:06d}_{safe_id}_overlay.png"
    patch_hit_path = output_dir / f"{args.sample_index:06d}_{safe_id}_patch_hit_voxels.png"
    summary_path = output_dir / f"{args.sample_index:06d}_{safe_id}_summary.json"

    title = (
        f"sample={args.sample_index} resolved_split={resolved_split} "
        f"sequence={sample_meta.get('sequence_id', 'unknown')} "
        f"target_frame={sample_meta.get('target_frame_id', 'unknown')}"
    )
    save_projection_figure(
        output_path=projection_path,
        point_cloud_range=point_cloud_range.detach().cpu().numpy(),
        lidar_points=lidar_points_np,
        voxel_centers=voxel_centers_np,
        voxel_size=voxel_size.detach().cpu().numpy(),
        projected_cam0=cam0_points,
        projected_cam1=cam1_points,
        projected_all=projected_all_np,
        title=title,
        voxel_cuboid_alpha=args.voxel_cuboid_alpha,
    )
    save_image_overlay_figure(
        output_path=overlay_path,
        images=last_frame_images,
        patch_uv=projection["patch_uv"],
        valid=projection["valid"],
        depth_patch=projection["depth_patch"],
        title=title,
    )
    save_patch_hit_voxel_figure(
        output_path=patch_hit_path,
        point_cloud_range=point_cloud_range.detach().cpu().numpy(),
        voxel_size=voxel_size.detach().cpu().numpy(),
        hit_centers=patch_hit_centers_np,
        hit_counts=patch_hit_counts_np,
        title=title,
    )

    summary = {
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "requested_model_weights": args.model_weights,
        "device": str(device),
        "requested_split": args.split,
        "resolved_split": resolved_split,
        "dataset_resolution": dataset_resolution,
        "sample_meta": sample_meta,
        "load_report": load_report,
        "counts": {
            "last_frame_lidar_points": int(last_frame_point_mask.sum().item()),
            "last_frame_voxel_features": int(voxel_centers.shape[0]),
            "cam02_valid_projected_patches": int(projection["valid"][0].sum().item()),
            "cam03_valid_projected_patches": int(projection["valid"][1].sum().item()),
            "patch_hit_voxels": int(patch_hit_counts.shape[0]),
            "max_patch_hits_per_voxel": int(patch_hit_counts.max().item()) if patch_hit_counts.numel() > 0 else 0,
        },
        "outputs": {
            "projection_figure": str(projection_path),
            "overlay_figure": str(overlay_path),
            "patch_hit_voxel_figure": str(patch_hit_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved projection figure to: {projection_path}")
    print(f"Saved overlay figure to: {overlay_path}")
    print(f"Saved patch-hit voxel figure to: {patch_hit_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
