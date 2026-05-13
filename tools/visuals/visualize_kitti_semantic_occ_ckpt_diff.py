#!/usr/bin/env python3
"""Visualize SemanticKITTI occupancy predictions against dense voxel GT.

Example:
  conda run -n fusion python tools/visuals/visualize_kitti_semantic_occ_ckpt_diff.py \
      --sample-index 0 \
      --checkpoint trainoutput/kitti_semantic_occ_mix_window_attn_early_ft_monoscene_head_cp_364x1218_occ_only/last.pth
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openmm-vggt")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS

import openmm_vggt  # noqa: F401
from openmm_vggt.utils.geometry import closed_form_inverse_se3


DEFAULT_CONFIG = REPO_ROOT / "configs" / "occupancy" / "kitti_semantic_occ_mix_window_attn_early_ft_monoscene_head_cp_364x1218_occ_only.py"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "trainoutput"
    / "kitti_semantic_occ_mix_window_attn_early_ft_monoscene_head_cp_364x1218_occ_only"
    / "last.pth"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visual"

CLASS_NAMES = (
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
)

CLASS_COLORS = np.asarray(
    [
        (0, 0, 0),
        (245, 150, 100),
        (245, 230, 100),
        (150, 60, 30),
        (180, 30, 80),
        (255, 0, 0),
        (30, 30, 255),
        (200, 40, 255),
        (90, 30, 150),
        (255, 0, 255),
        (255, 150, 255),
        (75, 0, 75),
        (75, 0, 175),
        (0, 200, 255),
        (50, 120, 255),
        (0, 175, 0),
        (0, 60, 135),
        (80, 240, 150),
        (150, 240, 255),
        (0, 0, 255),
    ],
    dtype=np.uint8,
)
INVALID_COLOR = np.asarray((180, 180, 180), dtype=np.uint8)

DIFF_NAMES = {
    0: "empty/correct",
    1: "correct occupied",
    2: "false positive",
    3: "false negative",
    4: "class mismatch",
}
DIFF_COLORS = np.asarray(
    [
        (18, 18, 18),
        (45, 170, 85),
        (220, 50, 47),
        (38, 139, 210),
        (238, 145, 34),
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ckpt occupancy prediction differences from dense voxel GT.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Occupancy config path.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint .pth path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for visualization outputs.")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--sample-index", type=int, default=0, help="First dataset sample index to visualize.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of consecutive samples to visualize.")
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="Explicit sample indices; overrides --sample-index.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cuda, cuda:0, cpu.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-3d-points", type=int, default=25000, help="Total max FP/FN/mismatch points in 3D figure.")
    parser.add_argument("--z-slices", default="4,8,12,16,20,24", help="Comma-separated z slice indices.")
    parser.add_argument("--image-size", nargs=2, type=int, default=None, metavar=("H", "W"))
    return parser.parse_args()


def resolve_device(name: str, force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def load_checkpoint(model: torch.nn.Module, path: Path, include_prefixes: tuple[str, ...] | None = None) -> str:
    ckpt = torch.load(str(path), map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:]: value for key, value in state_dict.items()}

    model_state = model.state_dict()
    model_keys = list(model_state)
    if include_prefixes:
        model_keys = [key for key in model_keys if matches_prefix(key, include_prefixes)]

    loaded = {}
    missing = 0
    mismatched = 0
    for key in model_keys:
        if key not in state_dict:
            missing += 1
            continue
        if tuple(state_dict[key].shape) != tuple(model_state[key].shape):
            mismatched += 1
            continue
        loaded[key] = state_dict[key]
    model.load_state_dict(loaded, strict=False)
    return f"loaded={len(loaded)} missing={missing} mismatched={mismatched}"


def normalize_extrinsics_to_first_frame(extrinsics: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = extrinsics.shape[:2]
    ext_h = torch.zeros(batch_size, seq_len, 4, 4, dtype=extrinsics.dtype, device=extrinsics.device)
    ext_h[..., :3, :] = extrinsics
    ext_h[..., 3, 3] = 1.0
    first_inv = closed_form_inverse_se3(ext_h[:, 0])
    return torch.matmul(ext_h, first_inv.unsqueeze(1))[..., :3, :]


def build_model_others(batch: dict[str, torch.Tensor], normalized_extrinsics: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "extrinsics": normalized_extrinsics,
        "intrinsics": batch["intrinsics"],
        "camera_to_world": batch["camera_to_world"],
        "lidar_to_world": batch["lidar_to_world"],
        "points": batch["points"],
        "point_mask": batch["point_mask"],
    }


def collate_single_sample(sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.unsqueeze(0).to(device, non_blocking=True)
        for key, value in sample.items()
        if isinstance(value, torch.Tensor)
    }


def align_target_to_pred(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape[0] == target.shape[0]:
        return target, valid
    if target.shape[0] <= 0 or pred.shape[0] % target.shape[0] != 0:
        raise ValueError(f"Batch mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    repeat = pred.shape[0] // target.shape[0]
    return target.repeat_interleave(repeat, 0), valid.repeat_interleave(repeat, 0)


def build_dataset(cfg: Config, split: str, image_size: tuple[int, int] | None):
    key = f"{split}_dataset"
    if key not in cfg:
        raise KeyError(f"Config missing {key}")
    dataset_cfg = cfg[key].copy()
    dataset_cfg.split = split
    dataset_cfg.require_dense_voxel_target = True
    dataset_cfg.color_jitter = None
    if image_size is not None:
        dataset_cfg.image_size = image_size
    if dataset_cfg.get("dense_voxel_root", None) is None:
        raise ValueError(f"{key}.dense_voxel_root is required for GT comparison")
    return DATASETS.build(dataset_cfg)


def extract_sample_meta(dataset, index: int) -> dict[str, Any]:
    meta: dict[str, Any] = {"sample_index": int(index)}
    if not hasattr(dataset, "samples") or not hasattr(dataset, "records"):
        return meta
    record_idx, last_idx = dataset.samples[index]
    record = dataset.records[record_idx]
    raw_indices = [last_idx - (dataset.n_time_steps - 1 - t) * dataset.stride for t in range(dataset.n_time_steps)]
    clamped_indices = [max(0, idx) for idx in raw_indices]
    frame_ids = [record.frame_ids[idx] for idx in clamped_indices]
    meta.update(
        {
            "sequence_id": record.sequence_id,
            "raw_drive_name": record.raw_drive_name,
            "time_frame_ids": frame_ids,
            "target_frame_id": frame_ids[-1],
        }
    )
    return meta


def semantic_bev(labels: np.ndarray, valid: np.ndarray, ignore_index: int) -> np.ndarray:
    valid_occ = valid & (labels != ignore_index) & (labels > 0)
    rev = valid_occ[..., ::-1]
    has = rev.any(axis=2)
    z_rev = rev.argmax(axis=2)
    z_idx = labels.shape[2] - 1 - z_rev
    out = np.zeros(labels.shape[:2], dtype=np.uint8)
    xs, ys = np.nonzero(has)
    out[xs, ys] = labels[xs, ys, z_idx[xs, ys]].astype(np.uint8)
    return out


def colorize_semantic(label_map: np.ndarray) -> np.ndarray:
    safe = np.clip(label_map.astype(np.int64), 0, len(CLASS_COLORS) - 1)
    return CLASS_COLORS[safe]


def build_diff_code(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, ignore_index: int) -> np.ndarray:
    valid = valid & (target != ignore_index)
    pred_occ = pred > 0
    gt_occ = target > 0
    code = np.zeros(target.shape, dtype=np.uint8)
    code[valid & pred_occ & gt_occ & (pred == target)] = 1
    code[valid & pred_occ & ~gt_occ] = 2
    code[valid & ~pred_occ & gt_occ] = 3
    code[valid & pred_occ & gt_occ & (pred != target)] = 4
    return code


def diff_bev(diff_code: np.ndarray) -> np.ndarray:
    out = np.zeros(diff_code.shape[:2], dtype=np.uint8)
    for code in (1, 2, 3, 4):
        out[(diff_code == code).any(axis=2)] = code
    return out


def colorize_diff(code_map: np.ndarray) -> np.ndarray:
    safe = np.clip(code_map.astype(np.int64), 0, len(DIFF_COLORS) - 1)
    return DIFF_COLORS[safe]


def add_diff_legend(ax) -> None:
    handles = [
        Patch(facecolor=DIFF_COLORS[idx] / 255.0, edgecolor="none", label=name)
        for idx, name in DIFF_NAMES.items()
        if idx != 0
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)


def save_bev_triptych(
    out_path: Path,
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    diff_code: np.ndarray,
    point_cloud_range: np.ndarray,
    ignore_index: int,
    title: str,
) -> None:
    gt_bev = semantic_bev(target, valid, ignore_index)
    pred_bev = semantic_bev(pred, valid, ignore_index)
    err_bev = diff_bev(diff_code)
    extent = [point_cloud_range[0], point_cloud_range[3], point_cloud_range[1], point_cloud_range[4]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, image, name in (
        (axes[0], colorize_semantic(gt_bev).transpose(1, 0, 2), "GT occupied top-down"),
        (axes[1], colorize_semantic(pred_bev).transpose(1, 0, 2), "Prediction occupied top-down"),
        (axes[2], colorize_diff(err_bev).transpose(1, 0, 2), "Difference"),
    ):
        ax.imshow(image, origin="lower", extent=extent, interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.12)
    add_diff_legend(axes[2])
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_z_slices(spec: str, z_size: int) -> list[int]:
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if 0 <= value < z_size:
            values.append(value)
    if not values:
        values = [z_size // 2]
    return values


def save_diff_slices(
    out_path: Path,
    diff_code: np.ndarray,
    point_cloud_range: np.ndarray,
    z_slices: list[int],
    title: str,
) -> None:
    cols = min(3, len(z_slices))
    rows = int(math.ceil(len(z_slices) / cols))
    cmap = ListedColormap(DIFF_COLORS / 255.0)
    norm = BoundaryNorm(np.arange(-0.5, len(DIFF_COLORS) + 0.5, 1), cmap.N)
    extent = [point_cloud_range[0], point_cloud_range[3], point_cloud_range[1], point_cloud_range[4]]

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False, constrained_layout=True)
    for ax, z_idx in zip(axes.reshape(-1), z_slices):
        ax.imshow(diff_code[:, :, z_idx].T, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"z slice {z_idx}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.12)
    for ax in axes.reshape(-1)[len(z_slices) :]:
        ax.axis("off")
    add_diff_legend(axes.reshape(-1)[0])
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def coords_to_centers(coords: np.ndarray, voxel_size: np.ndarray, point_cloud_range: np.ndarray) -> np.ndarray:
    return point_cloud_range[:3].reshape(1, 3) + (coords.astype(np.float32) + 0.5) * voxel_size.reshape(1, 3)


def sample_coords(mask: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.shape[0] <= max_count:
        return coords
    choice = rng.choice(coords.shape[0], size=max_count, replace=False)
    return coords[np.sort(choice)]


def save_3d_difference(
    out_path: Path,
    diff_code: np.ndarray,
    voxel_size: np.ndarray,
    point_cloud_range: np.ndarray,
    max_points: int,
    seed: int,
    title: str,
) -> None:
    rng = np.random.default_rng(seed)
    per_class = max(1, max_points // 3)
    specs = (
        (2, "false positive", "#dc322f"),
        (3, "false negative", "#268bd2"),
        (4, "class mismatch", "#ee9122"),
    )

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axbev = fig.add_subplot(1, 2, 2)
    for code, label, color in specs:
        coords = sample_coords(diff_code == code, per_class, rng)
        centers = coords_to_centers(coords, voxel_size, point_cloud_range)
        if centers.shape[0] == 0:
            continue
        ax3d.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=2.0, c=color, alpha=0.55, label=label)
        axbev.scatter(centers[:, 0], centers[:, 1], s=1.5, c=color, alpha=0.45, label=label)

    ax3d.set_xlim(point_cloud_range[0], point_cloud_range[3])
    ax3d.set_ylim(point_cloud_range[1], point_cloud_range[4])
    ax3d.set_zlim(point_cloud_range[2], point_cloud_range[5])
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("3D sampled errors")
    ax3d.view_init(elev=28, azim=-62)
    ax3d.legend(loc="upper right", fontsize=8)

    axbev.set_xlim(point_cloud_range[0], point_cloud_range[3])
    axbev.set_ylim(point_cloud_range[1], point_cloud_range[4])
    axbev.set_xlabel("x (m)")
    axbev.set_ylabel("y (m)")
    axbev.set_title("BEV sampled errors")
    axbev.grid(alpha=0.15)
    axbev.legend(loc="upper right", fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, ignore_index: int, num_classes: int) -> dict[str, Any]:
    valid = valid & (target != ignore_index)
    pred_valid = pred[valid]
    target_valid = target[valid]
    pred_occ = pred_valid != 0
    target_occ = target_valid != 0
    class_iou = []
    for class_idx in range(num_classes):
        pred_class = pred_valid == class_idx
        target_class = target_valid == class_idx
        union = np.logical_or(pred_class, target_class).sum()
        inter = np.logical_and(pred_class, target_class).sum()
        class_iou.append(float(inter / union) if union > 0 else None)
    semantic = [value for value in class_iou[1:] if value is not None]
    return {
        "valid_voxels": int(valid.sum()),
        "overall_acc": float((pred_valid == target_valid).sum() / max(target_valid.size, 1)),
        "sc_iou": float(np.logical_and(pred_occ, target_occ).sum() / max(np.logical_or(pred_occ, target_occ).sum(), 1)),
        "ssc_miou": float(sum(semantic) / max(len(semantic), 1)),
        "class_iou": class_iou,
        "counts": {
            "correct_occupied": int(((pred == target) & (target > 0) & valid).sum()),
            "false_positive": int(((pred > 0) & (target == 0) & valid).sum()),
            "false_negative": int(((pred == 0) & (target > 0) & valid).sum()),
            "class_mismatch": int(((pred > 0) & (target > 0) & (pred != target) & valid).sum()),
        },
    }


def run_one_sample(
    model: torch.nn.Module,
    dataset,
    sample_index: int,
    device: torch.device,
    cfg: Config,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    sample = dataset[sample_index]
    meta = extract_sample_meta(dataset, sample_index)
    batch = collate_single_sample(sample, device)

    with torch.inference_mode():
        norm_ext = normalize_extrinsics_to_first_frame(batch["extrinsics"])
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            preds = model(batch["images"], others=build_model_others(batch, norm_ext))
    if "occupancy_logits" not in preds:
        raise RuntimeError("Model did not return occupancy_logits")

    logits = preds["occupancy_logits"].float()
    target, valid = align_target_to_pred(logits, batch["occupancy_target"].long(), batch["occupancy_valid_mask"].bool())
    pred_np = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
    target_np = target[0].detach().cpu().numpy().astype(np.int64)
    valid_np = valid[0].detach().cpu().numpy().astype(bool)

    ignore_index = int(cfg.get("occupancy_ignore_index", 255))
    num_classes = int(cfg.get("occupancy_num_classes", 20))
    diff_code = build_diff_code(pred_np, target_np, valid_np, ignore_index)
    metrics = compute_metrics(pred_np, target_np, valid_np, ignore_index, num_classes)

    voxel_size = np.asarray(dataset.voxel_size, dtype=np.float32)
    point_cloud_range = np.asarray(dataset.point_cloud_range, dtype=np.float32)
    tag = f"{meta.get('sequence_id', args.split)}_{meta.get('target_frame_id', sample_index)}_idx{sample_index:06d}"
    title = f"{tag} | SC IoU {metrics['sc_iou']:.4f} | mIoU {metrics['ssc_miou']:.4f}"

    bev_path = output_dir / f"{tag}_bev_gt_pred_diff.png"
    slices_path = output_dir / f"{tag}_z_slices_diff.png"
    scatter_path = output_dir / f"{tag}_3d_errors.png"
    summary_path = output_dir / f"{tag}_summary.json"

    save_bev_triptych(bev_path, pred_np, target_np, valid_np, diff_code, point_cloud_range, ignore_index, title)
    save_diff_slices(slices_path, diff_code, point_cloud_range, parse_z_slices(args.z_slices, pred_np.shape[2]), title)
    save_3d_difference(scatter_path, diff_code, voxel_size, point_cloud_range, args.max_3d_points, args.seed, title)

    summary = {
        "meta": meta,
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "voxel_size": voxel_size.tolist(),
        "point_cloud_range": point_cloud_range.tolist(),
        "grid_shape": list(pred_np.shape),
        "class_names": CLASS_NAMES,
        "metrics": metrics,
        "outputs": {
            "bev_gt_pred_diff": str(bev_path),
            "z_slices_diff": str(slices_path),
            "3d_errors": str(scatter_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {summary_path}", flush=True)
    print(f"[saved] {bev_path}", flush=True)
    print(f"[saved] {slices_path}", flush=True)
    print(f"[saved] {scatter_path}", flush=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = Config.fromfile(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    image_size = tuple(args.image_size) if args.image_size is not None else None
    dataset = build_dataset(cfg, args.split, image_size=image_size)
    indices = args.indices if args.indices is not None and len(args.indices) > 0 else list(
        range(args.sample_index, args.sample_index + args.num_samples)
    )
    for index in indices:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"sample index {index} out of range [0, {len(dataset) - 1}] for split={args.split}")

    device = resolve_device(args.device, args.cpu)
    model = MODELS.build(cfg.model)
    include = tuple(cfg.get("checkpoint_include_prefixes", ()))
    load_msg = load_checkpoint(model, checkpoint, include_prefixes=include or None)
    model.to(device).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"config: {args.config}", flush=True)
    print(f"checkpoint: {checkpoint}", flush=True)
    print(f"checkpoint load: {load_msg}", flush=True)
    print(f"split={args.split} samples={len(dataset)} visualizing={indices}", flush=True)
    print(f"device: {device}", flush=True)

    for index in indices:
        run_one_sample(model, dataset, int(index), device, cfg, output_dir, args)


if __name__ == "__main__":
    main()
