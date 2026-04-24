#!/usr/bin/env python3
"""Visualize DDAD sparse-depth supervision for one sample across six cameras."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openmm-vggt")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmengine.config import Config
from mmengine.registry import DATASETS

import openmm_vggt  # noqa: F401
from openmm_vggt.datasets.ddad_depth_temporal import DEFAULT_CAMERA_NAMES


DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "ddad_ft" / "ddad_depth_3cam_010506_mix_window_attn_early_ft.py"
)
DEFAULT_DATASET_KEY = "val_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one DDAD sample's last-timestep six-camera sparse depth supervision."
    )
    parser.add_argument("sample_index", type=int, help="Dataset sample index.")
    parser.add_argument("output_dir", type=str, help="Directory to save the visualizations.")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def tensor_image_to_rgb(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image_np, 0.0, 1.0)


def colorize_sparse_depth(depth_map: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    color = np.zeros((*depth_map.shape, 3), dtype=np.float32)
    if not np.any(valid_mask):
        return color, 0.0, 1.0

    valid_depth = depth_map[valid_mask]
    vmin = float(valid_depth.min())
    vmax = float(valid_depth.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    norm = np.clip((depth_map - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    cmap = plt.get_cmap("turbo")
    colorized = cmap(norm)[..., :3].astype(np.float32)
    color[valid_mask] = colorized[valid_mask]
    return color, vmin, vmax


def make_coverage_overlay(rgb: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    if np.any(valid_mask):
        highlight = np.zeros_like(rgb)
        highlight[..., 0] = 1.0
        overlay[valid_mask] = 0.35 * overlay[valid_mask] + 0.65 * highlight[valid_mask]
    return overlay


def reshape_sample(sample: dict, cam_num: int) -> tuple[torch.Tensor, torch.Tensor]:
    images = sample["images"]
    depths = sample["depths"]
    seq_len, _, image_h, image_w = images.shape
    if seq_len % cam_num != 0:
        raise ValueError(f"Expected seq_len divisible by cam_num, got seq_len={seq_len}, cam_num={cam_num}")

    frame_count = seq_len // cam_num
    images = images.reshape(frame_count, cam_num, 3, image_h, image_w)
    depths = depths.reshape(frame_count, cam_num, image_h, image_w)
    return images, depths


def build_dataset():
    cfg = Config.fromfile(str(DEFAULT_CONFIG))
    if DEFAULT_DATASET_KEY not in cfg:
        raise KeyError(f"Config does not contain {DEFAULT_DATASET_KEY}")

    dataset_cfg = copy.deepcopy(cfg[DEFAULT_DATASET_KEY])
    dataset_cfg["camera_names"] = DEFAULT_CAMERA_NAMES
    dataset_cfg["image_size"] = tuple(cfg.image_size)
    dataset_cfg["n_time_steps"] = int(cfg.n_time_steps)
    dataset_cfg["stride"] = int(cfg.stride)
    dataset_cfg["return_lidar"] = True
    return DATASETS.build(dataset_cfg), cfg


def save_camera_figure(
    output_path: Path,
    camera_name: str,
    rgb: np.ndarray,
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    sequence_name: str,
    sample_index: int,
    depth_vmin: float,
    depth_vmax: float,
) -> dict[str, float | int | str]:
    coverage = float(valid_mask.mean())
    valid_depth = depth_map[valid_mask]
    coverage_pct = coverage * 100.0

    depth_color, _, _ = colorize_sparse_depth(depth_map, valid_mask)
    coverage_overlay = make_coverage_overlay(rgb, valid_mask)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
    fig.suptitle(
        f"sample={sample_index} sequence={sequence_name} camera={camera_name}",
        fontsize=12,
    )

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(depth_color)
    axes[1].set_title(f"Sparse Depth\n[{depth_vmin:.2f}, {depth_vmax:.2f}] m")
    axes[1].axis("off")

    axes[2].imshow(valid_mask.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Valid Mask")
    axes[2].axis("off")

    axes[3].imshow(coverage_overlay)
    axes[3].set_title(f"Coverage\n{coverage_pct:.4f}% ({int(valid_mask.sum())} px)")
    axes[3].axis("off")

    if valid_depth.size > 0:
        bins = min(50, max(10, int(np.sqrt(valid_depth.size))))
        axes[4].hist(valid_depth, bins=bins, color="#1f77b4", edgecolor="black", linewidth=0.5)
        axes[4].set_title("Depth Histogram")
        axes[4].set_xlabel("Depth (m)")
        axes[4].set_ylabel("Count")
    else:
        axes[4].text(0.5, 0.5, "No valid depth", ha="center", va="center", fontsize=12)
        axes[4].set_title("Depth Histogram")
        axes[4].set_xticks([])
        axes[4].set_yticks([])

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    stats: dict[str, float | int | str] = {
        "camera_name": camera_name,
        "coverage": coverage,
        "coverage_percent": coverage_pct,
        "valid_pixel_count": int(valid_mask.sum()),
        "output_path": str(output_path),
    }
    if valid_depth.size > 0:
        stats.update(
            {
                "depth_min_m": float(valid_depth.min()),
                "depth_max_m": float(valid_depth.max()),
                "depth_mean_m": float(valid_depth.mean()),
                "depth_median_m": float(np.median(valid_depth)),
            }
        )
    else:
        stats.update(
            {
                "depth_min_m": None,
                "depth_max_m": None,
                "depth_mean_m": None,
                "depth_median_m": None,
            }
        )
    return stats


def main() -> None:
    args = parse_args()
    dataset, cfg = build_dataset()

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample_index {args.sample_index} out of range [0, {len(dataset) - 1}]")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = dataset[args.sample_index]
    images, depths = reshape_sample(sample, cam_num=len(DEFAULT_CAMERA_NAMES))

    last_images = images[-1]
    last_depths = depths[-1]
    sequence_name = str(sample["sequence_name"])
    safe_sequence_name = sanitize_name(sequence_name)

    summary = {
        "config_path": str(DEFAULT_CONFIG),
        "dataset_key": DEFAULT_DATASET_KEY,
        "sample_index": args.sample_index,
        "sequence_name": sequence_name,
        "camera_names": list(DEFAULT_CAMERA_NAMES),
        "image_size": list(cfg.image_size),
        "n_time_steps": int(cfg.n_time_steps),
        "stride": int(cfg.stride),
        "figures": [],
    }

    for cam_idx, camera_name in enumerate(DEFAULT_CAMERA_NAMES):
        rgb = tensor_image_to_rgb(last_images[cam_idx])
        depth_map = last_depths[cam_idx].detach().cpu().numpy().astype(np.float32)
        valid_mask = depth_map > 0.0

        valid_depth = depth_map[valid_mask]
        if valid_depth.size > 0:
            depth_vmin = float(valid_depth.min())
            depth_vmax = float(valid_depth.max())
            if depth_vmax <= depth_vmin:
                depth_vmax = depth_vmin + 1.0
        else:
            depth_vmin, depth_vmax = 0.0, 1.0

        output_path = output_dir / f"{args.sample_index:06d}_{safe_sequence_name}_{camera_name}.png"
        stats = save_camera_figure(
            output_path=output_path,
            camera_name=camera_name,
            rgb=rgb,
            depth_map=depth_map,
            valid_mask=valid_mask,
            sequence_name=sequence_name,
            sample_index=args.sample_index,
            depth_vmin=depth_vmin,
            depth_vmax=depth_vmax,
        )
        summary["figures"].append(stats)

    summary_path = output_dir / f"{args.sample_index:06d}_{safe_sequence_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved 6 camera visualizations to: {output_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
