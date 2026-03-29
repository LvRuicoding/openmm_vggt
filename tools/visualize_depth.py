#!/usr/bin/env python3
"""Visualize depth predictions from a fine-tuned VGGT checkpoint on
KITTI val_selection_cropped images.

Usage::
    python tools/visualize_depth.py \
        configs/kitti_depth_stereo_ft_scaled.py \
        --checkpoint trainoutput/kitti_depth_stereo_ft_scaled/epoch_024.pth \
        --num-samples 6
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config
from mmengine.registry import MODELS

import openmm_vggt  # noqa: F401  triggers registration
from openmm_vggt.utils.geometry import closed_form_inverse_se3
from openmm_vggt.datasets.kitti_local_utils import (
    preprocess_rgb_like_demo,
    resolve_kitti_depth_root,
    load_rectified_intrinsics,
    load_camera_transform_imu_to_rectified,
    load_oxts_poses,
)


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_checkpoint(model: nn.Module, path: str) -> str:
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model_sd = model.state_dict()
    filtered, missing, shape_err = {}, [], []
    for k, v in model_sd.items():
        if k not in state_dict:
            missing.append(k)
        elif state_dict[k].shape != v.shape:
            shape_err.append((k, tuple(v.shape), tuple(state_dict[k].shape)))
        else:
            filtered[k] = state_dict[k]
    model.load_state_dict(filtered, strict=False)
    parts = [f"loaded={len(filtered)}"]
    if missing:
        parts.append(f"missing={len(missing)}")
    if shape_err:
        parts.append(f"shape_mismatch={len(shape_err)}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Extrinsics normalisation
# ---------------------------------------------------------------------------

def normalize_extrinsics_to_first_frame(extrinsics: torch.Tensor) -> torch.Tensor:
    B, S = extrinsics.shape[:2]
    ext_h = torch.zeros(B, S, 4, 4, dtype=extrinsics.dtype, device=extrinsics.device)
    ext_h[..., :3, :] = extrinsics
    ext_h[..., 3, 3] = 1.0
    first_inv = closed_form_inverse_se3(ext_h[:, 0])
    return torch.matmul(ext_h, first_inv.unsqueeze(1))[..., :3, :]


# ---------------------------------------------------------------------------
# Depth colorisation
# ---------------------------------------------------------------------------

def depth_to_colormap(depth_np: np.ndarray) -> np.ndarray:
    """Convert float depth array (H, W) to uint8 RGB via magma colormap."""
    valid = depth_np[depth_np > 0]
    if valid.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
    norm = np.clip((depth_np - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    colored = plt.get_cmap("magma")(norm)
    return (colored[:, :, :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Parse val_selection_cropped filename
# ---------------------------------------------------------------------------

def parse_val_stem(stem: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Pattern: {date(3)}_{drive(3)}_image_{frame_id}_image_{cam_num}  => 10 parts
    e.g. 2011_09_26_drive_0002_sync_image_0000000005_image_02
    Returns (date, drive_name, frame_id, camera) or None.
    """
    parts = stem.split("_")
    if len(parts) != 10:
        return None
    date = "_".join(parts[:3])        # 2011_09_26
    drive_name = "_".join(parts[:6])  # 2011_09_26_drive_0002_sync
    frame_id = parts[7]               # 0000000005
    camera = "_".join(parts[8:10])    # image_02
    return date, drive_name, frame_id, camera


# ---------------------------------------------------------------------------
# Build one 6-frame stereo sample
# ---------------------------------------------------------------------------

def build_stereo_sample(
    date: str,
    drive_name: str,
    frame_id: str,
    raw_root: Path,
    image_size: Tuple[int, int],
    n_time_steps: int = 3,
    stride: int = 1,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    """
    Build the S=n_time_steps*2 stereo input for the model.
    Layout: [t0_02, t0_03, t1_02, t1_03, t2_02, t2_03]
    Causal padding (scheme-A): clamp negative indices to 0.
    Returns (images [1,S,3,H,W], extrinsics [1,S,3,4], orig_rgb ndarray) or None.
    """
    raw_drive = raw_root / date / drive_name
    calib_root = raw_root / date

    if not raw_drive.is_dir():
        print(f"  [skip] raw drive not found: {raw_drive}")
        return None

    try:
        oxts_poses = load_oxts_poses(raw_drive)
    except Exception as e:
        print(f"  [skip] cannot load poses: {e}")
        return None

    cam_data: Dict[str, dict] = {}
    raw_hw = None
    for cam in ("image_02", "image_03"):
        rgb_dir = raw_drive / cam / "data"
        if not rgb_dir.is_dir():
            print(f"  [skip] missing rgb dir: {rgb_dir}")
            return None
        try:
            K_raw = load_rectified_intrinsics(calib_root / "calib_cam_to_cam.txt", cam)
            T_cam_imu = load_camera_transform_imu_to_rectified(calib_root, cam)
        except Exception as e:
            print(f"  [skip] calib error {cam}: {e}")
            return None

        img_paths: Dict[str, Path] = {p.stem: p for p in sorted(rgb_dir.glob("*.png"))}
        extrinsics: Dict[str, np.ndarray] = {}
        for fid, T_world_imu in oxts_poses.items():
            T_imu_world = np.linalg.inv(T_world_imu)
            extrinsics[fid] = (T_cam_imu @ T_imu_world)[:3, :4].astype(np.float32)

        if raw_hw is None and img_paths:
            w, h = Image.open(next(iter(img_paths.values()))).size
            raw_hw = (h, w)

        cam_data[cam] = dict(img_paths=img_paths, extrinsics=extrinsics)

    if raw_hw is None:
        return None

    all_fids = sorted(set(cam_data["image_02"]["img_paths"]) & set(oxts_poses))
    if frame_id not in all_fids:
        print(f"  [skip] frame {frame_id} not in drive {drive_name}")
        return None

    last_idx = all_fids.index(frame_id)
    raw_indices = [last_idx - (n_time_steps - 1 - t) * stride for t in range(n_time_steps)]
    clamped = [max(0, i) for i in raw_indices]
    time_fids = [all_fids[i] for i in clamped]

    images_list: List[torch.Tensor] = []
    extrinsics_list: List[torch.Tensor] = []
    orig_rgb: Optional[np.ndarray] = None

    for t_idx, fid in enumerate(time_fids):
        is_last = (t_idx == n_time_steps - 1)
        for cam in ("image_02", "image_03"):
            cd = cam_data[cam]
            if fid not in cd["img_paths"]:
                print(f"  [skip] frame {fid} missing for {cam}")
                return None
            img_path = cd["img_paths"][fid]
            if is_last and cam == "image_02":
                orig_rgb = np.array(Image.open(img_path).convert("RGB"))
            image, _ = preprocess_rgb_like_demo(img_path, image_size)
            images_list.append(image)
            ext = cd["extrinsics"].get(fid)
            if ext is None:
                print(f"  [skip] no extrinsics for {fid} {cam}")
                return None
            extrinsics_list.append(torch.from_numpy(ext.copy()))

    images = torch.stack(images_list).unsqueeze(0)        # [1, S, 3, H, W]
    extrinsics = torch.stack(extrinsics_list).unsqueeze(0)  # [1, S, 3, 4]
    return images, extrinsics, orig_rgb


# ---------------------------------------------------------------------------
# Save visualisation
# ---------------------------------------------------------------------------

def save_visualization(
    out_dir: Path,
    stem: str,
    orig_rgb: np.ndarray,
    pred_depth: np.ndarray,
    depth_scale: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_m = pred_depth * depth_scale

    # individual files
    Image.fromarray(orig_rgb).save(out_dir / f"{stem}_rgb.png")
    depth_color = depth_to_colormap(depth_m)
    Image.fromarray(depth_color).save(out_dir / f"{stem}_depth.png")

    # combined figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original image", fontsize=12)
    axes[0].axis("off")

    valid = depth_m[depth_m > 0]
    vmin = float(np.percentile(valid, 2)) if valid.size else 0.0
    vmax = float(np.percentile(valid, 98)) if valid.size else 1.0
    im = axes[1].imshow(depth_m, cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Predicted depth  (x{depth_scale})", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="depth (m)")
    plt.suptitle(stem, fontsize=9, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}_combined.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  saved -> {out_dir / stem}_*.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize VGGT depth on KITTI val_selection_cropped."
    )
    p.add_argument(
        "config",
        nargs="?",
        default=str(REPO_ROOT / "configs" / "kitti_depth_stereo_ft_scaled.py"),
    )
    p.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "trainoutput" / "kitti_depth_stereo_ft_scaled" / "epoch_024.pth"),
    )
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", nargs=2, type=int, default=None, metavar=("H", "W"))
    p.add_argument("--depth-scale", type=float, default=None)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "visualization"))
    p.add_argument("--camera", default="image_02")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config.fromfile(args.config)
    image_size = tuple(args.image_size) if args.image_size else tuple(cfg.image_size)
    assert image_size[0] % 14 == 0 and image_size[1] % 14 == 0
    n_time_steps = int(getattr(cfg, "n_time_steps", 3))
    stride = int(getattr(cfg, "stride", 1))
    depth_scale = (
        args.depth_scale if args.depth_scale is not None
        else float(cfg.get("depth_pred_scale", 1.0))
    )
    raw_root = Path(cfg.raw_root)
    depth_root = resolve_kitti_depth_root(cfg.depth_root)
    out_dir = Path(args.out_dir)

    print(f"Config        : {args.config}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"image_size    : {image_size}")
    print(f"depth_scale   : {depth_scale}")
    print(f"n_time_steps  : {n_time_steps}  stride={stride}")
    print(f"Output dir    : {out_dir}")

    # --- build model ---
    model = MODELS.build(dict(cfg.model))
    msg = load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    print(f"Checkpoint    : {msg}")

    # --- collect val_selection_cropped image_02 images ---
    val_img_root = depth_root / "depth_selection" / "val_selection_cropped" / "image"
    if not val_img_root.is_dir():
        raise FileNotFoundError(f"val_selection_cropped/image not found: {val_img_root}")

    all_images = sorted(
        p for p in val_img_root.glob("*.png")
        if p.stem.endswith(args.camera.replace("image_", "image_"))
        and parse_val_stem(p.stem) is not None
        and parse_val_stem(p.stem)[3] == args.camera
    )
    print(f"Found {len(all_images)} {args.camera} images in val_selection_cropped")

    random.seed(args.seed)
    selected = random.sample(all_images, min(args.num_samples, len(all_images)))
    print(f"Selected {len(selected)} samples (seed={args.seed})")

    # index of last cam02 in the S-frame sequence
    last_cam02_idx = (n_time_steps - 1) * 2  # [t0_02, t0_03, t1_02, t1_03, t2_02, t2_03]

    for img_path in selected:
        parsed = parse_val_stem(img_path.stem)
        if parsed is None:
            continue
        date, drive_name, frame_id, camera = parsed
        print(f"\nProcessing: {img_path.name}")

        result = build_stereo_sample(
            date=date,
            drive_name=drive_name,
            frame_id=frame_id,
            raw_root=raw_root,
            image_size=image_size,
            n_time_steps=n_time_steps,
            stride=stride,
        )
        if result is None:
            continue

        images, extrinsics, orig_rgb = result
        images = images.to(device)
        extrinsics = extrinsics.to(device)
        norm_ext = normalize_extrinsics_to_first_frame(extrinsics)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(images, others={"extrinsics": norm_ext})

        # depth: [B, S, H, W, 1] -> pick last cam02 -> [H, W]
        pred_depth = preds["depth"].squeeze(-1)           # [B, S, H, W]
        depth_np = pred_depth[0, last_cam02_idx].cpu().float().numpy()  # [H, W]

        stem = img_path.stem
        save_visualization(out_dir, stem, orig_rgb, depth_np, depth_scale)

    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
