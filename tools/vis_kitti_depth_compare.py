#!/usr/bin/env python3
"""Visualise GT (red) vs predicted (green) depth as coloured point clouds.

Loads one sample from the KITTI depth-completion val split
(3 time-steps x 2 cameras = 6 frames), runs model inference, then
unprojects both the GT sparse depth and the predicted dense depth of the
last time-step image_02 into world coordinates and displays them together
in a viser browser scene:

    RED   points = GT depth (sparse ground truth)
    GREEN points = predicted depth (dense model output)

Usage::

    python tools/vis_kitti_depth_compare.py \\
        configs/kitti_depth_stereo_ft_scaled.py \\
        --checkpoint trainoutput/kitti_depth_stereo_ft_scaled/epoch_024.pth \\
        --sample-index 0 \\
        --port 8080
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config
from mmengine.registry import MODELS, DATASETS

import openmm_vggt  # noqa: F401  registers datasets & models
from openmm_vggt.utils.geometry import (
    closed_form_inverse_se3,
    unproject_depth_map_to_point_map,
)


DEFAULT_CFG  = REPO_ROOT / "configs" / "kitti_depth_stereo_ft_scaled.py"
DEFAULT_CKPT = (
    REPO_ROOT / "trainoutput" / "kitti_depth_stereo_ft_scaled" / "epoch_024.pth"
)


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_checkpoint(model: nn.Module, path: str) -> str:
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if all(k.startswith("module.") for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    model_sd = model.state_dict()
    filtered, missing, shape_err = {}, [], []
    for k, v in model_sd.items():
        if k not in sd:
            missing.append(k)
        elif sd[k].shape != v.shape:
            shape_err.append((k, tuple(v.shape), tuple(sd[k].shape)))
        else:
            filtered[k] = sd[k]
    model.load_state_dict(filtered, strict=False)
    parts = [f"loaded={len(filtered)}"]
    if missing:
        parts.append(f"missing={len(missing)}")
    if shape_err:
        parts.append(f"shape_mismatch={shape_err[:3]}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Extrinsics normalisation (identical to eval_kitti_depth_stereo.py)
# ---------------------------------------------------------------------------

def normalize_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    """(B, S, 3, 4) -> (B, S, 3, 4) all relative to frame 0."""
    B, S = extrinsics.shape[:2]
    ext_h = torch.zeros(
        B, S, 4, 4, dtype=extrinsics.dtype, device=extrinsics.device
    )
    ext_h[..., :3, :] = extrinsics
    ext_h[..., 3, 3] = 1.0
    first_inv = closed_form_inverse_se3(ext_h[:, 0])  # (B, 4, 4)
    return torch.matmul(ext_h, first_inv.unsqueeze(1))[..., :3, :]  # (B, S, 3, 4)


# ---------------------------------------------------------------------------
# Viser visualisation
# ---------------------------------------------------------------------------

def viser_wrapper(
    images: np.ndarray,       # (S, 3, H, W)  float32 [0, 1]
    pts_gt: np.ndarray,       # (N_gt,  3)  world coords
    pts_pred: np.ndarray,     # (N_pred, 3)  world coords
    extrinsics: np.ndarray,   # (S, 3, 4)  normalised world-to-cam
    intrinsics: np.ndarray,   # (S, 3, 3)
    port: int = 8080,
    point_size: float = 0.02,
    background_mode: bool = False,
) -> viser.ViserServer:
    """Launch viser server and show GT (red) + predicted (green) depth clouds."""
    print(f"Starting viser server on port {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    S = images.shape[0]
    H, W = images.shape[2], images.shape[3]

    # ---- colours ----
    col_gt   = np.tile(np.array([255,   0,   0], dtype=np.uint8), (len(pts_gt),   1))
    col_pred = np.tile(np.array([  0, 255,   0], dtype=np.uint8), (len(pts_pred), 1))

    # ---- centre scene ----
    all_pts = np.concatenate([pts_gt, pts_pred], axis=0)
    scene_center = all_pts.mean(axis=0) if len(all_pts) > 0 else np.zeros(3)
    pts_gt_c   = (pts_gt   - scene_center).astype(np.float32)
    pts_pred_c = (pts_pred - scene_center).astype(np.float32)

    # cam-to-world for frustums
    ext_h = np.zeros((S, 4, 4), dtype=np.float64)
    ext_h[:, :3, :] = extrinsics
    ext_h[:, 3, 3]  = 1.0
    cam_to_world = closed_form_inverse_se3(ext_h)[:, :3, :]   # (S, 3, 4)
    cam_to_world[:, :, 3] -= scene_center

    # ---- GUI controls ----
    gui_show_gt   = server.gui.add_checkbox("Show GT depth (red)",   initial_value=True)
    gui_show_pred = server.gui.add_checkbox("Show Pred depth (green)", initial_value=True)
    gui_show_cams = server.gui.add_checkbox("Show Cameras",            initial_value=True)
    gui_pt_size   = server.gui.add_slider(
        "Point Size", min=0.002, max=0.15, step=0.002, initial_value=point_size
    )

    # ---- point clouds ----
    pcd_gt = server.scene.add_point_cloud(
        name="gt_depth",
        points=pts_gt_c,
        colors=col_gt,
        point_size=point_size,
        point_shape="circle",
    )
    pcd_pred = server.scene.add_point_cloud(
        name="pred_depth",
        points=pts_pred_c,
        colors=col_pred,
        point_size=point_size,
        point_shape="circle",
    )

    # ---- camera frustums ----
    frames:   List[viser.FrameHandle]         = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames() -> None:
        for f  in frames:   f.remove()
        frames.clear()
        for fr in frustums: fr.remove()
        frustums.clear()

        def attach_cb(
            frustum: viser.CameraFrustumHandle,
            frame: viser.FrameHandle,
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz     = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(range(S), desc="Adding cameras"):
            c2w = cam_to_world[img_id]   # (3, 4)
            T   = viser_tf.SE3.from_matrix(c2w)
            frame_h = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T.rotation().wxyz,
                position=T.translation(),
                axes_length=0.3,
                axes_radius=0.01,
                origin_radius=0.01,
            )
            frames.append(frame_h)

            img_hw = (images[img_id].transpose(1, 2, 0) * 255).astype(np.uint8)
            fy  = intrinsics[img_id, 1, 1]
            fov = 2.0 * np.arctan2(H / 2.0, fy)
            fr  = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum",
                fov=fov, aspect=W / H, scale=0.3,
                image=img_hw, line_width=1.5,
            )
            frustums.append(fr)
            attach_cb(fr, frame_h)

    # ---- callbacks ----
    @gui_show_gt.on_update
    def _(_) -> None:
        pcd_gt.visible = gui_show_gt.value

    @gui_show_pred.on_update
    def _(_) -> None:
        pcd_pred.visible = gui_show_pred.value

    @gui_show_cams.on_update
    def _(_) -> None:
        for f  in frames:   f.visible  = gui_show_cams.value
        for fr in frustums: fr.visible = gui_show_cams.value

    @gui_pt_size.on_update
    def _(_) -> None:
        pcd_gt.point_size   = gui_pt_size.value
        pcd_pred.point_size = gui_pt_size.value

    visualize_frames()
    print(f"Viser running — open http://localhost:{port} in your browser")
    print("  RED   = GT depth")
    print("  GREEN = predicted depth")

    if background_mode:
        threading.Thread(
            target=lambda: [time.sleep(0.01) or None for _ in iter(int, 1)],
            daemon=True,
        ).start()
    else:
        while True:
            time.sleep(0.01)

    return server


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare GT (red) vs predicted (green) depth via viser browser."
    )
    p.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CFG),
        help="mmengine config file.",
    )
    p.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT),
        help="Model checkpoint (.pth).",
    )
    p.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index into the val dataset.",
    )
    p.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Override depth_pred_scale from config.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viser web server.",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=0.02,
        help="Initial point size in the 3-D viewer.",
    )
    p.add_argument(
        "--background",
        action="store_true",
        help="Run viser server in background thread (non-blocking).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg          = Config.fromfile(args.config)
    depth_scale  = (
        args.depth_scale if args.depth_scale is not None
        else float(cfg.get("depth_pred_scale", 1.0))
    )
    image_size   = tuple(cfg.image_size)
    n_time_steps = int(getattr(cfg, "n_time_steps", 3))
    stride       = int(getattr(cfg, "stride", 1))

    ckpt_path = args.checkpoint
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("=" * 70)
    print(f"  config       : {Path(args.config).resolve()}")
    print(f"  checkpoint   : {Path(ckpt_path).resolve()}")
    print(f"  image_size   : {image_size}")
    print(f"  n_time_steps : {n_time_steps}  stride={stride}")
    print(f"  depth_scale  : {depth_scale}")
    print(f"  sample_index : {args.sample_index}")
    print("=" * 70)

    # ---- model ----
    print("Building model...")
    model = MODELS.build(dict(cfg.model))
    print(f"Checkpoint: {load_checkpoint(model, ckpt_path)}")
    model.to(device).eval()

    # ---- dataset ----
    print("Loading KITTI val dataset...")
    dataset = DATASETS.build(dict(
        type="KITTIDepthCompletionStereoDataset",
        depth_root=cfg.depth_root,
        raw_root=cfg.raw_root,
        split="val",
        n_time_steps=n_time_steps,
        stride=stride,
        image_size=image_size,
        strict=False,
    ))
    print(f"Val dataset: {len(dataset)} samples")

    sample_idx     = args.sample_index % len(dataset)
    sample         = dataset[sample_idx]
    S              = sample["images"].shape[0]            # n_time_steps * 2
    last_cam02_idx = (n_time_steps - 1) * 2              # evaluated slot

    imgs_b = sample["images"].unsqueeze(0).to(device)      # (1, S, 3, H, W)
    ext_b  = sample["extrinsics"].unsqueeze(0).to(device)  # (1, S, 3, 4)
    int_b  = sample["intrinsics"].unsqueeze(0).to(device)  # (1, S, 3, 3)
    deps_b = sample["depths"].unsqueeze(0)                  # (1, S, H, W)  cpu

    # normalise extrinsics relative to frame 0 (same as eval script)
    norm_ext = normalize_extrinsics(ext_b)  # (1, S, 3, 4)

    # ---- inference ----
    print("Running inference...")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
        if device.type == "cuda"
        else torch.float32
    )
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=dtype):
            preds = model(imgs_b, others={"extrinsics": norm_ext})

    # ---- predicted depth: (1, S, H, W, 1) -> (H, W) ----
    pred_depth_hw = (
        preds["depth"]
        .detach().float().cpu()
        .squeeze(0)[last_cam02_idx]   # (H, W, 1)
        .squeeze(-1).numpy()          # (H, W)
    ) * depth_scale
    pred_depth_hw = np.clip(pred_depth_hw, 0.0, None)

    # ---- GT depth: (1, S, H, W) -> (H, W),  invalid stored as -1 ----
    gt_depth_hw = deps_b[0, last_cam02_idx].numpy()
    gt_depth_hw = np.where(gt_depth_hw > 0, gt_depth_hw, 0.0)

    # extrinsic & intrinsic for the evaluated frame
    ext_np  = norm_ext[0, last_cam02_idx].float().cpu().numpy()  # (3, 4)
    intr_np = int_b[0, last_cam02_idx].float().cpu().numpy()     # (3, 3)

    print(f"GT   depth: min={gt_depth_hw[gt_depth_hw>0].min():.2f}  "
          f"max={gt_depth_hw.max():.2f} m  "
          f"valid_px={int((gt_depth_hw>0).sum())}")
    print(f"Pred depth: min={pred_depth_hw[pred_depth_hw>0].min():.3f}  "
          f"max={pred_depth_hw.max():.3f} m  "
          f"mean={pred_depth_hw[pred_depth_hw>0].mean():.3f}")

    # ---- unproject both depth maps to world points ----
    print("Unprojecting depth maps to point clouds...")

    def _unproject(depth_hw: np.ndarray) -> np.ndarray:
        """(H, W) depth -> (N, 3) valid world points."""
        world = unproject_depth_map_to_point_map(
            depth_hw[np.newaxis, :, :, np.newaxis],  # (1, H, W, 1)
            ext_np[np.newaxis],                      # (1, 3, 4)
            intr_np[np.newaxis],                     # (1, 3, 3)
        )  # (1, H, W, 3)
        pts   = world[0].reshape(-1, 3)
        valid = depth_hw.reshape(-1) > 0
        return pts[valid]

    pts_gt   = _unproject(gt_depth_hw)
    pts_pred = _unproject(pred_depth_hw)
    print(f"GT   cloud : {len(pts_gt):,} points  (red)")
    print(f"Pred cloud : {len(pts_pred):,} points  (green)")

    # extrinsics / intrinsics / images for all S frames (camera frustums)
    ext_all  = norm_ext.squeeze(0).float().cpu().numpy()   # (S, 3, 4)
    intr_all = int_b.squeeze(0).float().cpu().numpy()      # (S, 3, 3)
    imgs_np  = imgs_b.squeeze(0).float().cpu().numpy()     # (S, 3, H, W)

    print(f"depth range : {pred_depth_hw.min():.2f} ~ {pred_depth_hw.max():.2f} m")
    print(f"ext[0] (cam0 after norm):\n{ext_all[0]}")
    print(f"int[0]:\n{intr_all[0]}")

    print("Starting viser visualisation...")
    viser_wrapper(
        images=imgs_np,
        pts_gt=pts_gt,
        pts_pred=pts_pred,
        extrinsics=ext_all,
        intrinsics=intr_all,
        port=args.port,
        point_size=args.point_size,
        background_mode=args.background,
    )


if __name__ == "__main__":
    main()
