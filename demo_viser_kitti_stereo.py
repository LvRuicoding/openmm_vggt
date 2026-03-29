#!/usr/bin/env python3
"""KITTI stereo depth viser demo with fine-tuned VGGT_decoder_global.

Loads one sample from the KITTI depth-completion val split
(3 time-steps x 2 cameras = 6 frames), runs inference, and visualises
the predicted depth as a point cloud with viser.

Extrinsic convention
--------------------
- Dataset provides  T_cam_world  (world-to-cam, 3x4, OpenCV).
- Before inference we normalise all extrinsics relative to the first frame
  (same as eval_kitti_depth_stereo.py).
- For visualisation we use the **GT extrinsics** (normalised) so the camera
  frustums are placed correctly alongside the unprojected depth cloud.
  The intrinsics used for unprojection are also from GT so the geometry is
  self-consistent.

Usage::

    python demo_viser_kitti_stereo.py \\
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config
from mmengine.registry import MODELS, DATASETS

import openmm_vggt  # noqa: F401  registers datasets & models
from openmm_vggt.utils.geometry import (
    closed_form_inverse_se3,
    unproject_depth_map_to_point_map,
)


DEFAULT_CKPT = REPO_ROOT / "trainoutput" / "kitti_depth_stereo_ft_scaled" / "epoch_024.pth"
DEFAULT_CFG  = REPO_ROOT / "configs" / "kitti_depth_stereo_ft_scaled.py"


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_checkpoint(model: nn.Module, path: str) -> str:
    ckpt = torch.load(path, map_location="cpu")
    sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
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
    if missing:    parts.append(f"missing={len(missing)}")
    if shape_err:  parts.append(f"shape_mismatch={shape_err[:3]}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Extrinsics normalisation  (identical to eval_kitti_depth_stereo.py)
# ---------------------------------------------------------------------------

def normalize_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    """(B, S, 3, 4) -> (B, S, 3, 4) all relative to frame 0.

    Computes  E_norm_i = E_i @ inv(E_0)  for each frame i.
    E_i is world-to-cam, so E_norm_i is cam0-to-cam_i expressed as world-to-cam
    in the normalised frame (where cam0 is the origin).
    """
    B, S = extrinsics.shape[:2]
    ext_h = torch.zeros(B, S, 4, 4,
                        dtype=extrinsics.dtype, device=extrinsics.device)
    ext_h[..., :3, :] = extrinsics
    ext_h[..., 3, 3]  = 1.0
    # closed_form_inverse_se3 expects (N,4,4); ext_h[:,0] is (B,4,4)
    first_inv = closed_form_inverse_se3(ext_h[:, 0])   # (B,4,4)
    return torch.matmul(ext_h, first_inv.unsqueeze(1))[..., :3, :]  # (B,S,3,4)


# ---------------------------------------------------------------------------
# Viser visualisation
# ---------------------------------------------------------------------------

def viser_wrapper(
    images: np.ndarray,          # (S, 3, H, W)  float [0,1]
    depth: np.ndarray,           # (S, H, W, 1)  metres
    depth_conf: np.ndarray,      # (S, H, W)
    extrinsics: np.ndarray,      # (S, 3, 4)  world-to-cam (normalised GT)
    intrinsics: np.ndarray,      # (S, 3, 3)  GT intrinsics (resized)
    port: int = 8080,
    init_conf_threshold: float = 25.0,
    background_mode: bool = False,
) -> viser.ViserServer:
    """Unproject depth and visualise point cloud + camera frustums.

    Uses GT extrinsics & intrinsics for unprojection so the geometry is
    self-consistent.
    """
    print(f"Starting viser server on port {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    S = images.shape[0]
    H, W = depth.shape[1], depth.shape[2]

    # Unproject depth -> world points using GT extrinsics + GT intrinsics
    world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
    # world_points: (S, H, W, 3)

    colors      = images.transpose(0, 2, 3, 1)           # (S,H,W,3)
    points      = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat   = depth_conf.reshape(-1)

    # cam-to-world for camera frustum positions
    cam_to_world_mat = closed_form_inverse_se3(extrinsics)  # (S,4,4)
    cam_to_world     = cam_to_world_mat[:, :3, :]            # (S,3,4)

    # Centre the scene around the mean point cloud
    valid_mask   = conf_flat > 0
    if valid_mask.any():
        scene_center = np.mean(points[valid_mask], axis=0)
    else:
        scene_center = np.zeros(3)
    points_centered       = points - scene_center
    cam_to_world[..., -1] = cam_to_world[..., -1] - scene_center

    frame_indices = np.repeat(np.arange(S), H * W)

    # ----- GUI -----
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_conf        = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1,
        initial_value=init_conf_threshold)
    gui_frame_sel   = server.gui.add_dropdown(
        "Show Points from Frames",
        options=["All"] + [str(i) for i in range(S)],
        initial_value="All")

    init_thresh = np.percentile(conf_flat[conf_flat > 0], init_conf_threshold) \
                  if (conf_flat > 0).any() else 0.0
    init_mask   = (conf_flat >= init_thresh) & (conf_flat > 0)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_mask],
        colors=colors_flat[init_mask],
        point_size=0.02,
        point_shape="circle",
    )

    frames:   List[viser.FrameHandle]         = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames() -> None:
        for f  in frames:   f.remove()
        frames.clear()
        for fr in frustums: fr.remove()
        frustums.clear()

        def attach_cb(frustum: viser.CameraFrustumHandle,
                      frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz     = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(range(S), desc="Adding cameras"):
            c2w = cam_to_world[img_id]   # (3,4)
            T   = viser_tf.SE3.from_matrix(c2w)
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T.rotation().wxyz,
                position=T.translation(),
                axes_length=0.3,
                axes_radius=0.01,
                origin_radius=0.01,
            )
            frames.append(frame_axis)

            img_hw = (images[img_id].transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w   = img_hw.shape[:2]
            # Use real focal length from GT intrinsics for FOV
            fy     = intrinsics[img_id, 1, 1]
            fov    = 2 * np.arctan2(h / 2.0, fy)
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum",
                fov=fov, aspect=w / h, scale=0.3,
                image=img_hw, line_width=1.5,
            )
            frustums.append(frustum_cam)
            attach_cb(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        pct    = gui_conf.value
        valid  = conf_flat > 0
        if valid.any():
            thresh = np.percentile(conf_flat[valid], pct)
        else:
            thresh = 0.0
        c_mask = (conf_flat >= thresh) & (conf_flat > 0)
        if gui_frame_sel.value == "All":
            f_mask = np.ones(len(c_mask), dtype=bool)
        else:
            f_mask = frame_indices == int(gui_frame_sel.value)
        combined = c_mask & f_mask
        point_cloud.points = points_centered[combined]
        point_cloud.colors = colors_flat[combined]

    @gui_conf.on_update
    def _(_) -> None: update_point_cloud()

    @gui_frame_sel.on_update
    def _(_) -> None: update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        for f  in frames:   f.visible  = gui_show_frames.value
        for fr in frustums: fr.visible = gui_show_frames.value

    visualize_frames()
    print(f"Viser running — open http://localhost:{port} in your browser")

    if background_mode:
        threading.Thread(target=lambda: [time.sleep(0.01) or None
                                         for _ in iter(int, 1)],
                         daemon=True).start()
    else:
        while True:
            time.sleep(0.01)

    return server


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KITTI stereo depth viser demo.")
    p.add_argument("config", nargs="?", default=str(DEFAULT_CFG))
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--conf-threshold", type=float, default=25.0)
    p.add_argument("--depth-scale", type=float, default=None)
    p.add_argument("--background", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = Config.fromfile(args.config)
    depth_scale  = (args.depth_scale if args.depth_scale is not None
                    else float(cfg.get("depth_pred_scale", 1.0)))
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

    # Model
    print("Building model...")
    model = MODELS.build(dict(cfg.model))
    print(f"Checkpoint: {load_checkpoint(model, ckpt_path)}")
    model.to(device).eval()

    # Dataset
    print("Loading KITTI val sample...")
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

    sample = dataset[args.sample_index % len(dataset)]
    S = sample["images"].shape[0]   # n_time_steps*2 = 6
    print(f"S={S} frames (n_time_steps={n_time_steps} x 2 cameras)")

    imgs_b = sample["images"].unsqueeze(0).to(device)      # (1,S,3,H,W)
    ext_b  = sample["extrinsics"].unsqueeze(0).to(device)  # (1,S,3,4) world-to-cam
    int_b  = sample["intrinsics"].unsqueeze(0).to(device)  # (1,S,3,3)

    # Normalise extrinsics to first frame — exactly as eval script
    norm_ext = normalize_extrinsics(ext_b)   # (1,S,3,4)

    # Inference
    print("Running inference...")
    dtype = (torch.bfloat16 if device.type == "cuda" and
             torch.cuda.get_device_capability()[0] >= 8 else
             torch.float16 if device.type == "cuda" else torch.float32)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=dtype):
            preds = model(imgs_b, others={"extrinsics": norm_ext})

    # Predicted depth  (B,S,H,W,1) -> (S,H,W,1)
    pred_depth = preds["depth"].detach().float().cpu().numpy().squeeze(0)
    pred_depth = pred_depth * depth_scale
    if pred_depth.ndim == 3:
        pred_depth = pred_depth[..., np.newaxis]

    # Depth confidence  (B,S,H,W) -> (S,H,W)
    if "depth_conf" in preds:
        pred_conf = preds["depth_conf"].detach().float().cpu().numpy().squeeze(0)
    else:
        pred_conf = np.ones(pred_depth.shape[:3], dtype=np.float32)

    # GT extrinsics (normalised) and intrinsics for visualisation
    gt_ext  = norm_ext.squeeze(0).float().cpu().numpy()   # (S,3,4) world-to-cam
    gt_int  = int_b.squeeze(0).float().cpu().numpy()      # (S,3,3)
    imgs_np = imgs_b.squeeze(0).float().cpu().numpy()     # (S,3,H,W)

    print(f"depth shape : {pred_depth.shape}")
    print(f"conf  shape : {pred_conf.shape}")
    print(f"depth range : {pred_depth.min():.2f} ~ {pred_depth.max():.2f} m  mean={pred_depth.mean():.2f}")
    print(f"conf  range : {pred_conf.min():.4f} ~ {pred_conf.max():.4f}  mean={pred_conf.mean():.4f}")
    print(f"ext[0] (cam0, should be identity after norm):\n{gt_ext[0]}")
    print(f"ext[1] (cam1 = image_03, t0):\n{gt_ext[1]}")
    print(f"int[0]:\n{gt_int[0]}")

    print("Starting viser visualisation...")
    viser_wrapper(
        images=imgs_np,
        depth=pred_depth,
        depth_conf=pred_conf,
        extrinsics=gt_ext,
        intrinsics=gt_int,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        background_mode=args.background,
    )


if __name__ == "__main__":
    main()
 