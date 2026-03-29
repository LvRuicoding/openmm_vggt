#!/usr/bin/env python3
"""Official KITTI depth completion evaluation on val_selection_cropped.

Evaluates a fine-tuned checkpoint against the 1000 official KITTI val samples
in ``depth_selection/val_selection_cropped``.  Reports the four official metrics:

  RMSE   [mm]    root-mean-square error
  MAE    [mm]    mean absolute error
  iRMSE  [1/km]  root-mean-square error of inverse depth
  iMAE   [1/km]  mean absolute error of inverse depth

The model is run in monocular mode (cam_num=1) so that a window of
``--num-frames`` consecutive raw frames is used to predict depth at the
centre frame — the frame for which the official GT exists.

Usage (single GPU)::

    python tools/eval_kitti_depth.py configs/kitti_depth_stereo_ft.py \\
        --checkpoint trainoutput/kitti_depth_stereo_ft/best.pth

Usage (multi-GPU)::

    torchrun --nproc_per_node=4 tools/eval_kitti_depth.py \\
        configs/kitti_depth_stereo_ft.py \\
        --checkpoint trainoutput/kitti_depth_stereo_ft/best.pth
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openmm_vggt  # noqa: F401
from openmm_vggt.utils.geometry import closed_form_inverse_se3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official KITTI depth completion evaluation on val_selection_cropped."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(REPO_ROOT / "configs" / "kitti_depth_stereo_ft.py"),
        help="mmengine config (must define depth_root and raw_root).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pth to evaluate. Defaults to config checkpoint.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["image_02", "image_03"],
        help="Cameras to evaluate independently.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=6,
        help="Temporal window size (mono frames). Default 6.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Temporal stride. Default 1.",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Override image size (H W). Both must be multiples of 14.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap samples per camera (sanity checks).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def is_main() -> bool:
    return get_rank() == 0

def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)

def setup_distributed() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")

def cleanup_distributed() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Model / checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(model: nn.Module, path: str) -> str:
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model_state = model.state_dict()
    loaded, missing, shape_err = [], [], []
    filtered: Dict[str, torch.Tensor] = {}
    for k, v in model_state.items():
        if k not in state_dict:
            missing.append(k)
        elif state_dict[k].shape != v.shape:
            shape_err.append((k, tuple(v.shape), tuple(state_dict[k].shape)))
        else:
            filtered[k] = state_dict[k]
            loaded.append(k)
    model.load_state_dict(filtered, strict=False)
    parts = [f"loaded={len(loaded)}"]
    if missing:
        parts.append(f"missing={len(missing)}")
    if shape_err:
        parts.append(f"shape_mismatch={len(shape_err)}: {shape_err[:3]}")
    return " | ".join(parts)


def normalize_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    """Normalise so frame-0 is the coordinate origin."""
    B, S = extrinsics.shape[:2]
    ext_h = torch.zeros(B, S, 4, 4, dtype=extrinsics.dtype, device=extrinsics.device)
    ext_h[..., :3, :] = extrinsics
    ext_h[..., 3, 3] = 1.0
    first_inv = closed_form_inverse_se3(ext_h[:, 0])
    return torch.matmul(ext_h, first_inv.unsqueeze(1))[..., :3, :]


def collate_fn(batch):
    tensor_keys = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
    out = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    for k in batch[0]:
        if k not in out:
            out[k] = [b[k] for b in batch]
    return out


# ---------------------------------------------------------------------------
# Official KITTI metrics
# ---------------------------------------------------------------------------

class KITTIMetrics:
    """Four official KITTI depth completion metrics.

    Input depths are in **metres** (as returned by preprocess_depth_png).
    Following the official devkit:
      - RMSE / MAE  computed in millimetres  (depth_m * 1000)
      - iRMSE / iMAE computed in km⁻¹       (1000 / depth_m)
    Only pixels where gt > 0 are counted.
    """

    def __init__(self) -> None:
        self.n: float = 0.0
        self.sq_mm:  float = 0.0
        self.abs_mm: float = 0.0
        self.sq_ikm:  float = 0.0
        self.abs_ikm: float = 0.0

    def update(self, pred_m: torch.Tensor, gt_m: torch.Tensor) -> None:
        """Accumulate one (or more) predictions.

        Args:
            pred_m: predicted depth in metres, arbitrary shape.
            gt_m:   ground-truth depth in metres, same shape.
                    Pixels with gt <= 0 are ignored.
        """
        mask = gt_m > 0.0
        if not mask.any():
            return
        p = pred_m[mask].clamp_min(1e-6).double()
        g = gt_m[mask].double()

        diff_mm  = (p - g) * 1000.0
        diff_ikm = 1000.0 / p - 1000.0 / g

        self.n       += float(mask.sum())
        self.sq_mm   += float((diff_mm  ** 2).sum())
        self.abs_mm  += float(diff_mm.abs().sum())
        self.sq_ikm  += float((diff_ikm ** 2).sum())
        self.abs_ikm += float(diff_ikm.abs().sum())

    # distributed reduction
    def _as_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [self.n, self.sq_mm, self.abs_mm, self.sq_ikm, self.abs_ikm],
            dtype=torch.float64, device=device,
        )

    def reduce(self, device: torch.device) -> "KITTIMetrics":
        t = self._as_tensor(device)
        if is_dist():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        m = KITTIMetrics()
        m.n, m.sq_mm, m.abs_mm, m.sq_ikm, m.abs_ikm = (
            float(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4])
        )
        return m

    def result(self) -> Dict[str, float]:
        if self.n <= 0:
            raise RuntimeError("No valid pixels accumulated.")
        d = self.n
        return {
            "RMSE":   math.sqrt(self.sq_mm  / d),
            "MAE":    self.abs_mm  / d,
            "iRMSE":  math.sqrt(self.sq_ikm  / d),
            "iMAE":   self.abs_ikm / d,
            "n_pixels": d,
        }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_val_dataset(cfg: Config, camera: str, num_frames: int, stride: int,
                      image_size: tuple, max_samples: int | None):
    """Build KITTIDepthSelectionValDataset for one camera.

    This dataset reads directly from val_selection_cropped (cropped GT images
    + intrinsics) and fetches temporal context from kitti_raw.
    """
    kwargs = dict(
        type="KITTIDepthSelectionValDataset",
        depth_root=cfg.depth_root,
        raw_root=cfg.raw_root,
        image_size=tuple(image_size),
        camera=camera,
        num_frames=num_frames,
        stride=stride,
        strict=False,
    )
    if max_samples is not None:
        kwargs["max_samples"] = max_samples
    return DATASETS.build(kwargs)


# ---------------------------------------------------------------------------
# Evaluation loop for one camera
# ---------------------------------------------------------------------------

def evaluate_camera(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> KITTIMetrics:
    """Run inference over one camera's val split and accumulate metrics."""
    # Manual distributed sharding (no DistributedSampler needed for eval)
    indices = list(range(get_rank(), len(dataset), get_world_size()))
    sub = Subset(dataset, indices)

    loader = DataLoader(
        sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        drop_last=False,
    )

    # centre frame index inside the temporal window
    center_idx = dataset.num_frames // 2

    local = KITTIMetrics()
    bar = tqdm(
        loader,
        desc=f"  camera={dataset.camera} rank={get_rank()}",
        leave=False,
        disable=not is_main(),
    )

    with torch.inference_mode():
        for batch in bar:
            imgs   = batch["images"].to(device, non_blocking=True)    # B,S,3,H,W
            ext    = batch["extrinsics"].to(device, non_blocking=True) # B,S,3,4
            depths = batch["depths"].to(device, non_blocking=True)     # B,S,H,W

            norm_ext = normalize_extrinsics(ext)
            preds = model(imgs, others={"extrinsics": norm_ext})

            # predicted depth: B,S,H,W,1  ->  B,S,H,W
            pred_depth = preds["depth"].squeeze(-1)  # metres

            # only evaluate at the centre frame
            pred_c = pred_depth[:, center_idx]   # B,H,W
            gt_c   = depths[:, center_idx]       # B,H,W  (metres, -1 = invalid)

            local.update(pred_c, gt_c)

    return local.reduce(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = setup_distributed()

    try:
        cfg = Config.fromfile(args.config)

        # Resolve checkpoint
        ckpt_path = args.checkpoint or getattr(cfg, "checkpoint", None)
        if ckpt_path is None:
            raise ValueError(
                "No checkpoint specified. Use --checkpoint or set 'checkpoint' in config."
            )
        ckpt_path = str(ckpt_path)
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Resolve image size
        if args.image_size is not None:
            image_size = tuple(args.image_size)
        else:
            image_size = tuple(getattr(cfg, "image_size", (280, 518)))
        assert image_size[0] % 14 == 0 and image_size[1] % 14 == 0, (
            f"image_size {image_size} must have both dims as multiples of 14"
        )

        # Build model in monocular mode (cam_num=1) for val_selection_cropped
        model_cfg = dict(cfg.model)
        model_cfg["cam_num"] = 1  # mono temporal window
        model = MODELS.build(model_cfg)
        msg = load_checkpoint(model, ckpt_path)
        model.to(device)
        model.eval()

        log("=" * 70)
        log("KITTI depth completion — official val_selection_cropped evaluation")
        log(f"  config     : {Path(args.config).resolve()}")
        log(f"  checkpoint : {Path(ckpt_path).resolve()}")
        log(f"  ckpt load  : {msg}")
        log(f"  image_size : {image_size}")
        log(f"  num_frames : {args.num_frames}  stride={args.stride}")
        log(f"  cameras    : {args.cameras}")
        log(f"  distributed: world_size={get_world_size()}")
        log("=" * 70)

        all_metrics = KITTIMetrics()
        per_cam: Dict[str, Dict[str, float]] = {}

        for camera in args.cameras:
            dataset = build_val_dataset(
                cfg=cfg,
                camera=camera,
                num_frames=args.num_frames,
                stride=args.stride,
                image_size=image_size,
                max_samples=args.max_samples,
            )
            log(f"Evaluating {camera}: {len(dataset)} samples")

            cam_metrics = evaluate_camera(
                model=model,
                dataset=dataset,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            res = cam_metrics.result()
            per_cam[camera] = res
            # accumulate into combined
            all_metrics.n       += cam_metrics.n
            all_metrics.sq_mm   += cam_metrics.sq_mm
            all_metrics.abs_mm  += cam_metrics.abs_mm
            all_metrics.sq_ikm  += cam_metrics.sq_ikm
            all_metrics.abs_ikm += cam_metrics.abs_ikm

        log("")
        log("Results (official KITTI metrics):")
        header = f"  {'Camera':<14s}  {'RMSE[mm]':>10s}  {'MAE[mm]':>10s}  {'iRMSE[1/km]':>12s}  {'iMAE[1/km]':>11s}  {'n_pixels':>12s}"
        log(header)
        log("  " + "-" * (len(header) - 2))
        for cam, r in per_cam.items():
            log(
                f"  {cam:<14s}  {r['RMSE']:10.4f}  {r['MAE']:10.4f}"
                f"  {r['iRMSE']:12.4f}  {r['iMAE']:11.4f}  {r['n_pixels']:12.0f}"
            )
        if len(args.cameras) > 1:
            c = all_metrics.result()
            log("  " + "-" * (len(header) - 2))
            log(
                f"  {'combined':<14s}  {c['RMSE']:10.4f}  {c['MAE']:10.4f}"
                f"  {c['iRMSE']:12.4f}  {c['iMAE']:11.4f}  {c['n_pixels']:12.0f}"
            )
        log("")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
