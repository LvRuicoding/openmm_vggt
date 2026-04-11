#!/usr/bin/env python3
"""Evaluate a fine-tuned VGGT checkpoint on KITTI depth completion val split.

Input: 3 time-steps x 2 cameras = 6 frames, causal sliding window with
scheme-A padding (earliest frame repeated when history is insufficient).
Only the last time-step stereo pair is evaluated.

Official KITTI depth prediction metrics (main table):
    iRMSE  [1/km], iMAE [1/km], RMSE [mm], MAE [mm]

Supplementary metrics:
    Abs Rel, Sq Rel, SILog, delta < 1.25, delta < 1.25^2, delta < 1.25^3

Usage (single GPU)::
    python tools/eval_kitti_depth_stereo.py configs/kitti_depth_stereo_ft_scaled.py \\
        --checkpoint trainoutput/kitti_depth_stereo_ft/epoch_002.pth

Usage (multi-GPU)::
    torchrun --nproc_per_node=4 tools/eval_kitti_depth_stereo.py \\
        configs/kitti_depth_stereo_ft_scaled.py \\
        --checkpoint trainoutput/kitti_depth_stereo_ft/epoch_002.pth
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS

import openmm_vggt  # noqa: F401
from openmm_vggt.utils.geometry import closed_form_inverse_se3
from openmm_vggt.datasets.kitti_local_utils import (
    preprocess_depth_png,
    preprocess_rgb_like_demo,
    resize_intrinsics,
    resolve_kitti_depth_root,
    load_rectified_intrinsics,
    load_camera_transform_imu_to_rectified,
    load_oxts_poses,
)


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


DEFAULT_EVAL_CHECKPOINT_NAME = "epoch_002.pth"


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def matches_prefix(name: str, prefixes: Tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def load_checkpoint(
    model: nn.Module,
    path: str,
    include_prefixes: Optional[Tuple[str, ...]] = None,
) -> str:
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model_sd = model.state_dict()
    model_keys = list(model_sd.keys())
    if include_prefixes:
        model_keys = [k for k in model_keys if matches_prefix(k, include_prefixes)]
        if not model_keys:
            raise RuntimeError(
                f"No model keys matched include_prefixes={include_prefixes}"
            )

    filtered, missing, shape_err = {}, [], []
    for key in model_keys:
        value = model_sd[key]
        if key not in state_dict:
            missing.append(key)
        elif state_dict[key].shape != value.shape:
            shape_err.append((key, tuple(value.shape), tuple(state_dict[key].shape)))
        else:
            filtered[key] = state_dict[key]
    model.load_state_dict(filtered, strict=False)

    extras = sorted(set(state_dict) - set(model_sd))
    parts = [
        f"loaded={len(filtered)}",
        f"unexpected={len(extras)}",
    ]
    if missing:
        parts.append(f"missing={len(missing)}")
    if shape_err:
        parts.append(f"shape_mismatch={shape_err[:3]}")
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
# Metrics
# ---------------------------------------------------------------------------

class DepthMetrics:
    """Accumulates KITTI official + supplementary depth metrics.

    Depths are expected in metres.
    Official KITTI depth prediction metrics:
      RMSE  [mm], MAE  [mm], iRMSE [1/km], iMAE [1/km]
    Supplementary:
      Abs Rel, Sq Rel, SILog, delta<1.25, delta<1.25^2, delta<1.25^3
    """

    def __init__(self) -> None:
        # accumulators
        self.n: float = 0.0
        # official
        self.sq_mm:   float = 0.0
        self.abs_mm:  float = 0.0
        self.sq_ikm:  float = 0.0
        self.abs_ikm: float = 0.0
        # supplementary
        self.abs_rel:  float = 0.0
        self.sq_rel:   float = 0.0
        self.silog_sq: float = 0.0   # sum of (log p - log g)^2
        self.silog_lin: float = 0.0  # sum of  (log p - log g)
        self.d1: float = 0.0
        self.d2: float = 0.0
        self.d3: float = 0.0

    def update(self, pred_m: torch.Tensor, gt_m: torch.Tensor) -> None:
        """Accumulate one batch.

        Args:
            pred_m: predicted depth in metres, any shape.
            gt_m:   GT depth in metres, same shape. Pixels with gt<=0 ignored.
        """
        mask = gt_m > 0.0
        if not mask.any():
            return
        p = pred_m[mask].clamp_min(1e-6).double()
        g = gt_m[mask].double()
        n = float(mask.sum())

        # --- official KITTI depth prediction metrics ---
        diff_mm  = (p - g) * 1000.0          # convert m -> mm
        diff_ikm = 1000.0 / p - 1000.0 / g  # convert m -> km^-1, then difference
        self.sq_mm   += float((diff_mm  ** 2).sum())
        self.abs_mm  += float(diff_mm.abs().sum())
        self.sq_ikm  += float((diff_ikm ** 2).sum())
        self.abs_ikm += float(diff_ikm.abs().sum())

        # --- supplementary metrics ---
        self.abs_rel  += float((( p - g).abs() / g).sum())
        self.sq_rel   += float(((p - g) ** 2    / g).sum())
        log_diff = torch.log(p) - torch.log(g)
        self.silog_sq  += float((log_diff ** 2).sum())
        self.silog_lin += float(log_diff.sum())
        ratio = torch.max(p / g, g / p)
        self.d1 += float((ratio < 1.25     ).sum())
        self.d2 += float((ratio < 1.25 ** 2).sum())
        self.d3 += float((ratio < 1.25 ** 3).sum())

        self.n += n

    def _tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [self.n, self.sq_mm, self.abs_mm, self.sq_ikm, self.abs_ikm,
             self.abs_rel, self.sq_rel, self.silog_sq, self.silog_lin,
             self.d1, self.d2, self.d3],
            dtype=torch.float64, device=device,
        )

    def reduce(self, device: torch.device) -> "DepthMetrics":
        t = self._tensor(device)
        if is_dist():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        m = DepthMetrics()
        (m.n, m.sq_mm, m.abs_mm, m.sq_ikm, m.abs_ikm,
         m.abs_rel, m.sq_rel, m.silog_sq, m.silog_lin,
         m.d1, m.d2, m.d3) = [float(x) for x in t]
        return m

    def result(self) -> Dict[str, float]:
        if self.n <= 0:
            raise RuntimeError("No valid pixels accumulated.")
        d = self.n
        # SILog = sqrt( mean((log p - log g)^2) - (mean(log p - log g))^2 ) * 100
        silog = math.sqrt(max(self.silog_sq / d - (self.silog_lin / d) ** 2, 0.0)) * 100.0
        return {
            # official
            "iRMSE":  math.sqrt(self.sq_ikm / d),
            "iMAE":   self.abs_ikm / d,
            "RMSE":   math.sqrt(self.sq_mm  / d),
            "MAE":    self.abs_mm  / d,
            # supplementary
            "Abs Rel": self.abs_rel / d,
            "Sq Rel":  self.sq_rel  / d,
            "SILog":   silog,
            "d1":      self.d1 / d,
            "d2":      self.d2 / d,
            "d3":      self.d3 / d,
            "n_pixels": d,
        }


# ---------------------------------------------------------------------------
# Val dataset
# ---------------------------------------------------------------------------

class KITTIStereoValDataset(Dataset):
    """KITTI depth completion val split with 3-timestep stereo input.

    Mirrors training dataset exactly:
    - 3 time-steps x 2 cameras = 6 frames per sample
    - GT depth only at the last time-step's image_02
    - Scheme-A causal padding: repeat earliest frame when history insufficient
    """

    def __init__(
        self,
        depth_root: str,
        raw_root: str,
        image_size: Tuple[int, int] = (378, 1246),
        n_time_steps: int = 3,
        stride: int = 1,
        split: str = "val",
        strict: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        self.depth_root = resolve_kitti_depth_root(depth_root)
        self.raw_root = Path(raw_root)
        self.image_size = image_size
        self.n_time_steps = n_time_steps
        self.stride = stride
        self.split = split
        self.strict = strict
        self.max_samples = max_samples
        self.drives = self._load_drives()
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(
                f"No valid val samples (split={split}, n_time_steps={n_time_steps})."
            )

    def _load_drives(self) -> list:
        from openmm_vggt.datasets.kitti_depth_stereo import _StereoDrive
        split_root = self.depth_root / self.split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Val split not found: {split_root}")
        drives, skipped = [], 0
        for drive_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            date = drive_dir.name[:10]
            raw_drive = self.raw_root / date / drive_dir.name
            calib_root = self.raw_root / date
            if not raw_drive.is_dir():
                skipped += 1
                continue
            # Load poses (shared for both cameras)
            try:
                oxts_poses = load_oxts_poses(raw_drive)
            except Exception:
                skipped += 1
                continue
            cam_data: Dict[str, dict] = {}
            ok = True
            raw_hw = None
            for cam in ("image_02", "image_03"):
                rgb_dir = raw_drive / cam / "data"
                gt_dir = drive_dir / "proj_depth" / "groundtruth" / cam
                if not rgb_dir.is_dir():
                    ok = False; break
                img_paths = {p.stem: p for p in sorted(rgb_dir.glob("*.png"))}
                if not img_paths:
                    ok = False; break
                gt_paths = (
                    {p.stem: p for p in sorted(gt_dir.glob("*.png"))}
                    if gt_dir.is_dir() else {}
                )
                try:
                    K = load_rectified_intrinsics(
                        calib_root / "calib_cam_to_cam.txt", cam
                    )
                    T_cam_imu = load_camera_transform_imu_to_rectified(
                        calib_root, cam
                    )
                except Exception:
                    ok = False; break
                # Compute extrinsics for all frames
                extrinsics: Dict[str, "np.ndarray"] = {}
                for fid, T_world_imu in oxts_poses.items():
                    T_imu_world = np.linalg.inv(T_world_imu)
                    T_cam_world = T_cam_imu @ T_imu_world
                    extrinsics[fid] = T_cam_world[:3, :4].astype(np.float32)
                # frame_ids: rgb image exists AND pose exists (GT NOT required)
                frame_ids = sorted(set(img_paths.keys()) & set(extrinsics.keys()))
                if not frame_ids:
                    ok = False; break
                # Get raw image size from first image
                if raw_hw is None:
                    from PIL import Image as PILImage
                    first_img = img_paths[frame_ids[0]]
                    raw_w, raw_h = PILImage.open(first_img).size
                    raw_hw = (raw_h, raw_w)
                cam_data[cam] = {
                    "image_paths": img_paths,
                    "gt_paths": gt_paths,
                    "extrinsics": extrinsics,
                    "K": K,
                    "frame_ids": frame_ids,
                }
            if ok and "image_02" in cam_data and "image_03" in cam_data:
                frame_ids = sorted(
                    set(cam_data["image_02"]["frame_ids"])
                    & set(cam_data["image_03"]["frame_ids"])
                )
                if not frame_ids:
                    skipped += 1
                    continue
                drives.append(_StereoDrive(
                    name=drive_dir.name,
                    date=date,
                    raw_hw=raw_hw,
                    image_paths_02=cam_data["image_02"]["image_paths"],
                    image_paths_03=cam_data["image_03"]["image_paths"],
                    gt_paths_02=cam_data["image_02"]["gt_paths"],
                    frame_ids=frame_ids,
                    extrinsics_02=cam_data["image_02"]["extrinsics"],
                    extrinsics_03=cam_data["image_03"]["extrinsics"],
                    intrinsics_02=cam_data["image_02"]["K"],
                    intrinsics_03=cam_data["image_03"]["K"],
                ))
            else:
                skipped += 1
        log(f"[KITTIStereoValDataset] {len(drives)} drives, {skipped} skipped.")
        return sorted(drives, key=lambda d: d.name)

    def _build_samples(self) -> List[Tuple[int, int]]:
        samples = []
        for di, drive in enumerate(self.drives):
            for last_idx in range(len(drive.frame_ids)):
                if drive.frame_ids[last_idx] in drive.gt_paths_02:
                    samples.append((di, last_idx))
        if self.max_samples is not None:
            samples = samples[:self.max_samples]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        drive_idx, last_idx = self.samples[index]
        drive = self.drives[drive_idx]
        raw_indices = [
            last_idx - (self.n_time_steps - 1 - t) * self.stride
            for t in range(self.n_time_steps)
        ]
        clamped = [max(0, i) for i in raw_indices]
        time_frame_ids = [drive.frame_ids[i] for i in clamped]

        images_list, depths_list, extrinsics_list, intrinsics_list = [], [], [], []
        orig_hw = drive.raw_hw
        for t_idx, fid in enumerate(time_frame_ids):
            is_last = (t_idx == self.n_time_steps - 1)
            for cam in ("image_02", "image_03"):
                if cam == "image_02":
                    img_path = drive.image_paths_02[fid]
                    K_raw = drive.intrinsics_02
                    ext = drive.extrinsics_02[fid]
                    gt_path = drive.gt_paths_02.get(fid, None)
                else:
                    img_path = drive.image_paths_03[fid]
                    K_raw = drive.intrinsics_03
                    ext = drive.extrinsics_03[fid]
                    gt_path = None
                image, _ = preprocess_rgb_like_demo(img_path, self.image_size)
                images_list.append(image)
                if is_last and cam == "image_02" and gt_path is not None:
                    depth = preprocess_depth_png(gt_path, self.image_size)
                else:
                    depth = torch.full(self.image_size, -1.0, dtype=torch.float32)
                depths_list.append(depth)
                extrinsics_list.append(torch.from_numpy(ext.copy()))
                K = resize_intrinsics(K_raw, orig_hw=orig_hw, out_hw=self.image_size)
                intrinsics_list.append(torch.from_numpy(K))
        return {
            "images":     torch.stack(images_list),
            "depths":     torch.stack(depths_list),
            "extrinsics": torch.stack(extrinsics_list),
            "intrinsics": torch.stack(intrinsics_list),
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_batch(batch):
    keys = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
    return {k: torch.stack([item[k] for item in batch], dim=0) for k in keys}


class DistributedEvalSampler(Sampler[int]):
    """Shard eval data across ranks without padding or duplicated samples."""

    def __init__(self, dataset) -> None:
        if not is_dist():
            raise RuntimeError("DistributedEvalSampler requires initialized distributed mode.")
        self.dataset = dataset
        self.rank = get_rank()
        self.world_size = dist.get_world_size()

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return len(range(self.rank, len(self.dataset), self.world_size))


def duplicate_singleton_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Duplicate a singleton batch to avoid B=1 stride issues in model forward."""
    if batch["images"].shape[0] != 1:
        return batch
    return {k: torch.cat([v, v], dim=0) for k, v in batch.items()}


def resolve_checkpoint_path(args: argparse.Namespace, cfg: Config) -> str:
    if args.checkpoint is not None:
        return str(args.checkpoint)

    cfg_eval_checkpoint = cfg.get("eval_checkpoint", None)
    if cfg_eval_checkpoint is not None:
        return str(cfg_eval_checkpoint)

    output_dir = cfg.get("output_dir", None)
    if output_dir is not None:
        epoch_002_path = Path(output_dir) / DEFAULT_EVAL_CHECKPOINT_NAME
        if epoch_002_path.is_file():
            return str(epoch_002_path)

    ckpt_path = cfg.get("checkpoint", None)
    if ckpt_path is not None:
        return str(ckpt_path)

    raise ValueError(
        "No checkpoint specified. Use --checkpoint, set eval_checkpoint in config, "
        f"or ensure output_dir/{DEFAULT_EVAL_CHECKPOINT_NAME} exists."
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    depth_scale: float = 1.0,
    max_gt_depth_m: Optional[float] = None,
) -> DepthMetrics:
    """Run inference and accumulate metrics on the last-frame stereo pair."""
    local = DepthMetrics()
    bar = tqdm(
        data_loader,
        desc=f"Eval rank={get_rank()}",
        leave=False,
        disable=not is_main(),
    )

    with torch.inference_mode():
        for batch in bar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            metric_batch = batch
            model_batch = duplicate_singleton_batch(batch)

            imgs = model_batch["images"]
            ext = model_batch["extrinsics"]
            deps = metric_batch["depths"]

            norm_ext = normalize_extrinsics_to_first_frame(ext)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(
                    imgs,
                    others={
                        "extrinsics": norm_ext,
                        "intrinsics": model_batch["intrinsics"],
                    },
                )

            seq_len = imgs.shape[1]
            if seq_len < 2:
                raise RuntimeError(f"Expected stereo sequence, got seq_len={seq_len}")
            last_pair_start = seq_len - 2

            pred_depth = preds["depth"][: deps.shape[0]].squeeze(-1)
            pred_c = pred_depth[:, last_pair_start:] * depth_scale
            gt_c = deps[:, last_pair_start:]
            if max_gt_depth_m is not None:
                gt_c = gt_c.masked_fill(gt_c >= max_gt_depth_m, 0.0)

            local.update(pred_c.float(), gt_c.float())

    return local.reduce(device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VGGT on KITTI depth completion val split."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(REPO_ROOT / "configs" / "kitti_depth_stereo_ft_scaled.py"),
        help="mmengine config file.",
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint .pth to evaluate.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of val samples (for quick sanity checks).")
    parser.add_argument("--image-size", nargs=2, type=int, default=None,
                        metavar=("H", "W"),
                        help="Override image size (must be multiples of 14).")
    parser.add_argument("--depth-scale", type=float, default=None,
                        help="Multiply model depth output by this factor before computing metrics. "
                             "If not set, reads depth_pred_scale from config (default 1.0).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = setup_distributed()

    try:
        cfg = Config.fromfile(args.config)

        ckpt_path = resolve_checkpoint_path(args, cfg)
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if cfg.get("val_dataset", None) is None:
            raise ValueError("Config must define val_dataset for evaluation.")

        if args.image_size is not None:
            cfg.val_dataset.image_size = tuple(args.image_size)
        if args.max_samples is not None:
            cfg.val_dataset.max_samples = args.max_samples
        if cfg.get("val_dataloader", None) is not None:
            cfg.val_dataloader.batch_size = args.batch_size
            cfg.val_dataloader.num_workers = args.num_workers

        image_size = tuple(cfg.val_dataset.image_size)
        assert image_size[0] % 14 == 0 and image_size[1] % 14 == 0, (
            f"image_size {image_size} must be multiples of 14"
        )

        n_time_steps = int(cfg.val_dataset.get("n_time_steps", getattr(cfg, "n_time_steps", 3)))
        stride = int(cfg.val_dataset.get("stride", getattr(cfg, "stride", 1)))

        # Depth scale: CLI overrides config, config overrides default 1.0
        depth_scale = args.depth_scale if args.depth_scale is not None else float(cfg.get("depth_pred_scale", 1.0))

        # Build model
        model = MODELS.build(cfg.model)
        checkpoint_include_prefixes = tuple(cfg.get("checkpoint_include_prefixes", ()))
        msg = load_checkpoint(
            model,
            ckpt_path,
            include_prefixes=checkpoint_include_prefixes or None,
        )
        model.to(device).eval()

        log("=" * 70)
        log("KITTI Depth Completion — val split evaluation")
        log(f"  config      : {Path(args.config).resolve()}")
        log(f"  checkpoint  : {Path(ckpt_path).resolve()}")
        log(f"  ckpt status : {msg}")
        log(f"  image_size  : {image_size}")
        log(f"  n_time_steps: {n_time_steps}  stride={stride}")
        log(f"  world_size  : {get_world_size()}")
        log(f"  depth_scale : {depth_scale}")
        log("  singleton batches: duplicated inside eval only")
        log("=" * 70)

        dataset = DATASETS.build(cfg.val_dataset)
        val_sampler = (
            DistributedEvalSampler(dataset)
            if get_world_size() > 1
            else None
        )
        vdl_cfg = cfg.val_dataloader
        val_loader = DataLoader(
            dataset,
            batch_size=vdl_cfg.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=vdl_cfg.num_workers,
            pin_memory=vdl_cfg.get("pin_memory", True),
            collate_fn=collate_batch,
            drop_last=False,
        )
        log(f"Val samples: {len(dataset)}")
        log(f"Val batch_size: {vdl_cfg.batch_size}  num_workers={vdl_cfg.num_workers}")

        metrics = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            depth_scale=depth_scale,
        )
        res = metrics.result()

        if is_main():
            log("")
            log("Results:")
            log("-" * 50)
            log("  Official KITTI depth prediction metrics:")
            log(f"    iRMSE  [1/km] : {res['iRMSE']:.4f}")
            log(f"    iMAE   [1/km] : {res['iMAE']:.4f}")
            log(f"    RMSE   [mm]   : {res['RMSE']:.4f}")
            log(f"    MAE    [mm]   : {res['MAE']:.4f}")
            log("")
            log("  Supplementary metrics:")
            log(f"    Abs Rel       : {res['Abs Rel']:.4f}")
            log(f"    Sq Rel        : {res['Sq Rel']:.4f}")
            log(f"    SILog         : {res['SILog']:.4f}")
            log(f"    delta < 1.25  : {res['d1']:.4f}")
            log(f"    delta < 1.25^2: {res['d2']:.4f}")
            log(f"    delta < 1.25^3: {res['d3']:.4f}")
            log(f"    n_pixels      : {res['n_pixels']:.0f}")
            log("-" * 50)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
