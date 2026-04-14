#!/usr/bin/env python3
"""Evaluate a fine-tuned VGGT checkpoint on the vKITTI validation split.

This variant only evaluates pixels whose ground-truth depth is below 60 metres.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS

import openmm_vggt  # noqa: F401
from eval_kitti_depth import (
    DistributedEvalSampler,
    cleanup_distributed,
    collate_batch,
    evaluate,
    get_world_size,
    is_main,
    load_checkpoint,
    log,
    resolve_checkpoint_path,
    setup_distributed,
)
from torch.utils.data import DataLoader


DEFAULT_CONFIG = str(REPO_ROOT / "configs" / "vkitti_depth_stereo_ft_scaled_lt60m.py")
MAX_GT_DEPTH_M = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VGGT on the vKITTI validation split with GT depth < 60m."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG,
        help="mmengine config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pth to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of validation samples for quick sanity checks.",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Override image size (must be multiples of 14).",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help=(
            "Multiply model depth output by this factor before computing metrics. "
            "If not set, reads depth_pred_scale from config (default 1.0)."
        ),
    )
    return parser.parse_args()


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
        depth_scale = (
            args.depth_scale
            if args.depth_scale is not None
            else float(cfg.get("depth_pred_scale", 1.0))
        )

        model = MODELS.build(cfg.model)
        checkpoint_include_prefixes = tuple(cfg.get("checkpoint_include_prefixes", ()))
        msg = load_checkpoint(
            model,
            ckpt_path,
            include_prefixes=checkpoint_include_prefixes or None,
        )
        model.to(device).eval()

        log("=" * 70)
        log("vKITTI Depth — validation split evaluation (GT depth < 60m)")
        log(f"  config      : {Path(args.config).resolve()}")
        log(f"  checkpoint  : {Path(ckpt_path).resolve()}")
        log(f"  ckpt status : {msg}")
        log(f"  image_size  : {image_size}")
        log(f"  n_time_steps: {n_time_steps}  stride={stride}")
        log(f"  world_size  : {get_world_size()}")
        log(f"  depth_scale : {depth_scale}")
        log(f"  gt_depth <  : {MAX_GT_DEPTH_M} m")
        log("  singleton batches: duplicated inside eval only")
        log("=" * 70)

        dataset = DATASETS.build(cfg.val_dataset)
        val_sampler = DistributedEvalSampler(dataset) if get_world_size() > 1 else None
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
            max_gt_depth_m=MAX_GT_DEPTH_M,
        )
        res = metrics.result()

        if is_main():
            log("")
            log("Results:")
            log("-" * 50)
            log("  Depth metrics:")
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
