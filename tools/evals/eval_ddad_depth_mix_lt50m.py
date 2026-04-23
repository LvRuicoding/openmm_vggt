#!/usr/bin/env python3
"""Evaluate DDAD depth with GT supervision restricted to pixels within 50m by default.

This script reuses tools/eval_ddad_depth_mix.py and only overrides the metric
mask so that pixels with GT depth > threshold are ignored.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
from torch.utils.data import DataLoader

import tools.eval_ddad_depth_mix as base_eval


DEFAULT_CONFIG = str(REPO_ROOT / "configs" / "early" / "ddad_depth_6cam_mix_window_attn_early_ft.py")
DEFAULT_MAX_GT_DEPTH_M = 50.0

ACTIVE_MAX_GT_DEPTH_M = DEFAULT_MAX_GT_DEPTH_M


class DepthMetricsLT50M(base_eval.DepthMetrics):
    def update(self, pred_m, gt_m) -> None:
        mask = (gt_m > 0.0) & (gt_m < 655.0) & (gt_m <= ACTIVE_MAX_GT_DEPTH_M)
        if not mask.any():
            return
        pred = pred_m[mask].clamp_min(1e-6).double()
        gt = gt_m[mask].double()
        n = float(mask.sum())

        diff_mm = (pred - gt) * 1000.0
        diff_ikm = 1000.0 / pred - 1000.0 / gt
        self.sq_mm += float((diff_mm ** 2).sum())
        self.abs_mm += float(diff_mm.abs().sum())
        self.sq_ikm += float((diff_ikm ** 2).sum())
        self.abs_ikm += float(diff_ikm.abs().sum())

        self.abs_rel += float(((pred - gt).abs() / gt).sum())
        self.sq_rel += float((((pred - gt) ** 2) / gt).sum())
        log_diff = base_eval.torch.log(pred) - base_eval.torch.log(gt)
        self.silog_sq += float((log_diff ** 2).sum())
        self.silog_lin += float(log_diff.sum())
        ratio = base_eval.torch.max(pred / gt, gt / pred)
        self.d1 += float((ratio < 1.25).sum())
        self.d2 += float((ratio < 1.25 ** 2).sum())
        self.d3 += float((ratio < 1.25 ** 3).sum())
        self.n += n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DDAD depth with GT depth limited to <= 50m by default."
    )
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG, help="mmengine config file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint .pth to evaluate.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
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
        help="Multiply model depth output by this factor before computing metrics. If not set, reads config.",
    )
    parser.add_argument(
        "--supervision-source",
        type=str,
        default=None,
        choices=("batch_depth",),
        help="Override GT depth construction. Only batch_depth is supported.",
    )
    parser.add_argument(
        "--max-gt-depth-m",
        type=float,
        default=None,
        help="Only GT pixels with depth <= this threshold contribute to metrics. Default: read config or 50.",
    )
    return parser.parse_args()


def resolve_max_gt_depth_m(cfg: Config, cli_value: float | None) -> float:
    if cli_value is not None:
        return float(cli_value)
    if cfg.get("eval_max_gt_depth_m", None) is not None:
        return float(cfg.eval_max_gt_depth_m)
    if cfg.get("max_gt_depth_m", None) is not None:
        return float(cfg.max_gt_depth_m)
    return DEFAULT_MAX_GT_DEPTH_M


def format_metrics_report_lt50m(
    config_path: str,
    ckpt_path: str,
    batch_size: int,
    num_workers: int,
    depth_scale: float,
    supervision_source: str,
    evaluated_cameras: int,
    max_gt_depth_m: float,
    metrics,
) -> str:
    lines = [
        "DDAD Depth Metrics",
        f"config: {config_path}",
        f"checkpoint: {ckpt_path}",
        f"batch_size: {batch_size}",
        f"num_workers: {num_workers}",
        f"depth_scale: {depth_scale}",
        f"supervision_source: {supervision_source}",
        f"evaluated_cameras: {evaluated_cameras}",
        f"max_gt_depth_m: {max_gt_depth_m}",
        "",
    ]
    for key in ("iRMSE", "iMAE", "RMSE", "MAE", "Abs Rel", "Sq Rel", "SILog", "d1", "d2", "d3"):
        lines.append(f"{key}: {metrics[key]:.6f}")
    lines.append(f"n_pixels: {metrics['n_pixels']:.0f}")
    return "\n".join(lines) + "\n"


def save_metrics_report_lt50m(
    ckpt_path: str,
    config_path: str,
    batch_size: int,
    num_workers: int,
    depth_scale: float,
    supervision_source: str,
    evaluated_cameras: int,
    max_gt_depth_m: float,
    metrics,
) -> Path:
    ckpt = Path(ckpt_path)
    report_path = ckpt.with_name(f"{ckpt.stem}_ddad_eval_lt50m.txt")
    report_text = format_metrics_report_lt50m(
        config_path=config_path,
        ckpt_path=ckpt_path,
        batch_size=batch_size,
        num_workers=num_workers,
        depth_scale=depth_scale,
        supervision_source=supervision_source,
        evaluated_cameras=evaluated_cameras,
        max_gt_depth_m=max_gt_depth_m,
        metrics=metrics,
    )
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main() -> None:
    global ACTIVE_MAX_GT_DEPTH_M

    args = parse_args()
    device = base_eval.setup_distributed()

    try:
        cfg = Config.fromfile(args.config)
        ACTIVE_MAX_GT_DEPTH_M = resolve_max_gt_depth_m(cfg, args.max_gt_depth_m)

        ckpt_path = base_eval.resolve_checkpoint_path(args, cfg)
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if cfg.get("val_dataset", None) is None:
            raise ValueError("Config must define val_dataset for evaluation.")

        if args.image_size is not None:
            cfg.val_dataset.image_size = tuple(args.image_size)
        if args.max_samples is not None:
            cfg.val_dataset.max_samples = args.max_samples
        val_dataloader_cfg = cfg.get("val_dataloader", {})
        if val_dataloader_cfg:
            cfg.val_dataloader.batch_size = args.batch_size
            cfg.val_dataloader.num_workers = args.num_workers

        cfg.val_dataset.return_lidar = True
        image_size = tuple(cfg.val_dataset.image_size)
        assert image_size[0] % 14 == 0 and image_size[1] % 14 == 0, (
            f"image_size {image_size} must be multiples of 14"
        )

        depth_scale = args.depth_scale if args.depth_scale is not None else float(cfg.get("depth_pred_scale", 1.0))
        supervision_source = (
            args.supervision_source
            if args.supervision_source is not None
            else cfg.get("depth_supervision_source", "batch_depth")
        )

        base_eval.DepthMetrics = DepthMetricsLT50M

        model = MODELS.build(cfg.model)
        checkpoint_include_prefixes = tuple(cfg.get("checkpoint_include_prefixes", ()))
        msg = base_eval.load_checkpoint(
            model,
            ckpt_path,
            include_prefixes=checkpoint_include_prefixes or None,
        )
        base_eval.log(f"Loaded checkpoint: {msg}")
        base_eval.log(f"Using GT depth threshold <= {ACTIVE_MAX_GT_DEPTH_M:.2f}m for evaluation metrics")
        model.eval().to(device)
        if base_eval.is_dist():
            model = base_eval.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        dataset = DATASETS.build(cfg.val_dataset)
        sampler = base_eval.DistributedEvalSampler(dataset) if base_eval.is_dist() else None
        val_batch_size = int(val_dataloader_cfg.get("batch_size", args.batch_size))
        val_num_workers = int(val_dataloader_cfg.get("num_workers", args.num_workers))
        data_loader = DataLoader(
            dataset,
            batch_size=val_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=val_num_workers,
            pin_memory=bool(val_dataloader_cfg.get("pin_memory", True)),
            collate_fn=base_eval.collate_batch,
            drop_last=False,
        )

        metrics, evaluated_cameras = base_eval.evaluate(
            model,
            data_loader,
            device,
            depth_scale=depth_scale,
            supervision_source=supervision_source,
            amp_enabled=bool(cfg.get("amp", True)),
        )
        metrics = metrics.result()

        if base_eval.is_main():
            print("\nDDAD Depth Metrics", flush=True)
            print(f"{'cameras':>16s}: {evaluated_cameras}", flush=True)
            print(f"{'gt_mode':>16s}: {supervision_source}", flush=True)
            print(f"{'max_gt_depth_m':>16s}: {ACTIVE_MAX_GT_DEPTH_M:.2f}", flush=True)
            for key in ("iRMSE", "iMAE", "RMSE", "MAE", "Abs Rel", "Sq Rel", "SILog", "d1", "d2", "d3"):
                print(f"{key:>16s}: {metrics[key]:.6f}", flush=True)
            print(f"{'n_pixels':>16s}: {metrics['n_pixels']:.0f}", flush=True)
            report_path = save_metrics_report_lt50m(
                ckpt_path=ckpt_path,
                config_path=args.config,
                batch_size=val_batch_size,
                num_workers=val_num_workers,
                depth_scale=depth_scale,
                supervision_source=supervision_source,
                evaluated_cameras=evaluated_cameras,
                max_gt_depth_m=ACTIVE_MAX_GT_DEPTH_M,
                metrics=metrics,
            )
            print(f"\nSaved metrics to: {report_path}", flush=True)
    finally:
        base_eval.cleanup_distributed()


if __name__ == "__main__":
    main()
