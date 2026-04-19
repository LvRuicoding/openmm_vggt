#!/usr/bin/env python3
# Usage:
#   python tools/stat_ddad_depth_prediction_distribution.py \
#       configs/early/ddad_depth_6cam_mix_window_attn_early_ft.py \
#       --checkpoint trainoutput/ddad_depth_6cam_mix_window_attn_early_ft/epoch_004.pth
#
# Collect last-timestep 6-camera prediction-depth distributions on DDAD.
# By default it only counts predicted values at GT-valid evaluation pixels,
# where the GT mask is (gt > 0) & (gt < 655).
#
# Useful variants:
#   --pixel-selection all_pixels
#   --pixel-selection eval_valid_gt --gt-mask-max-depth-m 50
#   --pixel-selection all_pixels --pred-max-depth-m 50
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openmm-vggt")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS

import openmm_vggt  # noqa: F401
import tools.eval_ddad_depth_mix as base_eval


DEFAULT_CONFIG = str(REPO_ROOT / "configs" / "early" / "ddad_depth_6cam_mix_window_attn_early_ft.py")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "statics")
DEFAULT_BUCKET_EDGES = (0.0, 10.0, 20.0, 40.0, 80.0, 120.0)


class StreamingPredictionStats:
    def __init__(self, bucket_edges: Sequence[float], sample_stride: int) -> None:
        self.bucket_edges = np.asarray(bucket_edges, dtype=np.float64)
        self.sample_stride = max(int(sample_stride), 1)

        self.total_pixels = 0
        self.sum_depth = 0.0
        self.sum_sq_depth = 0.0
        self.min_depth = float("inf")
        self.max_depth = float("-inf")
        self.bucket_counts = np.zeros(len(self.bucket_edges) + 1, dtype=np.int64)
        self.sampled_chunks: List[np.ndarray] = []
        self.sampled_pixel_count = 0

    def update(self, values: torch.Tensor) -> None:
        flat_values = values.detach().reshape(-1).to(dtype=torch.float64).cpu().numpy()
        if flat_values.size == 0:
            return

        self.total_pixels += int(flat_values.size)
        self.sum_depth += float(flat_values.sum())
        self.sum_sq_depth += float(np.square(flat_values).sum())
        self.min_depth = min(self.min_depth, float(flat_values.min()))
        self.max_depth = max(self.max_depth, float(flat_values.max()))

        bucket_ids = np.digitize(flat_values, self.bucket_edges, right=False)
        self.bucket_counts += np.bincount(bucket_ids, minlength=self.bucket_counts.size)

        sampled = flat_values[:: self.sample_stride].astype(np.float32, copy=False)
        if sampled.size > 0:
            self.sampled_chunks.append(sampled.copy())
            self.sampled_pixel_count += int(sampled.size)

    def sampled_values(self) -> np.ndarray:
        if not self.sampled_chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self.sampled_chunks, axis=0)

    def reduce(self, device: torch.device) -> "StreamingPredictionStats":
        reduced = StreamingPredictionStats(self.bucket_edges.tolist(), self.sample_stride)
        reduced.total_pixels = self.total_pixels
        reduced.sum_depth = self.sum_depth
        reduced.sum_sq_depth = self.sum_sq_depth
        reduced.min_depth = self.min_depth
        reduced.max_depth = self.max_depth
        reduced.bucket_counts = self.bucket_counts.copy()
        reduced.sampled_pixel_count = self.sampled_pixel_count

        if base_eval.get_rank() == 0 and not base_eval.is_dist():
            return reduced
        if not base_eval.is_dist():
            return reduced

        scalars = torch.tensor(
            [
                float(self.total_pixels),
                float(self.sum_depth),
                float(self.sum_sq_depth),
                float(self.min_depth),
                float(self.max_depth),
            ],
            dtype=torch.float64,
            device=device,
        )
        bucket_counts = torch.tensor(self.bucket_counts, dtype=torch.int64, device=device)

        dist.all_reduce(scalars[:3], op=dist.ReduceOp.SUM)
        dist.all_reduce(scalars[3:4], op=dist.ReduceOp.MIN)
        dist.all_reduce(scalars[4:5], op=dist.ReduceOp.MAX)
        dist.all_reduce(bucket_counts, op=dist.ReduceOp.SUM)

        reduced.total_pixels = int(round(float(scalars[0].item())))
        reduced.sum_depth = float(scalars[1].item())
        reduced.sum_sq_depth = float(scalars[2].item())
        reduced.min_depth = float(scalars[3].item())
        reduced.max_depth = float(scalars[4].item())
        reduced.bucket_counts = bucket_counts.cpu().numpy()
        reduced.sampled_chunks = []
        reduced.sampled_pixel_count = 0
        return reduced

    def summary(self, sampled_values: np.ndarray) -> Dict[str, object]:
        if self.total_pixels == 0:
            raise RuntimeError("No predicted pixels were collected.")

        mean = self.sum_depth / float(self.total_pixels)
        variance = max(self.sum_sq_depth / float(self.total_pixels) - mean * mean, 0.0)
        percentiles = {
            "p01": 1.0,
            "p05": 5.0,
            "p10": 10.0,
            "p25": 25.0,
            "p50": 50.0,
            "p75": 75.0,
            "p90": 90.0,
            "p95": 95.0,
            "p99": 99.0,
            "p995": 99.5,
        }
        percentile_values = (
            {name: float(np.percentile(sampled_values, q)) for name, q in percentiles.items()}
            if sampled_values.size > 0
            else {name: None for name in percentiles}
        )

        bucket_summary = []
        total = max(self.total_pixels, 1)
        lower = float("-inf")
        for idx, count in enumerate(self.bucket_counts):
            if idx < len(self.bucket_edges):
                upper = float(self.bucket_edges[idx])
                label = f"< {upper:g}m" if idx == 0 else f"[{lower:g}, {upper:g})m"
            else:
                upper = float("inf")
                label = f">= {lower:g}m"
            bucket_summary.append(
                {
                    "label": label,
                    "count": int(count),
                    "ratio": float(count) / float(total),
                }
            )
            lower = upper

        return {
            "total_pixels": int(self.total_pixels),
            "sampled_pixel_count": int(sampled_values.size),
            "sample_stride": int(self.sample_stride),
            "min_m": float(self.min_depth),
            "max_m": float(self.max_depth),
            "mean_m": float(mean),
            "std_m": float(np.sqrt(variance)),
            "percentiles_m": percentile_values,
            "bucket_counts": bucket_summary,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect DDAD prediction-depth distributions on the last timestep cameras."
    )
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG, help="mmengine config file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint .pth to evaluate.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
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
        help="Multiply model depth output by this factor before collecting statistics.",
    )
    parser.add_argument(
        "--supervision-source",
        type=str,
        default=None,
        choices=("batch_depth", "projected_points"),
        help="Override GT depth construction. If not set, defaults to batch_depth for DDAD.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=128,
        help="Keep every N-th selected predicted pixel for approximate histograms/percentiles.",
    )
    parser.add_argument(
        "--pixel-selection",
        type=str,
        default="eval_valid_gt",
        choices=("all_pixels", "eval_valid_gt"),
        help="Collect all predicted pixels or only predictions at valid-GT evaluation pixels.",
    )
    parser.add_argument(
        "--gt-mask-min-depth-m",
        type=float,
        default=0.0,
        help="Minimum GT depth used when pixel-selection=eval_valid_gt. Mask uses gt > min_depth.",
    )
    parser.add_argument(
        "--gt-mask-max-depth-m",
        type=float,
        default=655.0,
        help="Maximum GT depth used when pixel-selection=eval_valid_gt. Mask uses gt < max_depth.",
    )
    parser.add_argument(
        "--pred-min-depth-m",
        type=float,
        default=None,
        help="Optional lower bound for predicted depth values kept in the distribution.",
    )
    parser.add_argument(
        "--pred-max-depth-m",
        type=float,
        default=None,
        help="Optional upper bound for predicted depth values kept in the distribution.",
    )
    return parser.parse_args()


def build_loader(cfg: Config, batch_size: int, num_workers: int, max_samples: int | None) -> Tuple[object, DataLoader]:
    if cfg.get("val_dataset", None) is None:
        raise ValueError("Config must define val_dataset.")
    if max_samples is not None:
        cfg.val_dataset.max_samples = max_samples
    if cfg.get("val_dataloader", None) is not None:
        cfg.val_dataloader.batch_size = batch_size
        cfg.val_dataloader.num_workers = num_workers

    cfg.val_dataset.return_lidar = True
    dataset = DATASETS.build(cfg.val_dataset)
    sampler = base_eval.DistributedEvalSampler(dataset) if base_eval.is_dist() else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=cfg.val_dataloader.get("pin_memory", True),
        collate_fn=base_eval.collate_batch,
        drop_last=False,
    )
    return dataset, loader


def build_eval_mask(gt_depth: torch.Tensor, min_depth_m: float, max_depth_m: float) -> torch.Tensor:
    return (gt_depth > float(min_depth_m)) & (gt_depth < float(max_depth_m))


def apply_prediction_value_filter(
    pred_values: torch.Tensor,
    pred_min_depth_m: float | None,
    pred_max_depth_m: float | None,
) -> torch.Tensor:
    if pred_values.numel() == 0:
        return pred_values
    keep = torch.ones_like(pred_values, dtype=torch.bool)
    if pred_min_depth_m is not None:
        keep &= pred_values > float(pred_min_depth_m)
    if pred_max_depth_m is not None:
        keep &= pred_values < float(pred_max_depth_m)
    return pred_values[keep]


def collect_prediction_distribution(
    cfg: Config,
    checkpoint_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    sample_stride: int,
    max_samples: int | None,
    depth_scale: float,
    supervision_source: str,
    pixel_selection: str,
    gt_mask_min_depth_m: float,
    gt_mask_max_depth_m: float,
    pred_min_depth_m: float | None,
    pred_max_depth_m: float | None,
    output_dir: Path,
) -> Dict[str, object] | None:
    model = MODELS.build(cfg.model)
    checkpoint_include_prefixes = tuple(cfg.get("checkpoint_include_prefixes", ()))
    checkpoint_status = base_eval.load_checkpoint(
        model,
        checkpoint_path,
        include_prefixes=checkpoint_include_prefixes or None,
    )
    model.to(device).eval()

    dataset, loader = build_loader(
        cfg=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
    )
    if base_eval.is_main():
        print(
            f"[ddad] dataset_len={len(dataset)} depth_scale={depth_scale} "
            f"pixel_selection={pixel_selection} supervision_source={supervision_source}",
            flush=True,
        )

    stats = StreamingPredictionStats(DEFAULT_BUCKET_EDGES, sample_stride)
    gt_selected_pixel_count = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            metric_batch = batch
            model_batch = base_eval.duplicate_singleton_batch(batch)
            imgs = model_batch["images"]
            norm_ext = base_eval.normalize_extrinsics_to_first_frame(model_batch["extrinsics"])

            with torch.amp.autocast(device_type=device.type, enabled=bool(cfg.get("amp", True)) and device.type == "cuda"):
                preds = model(
                    imgs,
                    others={
                        "extrinsics": norm_ext,
                        "intrinsics": model_batch["intrinsics"],
                        "camera_to_world": model_batch["camera_to_world"],
                        "lidar_to_world": model_batch["lidar_to_world"],
                        "points": model_batch["points"],
                        "point_mask": model_batch["point_mask"],
                    },
                )

            pred_depth = preds["depth"][: batch["images"].shape[0]].squeeze(-1) * depth_scale
            seq_len = pred_depth.shape[1]
            frame_count = metric_batch["points"].shape[1]
            if frame_count <= 0 or seq_len % frame_count != 0:
                raise RuntimeError(f"Cannot infer camera count from seq_len={seq_len}, frame_count={frame_count}")
            cam_num = seq_len // frame_count
            last_timestep_start = seq_len - cam_num
            pred_c = pred_depth[:, last_timestep_start:]

            if pixel_selection == "all_pixels":
                selected_pred = pred_c.float().reshape(-1)
            else:
                gt_depth = base_eval.build_depth_supervision_target(metric_batch, supervision_source)
                gt_c = gt_depth[:, last_timestep_start:]
                eval_mask = build_eval_mask(gt_c, gt_mask_min_depth_m, gt_mask_max_depth_m)
                gt_selected_pixel_count += int(eval_mask.sum().item())
                selected_pred = pred_c[eval_mask].float()

            selected_pred = apply_prediction_value_filter(
                selected_pred,
                pred_min_depth_m=pred_min_depth_m,
                pred_max_depth_m=pred_max_depth_m,
            )
            stats.update(selected_pred)

            if base_eval.is_main() and (batch_idx % 100 == 0 or batch_idx == len(loader)):
                print(f"[ddad] processed {batch_idx}/{len(loader)} batches", flush=True)

    shard_path = output_dir / f".ddad_prediction_sampled_values.rank{base_eval.get_rank():03d}.npy"
    np.save(shard_path, stats.sampled_values())
    if base_eval.is_dist():
        dist.barrier()

    reduced_stats = stats.reduce(device)
    gt_selected_tensor = torch.tensor([gt_selected_pixel_count], dtype=torch.int64, device=device)
    if base_eval.is_dist():
        dist.all_reduce(gt_selected_tensor, op=dist.ReduceOp.SUM)
    gt_selected_pixel_count = int(gt_selected_tensor.item())

    if not base_eval.is_main():
        return None

    sampled_parts = [
        np.load(output_dir / f".ddad_prediction_sampled_values.rank{rank:03d}.npy")
        for rank in range(dist.get_world_size() if base_eval.is_dist() else 1)
    ]
    sampled_values = np.concatenate(sampled_parts, axis=0) if sampled_parts else np.empty(0, dtype=np.float32)
    sampled_path = output_dir / "ddad_prediction_sampled_values.npy"
    np.save(sampled_path, sampled_values)
    for rank in range(dist.get_world_size() if base_eval.is_dist() else 1):
        temp_path = output_dir / f".ddad_prediction_sampled_values.rank{rank:03d}.npy"
        if temp_path.exists():
            temp_path.unlink()

    summary = reduced_stats.summary(sampled_values)
    summary["dataset_name"] = "ddad"
    summary["dataset_length"] = len(dataset)
    summary["config_path"] = str(Path(cfg.filename).resolve())
    summary["checkpoint_path"] = str(Path(checkpoint_path).resolve())
    summary["checkpoint_status"] = checkpoint_status
    summary["depth_scale"] = depth_scale
    summary["pixel_selection"] = pixel_selection
    summary["supervision_source"] = supervision_source
    summary["gt_mask_min_depth_m"] = gt_mask_min_depth_m
    summary["gt_mask_max_depth_m"] = gt_mask_max_depth_m
    summary["pred_min_depth_m"] = pred_min_depth_m
    summary["pred_max_depth_m"] = pred_max_depth_m
    summary["sampled_values_path"] = str(sampled_path)
    summary["gt_selected_pixel_count"] = gt_selected_pixel_count
    return summary


def make_histogram_plot(summary: Dict[str, object], output_path: Path) -> None:
    sampled_values = np.load(summary["sampled_values_path"]).astype(np.float32, copy=False)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=160)
    p995 = summary["percentiles_m"]["p995"]
    hist_upper = max(float(p995), 40.0) if p995 is not None else 40.0
    bins = np.linspace(0.0, hist_upper, 120)
    clipped = np.clip(sampled_values, 0.0, hist_upper)
    overflow_ratio = float(np.mean(sampled_values > hist_upper)) if sampled_values.size else 0.0
    ax.hist(clipped, bins=bins, density=True, color="#2d6cdf", alpha=0.85)
    ax.set_title(
        "DDAD prediction depth\n"
        f"selection={summary['pixel_selection']} sampled={summary['sampled_pixel_count']:,} "
        f"overflow>{hist_upper:.1f}m={overflow_ratio:.3%}"
    )
    ax.set_xlabel("Predicted depth (m)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_bucket_csv(summary: Dict[str, object], output_path: Path) -> None:
    labels = [bucket["label"] for bucket in summary["bucket_counts"]]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(labels)
        writer.writerow([f"{bucket['ratio']:.8f}" for bucket in summary["bucket_counts"]])


def write_stats_csv(summary: Dict[str, object], output_path: Path) -> None:
    percentile_names = list(summary["percentiles_m"].keys())
    fieldnames = [
        "dataset",
        "total_pixels",
        "gt_selected_pixel_count",
        "sampled_pixel_count",
        "sample_stride",
        "depth_scale",
        "pixel_selection",
        "supervision_source",
        "gt_mask_min_depth_m",
        "gt_mask_max_depth_m",
        "pred_min_depth_m",
        "pred_max_depth_m",
        "min_m",
        "max_m",
        "mean_m",
        "std_m",
        *percentile_names,
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        row = {
            "dataset": summary["dataset_name"],
            "total_pixels": summary["total_pixels"],
            "gt_selected_pixel_count": summary["gt_selected_pixel_count"],
            "sampled_pixel_count": summary["sampled_pixel_count"],
            "sample_stride": summary["sample_stride"],
            "depth_scale": summary["depth_scale"],
            "pixel_selection": summary["pixel_selection"],
            "supervision_source": summary["supervision_source"],
            "gt_mask_min_depth_m": summary["gt_mask_min_depth_m"],
            "gt_mask_max_depth_m": summary["gt_mask_max_depth_m"],
            "pred_min_depth_m": summary["pred_min_depth_m"],
            "pred_max_depth_m": summary["pred_max_depth_m"],
            "min_m": f"{summary['min_m']:.6f}",
            "max_m": f"{summary['max_m']:.6f}",
            "mean_m": f"{summary['mean_m']:.6f}",
            "std_m": f"{summary['std_m']:.6f}",
        }
        for key, value in summary["percentiles_m"].items():
            row[key] = "" if value is None else f"{value:.6f}"
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    device = base_eval.setup_distributed()
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = Config.fromfile(args.config)
        if args.image_size is not None:
            cfg.val_dataset.image_size = tuple(args.image_size)
        checkpoint_path = base_eval.resolve_checkpoint_path(args, cfg)
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        depth_scale = args.depth_scale if args.depth_scale is not None else float(cfg.get("depth_pred_scale", 1.0))
        supervision_source = args.supervision_source if args.supervision_source is not None else "batch_depth"

        summary = collect_prediction_distribution(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_stride=args.sample_stride,
            max_samples=args.max_samples,
            depth_scale=depth_scale,
            supervision_source=supervision_source,
            pixel_selection=args.pixel_selection,
            gt_mask_min_depth_m=args.gt_mask_min_depth_m,
            gt_mask_max_depth_m=args.gt_mask_max_depth_m,
            pred_min_depth_m=args.pred_min_depth_m,
            pred_max_depth_m=args.pred_max_depth_m,
            output_dir=output_dir,
        )
        if not base_eval.is_main():
            return

        histogram_path = output_dir / "ddad_depth_prediction_distribution_histogram.png"
        make_histogram_plot(summary, histogram_path)

        stats_csv_path = output_dir / "ddad_depth_prediction_distribution_stats.csv"
        write_stats_csv(summary, stats_csv_path)

        bucket_csv_path = output_dir / "ddad_depth_prediction_distribution_bucket_ratios.csv"
        write_bucket_csv(summary, bucket_csv_path)

        summary_path = output_dir / "ddad_depth_prediction_distribution_summary.json"
        payload = {
            "device": str(device),
            "world_size": dist.get_world_size() if base_eval.is_dist() else 1,
            "bucket_edges_m": list(DEFAULT_BUCKET_EDGES),
            "histogram_path": str(histogram_path),
            "stats_csv_path": str(stats_csv_path),
            "bucket_csv_path": str(bucket_csv_path),
            "summary": summary,
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print(f"Saved histogram: {histogram_path}", flush=True)
        print(f"Saved stats CSV: {stats_csv_path}", flush=True)
        print(f"Saved bucket CSV: {bucket_csv_path}", flush=True)
        print(f"Saved summary JSON: {summary_path}", flush=True)
    finally:
        base_eval.cleanup_distributed()


if __name__ == "__main__":
    main()
