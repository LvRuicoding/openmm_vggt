#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
from eval_kitti_depth import (
    DistributedEvalSampler,
    cleanup_distributed,
    collate_batch,
    duplicate_singleton_batch,
    get_rank,
    get_world_size,
    is_main,
    load_checkpoint,
    normalize_extrinsics_to_first_frame,
    setup_distributed,
)


DEFAULT_KITTI_CONFIG = str(REPO_ROOT / "configs" / "kitti_depth_stereo_ft_scaled.py")
DEFAULT_KITTI_CHECKPOINT = str(
    REPO_ROOT / "trainoutput" / "kitti_depth_stereo_ft_scaled" / "epoch_002.pth"
)
DEFAULT_VKITTI_CONFIG = str(REPO_ROOT / "configs" / "vkitti_depth_stereo_ft_scaled.py")
DEFAULT_VKITTI_CHECKPOINT = str(
    REPO_ROOT / "trainoutput" / "vkitti_depth_stereo_ft_scaled" / "epoch_004.pth"
)
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "statics")

DEFAULT_BUCKET_EDGES = (0.0, 10.0, 20.0, 40.0, 80.0, 120.0)


@dataclass
class DatasetSpec:
    name: str
    config_path: str
    checkpoint_path: str


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

    def update(self, pred_depth: torch.Tensor) -> None:
        values = pred_depth.detach().reshape(-1).to(dtype=torch.float64).cpu().numpy()
        if values.size == 0:
            return

        self.total_pixels += int(values.size)
        self.sum_depth += float(values.sum())
        self.sum_sq_depth += float(np.square(values).sum())
        self.min_depth = min(self.min_depth, float(values.min()))
        self.max_depth = max(self.max_depth, float(values.max()))

        bucket_ids = np.digitize(values, self.bucket_edges, right=False)
        self.bucket_counts += np.bincount(bucket_ids, minlength=self.bucket_counts.size)

        sampled = values[:: self.sample_stride].astype(np.float32, copy=False)
        if sampled.size > 0:
            self.sampled_chunks.append(sampled.copy())
            self.sampled_pixel_count += int(sampled.size)

    def _sampled_values(self) -> np.ndarray:
        if not self.sampled_chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self.sampled_chunks, axis=0)

    def _bucket_summary(self) -> List[Dict[str, float | int | str]]:
        total = max(self.total_pixels, 1)
        entries: List[Dict[str, float | int | str]] = []
        lower = float("-inf")
        for idx, count in enumerate(self.bucket_counts):
            if idx < len(self.bucket_edges):
                upper = float(self.bucket_edges[idx])
                label = f"< {upper:g}m" if idx == 0 else f"[{lower:g}, {upper:g})m"
            else:
                upper = float("inf")
                label = f">= {lower:g}m"
            entries.append(
                {
                    "label": label,
                    "count": int(count),
                    "ratio": float(count) / float(total),
                }
            )
            lower = upper
        return entries

    def summary(self, sampled_values: np.ndarray | None = None) -> Dict[str, object]:
        if self.total_pixels == 0:
            raise RuntimeError("No predicted pixels were collected.")

        sampled = self._sampled_values() if sampled_values is None else sampled_values
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
            {name: float(np.percentile(sampled, q)) for name, q in percentiles.items()}
            if sampled.size > 0
            else {name: None for name in percentiles}
        )

        return {
            "total_pixels": int(self.total_pixels),
            "sampled_pixel_count": int(sampled.size),
            "sample_stride": int(self.sample_stride),
            "min_m": float(self.min_depth),
            "max_m": float(self.max_depth),
            "mean_m": float(mean),
            "std_m": float(np.sqrt(variance)),
            "percentiles_m": percentile_values,
            "bucket_counts": self._bucket_summary(),
            "sampled_values_path": None,
        }

    def sampled_values(self) -> np.ndarray:
        return self._sampled_values()

    def reduce(self, device: torch.device) -> "StreamingPredictionStats":
        reduced = StreamingPredictionStats(
            bucket_edges=self.bucket_edges.tolist(),
            sample_stride=self.sample_stride,
        )
        reduced.total_pixels = self.total_pixels
        reduced.sum_depth = self.sum_depth
        reduced.sum_sq_depth = self.sum_sq_depth
        reduced.min_depth = self.min_depth
        reduced.max_depth = self.max_depth
        reduced.bucket_counts = self.bucket_counts.copy()
        reduced.sampled_pixel_count = self.sampled_pixel_count

        if get_world_size() == 1:
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
        bucket_counts = torch.tensor(
            self.bucket_counts,
            dtype=torch.int64,
            device=device,
        )

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect prediction-depth distributions on KITTI and vKITTI validation sets."
    )
    parser.add_argument("--kitti-config", type=str, default=DEFAULT_KITTI_CONFIG)
    parser.add_argument("--kitti-checkpoint", type=str, default=DEFAULT_KITTI_CHECKPOINT)
    parser.add_argument("--vkitti-config", type=str, default=DEFAULT_VKITTI_CONFIG)
    parser.add_argument("--vkitti-checkpoint", type=str, default=DEFAULT_VKITTI_CHECKPOINT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=128,
        help="Keep every N-th predicted pixel for approximate histograms/percentiles.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for each validation dataset, mainly for debugging.",
    )
    return parser.parse_args()


def pick_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resolve_device(device_arg: str | None) -> torch.device:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return setup_distributed()
    return pick_device(device_arg)


def build_loader(
    cfg: Config,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> Tuple[object, DataLoader]:
    if cfg.get("val_dataset", None) is None:
        raise ValueError("Config must define val_dataset.")

    if max_samples is not None:
        cfg.val_dataset.max_samples = max_samples
    if cfg.get("val_dataloader", None) is not None:
        cfg.val_dataloader.batch_size = batch_size
        cfg.val_dataloader.num_workers = num_workers

    dataset = DATASETS.build(cfg.val_dataset)
    sampler = DistributedEvalSampler(dataset) if get_world_size() > 1 else None
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=cfg.val_dataloader.get("pin_memory", True),
        collate_fn=collate_batch,
        drop_last=False,
    )
    return dataset, val_loader


def collect_prediction_distribution(
    spec: DatasetSpec,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    sample_stride: int,
    max_samples: int | None,
    output_dir: Path,
) -> Dict[str, object]:
    cfg = Config.fromfile(spec.config_path)
    depth_scale = float(cfg.get("depth_pred_scale", 1.0))
    model = MODELS.build(cfg.model)
    checkpoint_include_prefixes = tuple(cfg.get("checkpoint_include_prefixes", ()))
    checkpoint_status = load_checkpoint(
        model,
        spec.checkpoint_path,
        include_prefixes=checkpoint_include_prefixes or None,
    )
    model.to(device).eval()

    dataset, loader = build_loader(
        cfg=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
    )

    if is_main():
        print(
            f"[{spec.name}] dataset_len={len(dataset)} depth_scale={depth_scale} "
            f"device={device} world_size={get_world_size()}",
            flush=True,
        )
    stats = StreamingPredictionStats(
        bucket_edges=DEFAULT_BUCKET_EDGES,
        sample_stride=sample_stride,
    )

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            model_batch = duplicate_singleton_batch(batch)
            imgs = model_batch["images"]
            ext = model_batch["extrinsics"]
            norm_ext = normalize_extrinsics_to_first_frame(ext)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(
                    imgs,
                    others={
                        "extrinsics": norm_ext,
                        "intrinsics": model_batch["intrinsics"],
                    },
                )

            pred_depth = preds["depth"][: batch["images"].shape[0]].squeeze(-1) * depth_scale
            last_pair_start = pred_depth.shape[1] - 2
            pred_pair = pred_depth[:, last_pair_start:]
            stats.update(pred_pair)

            if is_main() and (batch_idx % 100 == 0 or batch_idx == len(loader)):
                print(
                    f"[{spec.name}] processed {batch_idx}/{len(loader)} batches",
                    flush=True,
                )

    shard_path = output_dir / f".{spec.name}_prediction_sampled_values.rank{get_rank():03d}.npy"
    np.save(shard_path, stats.sampled_values())
    if get_world_size() > 1:
        dist.barrier()

    reduced_stats = stats.reduce(device)
    if not is_main():
        return None

    sampled_parts = [
        np.load(output_dir / f".{spec.name}_prediction_sampled_values.rank{rank:03d}.npy")
        for rank in range(get_world_size())
    ]
    sampled_values = (
        np.concatenate(sampled_parts, axis=0)
        if sampled_parts
        else np.empty(0, dtype=np.float32)
    )
    sampled_path = output_dir / f"{spec.name}_prediction_sampled_values.npy"
    np.save(sampled_path, sampled_values)
    for rank in range(get_world_size()):
        temp_path = output_dir / f".{spec.name}_prediction_sampled_values.rank{rank:03d}.npy"
        if temp_path.exists():
            temp_path.unlink()

    summary = reduced_stats.summary(sampled_values=sampled_values)
    summary["sampled_values_path"] = str(sampled_path)
    summary["dataset_name"] = spec.name
    summary["dataset_length"] = len(dataset)
    summary["config_path"] = str(Path(spec.config_path).resolve())
    summary["checkpoint_path"] = str(Path(spec.checkpoint_path).resolve())
    summary["checkpoint_status"] = checkpoint_status
    summary["depth_scale"] = depth_scale
    return summary


def make_histogram_plot(
    summaries: Sequence[Dict[str, object]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(summaries), figsize=(8 * len(summaries), 5), dpi=160)
    if len(summaries) == 1:
        axes = [axes]

    for ax, summary in zip(axes, summaries):
        sampled_values = np.load(summary["sampled_values_path"]).astype(np.float32, copy=False)
        p995 = summary["percentiles_m"]["p995"]
        hist_upper = max(float(p995), 40.0)
        bins = np.linspace(0.0, hist_upper, 120)
        clipped = np.clip(sampled_values, 0.0, hist_upper)
        overflow_ratio = float(np.mean(sampled_values > hist_upper)) if sampled_values.size else 0.0

        ax.hist(clipped, bins=bins, density=True, color="#2d6cdf", alpha=0.85)
        ax.set_title(
            f"{summary['dataset_name']} prediction depth\n"
            f"sampled={summary['sampled_pixel_count']:,}, overflow>{hist_upper:.1f}m={overflow_ratio:.3%}"
        )
        ax.set_xlabel("Predicted depth (m)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_bucket_csv(summaries: Sequence[Dict[str, object]], output_path: Path) -> None:
    labels = [bucket["label"] for bucket in summaries[0]["bucket_counts"]]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", *labels])
        for summary in summaries:
            writer.writerow(
                [
                    summary["dataset_name"],
                    *[f"{bucket['ratio']:.8f}" for bucket in summary["bucket_counts"]],
                ]
            )


def write_stats_csv(summaries: Sequence[Dict[str, object]], output_path: Path) -> None:
    percentile_names = list(summaries[0]["percentiles_m"].keys())
    fieldnames = [
        "dataset",
        "total_pixels",
        "sampled_pixel_count",
        "sample_stride",
        "depth_scale",
        "min_m",
        "max_m",
        "mean_m",
        "std_m",
        *percentile_names,
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {
                "dataset": summary["dataset_name"],
                "total_pixels": summary["total_pixels"],
                "sampled_pixel_count": summary["sampled_pixel_count"],
                "sample_stride": summary["sample_stride"],
                "depth_scale": summary["depth_scale"],
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
    device = resolve_device(args.device)
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        specs = [
            DatasetSpec(
                name="kitti",
                config_path=args.kitti_config,
                checkpoint_path=args.kitti_checkpoint,
            ),
            DatasetSpec(
                name="vkitti",
                config_path=args.vkitti_config,
                checkpoint_path=args.vkitti_checkpoint,
            ),
        ]

        summaries = []
        for spec in specs:
            summary = collect_prediction_distribution(
                spec=spec,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sample_stride=args.sample_stride,
                max_samples=args.max_samples,
                output_dir=output_dir,
            )
            if summary is not None:
                summaries.append(summary)
            if get_world_size() > 1:
                dist.barrier()

        if not is_main():
            return

        histogram_path = output_dir / "depth_prediction_distribution_histograms.png"
        make_histogram_plot(summaries, histogram_path)

        stats_csv_path = output_dir / "depth_prediction_distribution_stats.csv"
        write_stats_csv(summaries, stats_csv_path)

        bucket_csv_path = output_dir / "depth_prediction_distribution_bucket_ratios.csv"
        write_bucket_csv(summaries, bucket_csv_path)

        summary_path = output_dir / "depth_prediction_distribution_summary.json"
        payload = {
            "device": str(device),
            "world_size": get_world_size(),
            "bucket_edges_m": list(DEFAULT_BUCKET_EDGES),
            "histogram_path": str(histogram_path),
            "stats_csv_path": str(stats_csv_path),
            "bucket_csv_path": str(bucket_csv_path),
            "datasets": {summary["dataset_name"]: summary for summary in summaries},
        }
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

        print(f"Saved summary to {summary_path}", flush=True)
        print(f"Saved histogram to {histogram_path}", flush=True)
        print(f"Saved stats csv to {stats_csv_path}", flush=True)
        print(f"Saved bucket csv to {bucket_csv_path}", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
