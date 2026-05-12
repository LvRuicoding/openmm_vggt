#!/usr/bin/env python3
"""Evaluate KITTI SemanticKITTI SSC occupancy models on the val split."""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS

import openmm_vggt  # noqa: F401
from openmm_vggt.utils.geometry import closed_form_inverse_se3


DEFAULT_CONFIG = REPO_ROOT / "configs" / "occupancy" / "kitti_semantic_occ_mix_window_attn_early_ft.py"
CLASS_NAMES = (
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def is_main() -> bool:
    return get_rank() == 0


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def align_occupancy_targets_to_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.shape[0] == target.shape[0]:
        return target, valid_mask
    if target.shape[0] <= 0 or logits.shape[0] % target.shape[0] != 0:
        raise ValueError(
            f"Occupancy batch mismatch: logits batch={logits.shape[0]}, target batch={target.shape[0]}"
        )
    repeat = logits.shape[0] // target.shape[0]
    return target.repeat_interleave(repeat, dim=0), valid_mask.repeat_interleave(repeat, dim=0)


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


def matches_prefix(name: str, prefixes: Tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def load_checkpoint(
    model: nn.Module,
    path: str,
    include_prefixes: Optional[Tuple[str, ...]] = None,
) -> str:
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:]: value for key, value in state_dict.items()}

    model_state = model.state_dict()
    model_keys = list(model_state.keys())
    if include_prefixes:
        model_keys = [key for key in model_keys if matches_prefix(key, include_prefixes)]
        if not model_keys:
            raise RuntimeError(f"No model keys matched include_prefixes={include_prefixes}")

    filtered = {}
    missing = []
    shape_err = []
    for key in model_keys:
        value = model_state[key]
        if key not in state_dict:
            missing.append(key)
        elif state_dict[key].shape != value.shape:
            shape_err.append((key, tuple(value.shape), tuple(state_dict[key].shape)))
        else:
            filtered[key] = state_dict[key]
    model.load_state_dict(filtered, strict=False)

    extras = sorted(set(state_dict) - set(model_state))
    parts = [f"loaded={len(filtered)}", f"unexpected={len(extras)}"]
    if missing:
        parts.append(f"missing={len(missing)}")
    if shape_err:
        parts.append(f"shape_mismatch={shape_err[:3]}")
    return " | ".join(parts)


def normalize_extrinsics_to_first_frame(extrinsics: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = extrinsics.shape[:2]
    ext_h = torch.zeros(batch_size, seq_len, 4, 4, dtype=extrinsics.dtype, device=extrinsics.device)
    ext_h[..., :3, :] = extrinsics
    ext_h[..., 3, 3] = 1.0
    first_inv = closed_form_inverse_se3(ext_h[:, 0])
    return torch.matmul(ext_h, first_inv.unsqueeze(1))[..., :3, :]


def collate_batch(batch):
    keys = [key for key in batch[0] if isinstance(batch[0][key], torch.Tensor)]
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


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


class SSCMetrics:
    def __init__(self, num_classes: int, ignore_index: int) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.total = 0.0
        self.correct = 0.0
        self.sc_intersection = 0.0
        self.sc_union = 0.0
        self.class_intersection = [0.0 for _ in range(num_classes)]
        self.class_union = [0.0 for _ in range(num_classes)]

    def update(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        valid = valid_mask.bool() & (target != self.ignore_index)
        if not valid.any():
            return

        pred_valid = pred[valid]
        target_valid = target[valid]
        self.total += float(target_valid.numel())
        self.correct += float((pred_valid == target_valid).sum().item())

        pred_occ = pred_valid != 0
        target_occ = target_valid != 0
        self.sc_intersection += float((pred_occ & target_occ).sum().item())
        self.sc_union += float((pred_occ | target_occ).sum().item())

        for class_idx in range(self.num_classes):
            pred_class = pred_valid == class_idx
            target_class = target_valid == class_idx
            self.class_intersection[class_idx] += float((pred_class & target_class).sum().item())
            self.class_union[class_idx] += float((pred_class | target_class).sum().item())

    def reduce(self, device: torch.device) -> "SSCMetrics":
        values = torch.tensor(
            [
                self.total,
                self.correct,
                self.sc_intersection,
                self.sc_union,
                *self.class_intersection,
                *self.class_union,
            ],
            dtype=torch.float64,
            device=device,
        )
        if is_dist():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)

        reduced = SSCMetrics(self.num_classes, self.ignore_index)
        reduced.total = float(values[0].item())
        reduced.correct = float(values[1].item())
        reduced.sc_intersection = float(values[2].item())
        reduced.sc_union = float(values[3].item())
        offset = 4
        reduced.class_intersection = [float(v.item()) for v in values[offset : offset + self.num_classes]]
        offset += self.num_classes
        reduced.class_union = [float(v.item()) for v in values[offset : offset + self.num_classes]]
        return reduced

    def result(self) -> Dict[str, float | list[float]]:
        if self.total <= 0:
            raise RuntimeError("No valid voxels accumulated.")
        class_iou = [
            self.class_intersection[idx] / self.class_union[idx]
            if self.class_union[idx] > 0
            else float("nan")
            for idx in range(self.num_classes)
        ]
        semantic_iou = [value for value in class_iou[1:] if not math.isnan(value)]
        return {
            "overall_acc": self.correct / self.total,
            "sc_iou": self.sc_intersection / max(self.sc_union, 1.0),
            "ssc_miou": sum(semantic_iou) / max(len(semantic_iou), 1),
            "empty_iou": class_iou[0],
            "class_iou": class_iou,
            "valid_voxels": self.total,
        }


def build_dense_voxel_subset(dataset) -> Subset:
    dense_root = getattr(dataset, "dense_voxel_root", None)
    if dense_root is None:
        raise ValueError("Evaluation requires official dense voxel GT; val_dataset.dense_voxel_root is not set.")
    dense_root = Path(dense_root)
    if not dense_root.is_dir():
        raise FileNotFoundError(f"Dense voxel root not found: {dense_root}")

    dense_indices = []
    skipped = 0
    for sample_idx, (record_idx, last_idx) in enumerate(dataset.samples):
        record = dataset.records[record_idx]
        frame_id = record.frame_ids[last_idx]
        label_path = dense_root / "sequences" / record.sequence_id / "voxels" / f"{frame_id}.label"
        if label_path.is_file():
            dense_indices.append(sample_idx)
        else:
            skipped += 1

    if not dense_indices:
        raise FileNotFoundError(
            "No official dense voxel labels were found for evaluation under "
            f"{dense_root}."
        )

    log(
        f"Dense GT subset: kept {len(dense_indices)} samples, skipped {skipped} samples without dense labels."
    )
    return Subset(dataset, dense_indices)


def build_model_others(batch: Dict[str, torch.Tensor], normalized_extrinsics: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "extrinsics": normalized_extrinsics,
        "intrinsics": batch["intrinsics"],
        "camera_to_world": batch["camera_to_world"],
        "lidar_to_world": batch["lidar_to_world"],
        "points": batch["points"],
        "point_mask": batch["point_mask"],
    }


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device, num_classes: int, ignore_index: int) -> SSCMetrics:
    local = SSCMetrics(num_classes=num_classes, ignore_index=ignore_index)
    bar = tqdm(data_loader, desc=f"Eval rank={get_rank()}", leave=False, disable=not is_main())

    with torch.inference_mode():
        for batch in bar:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            norm_ext = normalize_extrinsics_to_first_frame(batch["extrinsics"])
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(batch["images"], others=build_model_others(batch, norm_ext))
            if "occupancy_logits" not in preds:
                raise RuntimeError("Model did not return `occupancy_logits`.")
            occ_target, occ_valid_mask = align_occupancy_targets_to_logits(
                preds["occupancy_logits"],
                batch["occupancy_target"].long(),
                batch["occupancy_valid_mask"].bool(),
            )
            local.update(
                preds["occupancy_logits"].float(),
                occ_target,
                occ_valid_mask,
            )

    return local.reduce(device)


def format_report(config_path: str, ckpt_path: str, batch_size: int, num_workers: int, metrics: Dict[str, float | list[float]]) -> str:
    lines = [
        "SemanticKITTI SSC Metrics",
        f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"config: {config_path}",
        f"checkpoint: {ckpt_path}",
        f"split: val",
        f"batch_size: {batch_size}",
        f"num_workers: {num_workers}",
        "",
        f"SC IoU: {metrics['sc_iou']:.6f}",
        f"SSC mIoU: {metrics['ssc_miou']:.6f}",
        f"Overall Acc: {metrics['overall_acc']:.6f}",
        f"Empty IoU: {metrics['empty_iou']:.6f}",
        f"Valid Voxels: {metrics['valid_voxels']:.0f}",
        "",
        "Per-class IoU:",
    ]
    class_iou = metrics["class_iou"]
    assert isinstance(class_iou, list)
    for idx, value in enumerate(class_iou):
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
        if math.isnan(value):
            lines.append(f"{idx:02d} {name}: nan")
        else:
            lines.append(f"{idx:02d} {name}: {value:.6f}")
    return "\n".join(lines) + "\n"


def save_report(ckpt_path: str, report: str, out_file: Optional[str] = None) -> Optional[Path]:
    report_path = Path(out_file) if out_file else Path(ckpt_path).with_suffix(".txt")
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        log(f"WARNING: failed to save metrics to {report_path}: {exc}")
        if out_file:
            return None
        fallback_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "openmm_vggt_eval_metrics"
        fallback_path = fallback_dir / f"{Path(ckpt_path).stem}.txt"
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fallback_path.write_text(report, encoding="utf-8")
        except OSError as fallback_exc:
            log(f"WARNING: failed to save metrics fallback to {fallback_path}: {fallback_exc}")
            return None
        log(f"Saved metrics fallback to: {fallback_path}")
        return fallback_path
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KITTI SemanticKITTI SSC occupancy on val split.")
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG), help="mmengine config file.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth to evaluate.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap validation samples for quick checks.")
    parser.add_argument("--out-file", default=None, help="Optional path for the metrics .txt report.")
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Override image size (must be multiples of 14).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = setup_distributed()

    try:
        cfg = Config.fromfile(args.config)
        ckpt_path = str(args.checkpoint)
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if cfg.get("val_dataset", None) is None:
            raise ValueError("Config must define val_dataset for evaluation.")
        if cfg.val_dataset.get("dense_voxel_root", None) is None:
            raise ValueError("Config val_dataset must set dense_voxel_root for official SemanticKITTI SSC evaluation.")

        cfg.val_dataset.split = "val"
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

        model = MODELS.build(cfg.model)
        checkpoint_include_prefixes = tuple(cfg.get("checkpoint_include_prefixes", ()))
        msg = load_checkpoint(model, ckpt_path, include_prefixes=checkpoint_include_prefixes or None)
        model.to(device).eval()
        if is_dist():
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        dataset = DATASETS.build(cfg.val_dataset)
        dataset = build_dense_voxel_subset(dataset)
        sampler = DistributedEvalSampler(dataset) if is_dist() else None
        data_loader = DataLoader(
            dataset,
            batch_size=int(cfg.val_dataloader.get("batch_size", args.batch_size)),
            shuffle=False,
            sampler=sampler,
            num_workers=int(cfg.val_dataloader.get("num_workers", args.num_workers)),
            pin_memory=bool(cfg.val_dataloader.get("pin_memory", True)),
            collate_fn=collate_batch,
            drop_last=False,
        )

        log("=" * 70)
        log("SemanticKITTI SSC val evaluation")
        log(f"config: {args.config}")
        log(f"checkpoint: {ckpt_path}")
        log(f"checkpoint load: {msg}")
        log(f"samples: {len(dataset)}")
        log("=" * 70)

        metrics = evaluate(
            model,
            data_loader,
            device,
            num_classes=int(cfg.get("occupancy_num_classes", 20)),
            ignore_index=int(cfg.get("occupancy_ignore_index", 255)),
        ).result()

        if is_main():
            report = format_report(
                config_path=args.config,
                ckpt_path=ckpt_path,
                batch_size=int(cfg.val_dataloader.get("batch_size", args.batch_size)),
                num_workers=int(cfg.val_dataloader.get("num_workers", args.num_workers)),
                metrics=metrics,
            )
            print("\n" + report, flush=True)
            report_path = save_report(ckpt_path, report, out_file=args.out_file)
            if report_path is not None:
                print(f"Saved metrics to: {report_path}", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
