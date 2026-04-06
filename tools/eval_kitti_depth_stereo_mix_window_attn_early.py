#!/usr/bin/env python3
"""Evaluate mix_decoder_global_window_attn_early on KITTI depth completion val split.

This mirrors tools/eval_kitti_depth_stereo_mix.py but defaults to the
window-attention early-fusion config/checkpoint.

Usage:
  python tools/eval_kitti_depth_stereo_mix_window_attn_early.py \
      configs/kitti_depth_stereo_mix_window_attn_early_ft.py \
      --checkpoint trainoutput/kitti_depth_stereo_mix_window_attn_early_ft/epoch_002.pth
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS

import openmm_vggt  # noqa: F401
from openmm_vggt.utils.geometry import closed_form_inverse_se3


DEFAULT_EVAL_CHECKPOINT_NAME = "epoch_002.pth"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "kitti_depth_stereo_mix_window_attn_early_ft.py"


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


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


class DepthMetrics:
    def __init__(self) -> None:
        self.n = 0.0
        self.sq_mm = 0.0
        self.abs_mm = 0.0
        self.sq_ikm = 0.0
        self.abs_ikm = 0.0
        self.abs_rel = 0.0
        self.sq_rel = 0.0
        self.silog_sq = 0.0
        self.silog_lin = 0.0
        self.d1 = 0.0
        self.d2 = 0.0
        self.d3 = 0.0

    def update(self, pred_m: torch.Tensor, gt_m: torch.Tensor) -> None:
        mask = gt_m > 0.0
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
        log_diff = torch.log(pred) - torch.log(gt)
        self.silog_sq += float((log_diff ** 2).sum())
        self.silog_lin += float(log_diff.sum())
        ratio = torch.max(pred / gt, gt / pred)
        self.d1 += float((ratio < 1.25).sum())
        self.d2 += float((ratio < 1.25 ** 2).sum())
        self.d3 += float((ratio < 1.25 ** 3).sum())
        self.n += n

    def _tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [
                self.n,
                self.sq_mm,
                self.abs_mm,
                self.sq_ikm,
                self.abs_ikm,
                self.abs_rel,
                self.sq_rel,
                self.silog_sq,
                self.silog_lin,
                self.d1,
                self.d2,
                self.d3,
            ],
            dtype=torch.float64,
            device=device,
        )

    def reduce(self, device: torch.device) -> "DepthMetrics":
        tensor = self._tensor(device)
        if is_dist():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced = DepthMetrics()
        (
            reduced.n,
            reduced.sq_mm,
            reduced.abs_mm,
            reduced.sq_ikm,
            reduced.abs_ikm,
            reduced.abs_rel,
            reduced.sq_rel,
            reduced.silog_sq,
            reduced.silog_lin,
            reduced.d1,
            reduced.d2,
            reduced.d3,
        ) = [float(x) for x in tensor]
        return reduced

    def result(self) -> Dict[str, float]:
        if self.n <= 0:
            raise RuntimeError("No valid pixels accumulated.")
        silog = math.sqrt(max(self.silog_sq / self.n - (self.silog_lin / self.n) ** 2, 0.0)) * 100.0
        return {
            "iRMSE": math.sqrt(self.sq_ikm / self.n),
            "iMAE": self.abs_ikm / self.n,
            "RMSE": math.sqrt(self.sq_mm / self.n),
            "MAE": self.abs_mm / self.n,
            "Abs Rel": self.abs_rel / self.n,
            "Sq Rel": self.sq_rel / self.n,
            "SILog": silog,
            "d1": self.d1 / self.n,
            "d2": self.d2 / self.n,
            "d3": self.d3 / self.n,
            "n_pixels": self.n,
        }


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


def duplicate_singleton_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if batch["images"].shape[0] != 1:
        return batch
    return {key: torch.cat([value, value], dim=0) for key, value in batch.items()}


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


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    depth_scale: float = 1.0,
) -> DepthMetrics:
    local = DepthMetrics()
    bar = tqdm(
        data_loader,
        desc=f"Eval rank={get_rank()}",
        leave=False,
        disable=not is_main(),
    )

    with torch.inference_mode():
        for batch in bar:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            metric_batch = batch
            model_batch = duplicate_singleton_batch(batch)

            imgs = model_batch["images"]
            deps = metric_batch["depths"]
            norm_ext = normalize_extrinsics_to_first_frame(model_batch["extrinsics"])

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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

            seq_len = imgs.shape[1]
            if seq_len < 2:
                raise RuntimeError(f"Expected stereo sequence, got seq_len={seq_len}")
            last_pair_start = seq_len - 2

            pred_depth = preds["depth"][: deps.shape[0]].squeeze(-1)
            pred_c = pred_depth[:, last_pair_start:] * depth_scale
            gt_c = deps[:, last_pair_start:]
            local.update(pred_c.float(), gt_c.float())

    return local.reduce(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate mix_decoder_global_window_attn_early on KITTI depth completion val split."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG_PATH),
        help="mmengine config file.",
    )
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
        help="Multiply model depth output by this factor before computing metrics. "
        "If not set, reads depth_pred_scale from config.",
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

        cfg.val_dataset.return_lidar = True
        image_size = tuple(cfg.val_dataset.image_size)
        assert image_size[0] % 14 == 0 and image_size[1] % 14 == 0, (
            f"image_size {image_size} must be multiples of 14"
        )

        depth_scale = args.depth_scale if args.depth_scale is not None else float(cfg.get("depth_pred_scale", 1.0))

        model = MODELS.build(cfg.model)
        checkpoint_include_prefixes = tuple(cfg.get("checkpoint_include_prefixes", ()))
        msg = load_checkpoint(
            model,
            ckpt_path,
            include_prefixes=checkpoint_include_prefixes or None,
        )
        log(f"Loaded checkpoint: {msg}")
        model.eval().to(device)
        if is_dist():
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        dataset = DATASETS.build(cfg.val_dataset)
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

        metrics = evaluate(model, data_loader, device, depth_scale=depth_scale).result()
        if is_main():
            print("\nKITTI Depth Metrics", flush=True)
            for key in ("iRMSE", "iMAE", "RMSE", "MAE", "Abs Rel", "Sq Rel", "SILog", "d1", "d2", "d3"):
                print(f"{key:>8s}: {metrics[key]:.6f}", flush=True)
            print(f"{'n_pixels':>8s}: {metrics['n_pixels']:.0f}", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
