#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS

import openmm_vggt  # noqa: F401
from openmm_vggt.utils.geometry import closed_form_inverse_se3


def is_dist_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_dist_enabled() or dist.get_rank() == 0


def log(msg: str) -> None:
    if is_main_process():
        print(msg, flush=True)


def setup_distributed(args: argparse.Namespace) -> argparse.Namespace:
    args.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    args.rank = int(os.environ.get("RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not args.distributed:
        return args
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()
    return args


def cleanup_distributed() -> None:
    if is_dist_enabled():
        dist.barrier()
        dist.destroy_process_group()


def reduce_scalar_dict(metrics_sum: Dict[str, float], count: int, device: torch.device) -> Dict[str, float]:
    if count == 0:
        return {}
    keys = sorted(metrics_sum.keys())
    values = torch.tensor([metrics_sum[k] for k in keys] + [float(count)], device=device, dtype=torch.float64)
    if is_dist_enabled():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total = max(int(values[-1].item()), 1)
    return {k: float(values[i].item() / total) for i, k in enumerate(keys)}


def reduce_sum_dict(metrics_sum: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not metrics_sum:
        return {}
    keys = sorted(metrics_sum.keys())
    values = torch.tensor([metrics_sum[k] for k in keys], device=device, dtype=torch.float64)
    if is_dist_enabled():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {k: float(values[i].item()) for i, k in enumerate(keys)}


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_extrinsics_to_first_frame(extrinsics: torch.Tensor) -> torch.Tensor:
    bsz, seq, _, _ = extrinsics.shape
    extrinsics_h = torch.zeros((bsz, seq, 4, 4), dtype=extrinsics.dtype, device=extrinsics.device)
    extrinsics_h[..., :3, :] = extrinsics
    extrinsics_h[..., 3, 3] = 1.0
    first_inv = closed_form_inverse_se3(extrinsics_h[:, 0])
    normalized = torch.matmul(extrinsics_h, first_inv.unsqueeze(1))
    return normalized[..., :3, :]


def collate_batch(batch):
    keys = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
    return {k: torch.stack([item[k] for item in batch], dim=0) for k in keys}


def build_model_others(batch: Dict[str, torch.Tensor], normalized_extrinsics: torch.Tensor) -> Dict[str, torch.Tensor]:
    others = {
        "extrinsics": normalized_extrinsics,
        "intrinsics": batch["intrinsics"],
    }
    for key in ("camera_to_world", "lidar_to_world", "points", "point_mask"):
        if key in batch:
            others[key] = batch[key]
    return others


def matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def resolve_lr_multiplier(name: str, lr_multipliers: Dict[str, float]) -> Tuple[float, str]:
    best_prefix = ""
    best_multiplier = 1.0
    for prefix, multiplier in lr_multipliers.items():
        if matches_prefix(name, (prefix,)) and len(prefix) > len(best_prefix):
            best_prefix = prefix
            best_multiplier = float(multiplier)
    return best_multiplier, best_prefix or "default"


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    opt_cfg = cfg.optimizer
    if opt_cfg.type != "AdamW":
        raise ValueError(f"Unsupported optimizer type: {opt_cfg.type}")

    lr_multipliers = dict(cfg.get("lr_multipliers", {}))
    grouped_params: Dict[Tuple[str, float], list[nn.Parameter]] = defaultdict(list)
    group_counts: Dict[Tuple[str, float], int] = defaultdict(int)

    for name, param in model.named_parameters():
        multiplier, label = resolve_lr_multiplier(name, lr_multipliers)
        group_key = (label, multiplier)
        grouped_params[group_key].append(param)
        group_counts[group_key] += param.numel()

    param_groups = []
    for (label, multiplier), params in sorted(grouped_params.items()):
        param_groups.append(
            {
                "params": params,
                "lr": opt_cfg.lr * multiplier,
                "weight_decay": opt_cfg.weight_decay,
                "group_name": label,
                "lr_multiplier": multiplier,
            }
        )

    optimizer = torch.optim.AdamW(param_groups, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
    log(
        "Optimizer groups: "
        + " | ".join(
            f"{group['group_name']} lr={group['lr']:.2e} params={group_counts[(group['group_name'], group['lr_multiplier'])]}"
            for group in param_groups
        )
    )
    return optimizer


def load_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    include_prefixes: Tuple[str, ...] | None = None,
) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:]: value for key, value in state_dict.items()}

    model_state = model.state_dict()
    model_keys = list(model_state.keys())
    if include_prefixes:
        model_keys = [key for key in model_keys if matches_prefix(key, include_prefixes)]
        if not model_keys:
            raise RuntimeError(f"No model keys matched include_prefixes={include_prefixes}")

    missing = [k for k in model_keys if k not in state_dict]
    if missing:
        print(f"[load_checkpoint] WARNING: {len(missing)} keys not in checkpoint (will use random init): {missing[:20]}")

    mismatched = []
    filtered = {}
    for key in model_keys:
        if key in missing:
            continue
        if state_dict[key].shape != model_state[key].shape:
            mismatched.append((key, tuple(model_state[key].shape), tuple(state_dict[key].shape)))
        else:
            filtered[key] = state_dict[key]

    model.load_state_dict(filtered, strict=False)
    extras = sorted(set(state_dict) - set(model_state))

    loaded_prefix_counts = defaultdict(int)
    for key in filtered:
        loaded_prefix_counts[key.split(".", 1)[0]] += 1

    message_parts = [f"loaded {len(filtered)} tensors", f"ignored {len(extras)} unexpected tensors"]
    if mismatched:
        message_parts.append(f"shape mismatches: {mismatched[:5]}")
    if loaded_prefix_counts:
        message_parts.append(
            "prefixes[" + ", ".join(f"{prefix}={count}" for prefix, count in sorted(loaded_prefix_counts.items())) + "]"
        )
    return " | ".join(message_parts)


def set_trainable_state(
    model: nn.Module,
    epoch: int,
    freeze_prefixes: Tuple[str, ...],
    freeze_for_epochs: int,
) -> str:
    freeze_active = freeze_for_epochs > 0 and epoch <= freeze_for_epochs and bool(freeze_prefixes)
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        bare_name = name[len("module."):] if name.startswith("module.") else name
        should_freeze = freeze_active and matches_prefix(bare_name, freeze_prefixes)
        param.requires_grad = not should_freeze
        if param.requires_grad:
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()
    stage = "warmup" if freeze_active else "full_finetune"
    return f"{stage} trainable={trainable_count} frozen={frozen_count}"


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, step, cfg, scaler=None) -> None:
    if not is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "step": step,
            "config": cfg.filename,
        },
        path,
    )


def sync_device_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def append_loss_record(loss_file, step: int, loss: float, **extra_fields: float) -> None:
    if loss_file is None:
        return
    payload = {"step": step, "loss": loss}
    payload.update(extra_fields)
    loss_file.write(json.dumps(payload) + "\n")
    loss_file.flush()


def canonicalize_depth_tensor(depth: torch.Tensor, name: str) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    elif depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 4:
        raise ValueError(f"{name} must have shape [B, S, H, W] or singleton-channel equivalent, got {tuple(depth.shape)}")
    return depth


def compute_balanced_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    n_time_steps: int,
    cam_num: int,
    depth_loss_type: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    batch_size, sequence_len, height, width = pred_depth.shape
    expected_sequence_len = n_time_steps * cam_num
    if sequence_len != expected_sequence_len:
        raise ValueError(
            f"Depth sequence length {sequence_len} does not match n_time_steps({n_time_steps}) * cam_num({cam_num}) = {expected_sequence_len}"
        )
    valid_mask = valid_mask.reshape(batch_size, n_time_steps, cam_num, height, width)
    valid_float = valid_mask.to(dtype=pred_depth.dtype)
    pred_depth = pred_depth.reshape(batch_size, n_time_steps, cam_num, height, width)
    gt_depth = gt_depth.reshape(batch_size, n_time_steps, cam_num, height, width)

    if depth_loss_type == "l1":
        pixel_loss = torch.abs(pred_depth - gt_depth)
    elif depth_loss_type == "log1p_l1":
        safe_pred = torch.where(valid_mask, pred_depth, torch.zeros_like(pred_depth))
        safe_gt = torch.where(valid_mask, gt_depth, torch.zeros_like(gt_depth))
        pixel_loss = torch.abs(torch.log1p(safe_pred) - torch.log1p(safe_gt))
    else:
        raise ValueError(f"Unsupported depth_loss_type: {depth_loss_type}")

    pixel_count = valid_float.sum(dim=(-1, -2))
    camera_valid = pixel_count > 0
    camera_loss = (pixel_loss * valid_float).sum(dim=(-1, -2)) / pixel_count.clamp_min(1.0)
    camera_weight = camera_valid.to(dtype=pixel_loss.dtype)
    valid_camera_count = camera_weight.sum(dim=2)
    frame_valid = valid_camera_count > 0
    frame_loss = (camera_loss * camera_weight).sum(dim=2) / valid_camera_count.clamp_min(1.0)
    frame_weight = frame_valid.to(dtype=pixel_loss.dtype)
    valid_frame_count = frame_weight.sum(dim=1)
    sample_valid = valid_frame_count > 0
    sample_loss = (frame_loss * frame_weight).sum(dim=1) / valid_frame_count.clamp_min(1.0)
    sample_weight = sample_valid.to(dtype=pixel_loss.dtype)
    valid_sample_count = sample_weight.sum()
    depth_loss = (sample_loss * sample_weight).sum() / valid_sample_count.clamp_min(1.0)

    return depth_loss, {
        "depth_valid_pixels": valid_mask.sum(),
        "depth_valid_cameras": camera_valid.sum(),
        "depth_valid_frames": frame_valid.sum(),
        "depth_valid_samples": sample_valid.sum(),
    }


def compute_losses(
    predictions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    depth_weight: float,
    occupancy_weight: float,
    n_time_steps: int,
    cam_num: int,
    depth_pred_scale: float,
    depth_loss_type: str,
    occupancy_ignore_index: int,
    occupancy_class_weights: torch.Tensor | None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = torch.zeros((), device=batch["images"].device, dtype=torch.float32)
    metrics: Dict[str, float] = {"loss": 0.0}

    if "depth" in predictions and "depths" in batch and depth_weight > 0:
        with torch.amp.autocast(device_type=batch["images"].device.type, enabled=False):
            pred_depth = canonicalize_depth_tensor(predictions["depth"], "predictions['depth']")
            gt_depth = canonicalize_depth_tensor(batch["depths"].float(), "batch['depths']")
            pred_depth = pred_depth * depth_pred_scale
            valid_mask = (gt_depth > 0.0) & (gt_depth < 655.0)
            depth_loss, depth_stats = compute_balanced_depth_loss(
                pred_depth,
                gt_depth,
                valid_mask,
                n_time_steps=n_time_steps,
                cam_num=cam_num,
                depth_loss_type=depth_loss_type,
            )
        total = total + depth_weight * depth_loss
        metrics.update(
            {
                "loss": float(total.detach().cpu()),
                "depth_loss": float(depth_loss.detach().cpu()),
                "depth_valid_pixels": float(depth_stats["depth_valid_pixels"].detach().cpu()),
            }
        )

    if "occupancy_logits" in predictions and "occupancy_target" in batch and occupancy_weight > 0:
        occ_logits = predictions["occupancy_logits"].float()
        occ_target = batch["occupancy_target"].long().masked_fill(~batch["occupancy_valid_mask"].bool(), occupancy_ignore_index)
        occ_loss = F.cross_entropy(
            occ_logits,
            occ_target,
            ignore_index=occupancy_ignore_index,
            weight=occupancy_class_weights,
        )
        total = total + occupancy_weight * occ_loss
        occ_pred = occ_logits.argmax(dim=1)
        valid = occ_target != occupancy_ignore_index
        pred_occ = occ_pred[valid] != 0
        target_occ = occ_target[valid] != 0
        occ_inter = (pred_occ & target_occ).sum().item()
        occ_union = (pred_occ | target_occ).sum().item()
        metrics.update(
            {
                "loss": float(total.detach().cpu()),
                "occupancy_loss": float(occ_loss.detach().cpu()),
                "occupancy_iou": float(occ_inter / max(occ_union, 1)),
                "occupancy_valid_voxels": float(valid.sum().item()),
            }
        )

    return total, metrics


def accumulate_occupancy_stats(
    stats: Dict[str, float],
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> None:
    pred = logits.argmax(dim=1)
    valid = valid_mask.bool() & (target != ignore_index)
    if not valid.any():
        return
    pred_valid = pred[valid]
    target_valid = target[valid]
    pred_occ = pred_valid != 0
    target_occ = target_valid != 0
    stats["occ_intersection"] = stats.get("occ_intersection", 0.0) + float((pred_occ & target_occ).sum().item())
    stats["occ_union"] = stats.get("occ_union", 0.0) + float((pred_occ | target_occ).sum().item())
    for class_idx in range(num_classes):
        pred_class = pred_valid == class_idx
        target_class = target_valid == class_idx
        stats[f"class_{class_idx}_intersection"] = stats.get(f"class_{class_idx}_intersection", 0.0) + float((pred_class & target_class).sum().item())
        stats[f"class_{class_idx}_union"] = stats.get(f"class_{class_idx}_union", 0.0) + float((pred_class | target_class).sum().item())


def summarize_occupancy_stats(stats: Dict[str, float], num_classes: int) -> Dict[str, float]:
    if not stats:
        return {}
    occ_iou = stats.get("occ_intersection", 0.0) / max(stats.get("occ_union", 0.0), 1.0)
    semantic_ious = []
    for class_idx in range(1, num_classes):
        union = stats.get(f"class_{class_idx}_union", 0.0)
        if union > 0:
            semantic_ious.append(stats.get(f"class_{class_idx}_intersection", 0.0) / union)
    return {
        "occ_iou": float(occ_iou),
        "ssc_miou": float(sum(semantic_ious) / max(len(semantic_ious), 1)),
    }


def train(args: argparse.Namespace, cfg: Config) -> None:
    args = setup_distributed(args)
    set_seed(cfg.seed + args.rank)
    device = torch.device("cuda", args.local_rank if args.distributed else 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_ds = DATASETS.build(cfg.train_dataset)
    use_val = cfg.get("val_dataset", None) is not None and not args.no_eval
    val_ds = DATASETS.build(cfg.val_dataset) if use_val else None

    train_sampler = DistributedSampler(train_ds, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if args.distributed and val_ds is not None else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_dataloader.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.train_dataloader.num_workers,
        pin_memory=cfg.train_dataloader.get("pin_memory", True),
        collate_fn=collate_batch,
        drop_last=False,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.val_dataloader.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=cfg.val_dataloader.num_workers,
            pin_memory=cfg.val_dataloader.get("pin_memory", True),
            collate_fn=collate_batch,
            drop_last=False,
        )

    model = MODELS.build(cfg.model)
    checkpoint_include_prefixes = tuple(cfg.get("checkpoint_include_prefixes", ()))
    if cfg.checkpoint:
        log(f"Loaded checkpoint: {load_model_from_checkpoint(model, cfg.checkpoint, checkpoint_include_prefixes or None)}")
    model.to(device)

    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=cfg.scheduler.eta_min,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tb")) if is_main_process() else None
    loss_file = (output_dir / "loss_steps.jsonl").open("w", encoding="utf-8") if is_main_process() else None

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("amp", True) and device.type == "cuda")
    best_metric = -float("inf")
    global_step = 0
    freeze_prefixes = tuple(cfg.get("freeze_modules", ()))
    freeze_for_epochs = int(cfg.get("freeze_modules_for_epochs", 0))
    n_time_steps = int(cfg.get("n_time_steps", cfg.train_dataset.get("n_time_steps", 1)))
    cam_num = int(cfg.model.get("cam_num", 2))
    depth_weight = float(cfg.get("depth_weight", 0.0))
    occupancy_weight = float(cfg.get("occupancy_weight", 1.0))
    occupancy_ignore_index = int(cfg.get("occupancy_ignore_index", 255))
    occupancy_num_classes = int(cfg.get("occupancy_num_classes", 20))
    occupancy_class_weights = cfg.get("occupancy_class_weights", None)
    if occupancy_class_weights is not None:
        occupancy_class_weights = torch.tensor(occupancy_class_weights, dtype=torch.float32, device=device)

    for epoch in range(1, cfg.epochs + 1):
        stage_msg = set_trainable_state(model, epoch, freeze_prefixes, freeze_for_epochs)
        log(f"Epoch {epoch}: {stage_msg}")
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running: Dict[str, float] = {}
        num_batches = 0
        train_bar = tqdm(train_loader, desc=f"Train E{epoch}", leave=False, disable=not is_main_process())
        iter_end_time = time.perf_counter()
        for batch in train_bar:
            data_time = time.perf_counter() - iter_end_time
            sync_device_for_timing(device)
            h2d_start = time.perf_counter()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            sync_device_for_timing(device)
            h2d_time = time.perf_counter() - h2d_start

            normalized_extrinsics = normalize_extrinsics_to_first_frame(batch["extrinsics"])
            optimizer.zero_grad(set_to_none=True)

            sync_device_for_timing(device)
            forward_start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=cfg.get("amp", True) and device.type == "cuda"):
                predictions = model(batch["images"], others=build_model_others(batch, normalized_extrinsics))
            sync_device_for_timing(device)
            forward_time = time.perf_counter() - forward_start

            loss, metrics = compute_losses(
                predictions,
                batch,
                depth_weight=depth_weight,
                occupancy_weight=occupancy_weight,
                n_time_steps=n_time_steps,
                cam_num=cam_num,
                depth_pred_scale=cfg.get("depth_pred_scale", 1.0),
                depth_loss_type=cfg.get("depth_loss_type", "l1"),
                occupancy_ignore_index=occupancy_ignore_index,
                occupancy_class_weights=occupancy_class_weights,
            )

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                active_params = [p for p in model.parameters() if p.requires_grad]
                if active_params:
                    torch.nn.utils.clip_grad_norm_(active_params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value
            global_step += 1
            append_loss_record(loss_file, global_step, metrics["loss"], data_time=data_time, h2d_time=h2d_time, forward_time=forward_time)
            num_batches += 1
            iter_end_time = time.perf_counter()
            train_bar.set_postfix(loss=f"{metrics['loss']:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            if writer is not None and global_step % cfg.get("log_interval", 10) == 0:
                writer.add_scalar("train/loss_step", metrics["loss"], global_step)
                if "occupancy_loss" in metrics:
                    writer.add_scalar("train/occupancy_loss_step", metrics["occupancy_loss"], global_step)
                if "occupancy_iou" in metrics:
                    writer.add_scalar("train/occupancy_iou_step", metrics["occupancy_iou"], global_step)

        scheduler.step()
        train_avg = reduce_scalar_dict(running, num_batches, device)

        val_metrics: Dict[str, float] = {}
        occ_val_stats: Dict[str, float] = {}
        if val_loader is not None:
            model.eval()
            val_sum: Dict[str, float] = {}
            val_count = 0
            val_bar = tqdm(val_loader, desc=f"Val   E{epoch}", leave=False, disable=not is_main_process())
            with torch.no_grad():
                for batch in val_bar:
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    normalized_extrinsics = normalize_extrinsics_to_first_frame(batch["extrinsics"])
                    predictions = model(batch["images"], others=build_model_others(batch, normalized_extrinsics))
                    _, metrics = compute_losses(
                        predictions,
                        batch,
                        depth_weight=depth_weight,
                        occupancy_weight=occupancy_weight,
                        n_time_steps=n_time_steps,
                        cam_num=cam_num,
                        depth_pred_scale=cfg.get("depth_pred_scale", 1.0),
                        depth_loss_type=cfg.get("depth_loss_type", "l1"),
                        occupancy_ignore_index=occupancy_ignore_index,
                        occupancy_class_weights=occupancy_class_weights,
                    )
                    for key, value in metrics.items():
                        val_sum[key] = val_sum.get(key, 0.0) + value
                    if "occupancy_logits" in predictions:
                        occ_target = batch["occupancy_target"].long().masked_fill(~batch["occupancy_valid_mask"].bool(), occupancy_ignore_index)
                        accumulate_occupancy_stats(
                            occ_val_stats,
                            predictions["occupancy_logits"],
                            occ_target,
                            batch["occupancy_valid_mask"],
                            occupancy_num_classes,
                            occupancy_ignore_index,
                        )
                    val_count += 1
            val_metrics = reduce_scalar_dict(val_sum, val_count, device)
            occ_val_stats = reduce_sum_dict(occ_val_stats, device)
            val_metrics.update(summarize_occupancy_stats(occ_val_stats, occupancy_num_classes))

        log_line = f"Epoch {epoch}/{cfg.epochs} | train " + " ".join(f"{k}={v:.4f}" for k, v in train_avg.items())
        if val_metrics:
            log_line += " | val " + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        log(log_line)

        if writer is not None:
            for key, value in train_avg.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        save_checkpoint(output_dir / f"epoch_{epoch:03d}.pth", model, optimizer, scheduler, epoch, global_step, cfg, scaler)
        save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler, epoch, global_step, cfg, scaler)

        current_metric = val_metrics.get("ssc_miou", val_metrics.get("occupancy_iou", -float("inf")))
        if current_metric > best_metric:
            best_metric = current_metric
            save_checkpoint(output_dir / "best.pth", model, optimizer, scheduler, epoch, global_step, cfg, scaler)

    if writer is not None:
        writer.close()
    if loss_file is not None:
        loss_file.close()
    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune VGGT on KITTI semantic occupancy")
    parser.add_argument("config", help="Path to mmengine Python config file")
    parser.add_argument("--checkpoint", default=None, help="Override config checkpoint path")
    parser.add_argument("--output-dir", default=None, help="Override config output_dir")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-eval", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train_dataloader.batch_size = args.batch_size
        if cfg.get("val_dataloader", None) is not None:
            cfg.val_dataloader.batch_size = args.batch_size
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
    train(args, cfg)
