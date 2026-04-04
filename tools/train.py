#!/usr/bin/env python3
"""Fine-tune VGGT_decoder_global on vKITTI depth using mmengine config."""
from __future__ import annotations

import argparse
import os
import random
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
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config
from mmengine.registry import MODELS, DATASETS

import openmm_vggt  # noqa: F401 – triggers all register_module() decorators
from openmm_vggt.utils.geometry import closed_form_inverse_se3
from openmm_vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri


# ---------------------------------------------------------------------------
# distributed helpers
# ---------------------------------------------------------------------------

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
    values = torch.tensor(
        [metrics_sum[k] for k in keys] + [float(count)],
        device=device, dtype=torch.float64,
    )
    if is_dist_enabled():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total = max(int(values[-1].item()), 1)
    return {k: float(values[i].item() / total) for i, k in enumerate(keys)}


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------------------------------
# training utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_extrinsics_to_first_frame(extrinsics: torch.Tensor) -> torch.Tensor:
    B, S, _, _ = extrinsics.shape
    extrinsics_h = torch.zeros((B, S, 4, 4), dtype=extrinsics.dtype, device=extrinsics.device)
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
    for key in ("camera_to_world", "points", "point_mask"):
        if key in batch:
            others[key] = batch[key]
    return others


def matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def compute_losses(
    predictions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    normalized_extrinsics: torch.Tensor,
    depth_weight: float,
    camera_weight: float,
    pose_translation_weight: float,
    pose_rotation_weight: float,
    pose_fov_weight: float,
    depth_pred_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_depth = predictions["depth"].squeeze(-1) * depth_pred_scale
    gt_depth = batch["depths"]
    valid_mask = (gt_depth > 0.0) & (gt_depth < 655.0)
    depth_loss = F.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask]) if valid_mask.any() else pred_depth.new_zeros(())
    total = depth_weight * depth_loss
    metrics = {
        "loss": float(total.detach().cpu()),
        "depth_loss": float(depth_loss.detach().cpu()),
    }

    if "seq_enc_list" in predictions and "intrinsics" in batch and camera_weight > 0:
        image_hw = batch["images"].shape[-2:]
        gt_pose_enc = extri_intri_to_pose_encoding(
            normalized_extrinsics,
            batch["intrinsics"],
            image_size_hw=image_hw,
        )
        pred_pose_enc = predictions["seq_enc_list"][-1]
        pred_extrinsics, _ = pose_encoding_to_extri_intri(pred_pose_enc, image_size_hw=image_hw)

        pose_translation_loss = F.l1_loss(pred_extrinsics[..., :3, 3], normalized_extrinsics[..., :3, 3])
        pose_rotation_loss = F.l1_loss(pred_extrinsics[..., :3, :3], normalized_extrinsics[..., :3, :3])
        pose_fov_loss = F.l1_loss(pred_pose_enc[..., 7:], gt_pose_enc[..., 7:])

        camera_loss = (
            pose_translation_weight * pose_translation_loss
            + pose_rotation_weight * pose_rotation_loss
            + pose_fov_weight * pose_fov_loss
        )
        total = total + camera_weight * camera_loss
        metrics.update(
            {
                "loss": float(total.detach().cpu()),
                "camera_loss": float(camera_loss.detach().cpu()),
                "pose_translation_loss": float(pose_translation_loss.detach().cpu()),
                "pose_rotation_loss": float(pose_rotation_loss.detach().cpu()),
                "pose_fov_loss": float(pose_fov_loss.detach().cpu()),
            }
        )
    return total, metrics


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
        tensor = model_state[key]
        ckpt_tensor = state_dict[key]
        if ckpt_tensor.shape != tensor.shape:
            mismatched.append((key, tuple(tensor.shape), tuple(ckpt_tensor.shape)))
        else:
            filtered[key] = ckpt_tensor

    model.load_state_dict(filtered, strict=False)
    extras = sorted(set(state_dict) - set(model_state))

    loaded_prefix_counts = defaultdict(int)
    for key in filtered:
        prefix = key.split(".", 1)[0]
        loaded_prefix_counts[prefix] += 1

    message_parts = [
        f"loaded {len(filtered)} tensors",
        f"ignored {len(extras)} unexpected tensors",
    ]
    if mismatched:
        message_parts.append(f"shape mismatches: {mismatched[:5]}")
    if loaded_prefix_counts:
        prefix_msg = ", ".join(f"{prefix}={count}" for prefix, count in sorted(loaded_prefix_counts.items()))
        message_parts.append(f"prefixes[{prefix_msg}]")
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
        # DDP wraps the model, adding a "module." prefix to all parameter names.
        # Strip it so that freeze_prefixes like "aggregator" still match.
        bare_name = name[len("module."):] if name.startswith("module.") else name
        should_freeze = freeze_active and matches_prefix(bare_name, freeze_prefixes)
        param.requires_grad = not should_freeze
        if param.requires_grad:
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    stage = "depth_warmup" if freeze_active else "full_finetune"
    return f"{stage} trainable={trainable_count} frozen={frozen_count}"


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

    group_messages = [
        f"{group['group_name']} lr={group['lr']:.2e} params={group_counts[(group['group_name'], group['lr_multiplier'])]}"
        for group in param_groups
    ]
    log("Optimizer groups: " + " | ".join(group_messages))
    return optimizer


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, step, cfg, scaler=None) -> None:
    if not is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "config": cfg.filename,
    }, path)


# ---------------------------------------------------------------------------
# main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, cfg: Config) -> None:
    args = setup_distributed(args)
    set_seed(cfg.seed + args.rank)

    device = torch.device("cuda", args.local_rank if args.distributed else 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Build datasets via mmengine registry
    train_ds = DATASETS.build(cfg.train_dataset)
    use_val = cfg.get("val_dataset", None) is not None and not args.no_eval
    val_ds = DATASETS.build(cfg.val_dataset) if use_val else None

    train_sampler = DistributedSampler(train_ds, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if args.distributed and val_ds is not None else None

    dl_cfg = cfg.train_dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=dl_cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=dl_cfg.num_workers,
        pin_memory=dl_cfg.get("pin_memory", True),
        collate_fn=collate_batch,
        drop_last=False,
    )
    val_loader = None
    if val_ds is not None:
        vdl_cfg = cfg.val_dataloader
        val_loader = DataLoader(
            val_ds,
            batch_size=vdl_cfg.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=vdl_cfg.num_workers,
            pin_memory=vdl_cfg.get("pin_memory", True),
            collate_fn=collate_batch,
            drop_last=False,
        )

    # Build model via mmengine registry
    model = MODELS.build(cfg.model)
    checkpoint_include_prefixes = tuple(cfg.get("checkpoint_include_prefixes", ()))
    if cfg.checkpoint:
        msg = load_model_from_checkpoint(
            model,
            cfg.checkpoint,
            include_prefixes=checkpoint_include_prefixes or None,
        )
        log(f"Loaded checkpoint: {msg}")

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
    sch_cfg = cfg.scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=sch_cfg.eta_min,
    )

    output_dir = Path(cfg.output_dir)
    writer = SummaryWriter(log_dir=str(output_dir / "tb")) if is_main_process() else None

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("amp", True) and device.type == "cuda")
    best_val = float("inf")
    global_step = 0
    start_epoch = 1
    freeze_prefixes = tuple(cfg.get("freeze_modules", ()))
    if cfg.get("freeze_backbone", False) and not freeze_prefixes:
        freeze_prefixes = ("aggregator",)
    freeze_for_epochs = int(cfg.get("freeze_modules_for_epochs", 0))

    # ------------------------------------------------------------------
    # Resume: restore optimizer / scheduler / scaler / epoch from ckpt
    # ------------------------------------------------------------------
    if getattr(args, "resume", None):
        resume_path = args.resume
        if not Path(resume_path).is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location="cpu")
        # Model weights
        raw_sd = resume_ckpt["model"]
        if all(k.startswith("module.") for k in raw_sd):
            raw_sd = {k[7:]: v for k, v in raw_sd.items()}
        unwrap_model(model).load_state_dict(raw_sd, strict=True)
        log(f"[resume] model weights restored from {resume_path}")
        # Optimizer
        if resume_ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            log("[resume] optimizer state restored")
        # Scheduler
        if resume_ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
            log("[resume] scheduler state restored")
        # Scaler
        if resume_ckpt.get("scaler") is not None:
            scaler.load_state_dict(resume_ckpt["scaler"])
            log("[resume] grad scaler state restored")
        # Epoch / step counters
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("step", 0))
        log(f"[resume] resuming from epoch {start_epoch}, global_step={global_step}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        stage_msg = set_trainable_state(model, epoch, freeze_prefixes, freeze_for_epochs)
        log(f"Epoch {epoch}: {stage_msg}")
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running: Dict[str, float] = {}
        num_batches = 0
        train_bar = tqdm(train_loader, desc=f"Train E{epoch}", leave=False, disable=not is_main_process())
        for batch in train_bar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            normalized_extrinsics = normalize_extrinsics_to_first_frame(batch["extrinsics"])

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.get("amp", True) and device.type == "cuda"):
                predictions = model(batch["images"], others=build_model_others(batch, normalized_extrinsics))
                loss, metrics = compute_losses(
                    predictions,
                    batch,
                    normalized_extrinsics,
                    depth_weight=cfg.depth_weight,
                    camera_weight=cfg.get("camera_weight", 0.0),
                    pose_translation_weight=cfg.get("pose_translation_weight", 1.0),
                    pose_rotation_weight=cfg.get("pose_rotation_weight", 1.0),
                    pose_fov_weight=cfg.get("pose_fov_weight", 1.0),
                    depth_pred_scale=cfg.get("depth_pred_scale", 1.0),
                )

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                active_trainable_params = [p for p in model.parameters() if p.requires_grad]
                if active_trainable_params:
                    torch.nn.utils.clip_grad_norm_(active_trainable_params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value
            global_step += 1
            num_batches += 1

            cur_lr = optimizer.param_groups[0]["lr"]
            train_bar.set_postfix(loss=f"{metrics['loss']:.4f}", lr=f"{cur_lr:.2e}")

            if writer is not None and global_step % cfg.get("log_interval", 10) == 0:
                writer.add_scalar("train/loss_step", metrics["loss"], global_step)
                writer.add_scalar("train/depth_loss_step", metrics["depth_loss"], global_step)

        scheduler.step()
        train_avg = reduce_scalar_dict(running, num_batches, device)

        val_metrics: Dict[str, float] = {}
        if val_loader is not None:
            model.eval()
            val_sum: Dict[str, float] = {}
            val_count = 0
            val_bar = tqdm(val_loader, desc=f"Val   E{epoch}", leave=False, disable=not is_main_process())
            with torch.no_grad():
                for batch in val_bar:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    normalized_extrinsics = normalize_extrinsics_to_first_frame(batch["extrinsics"])
                    predictions = model(batch["images"], others=build_model_others(batch, normalized_extrinsics))
                    _, metrics = compute_losses(
                        predictions,
                        batch,
                        normalized_extrinsics,
                        depth_weight=cfg.depth_weight,
                        camera_weight=cfg.get("camera_weight", 0.0),
                        pose_translation_weight=cfg.get("pose_translation_weight", 1.0),
                        pose_rotation_weight=cfg.get("pose_rotation_weight", 1.0),
                        pose_fov_weight=cfg.get("pose_fov_weight", 1.0),
                        depth_pred_scale=cfg.get("depth_pred_scale", 1.0),
                    )
                    for key, value in metrics.items():
                        val_sum[key] = val_sum.get(key, 0.0) + value
                    val_count += 1
            val_metrics = reduce_scalar_dict(val_sum, val_count, device)

        log_line = (
            f"Epoch {epoch}/{cfg.epochs}"
            + " | train " + " ".join(f"{k}={v:.4f}" for k, v in train_avg.items())
        )
        if val_metrics:
            log_line += " | val " + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        log(log_line)

        if writer is not None:
            for k, v in train_avg.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        should_save = (epoch % cfg.save_every == 0) or (epoch == cfg.epochs)
        if should_save:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pth", model, optimizer, scheduler, epoch, global_step, cfg, scaler)
            save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler, epoch, global_step, cfg, scaler)
        if val_metrics and val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(output_dir / "best.pth", model, optimizer, scheduler, epoch, global_step, cfg, scaler)

    if writer is not None:
        writer.close()
    cleanup_distributed()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune VGGT on vKITTI depth")
    parser.add_argument("config", help="Path to mmengine Python config file")
    parser.add_argument("--checkpoint", default=None, help="Override config checkpoint path")
    parser.add_argument("--resume", default=None,
                        help="Resume full training state (model + optimizer + scheduler + epoch) "
                             "from a checkpoint saved by this script.")
    parser.add_argument("--output-dir", default=None, help="Override config output_dir")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-eval", action="store_true", default=False,
                        help="Disable validation loop during training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # CLI overrides
    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
    # When --resume is given, skip the initial pretrained checkpoint load so
    # that the full training state (model + optimizer + scheduler) is restored
    # exclusively from the resume checkpoint inside train().
    if args.resume is not None:
        cfg.checkpoint = None
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
