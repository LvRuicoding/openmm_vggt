#!/usr/bin/env python3
"""Fine-tune VGGT_decoder_global on vKITTI depth using mmengine config."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

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
    for key in ("camera_to_world", "lidar_to_world", "points", "point_mask"):
        if key in batch:
            others[key] = batch[key]
    return others


def matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def build_depth_supervision_target(
    batch: Dict[str, torch.Tensor],
    supervision_source: str,
) -> torch.Tensor:
    with torch.amp.autocast(device_type=batch["images"].device.type, enabled=False):
        if supervision_source != "batch_depth":
            raise ValueError(
                "Unsupported depth supervision source: "
                f"{supervision_source}. Only 'batch_depth' is supported."
            )
        return batch["depths"].float()


def geo_scal_loss(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255, eps: float = 1e-6) -> torch.Tensor:
    pred = F.softmax(pred, dim=1)
    empty_probs = pred[:, 0]
    nonempty_probs = 1.0 - empty_probs

    mask = target != ignore_index
    nonempty_target = (target != 0)[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]
    if nonempty_target.numel() == 0:
        return pred.sum() * 0.0

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum().clamp_min(eps)
    recall = intersection / nonempty_target.sum().clamp_min(eps)
    specificity = ((1.0 - nonempty_target) * empty_probs).sum() / (1.0 - nonempty_target).sum().clamp_min(eps)
    one = torch.ones((), device=pred.device, dtype=pred.dtype)
    return (
        F.binary_cross_entropy(precision.clamp(eps, 1.0 - eps), one)
        + F.binary_cross_entropy(recall.clamp(eps, 1.0 - eps), one)
        + F.binary_cross_entropy(specificity.clamp(eps, 1.0 - eps), one)
    )


def sem_scal_loss(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255, eps: float = 1e-6) -> torch.Tensor:
    pred = F.softmax(pred, dim=1)
    mask = target != ignore_index
    valid_target = target[mask]
    if valid_target.numel() == 0:
        return pred.sum() * 0.0

    loss = pred.new_zeros(())
    count = 0
    for class_idx in range(pred.shape[1]):
        class_prob = pred[:, class_idx][mask]
        class_target = (valid_target == class_idx).float()
        if class_target.sum() <= 0:
            continue
        count += 1
        intersection = (class_prob * class_target).sum()

        precision = intersection / class_prob.sum().clamp_min(eps)
        recall = intersection / class_target.sum().clamp_min(eps)
        specificity = ((1.0 - class_prob) * (1.0 - class_target)).sum() / (1.0 - class_target).sum().clamp_min(eps)
        one = torch.ones((), device=pred.device, dtype=pred.dtype)
        loss = loss + F.binary_cross_entropy(precision.clamp(eps, 1.0 - eps), one)
        loss = loss + F.binary_cross_entropy(recall.clamp(eps, 1.0 - eps), one)
        loss = loss + F.binary_cross_entropy(specificity.clamp(eps, 1.0 - eps), one)
    if count == 0:
        return pred.sum() * 0.0
    return loss / float(count)


def frustum_proportion_loss(
    pred: torch.Tensor,
    frustums_masks: torch.Tensor,
    frustums_class_dists: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_prob = F.softmax(pred, dim=1)
    batch_size, num_classes = pred_prob.shape[:2]
    pred_prob = pred_prob.reshape(batch_size, num_classes, -1)
    frustums_masks = frustums_masks.reshape(batch_size, frustums_masks.shape[1], -1).to(dtype=pred_prob.dtype)
    frustums_class_dists = frustums_class_dists.to(device=pred.device, dtype=pred_prob.dtype)

    cum_prob = torch.einsum("bcv,bfv->fc", pred_prob, frustums_masks)
    target_counts = frustums_class_dists.sum(dim=0)
    total_prob = cum_prob.sum(dim=1, keepdim=True)
    total_counts = target_counts.sum(dim=1, keepdim=True)
    valid_frustums = (total_prob.squeeze(1) > eps) & (total_counts.squeeze(1) > 0)
    if not valid_frustums.any():
        return pred.sum() * 0.0

    pred_proportion = cum_prob / total_prob.clamp_min(eps)
    target_proportion = target_counts / total_counts.clamp_min(eps)
    nonzero = (target_proportion > 0) & valid_frustums[:, None]
    kl = target_proportion * (target_proportion.clamp_min(eps).log() - pred_proportion.clamp_min(eps).log())
    return kl[nonzero].sum() / valid_frustums.sum().to(dtype=kl.dtype)


def downsample_target_object_priority(
    target: torch.Tensor,
    output_shape: Tuple[int, int, int],
    ignore_index: int,
    num_classes: int,
) -> torch.Tensor:
    """Downsample labels by majority vote, preferring occupied classes over empty."""
    batch_size, grid_x, grid_y, grid_z = target.shape
    out_x, out_y, out_z = output_shape
    if grid_x % out_x != 0 or grid_y % out_y != 0 or grid_z % out_z != 0:
        raise ValueError(f"Cannot downsample target shape {tuple(target.shape[1:])} to {output_shape}")
    fx, fy, fz = grid_x // out_x, grid_y // out_y, grid_z // out_z
    blocks = (
        target.reshape(batch_size, out_x, fx, out_y, fy, out_z, fz)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .reshape(batch_size, out_x * out_y * out_z, fx * fy * fz)
    )
    output = target.new_full((batch_size, out_x * out_y * out_z), ignore_index)
    for batch_idx in range(batch_size):
        for block_idx in range(blocks.shape[1]):
            labels = blocks[batch_idx, block_idx]
            labels = labels[labels != ignore_index]
            if labels.numel() == 0:
                continue
            occupied = labels[(labels > 0) & (labels < num_classes)]
            if occupied.numel() > 0:
                counts = torch.bincount(occupied, minlength=num_classes)
                output[batch_idx, block_idx] = torch.argmax(counts[1:]) + 1
            else:
                output[batch_idx, block_idx] = 0
    return output.reshape(batch_size, out_x, out_y, out_z)


def compute_context_prior_target(
    target_l3: torch.Tensor,
    ignore_index: int,
    n_relations: int = 4,
) -> torch.Tensor:
    batch_size, grid_x, grid_y, grid_z = target_l3.shape
    if n_relations != 4:
        raise ValueError("Only the 4-relation MonoScene context prior target is supported.")
    if grid_x % 2 != 0 or grid_y % 2 != 0 or grid_z % 2 != 0:
        raise ValueError(f"Context-prior target shape must be divisible by 2, got {tuple(target_l3.shape[1:])}")

    mega_x, mega_y, mega_z = grid_x // 2, grid_y // 2, grid_z // 2
    num_voxels = grid_x * grid_y * grid_z
    num_mega_voxels = mega_x * mega_y * mega_z
    matrices = target_l3.new_zeros((batch_size, n_relations, num_voxels, num_mega_voxels), dtype=torch.float32)

    for batch_idx in range(batch_size):
        label_row = target_l3[batch_idx].reshape(-1)
        valid_row = label_row != ignore_index
        for xx in range(mega_x):
            for yy in range(mega_y):
                for zz in range(mega_z):
                    col_idx = xx * (mega_y * mega_z) + yy * mega_z + zz
                    labels = target_l3[
                        batch_idx,
                        xx * 2 : xx * 2 + 2,
                        yy * 2 : yy * 2 + 2,
                        zz * 2 : zz * 2 + 2,
                    ].reshape(-1)
                    labels = torch.unique(labels[labels != ignore_index])
                    for label_col in labels:
                        same = label_row == label_col
                        label_col_nonempty = label_col != 0
                        row_nonempty = label_row != 0
                        matrices[batch_idx, 0, valid_row & same & label_col_nonempty, col_idx] = 1.0
                        matrices[batch_idx, 1, valid_row & (~same) & label_col_nonempty & row_nonempty, col_idx] = 1.0
                        matrices[batch_idx, 2, valid_row & same & (~label_col_nonempty), col_idx] = 1.0
                        matrices[batch_idx, 3, valid_row & (~same) & ((~row_nonempty) | (~label_col_nonempty)), col_idx] = 1.0
    return matrices


def context_prior_loss(pred_logits: torch.Tensor, target_matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    logits = []
    labels = []
    batch_size, n_relations, _, _ = pred_logits.shape
    for batch_idx in range(batch_size):
        logits.append(pred_logits[batch_idx].permute(0, 2, 1).reshape(n_relations, -1))
        labels.append(target_matrix[batch_idx].reshape(n_relations, -1))
    logits = torch.cat(logits, dim=1).T
    labels = torch.cat(labels, dim=1).T
    cnt_neg = (labels == 0).sum(dim=0).to(dtype=logits.dtype)
    cnt_pos = labels.sum(dim=0).to(dtype=logits.dtype)
    pos_weight = cnt_neg / cnt_pos.clamp_min(eps)
    return F.binary_cross_entropy_with_logits(logits, labels.to(dtype=logits.dtype), pos_weight=pos_weight)


def compute_balanced_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    n_time_steps: int,
    cam_num: int,
    depth_loss_type: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Average depth loss as batch -> frame -> camera -> valid pixels."""
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(f"pred_depth shape {tuple(pred_depth.shape)} != gt_depth shape {tuple(gt_depth.shape)}")
    if pred_depth.ndim != 4:
        raise ValueError(f"Expected depth tensors with shape [B, S, H, W], got {tuple(pred_depth.shape)}")

    batch_size, sequence_len, height, width = pred_depth.shape
    expected_sequence_len = n_time_steps * cam_num
    if sequence_len != expected_sequence_len:
        raise ValueError(
            f"Depth sequence length {sequence_len} does not match "
            f"n_time_steps({n_time_steps}) * cam_num({cam_num}) = {expected_sequence_len}"
        )

    valid_mask = valid_mask.reshape(batch_size, n_time_steps, cam_num, height, width)
    valid_float = valid_mask.to(dtype=pred_depth.dtype)

    pred_depth = pred_depth.reshape(batch_size, n_time_steps, cam_num, height, width)
    gt_depth = gt_depth.reshape(batch_size, n_time_steps, cam_num, height, width)

    if depth_loss_type == "l1":
        pixel_loss = torch.abs(pred_depth - gt_depth)
    elif depth_loss_type == "log1p_l1":
        safe_pred_depth = torch.where(valid_mask, pred_depth, torch.zeros_like(pred_depth))
        safe_gt_depth = torch.where(valid_mask, gt_depth, torch.zeros_like(gt_depth))
        pixel_loss = torch.abs(torch.log1p(safe_pred_depth) - torch.log1p(safe_gt_depth))
    else:
        raise ValueError(f"Unsupported depth_loss_type: {depth_loss_type}")

    pixel_count = valid_float.sum(dim=(-1, -2))
    camera_valid = pixel_count > 0
    camera_loss_sum = (pixel_loss * valid_float).sum(dim=(-1, -2))
    camera_loss = camera_loss_sum / pixel_count.clamp_min(1.0)

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


def get_last_frame_view_indices(
    sequence_len: int,
    n_time_steps: int,
    cam_num: int,
    device: torch.device,
) -> torch.Tensor:
    expected_sequence_len = n_time_steps * cam_num
    if sequence_len != expected_sequence_len:
        raise ValueError(
            f"Sequence length {sequence_len} does not match "
            f"n_time_steps({n_time_steps}) * cam_num({cam_num}) = {expected_sequence_len}"
        )
    start = (n_time_steps - 1) * cam_num
    return torch.arange(start, start + cam_num, device=device, dtype=torch.long)


def select_last_frame_views(
    tensor: torch.Tensor,
    n_time_steps: int,
    cam_num: int,
) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"Expected a tensor with a sequence dimension, got {tuple(tensor.shape)}")
    view_indices = get_last_frame_view_indices(
        tensor.shape[1],
        n_time_steps=n_time_steps,
        cam_num=cam_num,
        device=tensor.device,
    )
    return tensor.index_select(1, view_indices)


def intrinsics_to_fov(intrinsics: torch.Tensor, image_size_hw: Tuple[int, int]) -> torch.Tensor:
    image_h, image_w = image_size_hw
    fx = intrinsics[..., 0, 0].clamp_min(1e-6)
    fy = intrinsics[..., 1, 1].clamp_min(1e-6)
    fov_h = 2.0 * torch.atan((image_h * 0.5) / fy)
    fov_w = 2.0 * torch.atan((image_w * 0.5) / fx)
    return torch.stack([fov_h, fov_w], dim=-1)


def compute_losses(
    predictions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    normalized_extrinsics: torch.Tensor,
    depth_weight: float,
    camera_weight: float,
    pose_translation_weight: float,
    pose_rotation_weight: float,
    pose_fov_weight: float,
    occupancy_weight: float = 0.0,
    occupancy_ignore_index: int = 255,
    occupancy_class_weights: torch.Tensor | None = None,
    occupancy_num_classes: int = 20,
    sem_scal_weight: float = 0.0,
    geo_scal_weight: float = 0.0,
    frustum_proportion_weight: float = 0.0,
    context_prior_weight: float = 0.0,
    depth_pred_scale: float = 1.0,
    depth_supervision_source: str = "batch_depth",
    depth_loss_type: str = "l1",
    n_time_steps: int = 1,
    cam_num: int = 1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = torch.zeros((), device=batch["images"].device, dtype=torch.float32)
    metrics: Dict[str, float] = {"loss": 0.0}

    if "depth" in predictions and depth_weight > 0:
        with torch.amp.autocast(device_type=batch["images"].device.type, enabled=False):
            pred_depth = (predictions["depth"].squeeze(-1) * depth_pred_scale).float()
            gt_depth = build_depth_supervision_target(batch, depth_supervision_source)
            pred_depth = select_last_frame_views(pred_depth, n_time_steps=n_time_steps, cam_num=cam_num)
            gt_depth = select_last_frame_views(gt_depth, n_time_steps=n_time_steps, cam_num=cam_num)
            valid_mask = (gt_depth > 0.0) & (gt_depth < 655.0)
            depth_loss, depth_stats = compute_balanced_depth_loss(
                pred_depth,
                gt_depth,
                valid_mask,
                n_time_steps=1,
                cam_num=cam_num,
                depth_loss_type=depth_loss_type,
            )
        total = total + depth_weight * depth_loss
        metrics.update(
            {
                "loss": float(total.detach().cpu()),
                "depth_loss": float(depth_loss.detach().cpu()),
                "depth_valid_pixels": float(depth_stats["depth_valid_pixels"].detach().cpu()),
                "depth_valid_cameras": float(depth_stats["depth_valid_cameras"].detach().cpu()),
                "depth_valid_frames": float(depth_stats["depth_valid_frames"].detach().cpu()),
                "depth_valid_samples": float(depth_stats["depth_valid_samples"].detach().cpu()),
            }
        )

    if ("seq_enc_list" in predictions or "extrinsic" in predictions) and "intrinsics" in batch and camera_weight > 0:
        image_hw = batch["images"].shape[-2:]
        gt_extrinsics = select_last_frame_views(normalized_extrinsics, n_time_steps=n_time_steps, cam_num=cam_num)
        gt_intrinsics = select_last_frame_views(batch["intrinsics"], n_time_steps=n_time_steps, cam_num=cam_num)
        gt_pose_enc = extri_intri_to_pose_encoding(
            gt_extrinsics,
            gt_intrinsics,
            image_size_hw=image_hw,
        )
        if "extrinsic" in predictions:
            pred_extrinsics = select_last_frame_views(
                predictions["extrinsic"].float(),
                n_time_steps=n_time_steps,
                cam_num=cam_num,
            )
            if "intrinsic" in predictions:
                pred_intrinsics = select_last_frame_views(
                    predictions["intrinsic"].float(),
                    n_time_steps=n_time_steps,
                    cam_num=cam_num,
                )
                pred_fov = intrinsics_to_fov(pred_intrinsics, image_hw)
            else:
                pred_pose_enc = predictions["seq_enc_list"][-1]
                if pred_pose_enc.shape[1] != cam_num:
                    pred_pose_enc = select_last_frame_views(
                        pred_pose_enc,
                        n_time_steps=n_time_steps,
                        cam_num=cam_num,
                    )
                pred_fov = pred_pose_enc[..., 7:]
        else:
            pred_pose_enc = predictions["seq_enc_list"][-1]
            pred_extrinsics, _ = pose_encoding_to_extri_intri(pred_pose_enc, image_size_hw=image_hw)
            if pred_extrinsics.shape[1] != cam_num:
                pred_extrinsics = select_last_frame_views(
                    pred_extrinsics,
                    n_time_steps=n_time_steps,
                    cam_num=cam_num,
                )
                pred_pose_enc = select_last_frame_views(
                    pred_pose_enc,
                    n_time_steps=n_time_steps,
                    cam_num=cam_num,
                )
            pred_fov = pred_pose_enc[..., 7:]

        if pred_extrinsics.shape != gt_extrinsics.shape:
            raise ValueError(
                f"Camera extrinsic supervision shape mismatch: "
                f"pred {tuple(pred_extrinsics.shape)} vs gt {tuple(gt_extrinsics.shape)}"
            )
        if pred_fov.shape != gt_pose_enc[..., 7:].shape:
            raise ValueError(
                f"Camera FOV supervision shape mismatch: "
                f"pred {tuple(pred_fov.shape)} vs gt {tuple(gt_pose_enc[..., 7:].shape)}"
            )

        pose_translation_loss = F.l1_loss(pred_extrinsics[..., :3, 3], gt_extrinsics[..., :3, 3])
        pose_rotation_loss = F.l1_loss(pred_extrinsics[..., :3, :3], gt_extrinsics[..., :3, :3])
        pose_fov_loss = F.l1_loss(pred_fov, gt_pose_enc[..., 7:])

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

    if "occupancy_logits" in predictions and "occupancy_target" in batch and occupancy_weight > 0:
        occ_logits = predictions["occupancy_logits"].float()
        occ_target_raw, occ_valid_mask = align_occupancy_targets_to_logits(
            occ_logits,
            batch["occupancy_target"].long(),
            batch["occupancy_valid_mask"].bool(),
        )
        occ_target = occ_target_raw.masked_fill(~occ_valid_mask, occupancy_ignore_index)
        occ_loss = F.cross_entropy(
            occ_logits,
            occ_target,
            ignore_index=occupancy_ignore_index,
            weight=occupancy_class_weights,
        )
        total = total + occupancy_weight * occ_loss
        extra_occ_metrics = {}
        if sem_scal_weight > 0:
            sem_loss = sem_scal_loss(occ_logits, occ_target, ignore_index=occupancy_ignore_index)
            total = total + sem_scal_weight * sem_loss
            extra_occ_metrics["sem_scal_loss"] = float(sem_loss.detach().cpu())
        if geo_scal_weight > 0:
            geo_loss = geo_scal_loss(occ_logits, occ_target, ignore_index=occupancy_ignore_index)
            total = total + geo_scal_weight * geo_loss
            extra_occ_metrics["geo_scal_loss"] = float(geo_loss.detach().cpu())
        if (
            frustum_proportion_weight > 0
            and "frustums_masks" in batch
            and "frustums_class_dists" in batch
        ):
            frustums_masks, frustums_class_dists = align_frustum_targets_to_logits(
                occ_logits,
                batch["frustums_masks"].bool(),
                batch["frustums_class_dists"].float(),
            )
            fp_loss = frustum_proportion_loss(occ_logits, frustums_masks, frustums_class_dists)
            total = total + frustum_proportion_weight * fp_loss
            extra_occ_metrics["frustum_proportion_loss"] = float(fp_loss.detach().cpu())
        if context_prior_weight > 0 and "context_prior_logits" in predictions:
            cp_logits = predictions["context_prior_logits"].float()
            _, n_relations, context_count, voxel_count = cp_logits.shape
            # P_logits has shape [B, R, mega_voxels, l3_voxels].
            # Infer the l3 grid from the full target aspect ratio and voxel count.
            target_shape = occ_target.shape[1:]
            down_factor = round((int(np.prod(target_shape)) / float(voxel_count)) ** (1.0 / 3.0))
            if down_factor < 1:
                raise ValueError(f"Invalid context-prior down_factor inferred from {target_shape} and {voxel_count}")
            l3_shape = tuple(int(dim // down_factor) for dim in target_shape)
            if int(np.prod(l3_shape)) != voxel_count:
                raise ValueError(
                    f"Cannot infer context-prior l3 shape: target={tuple(target_shape)}, "
                    f"voxel_count={voxel_count}, inferred={l3_shape}"
                )
            expected_context_count = int(np.prod(tuple(dim // 2 for dim in l3_shape)))
            if expected_context_count != context_count:
                raise ValueError(
                    f"Context-prior mega-voxel count mismatch: logits={context_count}, "
                    f"expected={expected_context_count} from l3_shape={l3_shape}"
                )
            occ_target_l3 = downsample_target_object_priority(
                occ_target,
                output_shape=l3_shape,
                ignore_index=occupancy_ignore_index,
                num_classes=occupancy_num_classes,
            )
            cp_target = compute_context_prior_target(
                occ_target_l3,
                ignore_index=occupancy_ignore_index,
                n_relations=n_relations,
            )
            cp_loss = context_prior_loss(cp_logits, cp_target)
            total = total + context_prior_weight * cp_loss
            extra_occ_metrics["context_prior_loss"] = float(cp_loss.detach().cpu())
        occ_pred = occ_logits.argmax(dim=1)
        valid = occ_target != occupancy_ignore_index
        occ_acc = (occ_pred[valid] == occ_target[valid]).float().mean().item() if valid.any() else 0.0
        metrics.update(
            {
                "loss": float(total.detach().cpu()),
                "occupancy_loss": float(occ_loss.detach().cpu()),
                "occupancy_acc": float(occ_acc),
                "occupancy_valid_voxels": float(valid.sum().item()),
                **extra_occ_metrics,
            }
        )
    return total, metrics


def align_occupancy_targets_to_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if logits.shape[0] == target.shape[0]:
        return target, valid_mask
    if target.shape[0] <= 0 or logits.shape[0] % target.shape[0] != 0:
        raise ValueError(
            f"Occupancy batch mismatch: logits batch={logits.shape[0]}, target batch={target.shape[0]}"
        )
    repeat = logits.shape[0] // target.shape[0]
    target = target.repeat_interleave(repeat, dim=0)
    if valid_mask is not None:
        valid_mask = valid_mask.repeat_interleave(repeat, dim=0)
    return target, valid_mask


def align_frustum_targets_to_logits(
    logits: torch.Tensor,
    frustums_masks: torch.Tensor,
    frustums_class_dists: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_batch = logits.shape[0]
    if frustums_masks.shape[0] == logits_batch and frustums_masks.ndim == 5:
        return frustums_masks.to(device=logits.device), frustums_class_dists.to(device=logits.device)

    if frustums_masks.shape[0] == logits_batch and frustums_masks.ndim == 6:
        batch_size, view_count, frustum_count = frustums_masks.shape[:3]
        return (
            frustums_masks.reshape(batch_size, view_count * frustum_count, *frustums_masks.shape[3:]).to(device=logits.device),
            frustums_class_dists.reshape(batch_size, view_count * frustum_count, *frustums_class_dists.shape[3:]).to(device=logits.device),
        )

    if frustums_masks.ndim == 6 and frustums_masks.shape[0] * frustums_masks.shape[1] == logits_batch:
        return (
            frustums_masks.reshape(logits_batch, *frustums_masks.shape[2:]).to(device=logits.device),
            frustums_class_dists.reshape(logits_batch, *frustums_class_dists.shape[2:]).to(device=logits.device),
        )

    if frustums_masks.shape[0] > 0 and logits_batch % frustums_masks.shape[0] == 0:
        repeat = logits_batch // frustums_masks.shape[0]
        return (
            frustums_masks.repeat_interleave(repeat, dim=0).to(device=logits.device),
            frustums_class_dists.repeat_interleave(repeat, dim=0).to(device=logits.device),
        )

    raise ValueError(
        "Frustum target batch mismatch: "
        f"logits batch={logits_batch}, frustums_masks shape={tuple(frustums_masks.shape)}"
    )


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
    stats["correct"] = stats.get("correct", 0.0) + float((pred_valid == target_valid).sum().item())
    stats["total"] = stats.get("total", 0.0) + float(target_valid.numel())
    for class_idx in range(num_classes):
        pred_class = pred_valid == class_idx
        target_class = target_valid == class_idx
        stats[f"class_{class_idx}_intersection"] = stats.get(f"class_{class_idx}_intersection", 0.0) + float((pred_class & target_class).sum().item())
        stats[f"class_{class_idx}_union"] = stats.get(f"class_{class_idx}_union", 0.0) + float((pred_class | target_class).sum().item())


def summarize_occupancy_stats(stats: Dict[str, float], num_classes: int) -> Dict[str, float]:
    if not stats:
        return {}
    acc = stats.get("correct", 0.0) / max(stats.get("total", 0.0), 1.0)
    semantic_ious = []
    for class_idx in range(1, num_classes):
        union = stats.get(f"class_{class_idx}_union", 0.0)
        if union > 0:
            semantic_ious.append(stats.get(f"class_{class_idx}_intersection", 0.0) / union)
    return {
        "occupancy_acc": float(acc),
        "ssc_miou": float(sum(semantic_ious) / max(len(semantic_ious), 1)),
    }


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
    random_init_keys = set(missing)
    filtered = {}
    for key in model_keys:
        if key in missing:
            continue
        tensor = model_state[key]
        ckpt_tensor = state_dict[key]
        if ckpt_tensor.shape != tensor.shape:
            mismatched.append((key, tuple(tensor.shape), tuple(ckpt_tensor.shape)))
            random_init_keys.add(key)
        else:
            filtered[key] = ckpt_tensor

    model.load_state_dict(filtered, strict=False)
    model._random_init_param_names = {
        name for name, _ in model.named_parameters() if name in random_init_keys
    }
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
    if model._random_init_param_names:
        sample = sorted(model._random_init_param_names)[:20]
        message_parts.append(f"random-init params={len(model._random_init_param_names)} sample={sample}")
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
    random_init_param_names: Set[str] = set(getattr(unwrap_model(model), "_random_init_param_names", set()))
    random_init_multiplier = float(cfg.get("random_init_lr_multiplier", 1.0))
    grouped_params: Dict[Tuple[str, float], list[nn.Parameter]] = defaultdict(list)
    group_counts: Dict[Tuple[str, float], int] = defaultdict(int)

    for name, param in model.named_parameters():
        bare_name = name[len("module."):] if name.startswith("module.") else name
        multiplier, label = resolve_lr_multiplier(bare_name, lr_multipliers)
        if bare_name in random_init_param_names:
            multiplier *= random_init_multiplier
            label = f"random_init:{label}"
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


def build_loss_log_path(output_dir: Path) -> Path:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"loss_steps_{run_timestamp}.jsonl"


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
        T_max=max(int(sch_cfg.get("T_max", cfg.epochs)), 1),
        eta_min=sch_cfg.eta_min,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tb")) if is_main_process() else None
    loss_path = build_loss_log_path(output_dir) if is_main_process() else None
    loss_file = loss_path.open("w", encoding="utf-8") if loss_path is not None else None
    if loss_path is not None:
        log(f"Loss log path: {loss_path}")

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("amp", True) and device.type == "cuda")
    best_val = float("inf")
    global_step = 0
    start_epoch = 1
    freeze_prefixes = tuple(cfg.get("freeze_modules", ()))
    if cfg.get("freeze_backbone", False) and not freeze_prefixes:
        freeze_prefixes = ("aggregator",)
    freeze_for_epochs = int(cfg.get("freeze_modules_for_epochs", 0))
    n_time_steps = int(cfg.get("n_time_steps", cfg.train_dataset.get("n_time_steps", 1)))
    cam_num = int(cfg.model.get("cam_num", len(cfg.train_dataset.get("camera_names", ())) or 1))
    occupancy_weight = float(cfg.get("occupancy_weight", 0.0))
    occupancy_ignore_index = int(cfg.get("occupancy_ignore_index", 255))
    occupancy_num_classes = int(cfg.get("occupancy_num_classes", 20))
    occupancy_class_weights = cfg.get("occupancy_class_weights", None)
    if occupancy_class_weights is not None:
        occupancy_class_weights = torch.tensor(occupancy_class_weights, dtype=torch.float32, device=device)
    sem_scal_weight = float(cfg.get("sem_scal_weight", 0.0))
    geo_scal_weight = float(cfg.get("geo_scal_weight", 0.0))
    frustum_proportion_weight = float(cfg.get("frustum_proportion_weight", 0.0))
    context_prior_weight = float(cfg.get("context_prior_weight", 0.0))
    maximize_metric = occupancy_weight > 0
    best_val = -float("inf") if maximize_metric else float("inf")

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
        iter_end_time = time.perf_counter()
        for batch in train_bar:
            data_time = time.perf_counter() - iter_end_time

            sync_device_for_timing(device)
            h2d_start_time = time.perf_counter()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            sync_device_for_timing(device)
            h2d_time = time.perf_counter() - h2d_start_time

            sync_device_for_timing(device)
            prep_start_time = time.perf_counter()
            normalized_extrinsics = normalize_extrinsics_to_first_frame(batch["extrinsics"])
            optimizer.zero_grad(set_to_none=True)
            sync_device_for_timing(device)
            prep_time = time.perf_counter() - prep_start_time

            sync_device_for_timing(device)
            forward_start_time = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=cfg.get("amp", True) and device.type == "cuda"):
                predictions = model(batch["images"], others=build_model_others(batch, normalized_extrinsics))
            sync_device_for_timing(device)
            forward_time = time.perf_counter() - forward_start_time

            sync_device_for_timing(device)
            loss_start_time = time.perf_counter()
            loss, metrics = compute_losses(
                predictions,
                batch,
                normalized_extrinsics,
                depth_weight=cfg.depth_weight,
                camera_weight=cfg.get("camera_weight", 0.0),
                pose_translation_weight=cfg.get("pose_translation_weight", 1.0),
                pose_rotation_weight=cfg.get("pose_rotation_weight", 1.0),
                pose_fov_weight=cfg.get("pose_fov_weight", 1.0),
                occupancy_weight=occupancy_weight,
                occupancy_ignore_index=occupancy_ignore_index,
                occupancy_class_weights=occupancy_class_weights,
                occupancy_num_classes=occupancy_num_classes,
                sem_scal_weight=sem_scal_weight,
                geo_scal_weight=geo_scal_weight,
                frustum_proportion_weight=frustum_proportion_weight,
                context_prior_weight=context_prior_weight,
                depth_pred_scale=cfg.get("depth_pred_scale", 1.0),
                depth_supervision_source=cfg.get("depth_supervision_source", "batch_depth"),
                depth_loss_type=cfg.get("depth_loss_type", "l1"),
                n_time_steps=n_time_steps,
                cam_num=cam_num,
            )
            sync_device_for_timing(device)
            loss_time = time.perf_counter() - loss_start_time

            sync_device_for_timing(device)
            backward_start_time = time.perf_counter()
            scaler.scale(loss).backward()
            sync_device_for_timing(device)
            backward_time = time.perf_counter() - backward_start_time

            grad_clip_time = 0.0
            if cfg.grad_clip > 0:
                sync_device_for_timing(device)
                grad_clip_start_time = time.perf_counter()
                scaler.unscale_(optimizer)
                active_trainable_params = [p for p in model.parameters() if p.requires_grad]
                if active_trainable_params:
                    torch.nn.utils.clip_grad_norm_(active_trainable_params, cfg.grad_clip)
                sync_device_for_timing(device)
                grad_clip_time = time.perf_counter() - grad_clip_start_time

            sync_device_for_timing(device)
            optim_start_time = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            sync_device_for_timing(device)
            optim_time = time.perf_counter() - optim_start_time

            compute_time = (
                prep_time
                + forward_time
                + loss_time
                + backward_time
                + grad_clip_time
                + optim_time
            )

            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value
            global_step += 1
            append_loss_record(
                loss_file,
                global_step,
                metrics["loss"],
                data_time=data_time,
                h2d_time=h2d_time,
                prep_time=prep_time,
                forward_time=forward_time,
                loss_time=loss_time,
                backward_time=backward_time,
                grad_clip_time=grad_clip_time,
                optim_time=optim_time,
                compute_time=compute_time,
            )
            num_batches += 1
            iter_end_time = time.perf_counter()

            cur_lr = optimizer.param_groups[0]["lr"]
            train_bar.set_postfix(loss=f"{metrics['loss']:.4f}", lr=f"{cur_lr:.2e}")

            if writer is not None and global_step % cfg.get("log_interval", 10) == 0:
                writer.add_scalar("train/loss_step", metrics["loss"], global_step)
                if "depth_loss" in metrics:
                    writer.add_scalar("train/depth_loss_step", metrics["depth_loss"], global_step)
                if "occupancy_loss" in metrics:
                    writer.add_scalar("train/occupancy_loss_step", metrics["occupancy_loss"], global_step)
                if "occupancy_acc" in metrics:
                    writer.add_scalar("train/occupancy_acc_step", metrics["occupancy_acc"], global_step)

        scheduler.step()
        train_avg = reduce_scalar_dict(running, num_batches, device)

        val_metrics: Dict[str, float] = {}
        if val_loader is not None:
            model.eval()
            val_sum: Dict[str, float] = {}
            val_count = 0
            val_occ_stats: Dict[str, float] = {}
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
                        occupancy_weight=occupancy_weight,
                        occupancy_ignore_index=occupancy_ignore_index,
                        occupancy_class_weights=occupancy_class_weights,
                        occupancy_num_classes=occupancy_num_classes,
                        sem_scal_weight=sem_scal_weight,
                        geo_scal_weight=geo_scal_weight,
                        frustum_proportion_weight=frustum_proportion_weight,
                        context_prior_weight=context_prior_weight,
                        depth_pred_scale=cfg.get("depth_pred_scale", 1.0),
                        depth_supervision_source=cfg.get("depth_supervision_source", "batch_depth"),
                        depth_loss_type=cfg.get("depth_loss_type", "l1"),
                        n_time_steps=n_time_steps,
                        cam_num=cam_num,
                    )
                    for key, value in metrics.items():
                        val_sum[key] = val_sum.get(key, 0.0) + value
                    val_count += 1
                    if "occupancy_logits" in predictions:
                        occ_target_raw, occ_valid_mask = align_occupancy_targets_to_logits(
                            predictions["occupancy_logits"],
                            batch["occupancy_target"].long(),
                            batch["occupancy_valid_mask"].bool(),
                        )
                        occ_target = occ_target_raw.masked_fill(~occ_valid_mask, occupancy_ignore_index)
                        accumulate_occupancy_stats(
                            val_occ_stats,
                            predictions["occupancy_logits"],
                            occ_target,
                            occ_valid_mask,
                            occupancy_num_classes,
                            occupancy_ignore_index,
                        )
            val_metrics = reduce_scalar_dict(val_sum, val_count, device)
            val_occ_stats = reduce_sum_dict(val_occ_stats, device)
            val_metrics.update(summarize_occupancy_stats(val_occ_stats, occupancy_num_classes))

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
        if val_metrics:
            if maximize_metric:
                current_metric = val_metrics.get("ssc_miou", val_metrics.get("occupancy_acc", -float("inf")))
                if current_metric > best_val:
                    best_val = current_metric
                    save_checkpoint(output_dir / "best.pth", model, optimizer, scheduler, epoch, global_step, cfg, scaler)
            else:
                current_metric = val_metrics.get("loss", float("inf"))
                if current_metric < best_val:
                    best_val = current_metric
                    save_checkpoint(output_dir / "best.pth", model, optimizer, scheduler, epoch, global_step, cfg, scaler)

    if writer is not None:
        writer.close()
    if loss_file is not None:
        loss_file.close()
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
