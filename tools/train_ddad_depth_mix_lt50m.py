#!/usr/bin/env python3
"""Train DDAD depth with GT supervision restricted to pixels within 50m by default.

This script reuses tools/train.py end-to-end and only overrides the depth-valid
pixel mask used in both training and validation loss computation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config

import tools.train as base_train


DEFAULT_CONFIG = str(REPO_ROOT / "configs" / "early" / "ddad_depth_6cam_mix_window_attn_early_ft.py")
DEFAULT_MAX_GT_DEPTH_M = 50.0

ACTIVE_MAX_GT_DEPTH_M = DEFAULT_MAX_GT_DEPTH_M


def compute_losses_lt50m(
    predictions,
    batch,
    normalized_extrinsics,
    depth_weight,
    camera_weight,
    pose_translation_weight,
    pose_rotation_weight,
    pose_fov_weight,
    depth_pred_scale=1.0,
    depth_supervision_source="batch_depth",
):
    with base_train.torch.amp.autocast(device_type=batch["images"].device.type, enabled=False):
        pred_depth = (predictions["depth"].squeeze(-1) * depth_pred_scale).float()
        gt_depth = base_train.build_depth_supervision_target(batch, depth_supervision_source)
        valid_mask = (gt_depth > 0.0) & (gt_depth <= ACTIVE_MAX_GT_DEPTH_M)
        depth_loss = (
            base_train.F.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask])
            if valid_mask.any()
            else pred_depth.new_zeros(())
        )
    total = depth_weight * depth_loss
    metrics = {
        "loss": float(total.detach().cpu()),
        "depth_loss": float(depth_loss.detach().cpu()),
        "depth_valid_pixels": float(valid_mask.sum().detach().cpu()),
    }

    if "seq_enc_list" in predictions and "intrinsics" in batch and camera_weight > 0:
        image_hw = batch["images"].shape[-2:]
        gt_pose_enc = base_train.extri_intri_to_pose_encoding(
            normalized_extrinsics,
            batch["intrinsics"],
            image_size_hw=image_hw,
        )
        pred_pose_enc = predictions["seq_enc_list"][-1]
        pred_extrinsics, _ = base_train.pose_encoding_to_extri_intri(pred_pose_enc, image_size_hw=image_hw)

        pose_translation_loss = base_train.F.l1_loss(
            pred_extrinsics[..., :3, 3], normalized_extrinsics[..., :3, 3]
        )
        pose_rotation_loss = base_train.F.l1_loss(
            pred_extrinsics[..., :3, :3], normalized_extrinsics[..., :3, :3]
        )
        pose_fov_loss = base_train.F.l1_loss(pred_pose_enc[..., 7:], gt_pose_enc[..., 7:])

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DDAD depth with GT depth limited to <= 50m by default."
    )
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG, help="Path to mmengine Python config file")
    parser.add_argument("--checkpoint", default=None, help="Override config checkpoint path")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume full training state (model + optimizer + scheduler + epoch) from a training checkpoint.",
    )
    parser.add_argument("--output-dir", default=None, help="Override config output_dir")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-eval", action="store_true", default=False, help="Disable validation loop during training")
    parser.add_argument(
        "--max-gt-depth-m",
        type=float,
        default=None,
        help="Only GT pixels with depth <= this threshold contribute to train/val loss. Default: read config or 50.",
    )
    return parser.parse_args()


def resolve_max_gt_depth_m(cfg: Config, cli_value: float | None) -> float:
    if cli_value is not None:
        return float(cli_value)
    if cfg.get("train_max_gt_depth_m", None) is not None:
        return float(cfg.train_max_gt_depth_m)
    if cfg.get("max_gt_depth_m", None) is not None:
        return float(cfg.max_gt_depth_m)
    return DEFAULT_MAX_GT_DEPTH_M


def main() -> None:
    global ACTIVE_MAX_GT_DEPTH_M

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
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

    ACTIVE_MAX_GT_DEPTH_M = resolve_max_gt_depth_m(cfg, args.max_gt_depth_m)
    cfg.train_max_gt_depth_m = ACTIVE_MAX_GT_DEPTH_M
    cfg.eval_max_gt_depth_m = ACTIVE_MAX_GT_DEPTH_M
    cfg.max_gt_depth_m = ACTIVE_MAX_GT_DEPTH_M

    base_train.compute_losses = compute_losses_lt50m
    base_train.log(f"Using GT depth threshold <= {ACTIVE_MAX_GT_DEPTH_M:.2f}m for train/val loss")
    base_train.train(args, cfg)


if __name__ == "__main__":
    main()
