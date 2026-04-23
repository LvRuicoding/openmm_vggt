#!/usr/bin/env python3
"""Evaluate DDAD depth fine-tuning checkpoints for pure image-depth models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import tools.eval_ddad_depth_mix as base_eval


DEFAULT_CONFIG_PATH = base_eval.REPO_ROOT / "configs" / "ddad_depth_6cam_ft_scaled.py"
DEFAULT_DESCRIPTION = "Evaluate DDAD depth fine-tuning results for pure image-depth models."


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    depth_scale: float,
) -> Tuple[base_eval.DepthMetrics, int]:
    local = base_eval.DepthMetrics()
    evaluated_cameras = 0
    bar = tqdm(
        data_loader,
        desc=f"Eval rank={base_eval.get_rank()}",
        leave=False,
        disable=not base_eval.is_main(),
    )

    with torch.inference_mode():
        for batch in bar:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            metric_batch = batch
            model_batch = base_eval.duplicate_singleton_batch(batch)

            imgs = model_batch["images"]
            norm_ext = base_eval.normalize_extrinsics_to_first_frame(model_batch["extrinsics"])

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                preds = model(
                    imgs,
                    others={
                        "extrinsics": norm_ext,
                        "intrinsics": model_batch["intrinsics"],
                    },
                )

            gt_depth = metric_batch["depths"].float()
            pred_depth = preds["depth"][: gt_depth.shape[0]].squeeze(-1) * depth_scale
            local.update(pred_depth.float(), gt_depth)

            valid_slots = (gt_depth > 0.0).flatten(2).any(-1)
            if valid_slots.numel() > 0:
                evaluated_cameras = max(evaluated_cameras, int(valid_slots.sum(dim=1).max().item()))

    reduced = local.reduce(device)
    evaluated_cameras_tensor = torch.tensor([evaluated_cameras], dtype=torch.int64, device=device)
    if base_eval.is_dist():
        torch.distributed.all_reduce(evaluated_cameras_tensor, op=torch.distributed.ReduceOp.MAX)
    return reduced, int(evaluated_cameras_tensor.item())


def main() -> None:
    args = base_eval.parse_args(
        default_config_path=DEFAULT_CONFIG_PATH,
        description=DEFAULT_DESCRIPTION,
    )
    device = base_eval.setup_distributed()

    try:
        cfg = base_eval.Config.fromfile(args.config)
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

        cfg.val_dataset.return_lidar = False
        image_size = tuple(cfg.val_dataset.image_size)
        assert image_size[0] % 14 == 0 and image_size[1] % 14 == 0, (
            f"image_size {image_size} must be multiples of 14"
        )

        depth_scale = args.depth_scale if args.depth_scale is not None else float(cfg.get("depth_pred_scale", 1.0))
        supervision_source = "batch_depth"

        model = base_eval.MODELS.build(cfg.model)
        checkpoint_include_prefixes = tuple(cfg.get("checkpoint_include_prefixes", ()))
        msg = base_eval.load_checkpoint(
            model,
            ckpt_path,
            include_prefixes=checkpoint_include_prefixes or None,
        )
        base_eval.log(f"Loaded checkpoint: {msg}")
        model.eval().to(device)
        if base_eval.is_dist():
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        dataset = base_eval.DATASETS.build(cfg.val_dataset)
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

        metrics, evaluated_cameras = evaluate(
            model,
            data_loader,
            device,
            depth_scale=depth_scale,
        )
        metrics = metrics.result()

        if base_eval.is_main():
            print("\nDDAD Depth Metrics", flush=True)
            print(f"{'cameras':>8s}: {evaluated_cameras}", flush=True)
            print(f"{'gt_mode':>8s}: {supervision_source}", flush=True)
            for key in ("iRMSE", "iMAE", "RMSE", "MAE", "Abs Rel", "Sq Rel", "SILog", "d1", "d2", "d3"):
                print(f"{key:>8s}: {metrics[key]:.6f}", flush=True)
            print(f"{'n_pixels':>8s}: {metrics['n_pixels']:.0f}", flush=True)
            report_path = base_eval.save_metrics_report(
                ckpt_path=ckpt_path,
                config_path=args.config,
                batch_size=val_batch_size,
                num_workers=val_num_workers,
                depth_scale=depth_scale,
                supervision_source=supervision_source,
                evaluated_cameras=evaluated_cameras,
                metrics=metrics,
            )
            print(f"\nSaved metrics to: {report_path}", flush=True)
    finally:
        base_eval.cleanup_distributed()


if __name__ == "__main__":
    main()
