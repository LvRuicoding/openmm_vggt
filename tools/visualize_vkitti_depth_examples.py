#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS

import openmm_vggt  # noqa: F401
from eval_kitti_depth import (
    duplicate_singleton_batch,
    load_checkpoint,
    normalize_extrinsics_to_first_frame,
)


DEFAULT_CONFIG = str(REPO_ROOT / "configs" / "vkitti_depth_stereo_ft_scaled.py")
DEFAULT_CHECKPOINT = str(
    REPO_ROOT / "trainoutput" / "vkitti_depth_stereo_ft_scaled" / "last.pth"
)
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "visual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize dense depth predictions for a few vKITTI examples."
    )
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of validation samples to visualize when --indices is not set.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit dataset indices to visualize.",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Override validation image size.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Override depth prediction scaling factor.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device, e.g. cuda:0 or cpu. Defaults to cuda if available else cpu.",
    )
    return parser.parse_args()


def pick_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tensor_image_to_bgr(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    image_np = np.clip(image_np, 0.0, 1.0)
    image_rgb = (image_np * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def colorize_depth(
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    canvas = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if not np.any(valid_mask):
        return canvas

    clipped = np.clip(depth_m, vmin, vmax)
    norm = (clipped - vmin) / max(vmax - vmin, 1e-6)
    norm_u8 = (norm * 255.0).round().astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(norm_u8, colormap)
    canvas[valid_mask] = colored[valid_mask]
    return canvas


def colorize_error(error_m: np.ndarray, valid_mask: np.ndarray, emax: float) -> np.ndarray:
    canvas = np.zeros((*error_m.shape, 3), dtype=np.uint8)
    if not np.any(valid_mask):
        return canvas

    norm = np.clip(error_m / max(emax, 1e-6), 0.0, 1.0)
    norm_u8 = (norm * 255.0).round().astype(np.uint8)
    colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
    canvas[valid_mask] = colored[valid_mask]
    return canvas


def draw_panel_title(image: np.ndarray, title: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        title,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def make_camera_grid(
    rgb: np.ndarray,
    gt_m: np.ndarray,
    pred_m: np.ndarray,
) -> np.ndarray:
    valid_mask = gt_m > 0.0
    if np.any(valid_mask):
        valid_values = np.concatenate([gt_m[valid_mask], pred_m[valid_mask]])
        vmin = float(np.percentile(valid_values, 2.0))
        vmax = float(np.percentile(valid_values, 98.0))
        if vmax <= vmin:
            vmax = vmin + 1.0
        error_m = np.abs(pred_m - gt_m)
        emax = float(np.percentile(error_m[valid_mask], 98.0))
        if emax <= 0.0:
            emax = 1.0
    else:
        vmin, vmax, emax = 0.0, 1.0, 1.0
        error_m = np.abs(pred_m - gt_m)

    gt_vis = colorize_depth(gt_m, valid_mask, vmin=vmin, vmax=vmax)
    pred_vis = colorize_depth(pred_m, valid_mask, vmin=vmin, vmax=vmax)
    err_vis = colorize_error(error_m, valid_mask, emax=emax)

    panels = [
        draw_panel_title(rgb, "RGB"),
        draw_panel_title(gt_vis, f"GT depth [{vmin:.1f}, {vmax:.1f}] m"),
        draw_panel_title(pred_vis, "Pred depth"),
        draw_panel_title(err_vis, f"Abs error [0, {emax:.1f}] m"),
    ]
    return np.concatenate(panels, axis=1)


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def visualize_one_sample(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor | str],
    depth_scale: float,
    device: torch.device,
    output_dir: Path,
    sample_index: int,
) -> List[Dict[str, object]]:
    tensor_sample = {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if isinstance(value, torch.Tensor)
    }
    model_batch = duplicate_singleton_batch(tensor_sample)
    norm_ext = normalize_extrinsics_to_first_frame(model_batch["extrinsics"])

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            preds = model(
                model_batch["images"],
                others={
                    "extrinsics": norm_ext,
                    "intrinsics": model_batch["intrinsics"],
                },
            )

    images = tensor_sample["images"][0].detach().cpu()
    depths = tensor_sample["depths"][0].detach().cpu()
    pred_depth = preds["depth"][:1].squeeze(0).squeeze(-1).detach().cpu() * depth_scale

    seq_len = images.shape[0]
    last_pair_start = seq_len - 2
    sequence_name = sanitize_name(str(sample.get("sequence_name", f"sample_{sample_index:04d}")))

    records: List[Dict[str, object]] = []
    for cam_offset in range(2):
        seq_idx = last_pair_start + cam_offset
        rgb_bgr = tensor_image_to_bgr(images[seq_idx])
        gt_m = depths[seq_idx].numpy().astype(np.float32)
        pred_m = pred_depth[seq_idx].numpy().astype(np.float32)
        grid = make_camera_grid(rgb_bgr, gt_m, pred_m)

        stem = f"{sample_index:04d}_{sequence_name}_cam{cam_offset}"
        vis_path = output_dir / f"{stem}.png"
        npz_path = output_dir / f"{stem}.npz"
        cv2.imwrite(str(vis_path), grid)
        np.savez_compressed(
            npz_path,
            pred_depth_m=pred_m,
            gt_depth_m=gt_m,
            valid_mask=(gt_m > 0.0),
        )

        valid_mask = gt_m > 0.0
        mae = float(np.mean(np.abs(pred_m[valid_mask] - gt_m[valid_mask]))) if np.any(valid_mask) else None
        records.append(
            {
                "sample_index": sample_index,
                "camera_index": cam_offset,
                "sequence_name": str(sample.get("sequence_name", "")),
                "visual_path": str(vis_path),
                "depth_npz_path": str(npz_path),
                "valid_pixels": int(valid_mask.sum()),
                "mae_m": mae,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get("val_dataset", None) is None:
        raise ValueError("Config must define val_dataset.")

    if args.image_size is not None:
        cfg.val_dataset.image_size = tuple(args.image_size)

    depth_scale = (
        float(args.depth_scale)
        if args.depth_scale is not None
        else float(cfg.get("depth_pred_scale", 1.0))
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    model = MODELS.build(cfg.model)
    msg = load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    dataset = DATASETS.build(cfg.val_dataset)
    if args.indices is not None and len(args.indices) > 0:
        sample_indices = args.indices
    else:
        sample_indices = list(range(min(args.num_examples, len(dataset))))

    sample_indices = [idx for idx in sample_indices if 0 <= idx < len(dataset)]
    if not sample_indices:
        raise ValueError("No valid dataset indices selected.")

    summary: Dict[str, object] = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_status": msg,
        "output_dir": str(output_dir.resolve()),
        "device": str(device),
        "depth_scale": depth_scale,
        "dataset_length": len(dataset),
        "sample_indices": sample_indices,
        "records": [],
    }

    print(f"Using device: {device}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Depth scale: {depth_scale}", flush=True)
    print(f"Writing outputs to: {output_dir}", flush=True)

    for sample_index in sample_indices:
        sample = dataset[sample_index]
        records = visualize_one_sample(
            model=model,
            sample=sample,
            depth_scale=depth_scale,
            device=device,
            output_dir=output_dir,
            sample_index=sample_index,
        )
        summary["records"].extend(records)
        print(
            f"Saved sample {sample_index}: "
            + ", ".join(Path(record["visual_path"]).name for record in records),
            flush=True,
        )

    summary_path = output_dir / "vkitti_depth_examples_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
