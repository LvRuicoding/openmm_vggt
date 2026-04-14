#!/usr/bin/env python3
"""Visualize DDAD voxel projections on images using the early-fusion geometry path."""
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config
from mmengine.registry import DATASETS

import openmm_vggt  # noqa: F401
from tools.visualize_voxel_projection import (
    FusionProjector,
    concat_h,
    draw_patch_heatmap,
    draw_points_overlay,
    make_contact_sheet,
    tensor_image_to_pil,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DDAD voxel-to-image projections.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "early" / "ddad_depth_6cam_mix_window_attn_early_ft.py"),
        help="Path to the DDAD early-fusion config.",
    )
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "visual" / "ddad_voxel_projection"),
        help="Directory to save the visualizations.",
    )
    parser.add_argument("--max-raw-points", type=int, default=4000)
    parser.add_argument("--max-voxel-points", type=int, default=2500)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    cfg = Config.fromfile(args.config)
    dataset_cfg_name = f"{args.split}_dataset"
    if dataset_cfg_name not in cfg:
        raise KeyError(f"Config does not contain {dataset_cfg_name}")

    dataset_cfg = copy.deepcopy(cfg[dataset_cfg_name])
    dataset_cfg["split"] = args.split
    dataset_cfg["return_lidar"] = True
    dataset = DATASETS.build(dataset_cfg)

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} out of range [0, {len(dataset) - 1}]")

    model_cfg = cfg.model
    projector = FusionProjector(
        patch_size=int(model_cfg.get("patch_size", 14)),
        voxel_size=tuple(model_cfg.get("voxel_size", (0.4, 0.4, 0.8))),
        point_cloud_range=tuple(model_cfg.get("point_cloud_range", (0.0, -40.0, -3.0, 80.0, 40.0, 3.0))),
        voxel_encoder_filters=tuple(model_cfg.get("voxel_encoder_filters", (128, 128))),
    )

    sample = dataset[args.sample_index]
    sequence_name = str(sample["sequence_name"])
    output_dir = Path(args.output_dir) / args.split / f"{args.sample_index:05d}_{sequence_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sample["images"]
    intrinsics = sample["intrinsics"]
    camera_to_world = sample["camera_to_world"]
    lidar_to_world = sample["lidar_to_world"]
    points = sample["points"]
    point_mask = sample["point_mask"]

    sequence_length, _, image_h, image_w = images.shape
    cam_num = int(model_cfg.get("cam_num", 6))
    frame_count = points.shape[0]
    if sequence_length != frame_count * cam_num:
        raise ValueError(
            f"Expected sequence_length == frame_count * cam_num, got "
            f"{sequence_length} vs {frame_count} * {cam_num}"
        )

    images = images.reshape(frame_count, cam_num, 3, image_h, image_w)
    intrinsics = intrinsics.reshape(frame_count, cam_num, 3, 3)
    camera_to_world = camera_to_world.reshape(frame_count, cam_num, 4, 4)

    print(
        f"[visualize_ddad_voxel_projection] split={args.split} sample={args.sample_index} "
        f"sequence={sequence_name} frames={frame_count} cam_num={cam_num}",
        flush=True,
    )

    row_images = []
    summary_lines = [
        f"sequence: {sequence_name}",
        f"sample_index: {args.sample_index}",
        f"image_hw: {image_h}x{image_w}",
        f"frame_count: {frame_count}",
        f"cam_num: {cam_num}",
        f"patch_size: {projector.patch_size}",
    ]

    for frame_idx in range(frame_count):
        frame_points = points[frame_idx : frame_idx + 1]
        frame_mask = point_mask[frame_idx : frame_idx + 1]
        frame_lidar_to_world = lidar_to_world[frame_idx : frame_idx + 1]

        voxel_features, voxel_coords = projector.encode_voxels(frame_points, frame_mask, frame_lidar_to_world)
        del voxel_features
        voxel_batch_ids = voxel_coords[:, 0].long()
        voxel_centers_local = projector.voxel_coords_to_centers(voxel_coords)
        voxel_centers_world = projector.local_voxel_centers_to_world(
            voxel_centers_local,
            voxel_batch_ids,
            frame_lidar_to_world,
        )

        valid_world_points = frame_points[0, frame_mask[0], :3]
        raw_bundle = projector.project_world_points(
            valid_world_points,
            intrinsics[frame_idx],
            camera_to_world[frame_idx],
            (image_h, image_w),
        )
        voxel_bundle = projector.project_world_points(
            voxel_centers_world,
            intrinsics[frame_idx],
            camera_to_world[frame_idx],
            (image_h, image_w),
        )

        summary_lines.append(
            f"frame {frame_idx}: raw_points={valid_world_points.shape[0]} voxels={voxel_centers_world.shape[0]}"
        )

        for cam_idx in range(cam_num):
            base = tensor_image_to_pil(images[frame_idx, cam_idx])
            cam_tag = f"frame {frame_idx} cam {cam_idx}"
            raw_panel = draw_points_overlay(
                base.copy(),
                raw_bundle.uv[:, cam_idx, :],
                raw_bundle.visible[:, cam_idx],
                raw_bundle.depth[:, cam_idx],
                title=f"{cam_tag} | raw points",
                point_radius=max(1, args.point_radius - 1),
                color=(0, 200, 255),
                max_points=args.max_raw_points,
                rng=rng,
            )
            voxel_panel = draw_points_overlay(
                base.copy(),
                voxel_bundle.uv[:, cam_idx, :],
                voxel_bundle.visible[:, cam_idx],
                voxel_bundle.depth[:, cam_idx],
                title=f"{cam_tag} | voxel centers",
                point_radius=args.point_radius,
                color=(255, 80, 80),
                max_points=args.max_voxel_points,
                rng=rng,
            )
            patch_panel = draw_patch_heatmap(
                base.copy(),
                voxel_bundle.patch_x[:, cam_idx],
                voxel_bundle.patch_y[:, cam_idx],
                voxel_bundle.visible[:, cam_idx],
                patch_size=projector.patch_size,
                title=f"{cam_tag} | patch bins",
            )
            triptych = concat_h([raw_panel, voxel_panel, patch_panel])
            row_images.append(triptych)
            triptych.save(output_dir / f"frame_{frame_idx:02d}_cam_{cam_idx:02d}.png")

    contact_sheet = make_contact_sheet(row_images, columns=1)
    contact_sheet_path = output_dir / "projection_overview.png"
    contact_sheet.save(contact_sheet_path)

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[visualize_ddad_voxel_projection] saved overview: {contact_sheet_path}", flush=True)
    print(f"[visualize_ddad_voxel_projection] saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
