#!/usr/bin/env python3
"""Visualize how the early-fusion model projects voxels onto KITTI images.

The script mirrors the geometry used by ``mix_decoder_global_early``:
1. World-space LiDAR points are transformed into the LiDAR-local frame.
2. Points are voxelized with ``PCDetDynamicVoxelVFE``.
3. Voxel centers are transformed back to world coordinates.
4. Voxel centers are projected into each camera and mapped to image patches.

For every frame/camera pair, the script saves a 3-panel visualization:
  - raw LiDAR point projections
  - voxel-center projections
  - patch heatmap after voxel-to-patch assignment

Example:
  python tools/visualize_voxel_projection.py \
      --config configs/kitti_depth_stereo_mix_ft.py \
      --split val \
      --sample-index 0
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config
from mmengine.registry import DATASETS

import openmm_vggt  # noqa: F401
from openmm_vggt.models.pcdet_dynamic_voxel_vfe import PCDetDynamicVoxelVFE


@dataclass
class ProjectionBundle:
    uv: torch.Tensor
    depth: torch.Tensor
    visible: torch.Tensor
    patch_x: torch.Tensor
    patch_y: torch.Tensor


class FusionProjector:
    """Minimal geometry helper matching the fusion model's projection path."""

    def __init__(
        self,
        patch_size: int,
        voxel_size: tuple[float, float, float],
        point_cloud_range: tuple[float, float, float, float, float, float],
        voxel_encoder_filters: tuple[int, ...],
    ) -> None:
        self.patch_size = patch_size
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        grid_size = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
        ]
        self.voxel_encoder = PCDetDynamicVoxelVFE(
            num_point_features=4,
            voxel_size=voxel_size,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            use_norm=True,
            with_distance=False,
            use_absolute_xyz=True,
            num_filters=voxel_encoder_filters,
        )

    def world_points_to_local(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        lidar_to_world: torch.Tensor,
    ) -> torch.Tensor:
        valid_indices = torch.nonzero(point_mask, as_tuple=False)
        if valid_indices.shape[0] == 0:
            return points.new_zeros((0, 5))

        batch_ids = valid_indices[:, 0]
        selected_points = points[batch_ids, valid_indices[:, 1]]
        xyz1 = torch.cat([selected_points[:, :3], torch.ones_like(selected_points[:, :1])], dim=-1)
        world_to_lidar = torch.inverse(lidar_to_world[batch_ids])
        local_xyz = torch.matmul(world_to_lidar, xyz1.unsqueeze(-1)).squeeze(-1)[..., :3]

        local_points = selected_points.clone()
        local_points[:, :3] = local_xyz
        return torch.cat([batch_ids.to(points.dtype).unsqueeze(-1), local_points], dim=-1)

    def encode_voxels(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        lidar_to_world: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_points = self.world_points_to_local(points, point_mask, lidar_to_world)
        return self.voxel_encoder(flat_points)

    def voxel_coords_to_centers(self, voxel_coords: torch.Tensor) -> torch.Tensor:
        xyz_index = voxel_coords[:, [3, 2, 1]].to(self.voxel_size.dtype)
        return self.point_cloud_range[:3] + (xyz_index + 0.5) * self.voxel_size

    def local_voxel_centers_to_world(
        self,
        voxel_centers: torch.Tensor,
        voxel_batch_ids: torch.Tensor,
        lidar_to_world: torch.Tensor,
    ) -> torch.Tensor:
        if voxel_centers.shape[0] == 0:
            return voxel_centers
        xyz1 = torch.cat([voxel_centers, torch.ones_like(voxel_centers[..., :1])], dim=-1)
        return torch.matmul(lidar_to_world[voxel_batch_ids], xyz1.unsqueeze(-1)).squeeze(-1)[..., :3]

    def project_world_points(
        self,
        world_points: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> ProjectionBundle:
        image_h, image_w = image_hw
        if world_points.shape[0] == 0:
            empty = world_points.new_zeros((0, intrinsics.shape[0]))
            empty_long = torch.zeros((0, intrinsics.shape[0]), dtype=torch.long, device=world_points.device)
            empty_bool = torch.zeros((0, intrinsics.shape[0]), dtype=torch.bool, device=world_points.device)
            return ProjectionBundle(
                uv=world_points.new_zeros((0, intrinsics.shape[0], 2)),
                depth=empty,
                visible=empty_bool,
                patch_x=empty_long,
                patch_y=empty_long,
            )

        xyz1 = torch.cat([world_points, torch.ones_like(world_points[:, :1])], dim=-1)
        world_to_camera = torch.inverse(camera_to_world)
        cam_coords = torch.matmul(world_to_camera, xyz1[:, None, :, None]).squeeze(-1)[..., :3]
        img_coords = torch.matmul(intrinsics, cam_coords.unsqueeze(-1)).squeeze(-1)
        depth = img_coords[..., 2]
        safe_depth = torch.clamp(depth, min=1e-5)
        u = img_coords[..., 0] / safe_depth
        v = img_coords[..., 1] / safe_depth
        patch_x = torch.floor(u / self.patch_size).to(torch.long)
        patch_y = torch.floor(v / self.patch_size).to(torch.long)
        visible = (
            (depth > 1e-5)
            & (u >= 0)
            & (u < image_w)
            & (v >= 0)
            & (v < image_h)
        )
        return ProjectionBundle(
            uv=torch.stack([u, v], dim=-1),
            depth=depth,
            visible=visible,
            patch_x=patch_x,
            patch_y=patch_y,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize voxel-to-image projection for the fusion model.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "kitti_depth_stereo_mix_ft.py"),
        help="Path to the fusion config.",
    )
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "visualizetions"),
        help="Directory to save the visualizations.",
    )
    parser.add_argument("--max-raw-points", type=int, default=4000)
    parser.add_argument("--max-voxel-points", type=int, default=2500)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def sample_indices(total: int, limit: int, rng: random.Random) -> list[int]:
    if total <= limit:
        return list(range(total))
    return sorted(rng.sample(range(total), limit))


def draw_points_overlay(
    image: Image.Image,
    uv: torch.Tensor,
    visible: torch.Tensor,
    depth: torch.Tensor,
    title: str,
    point_radius: int,
    color: tuple[int, int, int],
    max_points: int,
    rng: random.Random,
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    visible_indices = torch.nonzero(visible, as_tuple=False).flatten().tolist()
    visible_indices = sample_indices(len(visible_indices), max_points, rng)
    selected = [torch.nonzero(visible, as_tuple=False).flatten().tolist()[idx] for idx in visible_indices]

    if selected:
        depth_values = depth[selected]
        depth_min = float(depth_values.min())
        depth_max = float(depth_values.max())
    else:
        depth_min = 0.0
        depth_max = 1.0

    for idx in selected:
        x = float(uv[idx, 0])
        y = float(uv[idx, 1])
        alpha = 210
        if depth_max > depth_min:
            norm = (float(depth[idx]) - depth_min) / (depth_max - depth_min)
            alpha = int(120 + 120 * (1.0 - norm))
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            fill=(color[0], color[1], color[2], alpha),
            outline=None,
        )

    blended = Image.alpha_composite(canvas, overlay).convert("RGB")
    annotate_image(
        blended,
        [
            title,
            f"visible: {int(visible.sum().item())}",
            f"drawn: {len(selected)}",
        ],
    )
    return blended


def draw_patch_heatmap(
    image: Image.Image,
    patch_x: torch.Tensor,
    patch_y: torch.Tensor,
    visible: torch.Tensor,
    patch_size: int,
    title: str,
) -> Image.Image:
    image_h, image_w = image.height, image.width
    patch_h = image_h // patch_size
    patch_w = image_w // patch_size
    counts = torch.zeros((patch_h, patch_w), dtype=torch.int32)
    visible_patch_x = patch_x[visible]
    visible_patch_y = patch_y[visible]
    for px, py in zip(visible_patch_x.tolist(), visible_patch_y.tolist()):
        counts[py, px] += 1

    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    max_count = int(counts.max().item()) if counts.numel() > 0 else 0
    occupied = int((counts > 0).sum().item())

    for py in range(patch_h):
        for px in range(patch_w):
            count = int(counts[py, px].item())
            if count <= 0:
                continue
            strength = math.log1p(count) / math.log1p(max_count) if max_count > 0 else 0.0
            fill = (
                255,
                int(180 * (1.0 - strength)),
                0,
                int(60 + 140 * strength),
            )
            x0 = px * patch_size
            y0 = py * patch_size
            x1 = x0 + patch_size
            y1 = y0 + patch_size
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(255, 255, 255, 48))

    blended = Image.alpha_composite(canvas, overlay).convert("RGB")
    annotate_image(
        blended,
        [
            title,
            f"occupied patches: {occupied}",
            f"max voxels/patch: {max_count}",
        ],
    )
    return blended


def annotate_image(image: Image.Image, lines: Iterable[str]) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_lines = list(lines)
    if not text_lines:
        return
    heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in text_lines]
    box_h = sum(heights) + 4 * len(text_lines) + 8
    max_w = max(font.getbbox(line)[2] - font.getbbox(line)[0] for line in text_lines) + 12
    draw.rectangle((8, 8, 8 + max_w, 8 + box_h), fill=(0, 0, 0))
    y = 12
    for line in text_lines:
        draw.text((14, y), line, fill=(255, 255, 255), font=font)
        y += (font.getbbox(line)[3] - font.getbbox(line)[1]) + 4


def make_contact_sheet(images: list[Image.Image], columns: int, bg_color: tuple[int, int, int] = (24, 24, 24)) -> Image.Image:
    if not images:
        raise ValueError("No images to combine.")
    width, height = images[0].size
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * width, rows * height), color=bg_color)
    for idx, image in enumerate(images):
        x = (idx % columns) * width
        y = (idx // columns) * height
        sheet.paste(image, (x, y))
    return sheet


def concat_h(images: list[Image.Image], spacing: int = 8, bg_color: tuple[int, int, int] = (24, 24, 24)) -> Image.Image:
    width = sum(image.width for image in images) + spacing * (len(images) - 1)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), color=bg_color)
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width + spacing
    return canvas


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
    sequence_name = sample["sequence_name"]
    output_dir = Path(args.output_dir) / f"{args.sample_index:05d}_{sequence_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sample["images"]
    intrinsics = sample["intrinsics"]
    camera_to_world = sample["camera_to_world"]
    lidar_to_world = sample["lidar_to_world"]
    points = sample["points"]
    point_mask = sample["point_mask"]

    sequence_length, _, image_h, image_w = images.shape
    cam_num = int(model_cfg.get("cam_num", 2))
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
        f"[visualize_voxel_projection] split={args.split} sample={args.sample_index} "
        f"sequence={sequence_name} frames={frame_count} cam_num={cam_num}",
        flush=True,
    )

    row_images: list[Image.Image] = []
    summary_lines = [
        f"sequence: {sequence_name}",
        f"sample_index: {args.sample_index}",
        f"image_hw: {image_h}x{image_w}",
        f"patch_size: {projector.patch_size}",
        f"voxel_size: {tuple(float(x) for x in projector.voxel_size.tolist())}",
        f"point_cloud_range: {tuple(float(x) for x in projector.point_cloud_range.tolist())}",
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
            raw_panel = draw_points_overlay(
                base.copy(),
                raw_bundle.uv[:, cam_idx, :],
                raw_bundle.visible[:, cam_idx],
                raw_bundle.depth[:, cam_idx],
                title=f"frame {frame_idx} cam {cam_idx} | raw points",
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
                title=f"frame {frame_idx} cam {cam_idx} | voxel centers",
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
                title=f"frame {frame_idx} cam {cam_idx} | patch bins",
            )
            triptych = concat_h([raw_panel, voxel_panel, patch_panel])
            row_images.append(triptych)

            panel_path = output_dir / f"frame_{frame_idx:02d}_cam_{cam_idx:02d}.png"
            triptych.save(panel_path)

    contact_sheet = make_contact_sheet(row_images, columns=1)
    contact_sheet_path = output_dir / "projection_overview.png"
    contact_sheet.save(contact_sheet_path)

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[visualize_voxel_projection] saved overview: {contact_sheet_path}", flush=True)
    print(f"[visualize_voxel_projection] saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
