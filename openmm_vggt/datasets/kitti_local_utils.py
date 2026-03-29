#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_IMAGE_HW = (280, 518)
EARTH_RADIUS = 6378137.0


def parse_calib_file(calib_path: Path) -> Dict[str, np.ndarray]:
    values: Dict[str, np.ndarray] = {}
    with calib_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            raw_value = raw_value.strip()
            if not raw_value:
                continue
            try:
                array = np.fromstring(raw_value, sep=" ", dtype=np.float64)
            except ValueError:
                continue
            if array.size > 0:
                values[key] = array
    return values


def build_se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.reshape(3, 3)
    transform[:3, 3] = translation.reshape(3)
    return transform


def rotx(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def roty(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def rotz(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def resolve_kitti_depth_root(root: str | Path) -> Path:
    root = Path(root)
    if (root / "KITTI_depth_completion").is_dir():
        return root / "KITTI_depth_completion"
    return root


def camera_name_to_id(camera: str) -> str:
    camera_id = camera.split("_")[-1]
    if camera_id not in {"00", "01", "02", "03"}:
        raise ValueError(f"Unsupported KITTI camera name: {camera}")
    return camera_id


def load_rectified_intrinsics(calib_path: Path, camera: str) -> np.ndarray:
    camera_id = camera_name_to_id(camera)
    calib = parse_calib_file(calib_path)
    key = f"P_rect_{camera_id}"
    if key not in calib:
        raise KeyError(f"Missing {key} in calibration file: {calib_path}")
    return calib[key].reshape(3, 4)[:, :3].astype(np.float32)


def load_camera_transform_imu_to_rectified(camera_root: Path, camera: str) -> np.ndarray:
    camera_id = camera_name_to_id(camera)
    cam_calib = parse_calib_file(camera_root / "calib_cam_to_cam.txt")
    velo_calib = parse_calib_file(camera_root / "calib_velo_to_cam.txt")
    imu_calib = parse_calib_file(camera_root / "calib_imu_to_velo.txt")

    t_velo_imu = build_se3(imu_calib["R"], imu_calib["T"])
    t_cam0_velo = build_se3(velo_calib["R"], velo_calib["T"])
    t_cam_cam0 = build_se3(cam_calib[f"R_{camera_id}"], cam_calib[f"T_{camera_id}"])
    t_rect_cam = build_se3(cam_calib[f"R_rect_{camera_id}"], np.zeros(3, dtype=np.float64))

    t_rect_imu = t_rect_cam @ t_cam_cam0 @ t_cam0_velo @ t_velo_imu
    return t_rect_imu.astype(np.float64)


def resize_intrinsics(intrinsics: np.ndarray, orig_hw: Tuple[int, int], out_hw: Tuple[int, int]) -> np.ndarray:
    orig_h, orig_w = orig_hw
    out_h, out_w = out_hw
    intrinsics = intrinsics.copy().astype(np.float32)
    intrinsics[0, 0] *= out_w / float(orig_w)
    intrinsics[0, 2] *= out_w / float(orig_w)
    intrinsics[1, 1] *= out_h / float(orig_h)
    intrinsics[1, 2] *= out_h / float(orig_h)
    return intrinsics


def crop_intrinsics(intrinsics: np.ndarray, left: int, top: int) -> np.ndarray:
    intrinsics = intrinsics.copy().astype(np.float32)
    intrinsics[0, 2] -= float(left)
    intrinsics[1, 2] -= float(top)
    return intrinsics


def load_selection_intrinsics(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32).reshape(3, 3)


def infer_crop_box(
    raw_intrinsics: np.ndarray,
    cropped_intrinsics: np.ndarray,
    raw_hw: Tuple[int, int],
    cropped_hw: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    raw_h, raw_w = raw_hw
    crop_h, crop_w = cropped_hw
    left = int(round(float(raw_intrinsics[0, 2] - cropped_intrinsics[0, 2])))
    top = int(round(float(raw_intrinsics[1, 2] - cropped_intrinsics[1, 2])))
    right = min(left + crop_w, raw_w)
    bottom = min(top + crop_h, raw_h)
    return left, top, right, bottom


def read_oxts_packet(oxts_path: Path) -> Tuple[float, float, float, float, float, float]:
    values = np.loadtxt(oxts_path, dtype=np.float64)
    if values.ndim != 1 or values.size < 6:
        raise ValueError(f"Unexpected OXTS packet format: {oxts_path}")
    return tuple(float(v) for v in values[:6])


def oxts_to_pose(
    packet: Tuple[float, float, float, float, float, float],
    scale: float,
    origin: np.ndarray,
) -> np.ndarray:
    lat, lon, alt, roll, pitch, yaw = packet
    tx = scale * lon * math.pi * EARTH_RADIUS / 180.0
    ty = scale * EARTH_RADIUS * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    tz = alt
    translation = np.array([tx, ty, tz], dtype=np.float64) - origin
    rotation = rotz(yaw) @ roty(pitch) @ rotx(roll)
    return build_se3(rotation, translation)


def load_oxts_poses(drive_root: Path) -> Dict[str, np.ndarray]:
    oxts_root = drive_root / "oxts" / "data"
    if not oxts_root.is_dir():
        raise FileNotFoundError(f"Missing OXTS directory: {oxts_root}")

    oxts_files = sorted(oxts_root.glob("*.txt"))
    if not oxts_files:
        raise FileNotFoundError(f"No OXTS files found under {oxts_root}")

    first_packet = read_oxts_packet(oxts_files[0])
    lat0 = first_packet[0]
    scale = math.cos(lat0 * math.pi / 180.0)
    tx0 = scale * first_packet[1] * math.pi * EARTH_RADIUS / 180.0
    ty0 = scale * EARTH_RADIUS * math.log(math.tan((90.0 + first_packet[0]) * math.pi / 360.0))
    tz0 = first_packet[2]
    origin = np.array([tx0, ty0, tz0], dtype=np.float64)

    poses: Dict[str, np.ndarray] = {}
    for oxts_path in oxts_files:
        poses[oxts_path.stem] = oxts_to_pose(read_oxts_packet(oxts_path), scale=scale, origin=origin)
    return poses


def preprocess_rgb_like_demo(
    image_path: Path,
    out_hw: Tuple[int, int],
    crop_box: Tuple[int, int, int, int] | None = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    out_h, out_w = out_hw
    image = Image.open(image_path)
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    image = image.convert("RGB")
    if crop_box is not None:
        image = image.crop(crop_box)
    orig_w, orig_h = image.size
    image = image.resize((out_w, out_h), Image.Resampling.BICUBIC)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    return image_tensor, (orig_h, orig_w)


def preprocess_depth_png(
    depth_path: Path,
    out_hw: Tuple[int, int],
    crop_box: Tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    out_h, out_w = out_hw
    depth_png = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_png is None:
        raise FileNotFoundError(depth_path)
    if crop_box is not None:
        left, top, right, bottom = crop_box
        depth_png = depth_png[top:bottom, left:right]
    depth_png = cv2.resize(depth_png, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    depth = depth_png.astype(np.float32) / 256.0
    depth[depth_png == 0] = -1.0
    return torch.from_numpy(depth)
