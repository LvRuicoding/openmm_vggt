#!/usr/bin/env python3
"""Merge KITTI Odometry images/calibration with SemanticKITTI labels.

Output layout:

  semantickitti/
    poses/
    sequences/
      00/
        calib.txt
        times.txt
        image_0/
        image_1/
        image_2/
        image_3/
        velodyne/
        voxels/

By default the script creates symlinks to avoid duplicating the large image
folders. Use ``--mode copy`` if you need a physically copied dataset.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_ODOMETRY_ROOT = Path(
    "/home/dataset-local/lr/code/openmm_vggt/data/"
    "OpenDataLab___KITTI_Odometry_2012/dataset"
)
DEFAULT_SEMANTIC_ROOT = Path(
    "/home/dataset-local/lr/code/openmm_vggt/data/kitti_semantic"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/dataset-local/lr/code/openmm_vggt/data/semantickitti"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge KITTI Odometry and SemanticKITTI-style labels."
    )
    parser.add_argument("--odometry-root", type=Path, default=DEFAULT_ODOMETRY_ROOT)
    parser.add_argument("--semantic-root", type=Path, default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How files are materialized in the output dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files/symlinks.",
    )
    parser.add_argument(
        "--image-dirs",
        nargs="+",
        default=("image_0", "image_1", "image_2", "image_3"),
        help="Image folders copied from odometry sequences.",
    )
    parser.add_argument(
        "--skip-times",
        action="store_true",
        help="Do not place times.txt under each output sequence.",
    )
    parser.add_argument(
        "--skip-velodyne",
        action="store_true",
        help="Do not place velodyne scans under each output sequence.",
    )
    return parser.parse_args()


def ensure_source(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def is_same_symlink(dst: Path, src: Path) -> bool:
    return dst.is_symlink() and Path(os.readlink(dst)) == src


def prepare_destination(dst: Path, src: Path, overwrite: bool) -> bool:
    """Return True when dst should be written."""
    if not dst.exists() and not dst.is_symlink():
        dst.parent.mkdir(parents=True, exist_ok=True)
        return True

    if is_same_symlink(dst, src):
        return False

    if not overwrite:
        return False

    if dst.is_dir() and not dst.is_symlink():
        shutil.rmtree(dst)
    else:
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    return True


def materialize_file(src: Path, dst: Path, mode: str, overwrite: bool) -> bool:
    src = src.resolve()
    if not prepare_destination(dst, src, overwrite):
        return False

    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return True


def materialize_tree(src_dir: Path, dst_dir: Path, mode: str, overwrite: bool) -> int:
    if not src_dir.exists():
        return 0

    written = 0
    for src in sorted(p for p in src_dir.rglob("*") if p.is_file()):
        rel = src.relative_to(src_dir)
        if materialize_file(src, dst_dir / rel, mode, overwrite):
            written += 1
    return written


def sequence_ids(odometry_root: Path, semantic_root: Path) -> list[str]:
    seqs: set[str] = set()
    odo_seq_root = odometry_root / "sequences"
    if odo_seq_root.exists():
        seqs.update(p.name for p in odo_seq_root.iterdir() if p.is_dir())
    if semantic_root.exists():
        seqs.update(p.name for p in semantic_root.iterdir() if p.is_dir())
    return sorted(seq for seq in seqs if seq.isdigit())


def copy_poses(odometry_root: Path, output_root: Path, mode: str, overwrite: bool) -> int:
    poses_dir = odometry_root / "poses"
    if not poses_dir.exists():
        return 0

    written = 0
    for pose_file in sorted(poses_dir.glob("*.txt")):
        if materialize_file(pose_file, output_root / "poses" / pose_file.name, mode, overwrite):
            written += 1
    return written


def merge_sequence(
    seq: str,
    odometry_root: Path,
    semantic_root: Path,
    output_root: Path,
    mode: str,
    overwrite: bool,
    image_dirs: tuple[str, ...],
    include_times: bool,
    include_velodyne: bool,
) -> dict[str, int]:
    odo_seq = odometry_root / "sequences" / seq
    sem_seq = semantic_root / seq
    out_seq = output_root / "sequences" / seq
    stats = {
        "calib": 0,
        "times": 0,
        "voxels": 0,
        "velodyne": 0,
    }
    for image_dir in image_dirs:
        stats[image_dir] = 0

    calib_src = odo_seq / "calib.txt"
    if not calib_src.exists():
        calib_src = sem_seq / "calib.txt"
    if calib_src.exists():
        stats["calib"] = int(
            materialize_file(calib_src, out_seq / "calib.txt", mode, overwrite)
        )

    if include_times and (odo_seq / "times.txt").exists():
        stats["times"] = int(
            materialize_file(odo_seq / "times.txt", out_seq / "times.txt", mode, overwrite)
        )

    for image_dir in image_dirs:
        stats[image_dir] = materialize_tree(
            odo_seq / image_dir, out_seq / image_dir, mode, overwrite
        )

    voxel_src = sem_seq / "voxels"
    label_src = sem_seq / "labels"
    if voxel_src.exists():
        stats["voxels"] = materialize_tree(voxel_src, out_seq / "voxels", mode, overwrite)
    elif label_src.exists():
        out_voxels = out_seq / "voxels"
        for label_file in sorted(label_src.glob("*.label")):
            if materialize_file(label_file, out_voxels / label_file.name, mode, overwrite):
                stats["voxels"] += 1

    if include_velodyne:
        velodyne_src = sem_seq / "velodyne"
        if not velodyne_src.exists():
            velodyne_src = odo_seq / "velodyne"
        stats["velodyne"] = materialize_tree(
            velodyne_src, out_seq / "velodyne", mode, overwrite
        )

    return stats


def main() -> None:
    args = parse_args()
    odometry_root = args.odometry_root.expanduser().resolve()
    semantic_root = args.semantic_root.expanduser().resolve()
    output_root = args.output_root.expanduser()

    ensure_source(odometry_root, "KITTI odometry root")
    ensure_source(semantic_root, "SemanticKITTI root")
    output_root.mkdir(parents=True, exist_ok=True)

    pose_count = copy_poses(odometry_root, output_root, args.mode, args.overwrite)
    totals = {
        "sequences": 0,
        "calib": 0,
        "times": 0,
        "voxels": 0,
        "velodyne": 0,
    }
    image_dirs = tuple(args.image_dirs)
    for image_dir in image_dirs:
        totals[image_dir] = 0

    for seq in sequence_ids(odometry_root, semantic_root):
        stats = merge_sequence(
            seq=seq,
            odometry_root=odometry_root,
            semantic_root=semantic_root,
            output_root=output_root,
            mode=args.mode,
            overwrite=args.overwrite,
            image_dirs=image_dirs,
            include_times=not args.skip_times,
            include_velodyne=not args.skip_velodyne,
        )
        totals["sequences"] += 1
        for key, value in stats.items():
            totals[key] += value
        print(
            f"{seq}: calib={stats['calib']} times={stats['times']} "
            + " ".join(f"{d}={stats[d]}" for d in image_dirs)
            + f" velodyne={stats['velodyne']} voxels={stats['voxels']}"
        )

    print("\nDone")
    print(f"Output: {output_root}")
    print(f"Mode: {args.mode}")
    print(f"Poses written: {pose_count}")
    print(
        "Totals: "
        f"sequences={totals['sequences']} calib={totals['calib']} "
        f"times={totals['times']} "
        + " ".join(f"{d}={totals[d]}" for d in image_dirs)
        + f" velodyne={totals['velodyne']} voxels={totals['voxels']}"
    )


if __name__ == "__main__":
    main()
