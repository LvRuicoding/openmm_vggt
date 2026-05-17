#!/usr/bin/env python3
"""Copy GT voxel files into a separate folder per sequence."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_VOXEL_SEQUENCES = Path(
    "/home/dataset-local/lr/code/openmm_vggt/data/odemetry_voxels/sequences"
)
DEFAULT_TARGET_SEQUENCES = Path(
    "/home/dataset-local/lr/code/openmm_vggt/data/semantickitti/sequences"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy odometry GT voxel files to a separate folder under "
            "semantickitti/sequences/*."
        )
    )
    parser.add_argument("--source-sequences", type=Path, default=DEFAULT_VOXEL_SEQUENCES)
    parser.add_argument("--target-sequences", type=Path, default=DEFAULT_TARGET_SEQUENCES)
    parser.add_argument(
        "--target-folder",
        default="gt_voxels",
        help="Folder created under each target sequence for copied GT voxel files.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help="How voxel files are materialized in the target dataset.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip target files that already exist.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=(".bin", ".invalid", ".label", ".occluded"),
        help="Voxel file suffixes to copy.",
    )
    return parser.parse_args()


def prepare_destination(dst: Path, overwrite: bool) -> bool:
    if not dst.exists() and not dst.is_symlink():
        dst.parent.mkdir(parents=True, exist_ok=True)
        return True

    if not overwrite:
        return False

    if dst.is_dir() and not dst.is_symlink():
        shutil.rmtree(dst)
    else:
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    return True


def materialize(src: Path, dst: Path, mode: str, overwrite: bool) -> bool:
    if not prepare_destination(dst, overwrite):
        return False

    src = src.resolve()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return True


def main() -> None:
    args = parse_args()
    source_sequences = args.source_sequences.expanduser().resolve()
    target_sequences = args.target_sequences.expanduser()
    overwrite = not args.no_overwrite
    extensions = tuple(args.extensions)

    if not source_sequences.exists():
        raise FileNotFoundError(f"Source sequences not found: {source_sequences}")
    target_sequences.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_skipped = 0
    for src_seq in sorted(p for p in source_sequences.iterdir() if p.is_dir()):
        src_voxels = src_seq / "voxels"
        if not src_voxels.exists():
            continue

        dst_voxels = target_sequences / src_seq.name / args.target_folder
        written = 0
        skipped = 0
        for src_file in sorted(p for p in src_voxels.rglob("*") if p.is_file()):
            if extensions and src_file.suffix not in extensions:
                continue
            dst_file = dst_voxels / src_file.relative_to(src_voxels)
            if materialize(src_file, dst_file, args.mode, overwrite):
                written += 1
            else:
                skipped += 1

        total_written += written
        total_skipped += skipped
        print(f"{src_seq.name}: written={written} skipped={skipped}")

    print("\nDone")
    print(f"Source: {source_sequences}")
    print(f"Target: {target_sequences}")
    print(f"Target folder: {args.target_folder}")
    print(f"Mode: {args.mode}")
    print(f"Overwrite: {overwrite}")
    print(f"Total written: {total_written}")
    print(f"Total skipped: {total_skipped}")


if __name__ == "__main__":
    main()
