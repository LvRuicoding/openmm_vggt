#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mmengine.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openmm_vggt.datasets.kitti_semantic_occ_depth import KITTISemanticOccupancyDepthDataset


def _dataset_cfg(cfg: Config, split: str, depth_root: str | None) -> dict[str, Any]:
    source = cfg.train_dataset if split == "train" else cfg.val_dataset
    dataset_cfg = dict(source)
    dataset_cfg.pop("type", None)
    dataset_cfg["require_depth_gt"] = False
    if depth_root is not None:
        dataset_cfg["depth_root"] = depth_root
    elif "depth_root" not in dataset_cfg:
        cfg_depth_root = cfg.get("depth_root", None)
        if cfg_depth_root is None:
            raise ValueError("depth_root is not set in the dataset config or top-level config.")
        dataset_cfg["depth_root"] = cfg_depth_root
    return dataset_cfg


def _count_missing(
    cfg: Config,
    split: str,
    cameras: tuple[str, ...],
    depth_root: str | None,
    list_limit: int,
) -> dict[str, Any]:
    dataset = KITTISemanticOccupancyDepthDataset(**_dataset_cfg(cfg, split, depth_root))

    missing_by_camera: Counter[str] = Counter()
    per_sequence: dict[str, Counter[str]] = defaultdict(Counter)
    missing_examples: list[dict[str, Any]] = []
    missing_any = 0
    missing_all = 0

    for sample_index, (record_idx, last_idx) in enumerate(dataset.samples):
        record = dataset.records[record_idx]
        semantic_frame_id = record.frame_ids[last_idx]
        raw_frame_id = record.raw_frame_ids[semantic_frame_id]
        missing_cameras = [
            camera_name
            for camera_name in cameras
            if dataset._get_depth_gt_path(record, semantic_frame_id, camera_name) is None
        ]

        seq_stats = per_sequence[record.sequence_id]
        seq_stats["samples"] += 1
        if missing_cameras:
            missing_any += 1
            seq_stats["missing_any"] += 1
            for camera_name in missing_cameras:
                missing_by_camera[camera_name] += 1
                seq_stats[f"missing_{camera_name}"] += 1
            if len(missing_cameras) == len(cameras):
                missing_all += 1
                seq_stats["missing_all"] += 1
            if len(missing_examples) < list_limit:
                missing_examples.append(
                    {
                        "sample_index": sample_index,
                        "sequence_id": record.sequence_id,
                        "semantic_frame_id": semantic_frame_id,
                        "raw_drive": record.raw_drive_name,
                        "raw_frame_id": raw_frame_id,
                        "missing_cameras": missing_cameras,
                    }
                )

    total = len(dataset.samples)
    complete = total - missing_any
    return {
        "split": split,
        "cameras": list(cameras),
        "samples_checked": total,
        "complete_samples": complete,
        "missing_any_camera": missing_any,
        "missing_all_selected_cameras": missing_all,
        "missing_by_camera": dict(sorted(missing_by_camera.items())),
        "missing_any_ratio": missing_any / total if total else 0.0,
        "per_sequence": {
            sequence_id: dict(stats)
            for sequence_id, stats in sorted(per_sequence.items())
        },
        "examples": missing_examples,
    }


def _print_report(result: dict[str, Any]) -> None:
    print(f"split: {result['split']}")
    print(f"cameras: {', '.join(result['cameras'])}")
    print(f"samples_checked: {result['samples_checked']}")
    print(f"complete_samples: {result['complete_samples']}")
    print(f"missing_any_camera: {result['missing_any_camera']} ({result['missing_any_ratio']:.4%})")
    print(f"missing_all_selected_cameras: {result['missing_all_selected_cameras']}")
    for camera_name, count in result["missing_by_camera"].items():
        print(f"missing_{camera_name}: {count}")

    print("\nper_sequence:")
    for sequence_id, stats in result["per_sequence"].items():
        print(
            f"  {sequence_id}: samples={stats.get('samples', 0)} "
            f"missing_any={stats.get('missing_any', 0)} "
            f"missing_all={stats.get('missing_all', 0)}"
        )

    if result["examples"]:
        print("\nexamples:")
        for item in result["examples"]:
            print(
                "  "
                f"sample_index={item['sample_index']} "
                f"seq={item['sequence_id']} "
                f"semantic_frame={item['semantic_frame_id']} "
                f"raw_drive={item['raw_drive']} "
                f"raw_frame={item['raw_frame_id']} "
                f"missing={','.join(item['missing_cameras'])}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count SemanticKITTI occupancy samples whose KITTI depth GT is missing."
    )
    parser.add_argument("config", type=str, help="Occupancy config path.")
    parser.add_argument("--split", choices=("train", "val", "all"), default="train")
    parser.add_argument("--depth-root", type=str, default=None, help="Override KITTI depth completion root.")
    parser.add_argument(
        "--cameras",
        type=str,
        default="image_02,image_03",
        help="Comma-separated cameras to require, e.g. image_02 or image_02,image_03.",
    )
    parser.add_argument("--list-missing", type=int, default=20, help="Number of missing examples to print per split.")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to write JSON results.")
    args = parser.parse_args()

    cameras = tuple(camera.strip() for camera in args.cameras.split(",") if camera.strip())
    if not cameras:
        raise ValueError("--cameras must contain at least one camera.")

    cfg = Config.fromfile(args.config)
    splits = ("train", "val") if args.split == "all" else (args.split,)
    results = [
        _count_missing(cfg, split, cameras, args.depth_root, max(int(args.list_missing), 0))
        for split in splits
    ]

    for idx, result in enumerate(results):
        if idx > 0:
            print("\n" + "=" * 80 + "\n")
        _print_report(result)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote: {output_path}")


if __name__ == "__main__":
    main()
