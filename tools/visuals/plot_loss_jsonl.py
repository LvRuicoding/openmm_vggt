#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openmm-vggt")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a loss curve from one JSONL file.")
    parser.add_argument("loss_jsonl", type=str, help="Path to loss_steps.jsonl.")
    parser.add_argument("output_image", type=str, help="Path to the output image file.")
    return parser.parse_args()


def load_loss_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps: list[float] = []
    losses: list[float] = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_idx} is not valid JSON: {exc}") from exc

            if not isinstance(record, dict):
                continue
            if "step" not in record or "loss" not in record:
                raise KeyError(f"{path}:{line_idx} is missing `step` or `loss`.")

            steps.append(float(record["step"]))
            losses.append(float(record["loss"]))

    if not steps:
        raise ValueError(f"No valid step/loss points found in {path}")

    step_arr = np.asarray(steps, dtype=np.float64)
    loss_arr = np.asarray(losses, dtype=np.float64)
    order = np.argsort(step_arr)
    return step_arr[order], loss_arr[order]


def plot_loss_curve(loss_jsonl: Path, output_image: Path) -> Path:
    if not loss_jsonl.is_file():
        raise FileNotFoundError(f"JSONL file not found: {loss_jsonl}")

    if output_image.exists() and output_image.is_dir():
        output_image = output_image / "loss_curve.png"
    elif output_image.suffix == "":
        output_image = output_image / "loss_curve.png"

    output_image.parent.mkdir(parents=True, exist_ok=True)
    steps, losses = load_loss_series(loss_jsonl)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, linewidth=1.6, alpha=0.9)
    plt.title("Training Loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_image, dpi=160)
    plt.close()
    return output_image


def main() -> None:
    args = parse_args()
    loss_jsonl = Path(args.loss_jsonl).expanduser().resolve()
    output_image = Path(args.output_image).expanduser().resolve()
    saved_path = plot_loss_curve(loss_jsonl, output_image)
    print(f"Saved plot to: {saved_path}")


if __name__ == "__main__":
    main()
