#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openmm-vggt")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_JSONL_NAME = "loss_steps.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot loss curves from one or more JSONL files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "JSONL file paths or experiment directories. "
            f"If a directory is given, `{DEFAULT_JSONL_NAME}` is used."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output image path. Defaults to ./loss_curve.png.",
    )
    parser.add_argument(
        "--x-key",
        type=str,
        default="step",
        help="Key used for the x-axis. Supports dotted paths such as `metrics.step`.",
    )
    parser.add_argument(
        "--y-key",
        type=str,
        default="loss",
        help="Key used for the y-axis. Supports dotted paths such as `metrics.loss`.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for each input curve. Must match the number of inputs.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Training Loss",
        help="Figure title.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving-average window size. Set to 1 to disable smoothing.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Maximum plotted points per curve after optional smoothing.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(10.0, 6.0),
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Line alpha.",
    )
    return parser.parse_args()


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_dir():
        path = path / DEFAULT_JSONL_NAME
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    return path


def get_nested_value(record: dict, key: str):
    current = record
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(key)
        current = current[part]
    return current


def load_xy_series(path: Path, x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []

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

            try:
                x_value = get_nested_value(record, x_key)
                y_value = get_nested_value(record, y_key)
            except KeyError as exc:
                raise KeyError(
                    f"{path}:{line_idx} is missing key `{exc.args[0]}`"
                ) from exc

            xs.append(float(x_value))
            ys.append(float(y_value))

    if not xs:
        raise ValueError(f"No valid `{x_key}` / `{y_key}` points found in {path}")

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    order = np.argsort(x_arr)
    return x_arr[order], y_arr[order]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size <= 1:
        return values
    window = min(window, values.size)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid")


def smooth_x(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size <= 1:
        return x
    window = min(window, x.size)
    return x[window - 1 :]


def maybe_downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or x.size <= max_points:
        return x, y
    indices = np.linspace(0, x.size - 1, num=max_points, dtype=np.int64)
    return x[indices], y[indices]


def default_label(path: Path) -> str:
    if path.name == DEFAULT_JSONL_NAME and path.parent.name:
        return path.parent.name
    return path.stem


def plot_curves(
    paths: Sequence[Path],
    labels: Sequence[str],
    args: argparse.Namespace,
) -> Path:
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else Path.cwd().resolve() / "loss_curve.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=tuple(args.figsize))

    for path, label in zip(paths, labels):
        x, y = load_xy_series(path, x_key=args.x_key, y_key=args.y_key)
        y_plot = moving_average(y, args.smooth_window)
        x_plot = smooth_x(x, args.smooth_window)
        x_plot, y_plot = maybe_downsample(x_plot, y_plot, args.max_points)
        plt.plot(x_plot, y_plot, linewidth=1.6, alpha=args.alpha, label=label)

    plt.title(args.title)
    plt.xlabel(args.x_key)
    plt.ylabel(args.y_key)
    plt.grid(True, linestyle="--", alpha=0.3)
    if len(labels) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi)
    plt.close()
    return output_path


def main() -> None:
    args = parse_args()
    paths = [resolve_input_path(item) for item in args.inputs]

    if args.labels is not None and len(args.labels) != len(paths):
        raise ValueError("`--labels` count must match the number of input paths.")

    labels = args.labels or [default_label(path) for path in paths]
    output_path = plot_curves(paths=paths, labels=labels, args=args)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
