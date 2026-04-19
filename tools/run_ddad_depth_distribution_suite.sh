#!/usr/bin/env bash
# Usage:
#   bash tools/run_ddad_depth_distribution_suite.sh
#   bash tools/run_ddad_depth_distribution_suite.sh <config> [checkpoint]
#
# This script runs four DDAD depth-distribution jobs sequentially with 4 GPUs:
#   1) all predicted pixels
#   2) all predicted pixels with predicted depth < 50m
#   3) predictions at valid GT evaluation pixels
#   4) predictions at valid GT evaluation pixels with predicted depth < 50m
#
# Outputs are saved under:
#   /home/dataset-local/lr/code/openmm_vggt/visual/
# with one subdirectory per distribution.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${1:-${REPO_ROOT}/configs/early/ddad_depth_6cam_mix_window_attn_early_ft.py}"
CHECKPOINT_PATH="${2:-}"
OUTPUT_ROOT="${REPO_ROOT}/visual"
STAT_SCRIPT="${REPO_ROOT}/tools/stat_ddad_depth_prediction_distribution.py"

COMMON_ARGS=("${CONFIG_PATH}")
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  COMMON_ARGS+=("--checkpoint" "${CHECKPOINT_PATH}")
fi

run_case() {
  local case_name="$1"
  shift
  local case_output_dir="${OUTPUT_ROOT}/${case_name}"
  echo "[run_ddad_depth_distribution_suite] case=${case_name}"
  torchrun --nproc_per_node=4 "${STAT_SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --output-dir "${case_output_dir}" \
    "$@"
}

run_case "ddad_depth_distribution_all_pixels" \
  --pixel-selection all_pixels

run_case "ddad_depth_distribution_all_pixels_lt50m" \
  --pixel-selection all_pixels \
  --pred-max-depth-m 50

run_case "ddad_depth_distribution_eval_valid_gt" \
  --pixel-selection eval_valid_gt \
  --gt-mask-min-depth-m 0 \
  --gt-mask-max-depth-m 655

run_case "ddad_depth_distribution_eval_valid_gt_lt50m" \
  --pixel-selection eval_valid_gt \
  --gt-mask-min-depth-m 0 \
  --gt-mask-max-depth-m 655 \
  --pred-max-depth-m 50

echo "[run_ddad_depth_distribution_suite] all cases finished"
