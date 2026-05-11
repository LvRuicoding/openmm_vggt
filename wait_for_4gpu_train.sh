#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-0,1,2,3}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
THRESHOLD_MIB="${THRESHOLD_MIB:-10240}"
CONFIG="./configs/occupancy/kitti_semantic_occ_multi_fusion_monoscene_head_cp_364x1218.py"

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
EXPECTED_GPU_COUNT="${#GPU_ID_ARRAY[@]}"
GPU_USAGES=()

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

format_usages() {
  local output=""
  local i

  for i in "${!GPU_ID_ARRAY[@]}"; do
    output+="GPU ${GPU_ID_ARRAY[$i]}: ${GPU_USAGES[$i]} MiB"
    if [[ "${i}" -lt "$((EXPECTED_GPU_COUNT - 1))" ]]; then
      output+=", "
    fi
  done

  printf '%s' "${output}"
}

all_gpus_below_threshold() {
  local usage

  mapfile -t GPU_USAGES < <(
    nvidia-smi \
      --id="${GPU_IDS}" \
      --query-gpu=memory.used \
      --format=csv,noheader,nounits
  )

  if [[ "${#GPU_USAGES[@]}" -ne "${EXPECTED_GPU_COUNT}" ]]; then
    echo "[$(timestamp)] Expected ${EXPECTED_GPU_COUNT} GPUs (${GPU_IDS}), got ${#GPU_USAGES[@]} from nvidia-smi." >&2
    return 1
  fi

  for usage in "${GPU_USAGES[@]}"; do
    usage="${usage//[[:space:]]/}"
    if [[ ! "${usage}" =~ ^[0-9]+$ ]]; then
      echo "[$(timestamp)] Invalid nvidia-smi memory value: ${usage}" >&2
      return 1
    fi
    if (( usage >= THRESHOLD_MIB )); then
      return 1
    fi
  done

  return 0
}

echo "[$(timestamp)] Waiting for GPUs ${GPU_IDS} to all use less than ${THRESHOLD_MIB} MiB."

while true; do
  if all_gpus_below_threshold; then
    echo "[$(timestamp)] GPU memory is available: $(format_usages)"
    echo "[$(timestamp)] Starting training."
    exec env CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun --nproc_per_node=4 tools/train.py "${CONFIG}" --batch-size 2 --no-eval
  fi

  if [[ "${#GPU_USAGES[@]}" -eq "${EXPECTED_GPU_COUNT}" ]]; then
    echo "[$(timestamp)] Still waiting: $(format_usages)"
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done
