#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
THRESHOLD_MB="${THRESHOLD_MB:-5120}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

cd /home/dataset-local/lr/code/openmm_vggt

echo "Watching GPUs ${GPU_IDS}; will start when every listed GPU uses < ${THRESHOLD_MB} MiB."
echo "Check interval: ${INTERVAL_SECONDS}s"

while true; do
  mapfile -t used_mb < <(
    nvidia-smi \
      --id="${GPU_IDS}" \
      --query-gpu=memory.used \
      --format=csv,noheader,nounits
  )

  if [[ "${#used_mb[@]}" -ne 4 ]]; then
    echo "$(date '+%F %T') expected 4 GPUs, got ${#used_mb[@]} values: ${used_mb[*]:-none}"
    sleep "${INTERVAL_SECONDS}"
    continue
  fi

  all_idle=1
  status=()
  for idx in "${!used_mb[@]}"; do
    mem="${used_mb[$idx]//[[:space:]]/}"
    status+=("gpu${idx}=${mem}MiB")
    if (( mem >= THRESHOLD_MB )); then
      all_idle=0
    fi
  done

  echo "$(date '+%F %T') ${status[*]}"

  if (( all_idle )); then
    echo "$(date '+%F %T') all GPUs are below ${THRESHOLD_MB} MiB; starting training."
    exec torchrun --nproc_per_node=4 tools/train.py \
      configs/occupancy/kitti_semantic_occ_mix_window_attn_early_ft_monoscene_head_cp_364x1218_occ_only.py \
      --batch-size 1 \
      --no-eval
  fi

  sleep "${INTERVAL_SECONDS}"
done
