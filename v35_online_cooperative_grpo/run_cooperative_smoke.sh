#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/apulis-dev/userdata/VLMTSCS}
VERL_ROOT=${VERL_ROOT:-${REPO_ROOT}/training/verl}
MODEL_PATH=${MODEL_PATH:-/home/apulis-dev/userdata/VLMTSCS/training/LlamaFactory/saves/qwen35-4b/merged/v35_four_video_context_reasoning_512x960_retrain}
API_URL=${API_URL:-http://localhost:8088/v1/chat/completions}
WORK_DIR=${WORK_DIR:-${REPO_ROOT}/training/LlamaFactory/runs/v35_cooperative_smoke}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-${REPO_ROOT}/grpo_v30_offline_local_video_dataset_reduced_pixels/train.jsonl}
CITY=${CITY:-jinan}
ROLLOUT_N=${ROLLOUT_N:-6}

cd "${VERL_ROOT}"
mkdir -p "${WORK_DIR}/sumo" "${WORK_DIR}/snapshots"

python -m v35_online_cooperative_grpo.ray_vlm_smoke \
  --repo-root "${REPO_ROOT}" \
  --work-root "${WORK_DIR}/sumo" \
  --snapshot-dir "${WORK_DIR}/snapshots" \
  --api-url "${API_URL}" \
  --model "${MODEL_PATH}" \
  --prompt-template "${PROMPT_TEMPLATE}" \
  --cities "${CITY}" \
  --rollout-n "${ROLLOUT_N}" \
  --trajectory-output "${WORK_DIR}/trajectories.jsonl"
