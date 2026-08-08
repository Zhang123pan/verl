#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/home/apulis-dev/userdata/VLMTSCS/training/LlamaFactory/saves/qwen35-4b/merged/v35_four_video_context_reasoning}
DATA_DIR=${DATA_DIR:-/home/apulis-dev/userdata/VLMTSCS/grpo_v30_offline_local_video_dataset}
WORK_DIR=${WORK_DIR:-/home/apulis-dev/userdata/VLMTSCS/training/LlamaFactory/runs/v35_4b_offline_grpo}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
N_GPUS=${N_GPUS:-4}
ROLLOUT_TP=${ROLLOUT_TP:-4}
FSDP_SIZE=${FSDP_SIZE:-4}
ROLLOUT_N=${ROLLOUT_N:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-61440}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REWARD_FILE=${REWARD_FILE:-${SCRIPT_DIR}/v35_offline_grpo_reward.py}

TRAIN_FILE="${DATA_DIR}/train.jsonl"
VAL_FILE="${DATA_DIR}/val.jsonl"
OUTPUT_DIR="${WORK_DIR}/checkpoints"
LOG_DIR="${WORK_DIR}/logs"
export CUDA_VISIBLE_DEVICES MODEL_PATH V35_GRPO_MODEL_PATH=${MODEL_PATH}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/${USER:-user}_v35_grpo_ray}
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

for path in "${MODEL_PATH}/config.json" "${TRAIN_FILE}" "${VAL_FILE}" "${REWARD_FILE}"; do
    [[ -e "${path}" ]] || { echo "Missing: ${path}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RAY_TMPDIR}"
stamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/v35_4b_offline_grpo_${stamp}.log"

echo "model=${MODEL_PATH}"
echo "data=${DATA_DIR} train=$(wc -l < "${TRAIN_FILE}") val=$(wc -l < "${VAL_FILE}")"
echo "gpus=${CUDA_VISIBLE_DEVICES} rollout_n=${ROLLOUT_N} max_response=${MAX_RESPONSE_LENGTH}"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.video_key=videos \
    data.image_key=images \
    data.image_patch_size=16 \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    reward.custom_reward_function.path="${REWARD_FILE}" \
    reward.custom_reward_function.name=compute_score \
    trainer.logger='["console"]' \
    trainer.project_name=v35_offline_grpo \
    trainer.experiment_name=v35_4b_offline_grpo \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=1 \
    trainer.save_freq=1 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo "completed; log=${LOG_FILE}"
