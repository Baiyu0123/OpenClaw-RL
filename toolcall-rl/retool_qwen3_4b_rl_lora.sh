#!/bin/bash
# ===================================================================
# Toolcall-RL: Qwen3-4B LoRA Training Script
# Math problem solving with Python tool calling (DAPO-Math-17k)
# Uses FSDP backend (no transformer_engine needed)
# ===================================================================

# cleanup for rerun
pkill -9 sglang 2>/dev/null
sleep 3
ray stop --force 2>/dev/null
pkill -9 ray 2>/dev/null
pkill -9 python 2>/dev/null
sleep 3
pkill -9 ray 2>/dev/null
pkill -9 python 2>/dev/null

set -ex

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ---- GPU Allocation ----
# 8 GPUs total: 4 actor (LoRA training), 4 rollout (inference)
NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
    exit 1
fi

# ---- Ray Health Checks ----
export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

# ---- Paths ----
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SCRIPT_DIR}/../Megatron-LM"}

HF_CKPT=${HF_CKPT:-"/data/openclaw-rl/models/qwen3-4b-retool-sft"}
REF_LOAD=${REF_LOAD:-"${HF_CKPT}"}
SAVE_CKPT=${SAVE_CKPT:-"/data/openclaw-rl/ckpt/qwen3-4b-sft-retool-rl-lora"}
RESUME_LOAD=${RESUME_LOAD:-${SAVE_CKPT}}

# ---- Checkpoint ----
CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 20
)

# ---- Rollout / Data ----
ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA:-/data/openclaw-rl/data/dapo_math_17k.jsonl}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 4
   --n-samples-per-prompt 4
   --rollout-max-response-len 6144
   --rollout-max-context-len 12288
   --rollout-temperature 1

   --num-steps-per-rollout 2
   --balance-data
)

# ---- Performance (FSDP + LoRA) ----
PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-2048}"
   --gradient-checkpointing
   --log-probs-chunk-size 256
)

# ---- LoRA ----
LORA_ARGS=(
   --use-lora
   --lora-rank 16
   --lora-alpha 32
   --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)

# ---- GRPO ----
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type k3
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# ---- Optimizer ----
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ---- SGLang Inference ----
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.6
)

# ---- Eval ----
EVAL_ARGS=()

# ---- W&B (optional) ----
USE_WANDB=${USE_WANDB:-0}
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ "${USE_WANDB}" = "1" ] && [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project slime_retool
    --wandb-group qwen3-4B-rl-retool-lora
    --wandb-key "${WANDB_KEY_VALUE}"
  )
else
  WANDB_ARGS=()
fi

# ---- Custom generate / reward ----
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

# ---- Launch ----
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

CONDA_PREFIX_PATH="$(dirname $(which nvcc)):${PATH}"
CUDA_HOME_DIR="$(dirname $(dirname $(which nvcc)))"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"PATH\": \"${CONDA_PREFIX_PATH}\",
    \"CUDA_HOME\": \"${CUDA_HOME_DIR}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --train-backend fsdp \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${LORA_ARGS[@]}
