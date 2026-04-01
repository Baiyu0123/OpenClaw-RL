#!/bin/bash
#
# Tool-call RL for local 8×A100-40GB.
#
# Usage:
#   conda activate openclaw-rl
#   cd /home/test6/OpenClaw-RL/slime
#   bash ../toolcall-rl/retool_qwen3_4b_rl_local.sh
#
# IMPORTANT: This script does NOT kill existing processes.
# Make sure GPUs are free before running.
#

set -ex

# ─── Sanity check: abort if GPUs are still busy ───
BUSY_GPUS=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '$1 > 5' | wc -l)
BUSY_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 5000' | wc -l)
if (( BUSY_GPUS > 0 || BUSY_MEM > 0 )); then
    echo "============================================"
    echo "WARNING: ${BUSY_GPUS} GPUs show >5% utilization, ${BUSY_MEM} GPUs have >5GB memory used."
    echo "Existing experiments may still be running."
    echo "Press Ctrl+C within 10s to abort, or wait to continue."
    echo "============================================"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
    sleep 10
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ─── GPU layout: 8 GPUs total ───
# 4 for training actor, 4 for SGLang rollout inference
NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS(${ACTOR_GPUS}) + ROLLOUT_GPUS(${ROLLOUT_GPUS}) > NUM_GPUS(${NUM_GPUS})"
    exit 1
fi

# ─── Ray health-check tuning ───
export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

# ─── NVLink detection ───
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$(( NVLINK_COUNT > 0 ? 1 : 0 ))
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ─── Paths ───
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SCRIPT_DIR}/../Megatron-LM"}

# Load Qwen3-4B model architecture args
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

# ─── Checkpoints (all on /data) ───
HF_CKPT=${HF_CKPT:-/data/openclaw-rl/models/qwen3-4b-retool-sft}
# Bridge mode: no need for torch_dist ref model.
# ref-load is omitted; bridge mode falls back to hf-checkpoint for reference.
SAVE_CKPT=${SAVE_CKPT:-/data/openclaw-rl/ckpt/qwen3-4b-retool-rl-local}

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${HF_CKPT}"
    --save "${SAVE_CKPT}"
    --save-interval 20
    --rotary-base 5000000
    --megatron-to-hf-mode bridge
)

# ─── Data ───
PROMPT_DATA=${PROMPT_DATA:-/data/openclaw-rl/data/dapo_math_17k.jsonl}
EVAL_DATA=${EVAL_DATA:-/data/openclaw-rl/data/aime-2024.jsonl}

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --reward-key score
    --num-rollout 3000
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --rollout-max-response-len 8192
    --rollout-max-context-len 16384
    --rollout-temperature 1
    --num-steps-per-rollout 2
    --balance-data
)

EVAL_ARGS=(
    --eval-interval 20
    --eval-prompt-data aime "${EVAL_DATA}"
    --n-samples-per-eval-prompt 16
    --eval-max-response-len 16384
    --eval-max-context-len 32768
    --eval-top-p 1
    --eval-reward-key acc
)

# ─── Performance: adapted for A100-40GB ───
# Reduced max-tokens-per-gpu from 16384 to 8192 for 40GB cards
PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --log-probs-chunk-size 1024
)

# ─── GRPO ───
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.01
    --kl-loss-type k3
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

# ─── Optimizer ───
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

# ─── Wandb (optional, off by default) ───
WANDB_ARGS=()
if [ -n "${WANDB_KEY}" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project slime_retool
        --wandb-group qwen3-4B-rl_retool_local
        --wandb-key "${WANDB_KEY}"
    )
fi

# ─── SGLang rollout engine ───
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.6
)

# ─── Misc ───
MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

# ─── Custom generate/reward functions ───
CUSTOM_ARGS=(
    --custom-generate-function-path generate_with_retool.generate
    --custom-rm-path generate_with_retool.reward_func
)

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}

# ─── Launch Ray cluster ───
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus ${NUM_GPUS} \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\"
  }
}"

# ─── Submit training job ───
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node ${ACTOR_GPUS} \
    --rollout-num-gpus ${ROLLOUT_GPUS} \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${CUSTOM_ARGS[@]}
