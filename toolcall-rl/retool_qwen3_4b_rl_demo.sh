#!/bin/bash
#
# ── DEMO 脚本 ──
# 目标：快速验证整条链路是否跑通，看到 loss/reward 日志即成功。
# 预计耗时：10-20 分钟（含模型加载 ~5min + 50 步训练）
#
# GPU 需求：8 × A100-40GB（4 Actor + 4 Rollout）
#
# 用法：
#   conda activate openclaw-rl
#   cd /home/test6/OpenClaw-RL/slime
#   bash ../toolcall-rl/retool_qwen3_4b_rl_demo.sh
#

set -ex

# ─── GPU 占用检查 ───
BUSY_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 5000' | wc -l)
if (( BUSY_MEM > 0 )); then
    echo "============================================"
    echo "WARNING: ${BUSY_MEM} GPUs have >5GB used."
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
    echo "按 Ctrl+C 中止，或等 10s 继续..."
    echo "============================================"
    sleep 10
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$(( NVLINK_COUNT > 0 ? 1 : 0 ))

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SCRIPT_DIR}/../Megatron-LM"}
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

HF_CKPT=${HF_CKPT:-/data/openclaw-rl/models/qwen3-4b-retool-sft}
SAVE_CKPT=${SAVE_CKPT:-/data/openclaw-rl/ckpt/qwen3-4b-retool-rl-demo}

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${HF_CKPT}"
    --save "${SAVE_CKPT}"
    --save-interval 50          # demo 只跑 50 步，结束时保一次
    --rotary-base 5000000
    --megatron-to-hf-mode bridge
)

# ─── Demo 规模：缩小到可以快速跑完 ───
#   原版: batch=32, n_samples=8, rollout=3000 → 约48h
#   Demo: batch=4,  n_samples=4, rollout=50  → 约15min
ROLLOUT_ARGS=(
    --prompt-data /data/openclaw-rl/data/dapo_math_17k.jsonl
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --reward-key score
    --num-rollout 50            # 只跑 50 步就停
    --rollout-batch-size 4      # 每步 4 道题（原版 32）
    --n-samples-per-prompt 4    # 每题 4 条轨迹（原版 8）
    --rollout-max-response-len 4096   # 缩短响应长度（原版 8192）
    --rollout-max-context-len 8192    # 缩短上下文（原版 16384）
    --rollout-temperature 1
    --num-steps-per-rollout 2
    --balance-data
)

# ─── Demo 评估：只用 5 道题，快速出分 ───
EVAL_ARGS=(
    --eval-interval 25          # 25 步评估一次（共评 2 次）
    --eval-prompt-data aime /data/openclaw-rl/data/aime-2024.jsonl
    --n-samples-per-eval-prompt 2   # 每题 2 条（原版 16）
    --eval-max-response-len 8192
    --eval-max-context-len 16384
    --eval-top-p 1
    --eval-reward-key acc
)

# ─── 性能：A100-40GB ───
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
    --max-tokens-per-gpu 8192   # 40GB 适配
    --log-probs-chunk-size 1024
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.01
    --kl-loss-type k3
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

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

WANDB_ARGS=()
if [ -n "${WANDB_KEY}" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project slime_retool
        --wandb-group qwen3-4B-demo
        --wandb-key "${WANDB_KEY}"
    )
fi

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

CUSTOM_ARGS=(
    --custom-generate-function-path generate_with_retool.generate
    --custom-rm-path generate_with_retool.reward_func
)

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}
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

echo ""
echo "════════════════════════════════════════════"
echo "  DEMO 模式：50步 × 4题 × 4轨迹 = 800条轨迹"
echo "  预计 15-20 分钟跑完"
echo "  看到 'rollout/reward' 和 'train/loss' 即正常"
echo "════════════════════════════════════════════"
echo ""

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
