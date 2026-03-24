#!/usr/bin/env bash
# =============================================================
#  OpenClaw-RL  一键启动脚本
#  用法:
#    bash start.sh              # 用默认配置启动
#    bash start.sh --no-lora    # 不用 LoRA（全量微调）
#    bash start.sh --dry-run    # 只打印命令，不实际执行
# =============================================================
set -euo pipefail

# ── 颜色 ──────────────────────────────────────────────────────
RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'
CYAN='\033[36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
die()     { echo -e "${RED}[ERR]${RESET}  $*" >&2; exit 1; }

# ── 路径 ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_ROOT="${REPO_ROOT}/slime"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# ── 参数解析 ──────────────────────────────────────────────────
USE_LORA=1
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --no-lora)  USE_LORA=0 ;;
    --dry-run)  DRY_RUN=1 ;;
    *) warn "未知参数: $arg，忽略" ;;
  esac
done

# ── 可配置变量（可在外部通过环境变量覆盖）────────────────────
CONDA_ENV="${CONDA_ENV:-openclaw-rl}"
HF_CKPT="${HF_CKPT:-/data/openclaw-rl/models/Qwen3-4B}"
REF_LOAD="${REF_LOAD:-${HF_CKPT}}"
SAVE_CKPT="${SAVE_CKPT:-/data/openclaw-rl/ckpt/qwen3-4b-openclaw-rl-lora}"
PRM_MODEL_PATH="${PRM_MODEL_PATH:-${HF_CKPT}}"

NUM_GPUS="${NUM_GPUS:-4}"
ACTOR_GPUS="${ACTOR_GPUS:-2}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
PRM_GPUS="${PRM_GPUS:-1}"

API_PORT="${API_PORT:-30000}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
RECORD_FILE="${RESULTS_DIR}/qwen3_4b_lora_record.jsonl"

# ── Banner ────────────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "╔═══════════════════════════════════════════════╗"
echo "║          OpenClaw-RL  启动脚本                ║"
echo "╚═══════════════════════════════════════════════╝"
echo -e "${RESET}"
info "模型路径  : ${HF_CKPT}"
info "Checkpoint: ${SAVE_CKPT}"
info "GPU 分配  : actor=${ACTOR_GPUS}  rollout=${ROLLOUT_GPUS}  prm=${PRM_GPUS}  total=${NUM_GPUS}"
info "API 端口  : ${API_PORT}"
info "LoRA      : $([ $USE_LORA -eq 1 ] && echo on || echo off)"
echo ""

# ── 前置检查 ──────────────────────────────────────────────────
[[ -d "${HF_CKPT}" ]]   || die "模型路径不存在: ${HF_CKPT}"
[[ -d "${SLIME_ROOT}" ]] || die "slime 目录不存在: ${SLIME_ROOT}"

# 检查 conda 环境是否存在
conda env list 2>/dev/null | grep -q "^${CONDA_ENV}" \
  || die "conda 环境 '${CONDA_ENV}' 不存在，请先创建"

# GPU 数量是否足够
(( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS <= NUM_GPUS )) \
  || die "GPU 数量不足：actor(${ACTOR_GPUS})+rollout(${ROLLOUT_GPUS})+prm(${PRM_GPUS}) > ${NUM_GPUS}"

# conda 环境里 ray/python 是否存在
conda run -n "${CONDA_ENV}" ray --version &>/dev/null \
  || die "conda 环境 '${CONDA_ENV}' 中找不到 ray"

[[ $DRY_RUN -eq 1 ]] && { warn "dry-run 模式，退出（不执行）"; exit 0; }

# ── 停掉旧进程 ────────────────────────────────────────────────
info "清理旧进程..."
pkill -9 sglang 2>/dev/null || true
sleep 2
conda run -n "${CONDA_ENV}" ray stop --force 2>/dev/null || true
pkill -9 ray    2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray    2>/dev/null || true
pkill -9 python 2>/dev/null || true
success "旧进程已清理"

# ── 环境变量 ──────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60
export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export OPENCLAW_RECORD_ENABLED="${RECORD_ENABLED}"
export OPENCLAW_RECORD_FILE="${RECORD_FILE}"
export SERVED_MODEL_NAME="qwen3-4b"
export HOST="0.0.0.0"
export PORT="${API_PORT}"
export TP="${TP:-1}"
export CONTEXT_LENGTH="32768"
export MEM_FRACTION_STATIC="0.85"
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-3}"

# ── 启动 Ray ──────────────────────────────────────────────────
info "启动 Ray 集群..."
conda run -n "${CONDA_ENV}" \
  ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265
success "Ray 集群已启动  dashboard -> http://${MASTER_ADDR}:8265"

# ── 日志文件 ──────────────────────────────────────────────────
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RAY_LOG_FILE="${RESULTS_DIR}/ray_${RUN_TIMESTAMP}.log"
ln -sf "${RAY_LOG_FILE}" "${RESULTS_DIR}/ray_latest.log"
info "Ray 日志  : ${RAY_LOG_FILE}"
info "          (软链接 -> results/ray_latest.log)"

# ── 构建参数 ──────────────────────────────────────────────────
CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load      "${REF_LOAD}"
  --save          "${SAVE_CKPT}"
  --save-interval 1
)

ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --rollout-function-path openclaw_rollout.generate_rollout_openclaw
  --num-rollout 100000000
  --rollout-batch-size 4
  --n-samples-per-prompt 1
  --rollout-max-response-len 8192
  --rollout-max-context-len 32768
  --rollout-temperature 0.6
  --reward-key score
  --num-steps-per-rollout 1
)

PERF_ARGS=(
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
  --gradient-checkpointing
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --disable-rewards-normalization
  --use-kl-loss
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-5
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

LORA_ARGS=()
if [[ $USE_LORA -eq 1 ]]; then
  LORA_ARGS=(
    --use-lora
    --lora-rank 16
    --lora-alpha 32
    --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
  )
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${TP}"
  --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
  --sglang-mem-fraction-static 0.85
  --sglang-context-length 32768
  --sglang-reasoning-parser qwen3
)

PRM_ARGS=(
  --prm-enable
  --prm-num-gpus "${PRM_GPUS}"
  --prm-num-gpus-per-engine "${PRM_TP:-${TP}}"
  --prm-model-path "${PRM_MODEL_PATH}"
  --prm-m "${PRM_M}"
  --prm-temperature "${PRM_TEMPERATURE:-0.6}"
  --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-4096}"
)

CUSTOM_ARGS=(
  --custom-generate-function-path openclaw_api_server.generate
  --custom-rm-path openclaw_api_server.reward_func
)

WANDB_ARGS=()
WANDB_KEY_VALUE="${WANDB_KEY:-${WANDB_API_KEY:-}}"
if [[ "${USE_WANDB:-1}" == "1" && -n "${WANDB_KEY_VALUE}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-openclaw_rl}"
    --wandb-group   "qwen3-4b-openclaw-rl-lora"
    --wandb-key     "${WANDB_KEY_VALUE}"
  )
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

# ── 提交 Ray job ──────────────────────────────────────────────
info "提交 Ray job..."
conda run -n "${CONDA_ENV}" \
  ray job submit \
    --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    --no-wait \
    -- python3 "${SLIME_ROOT}/train_async.py" \
    --train-backend fsdp \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${ACTOR_GPUS}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    --num-gpus-per-node "${NUM_GPUS}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "${PRM_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
  2>&1 | tee -a "${RAY_LOG_FILE}"

# 拿到 job id
JOB_ID=$(conda run -n "${CONDA_ENV}" \
  ray job list 2>/dev/null \
  | grep -oP 'raysubmit_\w+' | tail -1)
success "Job 已提交: ${JOB_ID}"

# 把 job id 写到文件，方便 stop.sh 使用
echo "${JOB_ID}" > "${RESULTS_DIR}/.last_job_id"

# 后台把 job 日志持续追写到文件
(
  conda run -n "${CONDA_ENV}" \
    ray job logs --follow "${JOB_ID}" 2>/dev/null \
  | tee -a "${RAY_LOG_FILE}" > /dev/null
) &
LOG_TAIL_PID=$!
echo "${LOG_TAIL_PID}" > "${RESULTS_DIR}/.log_tail_pid"

# ── 等待服务就绪 ──────────────────────────────────────────────
info "等待 API 服务就绪 (http://localhost:${API_PORT}/health)..."
WAIT_TIMEOUT=300   # 最多等 5 分钟
ELAPSED=0
while true; do
  if curl -sf "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
    break
  fi
  if (( ELAPSED >= WAIT_TIMEOUT )); then
    warn "超时 ${WAIT_TIMEOUT}s，服务可能还在加载，请手动确认"
    warn "  tail -f ${RAY_LOG_FILE}"
    break
  fi
  sleep 5
  ELAPSED=$(( ELAPSED + 5 ))
  echo -ne "  已等待 ${ELAPSED}s...\r"
done

echo ""
echo -e "${GREEN}${BOLD}"
echo "╔══════════════════════════════════════════════════╗"
echo "║   ✅  OpenClaw 已就绪，可以开始对话！           ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "  API     : ${CYAN}http://localhost:${API_PORT}${RESET}"
echo -e "  对话    : ${CYAN}conda run -n ${CONDA_ENV} python3 ${SCRIPT_DIR}/chat.py${RESET}"
echo -e "  实时日志: ${CYAN}tail -f ${RAY_LOG_FILE}${RESET}"
echo -e "  停止    : ${CYAN}bash ${SCRIPT_DIR}/stop.sh${RESET}"
echo ""
