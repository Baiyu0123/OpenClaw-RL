#!/usr/bin/env bash
# =============================================================
#  OpenClaw-OPD  一键启动脚本（Top-K Distillation LoRA）
#  用法:
#    bash start_opd.sh           # 默认启动
#    bash start_opd.sh --dry-run # 只打印命令，不实际执行
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
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)  DRY_RUN=1 ;;
    *) warn "未知参数: $arg，忽略" ;;
  esac
done

# ── 可配置变量（可在外部通过环境变量覆盖）────────────────────
CONDA_ENV="${CONDA_ENV:-openclaw-rl}"
HF_CKPT="${HF_CKPT:-/data/openclaw-rl/models/Qwen3-4B}"
REF_LOAD="${REF_LOAD:-${HF_CKPT}}"
SAVE_CKPT="${SAVE_CKPT:-/data/openclaw-rl/ckpt/qwen3-4b-openclaw-opd-topk-lora}"
PRM_MODEL_PATH="${PRM_MODEL_PATH:-${HF_CKPT}}"

NUM_GPUS="${NUM_GPUS:-4}"
ACTOR_GPUS="${ACTOR_GPUS:-2}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
PRM_GPUS="${PRM_GPUS:-1}"

API_PORT="${API_PORT:-30000}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
RECORD_FILE="${RESULTS_DIR}/qwen3_4b_opd_record.jsonl"

TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-3}"

# ── Banner ────────────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "╔═══════════════════════════════════════════════╗"
echo "║       OpenClaw-OPD  启动脚本                  ║"
echo "╚═══════════════════════════════════════════════╝"
echo -e "${RESET}"
info "模型路径  : ${HF_CKPT}"
info "Checkpoint: ${SAVE_CKPT}"
info "GPU 分配  : actor=${ACTOR_GPUS}  rollout=${ROLLOUT_GPUS}  prm=${PRM_GPUS}  total=${NUM_GPUS}"
info "API 端口  : ${API_PORT}"
info "OPD 模式  : Top-K Distillation (topk=50, lr=1e-6)"
echo ""

# ── 前置检查 ──────────────────────────────────────────────────
[[ -d "${HF_CKPT}" ]]   || die "模型路径不存在: ${HF_CKPT}"
[[ -d "${SLIME_ROOT}" ]] || die "slime 目录不存在: ${SLIME_ROOT}"

conda env list 2>/dev/null | grep -q "^${CONDA_ENV}" \
  || die "conda 环境 '${CONDA_ENV}' 不存在，请先创建"

(( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS <= NUM_GPUS )) \
  || die "GPU 数量不足：actor(${ACTOR_GPUS})+rollout(${ROLLOUT_GPUS})+prm(${PRM_GPUS}) > ${NUM_GPUS}"

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
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${TEACHER_LP_MAX_CONCURRENCY}"

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
  --rollout-function-path openclaw_opd_rollout.generate_rollout_openclaw_opd
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

# OPD 使用 Top-K Distillation loss，不用 GRPO advantage
OPD_ARGS=(
  --loss-type custom_loss
  --custom-loss-function-path topk_distillation_loss.topk_distillation_loss_function
  --distill-topk 50
  --disable-compute-advantages-and-returns
  --disable-rewards-normalization
  --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

LORA_ARGS=(
  --use-lora
  --lora-rank 16
  --lora-alpha 32
  --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)

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
  --custom-generate-function-path openclaw_opd_api_server.generate
  --custom-rm-path openclaw_opd_api_server.reward_func
)

WANDB_ARGS=()
WANDB_KEY_VALUE="${WANDB_KEY:-${WANDB_API_KEY:-}}"
if [[ "${USE_WANDB:-1}" == "1" && -n "${WANDB_KEY_VALUE}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-openclaw_rl}"
    --wandb-group   "qwen3-4b-openclaw-opd-topk-lora"
    --wandb-key     "${WANDB_KEY_VALUE}"
  )
fi

# ── RUNTIME_ENV_JSON：所有 env var 必须在这里显式传入 Ray worker ──
# 注意：conda run 会将 HOST 覆盖为架构字符串，必须在 JSON 里强制设置
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"HOST\": \"0.0.0.0\",
    \"PORT\": \"${API_PORT}\",
    \"OPENCLAW_RECORD_ENABLED\": \"${RECORD_ENABLED}\",
    \"OPENCLAW_RECORD_FILE\": \"${RECORD_FILE}\",
    \"SERVED_MODEL_NAME\": \"qwen3-4b\",
    \"TP\": \"${TP:-1}\",
    \"CONTEXT_LENGTH\": \"32768\",
    \"MEM_FRACTION_STATIC\": \"0.85\",
    \"REASONING_PARSER\": \"qwen3\",
    \"TOOL_CALL_PARSER\": \"${TOOL_CALL_PARSER:-qwen25}\",
    \"PRM_M\": \"${PRM_M:-3}\",
    \"OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY\": \"${TEACHER_LP_MAX_CONCURRENCY}\"
  }
}"

# ── 提交 Ray job ──────────────────────────────────────────────
info "提交 Ray job（OPD 模式）..."
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
    "${OPD_ARGS[@]}" \
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

echo "${JOB_ID}" > "${RESULTS_DIR}/.last_job_id"

# Ray 的 job driver log 就是最完整的输出，等待它出现后建软链接
info "等待 job driver log 文件出现..."
DRIVER_LOG=""
for i in $(seq 1 30); do
  DRIVER_LOG=$(ls /tmp/ray/session_latest/logs/job-driver-${JOB_ID}.log 2>/dev/null || true)
  [[ -n "${DRIVER_LOG}" ]] && break
  sleep 2
done

if [[ -n "${DRIVER_LOG}" ]]; then
  # 把 ray_latest.log 软链接直接指向 job driver log（真实输出）
  ln -sf "${DRIVER_LOG}" "${RESULTS_DIR}/ray_latest.log"
  success "job driver log: ${DRIVER_LOG}"
  success "软链接已更新: results/ray_latest.log -> ${DRIVER_LOG}"
  echo "${DRIVER_LOG}" > "${RESULTS_DIR}/.driver_log_path"
  # 后台持续追加到带时间戳的副本（可选）
  (tail -F "${DRIVER_LOG}" >> "${RAY_LOG_FILE}" 2>/dev/null) &
  LOG_TAIL_PID=$!
else
  warn "未找到 job driver log，回退到 ray job logs --follow"
  (conda run -n "${CONDA_ENV}" ray job logs --follow "${JOB_ID}" 2>/dev/null \
    | tee -a "${RAY_LOG_FILE}" > /dev/null) &
  LOG_TAIL_PID=$!
fi
echo "${LOG_TAIL_PID}" > "${RESULTS_DIR}/.log_tail_pid"

# ── 等待服务就绪 ──────────────────────────────────────────────
info "等待 API 服务就绪..."
WAIT_TIMEOUT=360
ELAPSED=0
while true; do
  for base in "http://localhost:${API_PORT}" "http://127.0.0.1:${API_PORT}" "http://${MASTER_ADDR}:${API_PORT}"; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "${base}/v1/chat/completions" 2>/dev/null) || true
    if [[ "${code}" == "405" || "${code}" == "200" ]]; then
      API_PORT_URL="${base}"
      break 2
    fi
  done
  if grep -q "your model is fired up" "${RAY_LOG_FILE}" 2>/dev/null; then
    ACTUAL_HOST=$(grep "proxy" "${RAY_LOG_FILE}" 2>/dev/null \
      | grep -oP '(?<=proxy )[^ ]+' | tail -1 | cut -d: -f1)
    [[ -n "${ACTUAL_HOST}" ]] && API_PORT_URL="http://${ACTUAL_HOST}:${API_PORT}"
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
echo "║   ✅  OpenClaw-OPD 已就绪，可以开始对话！       ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "  API     : ${CYAN}${API_PORT_URL:-http://localhost:${API_PORT}}${RESET}"
echo -e "  对话    : ${CYAN}conda activate ${CONDA_ENV} && python3 $(cd "${SCRIPT_DIR}/../openclaw-rl" &>/dev/null && pwd)/chat.py --url ${API_PORT_URL:-http://localhost:${API_PORT}}${RESET}"
echo -e "  实时日志: ${CYAN}tail -f ${RAY_LOG_FILE}${RESET}"
echo -e "  OPD hint: ${CYAN}grep 'OpenClaw-OPD' ${RAY_LOG_FILE}${RESET}"
echo -e "  停止    : ${CYAN}bash ${SCRIPT_DIR}/stop_opd.sh${RESET}"
echo ""
