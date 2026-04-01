#!/usr/bin/env bash
# =============================================================
#  OpenClaw-Combined  一键启动脚本（Binary RL + OPD, Megatron TP=4）
#  用法:
#    bash start_combine.sh           # 全量实验 (batch=16)
#    bash start_combine.sh --demo    # Demo 模式 (batch=4)
# =============================================================
set -euo pipefail

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
MEGATRON_ROOT="${REPO_ROOT}/Megatron-LM"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# ── Megatron 模型定义（上游配置） ─────────────────────────────
source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

# ── 参数解析 ──────────────────────────────────────────────────
DEMO_MODE=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --demo)    DEMO_MODE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) warn "未知参数: $arg，忽略" ;;
  esac
done

# ── 可配置变量 ────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-openclaw-rl}"
HF_CKPT="${HF_CKPT:-/data/openclaw-rl/models/Qwen3-4B-Thinking-2507}"
REF_LOAD="${REF_LOAD:-${HF_CKPT}}"
SAVE_CKPT="${SAVE_CKPT:-/data/openclaw-rl/ckpt/qwen3-4b-openclaw-combine-full}"
PRM_MODEL_PATH="${PRM_MODEL_PATH:-${HF_CKPT}}"

NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
PRM_GPUS="${PRM_GPUS:-2}"

API_PORT="${API_PORT:-30000}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
RECORD_FILE="${RESULTS_DIR}/qwen3_4b_combine_record.jsonl"
RECORD_PRM_FILE="${RESULTS_DIR}/qwen3_4b_combine_record_prm.jsonl"

# ── Demo vs 全量 ──────────────────────────────────────────────
if [[ $DEMO_MODE -eq 1 ]]; then
  ROLLOUT_BATCH_SIZE=4
  MODE_LABEL="DEMO (batch=4)"
else
  ROLLOUT_BATCH_SIZE=16
  MODE_LABEL="FULL (batch=16, paper setting)"
fi

# ── Banner ─────────────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "╔════════════════════════════════════════════════════╗"
echo "║  OpenClaw-Combined (RL+OPD, Megatron TP=4)        ║"
echo "╚════════════════════════════════════════════════════╝"
echo -e "${RESET}"
info "模式       : ${MODE_LABEL}"
info "模型路径   : ${HF_CKPT}"
info "Checkpoint : ${SAVE_CKPT}"
info "GPU 分配   : actor=${ACTOR_GPUS}  rollout=${ROLLOUT_GPUS}  prm=${PRM_GPUS}  total=${NUM_GPUS}"
info "API 端口   : ${API_PORT}"
echo ""

# ── 前置检查 ──────────────────────────────────────────────────
[[ -d "${HF_CKPT}" ]]   || die "模型路径不存在: ${HF_CKPT}"
[[ -d "${SLIME_ROOT}" ]] || die "slime 目录不存在: ${SLIME_ROOT}"

conda env list 2>/dev/null | grep -q "^${CONDA_ENV}" \
  || die "conda 环境 '${CONDA_ENV}' 不存在"

(( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS <= NUM_GPUS )) \
  || die "GPU 不足: actor(${ACTOR_GPUS})+rollout(${ROLLOUT_GPUS})+prm(${PRM_GPUS}) > ${NUM_GPUS}"

conda run -n "${CONDA_ENV}" ray --version &>/dev/null \
  || die "conda 环境中找不到 ray"

[[ $DRY_RUN -eq 1 ]] && { warn "dry-run 模式，退出"; exit 0; }

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
export HOST="0.0.0.0"
export PORT="${API_PORT}"
export OPENCLAW_RECORD_ENABLED="${RECORD_ENABLED}"
export OPENCLAW_RECORD_FILE="${RECORD_FILE}"
export OPENCLAW_RECORD_PRM_FILE="${RECORD_PRM_FILE}"
export SERVED_MODEL_NAME="qwen3-4b"
export TP="${TP:-2}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export MEM_FRACTION_STATIC="0.80"
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-1}"
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-1}"
export OPENCLAW_COMBINE_W_RL="${OPENCLAW_COMBINE_W_RL:-1.0}"
export OPENCLAW_COMBINE_W_OPD="${OPENCLAW_COMBINE_W_OPD:-1.0}"
export OPENCLAW_EVAL_MODE="${OPENCLAW_EVAL_MODE:-1}"

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

# ── 构建参数 ──────────────────────────────────────────────────
CKPT_ARGS=(
  --megatron-to-hf-mode bridge
  --hf-checkpoint "${HF_CKPT}"
  --ref-load      "${REF_LOAD}"
  --save          "${SAVE_CKPT}"
  --save-interval 1
  --rotary-base "${ROTARY_BASE:-5000000}"
)

ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --rollout-function-path openclaw_combine_rollout.generate_rollout_openclaw_combine
  --num-rollout 100000000
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt 1
  --rollout-max-response-len 8192
  --rollout-max-context-len 32768
  --rollout-temperature 1.0
  --reward-key score
  --num-steps-per-rollout 1
)

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
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
  --log-probs-chunk-size 1024
)

# Combined: Binary RL (GRPO) + OPD (teacher distillation), same as paper §5.1.1
# LR=1e-5, KL=0 as per paper
COMBINE_ARGS=(
  --advantage-estimator grpo
  --disable-rewards-normalization
  --loss-type custom_loss
  --custom-loss-function-path combine_loss.combine_loss_function
  --use-kl-loss
  --kl-loss-coef 0.01
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
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 2
  --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
  --sglang-mem-fraction-static 0.80
  --sglang-context-length 32768
  --sglang-reasoning-parser qwen3
)

PRM_ARGS=(
  --prm-enable
  --prm-num-gpus "${PRM_GPUS}"
  --prm-num-gpus-per-engine 2
  --prm-model-path "${PRM_MODEL_PATH}"
  --prm-m "${PRM_M}"
  --prm-temperature "${PRM_TEMPERATURE:-0.6}"
  --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-8192}"
)

CUSTOM_ARGS=(
  --custom-generate-function-path openclaw_combine_api_server.generate
  --custom-rm-path openclaw_combine_api_server.reward_func
)

WANDB_ARGS=()
WANDB_KEY_VALUE="${WANDB_KEY:-${WANDB_API_KEY:-}}"
if [[ "${USE_WANDB:-1}" == "1" && -n "${WANDB_KEY_VALUE}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-openclaw_rl}"
    --wandb-group   "qwen3-4b-combine-full-sysprompt"
    --wandb-key     "${WANDB_KEY_VALUE}"
  )
fi

# ── RUNTIME_ENV_JSON：所有 env var 必须在这里传入 Ray worker ──
# 注意：openclaw-combine 依赖 openclaw-opd 的基类，PYTHONPATH 必须包含两个目录
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SCRIPT_DIR}:${SCRIPT_DIR}/../openclaw-opd:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"HOST\": \"0.0.0.0\",
    \"PORT\": \"${API_PORT}\",
    \"OPENCLAW_RECORD_ENABLED\": \"${RECORD_ENABLED}\",
    \"OPENCLAW_RECORD_FILE\": \"${RECORD_FILE}\",
    \"SERVED_MODEL_NAME\": \"qwen3-4b\",
    \"TP\": \"${TP}\",
    \"CONTEXT_LENGTH\": \"${CONTEXT_LENGTH}\",
    \"MEM_FRACTION_STATIC\": \"0.80\",
    \"REASONING_PARSER\": \"qwen3\",
    \"TOOL_CALL_PARSER\": \"${TOOL_CALL_PARSER:-qwen25}\",
    \"PRM_M\": \"${PRM_M:-1}\",
    \"OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY\": \"${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-1}\",
    \"OPENCLAW_COMBINE_W_RL\": \"${OPENCLAW_COMBINE_W_RL:-1.0}\",
    \"OPENCLAW_COMBINE_W_OPD\": \"${OPENCLAW_COMBINE_W_OPD:-1.0}\",
    \"OPENCLAW_EVAL_MODE\": \"${OPENCLAW_EVAL_MODE:-1}\"
  }
}"

# ── 提交 Ray job ──────────────────────────────────────────────
info "提交 Ray job（Combined RL+OPD 模式，batch=${ROLLOUT_BATCH_SIZE}）..."
conda run -n "${CONDA_ENV}" \
  ray job submit \
    --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    --no-wait \
    -- python3 "${SLIME_ROOT}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${ACTOR_GPUS}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    --num-gpus-per-node "${NUM_GPUS}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${COMBINE_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "${PRM_ARGS[@]}" \
  2>&1 | tee -a "${RAY_LOG_FILE}"

# 拿到 job id
JOB_ID=$(conda run -n "${CONDA_ENV}" \
  ray job list 2>/dev/null \
  | grep -oP 'raysubmit_\w+' | tail -1)
success "Job 已提交: ${JOB_ID}"
echo "${JOB_ID}" > "${RESULTS_DIR}/.last_job_id"

# 等待 job driver log 出现后建软链接（真实输出）
info "等待 job driver log..."
DRIVER_LOG=""
for i in $(seq 1 30); do
  DRIVER_LOG=$(ls /tmp/ray/session_latest/logs/job-driver-${JOB_ID}.log 2>/dev/null || true)
  [[ -n "${DRIVER_LOG}" ]] && break
  sleep 2
done

if [[ -n "${DRIVER_LOG}" ]]; then
  ln -sf "${DRIVER_LOG}" "${RESULTS_DIR}/ray_latest.log"
  success "日志软链接: results/ray_latest.log -> ${DRIVER_LOG}"
  echo "${DRIVER_LOG}" > "${RESULTS_DIR}/.driver_log_path"
  (tail -F "${DRIVER_LOG}" >> "${RAY_LOG_FILE}" 2>/dev/null) &
  LOG_TAIL_PID=$!
else
  warn "未找到 driver log，回退到 ray job logs --follow"
  (conda run -n "${CONDA_ENV}" ray job logs --follow "${JOB_ID}" 2>/dev/null \
    | tee -a "${RAY_LOG_FILE}" > /dev/null) &
  LOG_TAIL_PID=$!
fi
echo "${LOG_TAIL_PID}" > "${RESULTS_DIR}/.log_tail_pid"

# ── 等待 API 就绪 ──────────────────────────────────────────────
info "等待 API 服务就绪（最长 360s）..."
WAIT_TIMEOUT=360
ELAPSED=0
API_PORT_URL=""
while true; do
  for base in "http://localhost:${API_PORT}" "http://127.0.0.1:${API_PORT}" "http://${MASTER_ADDR}:${API_PORT}"; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "${base}/v1/chat/completions" 2>/dev/null) || true
    if [[ "${code}" == "405" || "${code}" == "200" ]]; then
      API_PORT_URL="${base}"
      break 2
    fi
  done
  if grep -q "your model is fired up" "${RESULTS_DIR}/ray_latest.log" 2>/dev/null; then
    [[ -z "${API_PORT_URL}" ]] && API_PORT_URL="http://localhost:${API_PORT}"
    break
  fi
  if (( ELAPSED >= WAIT_TIMEOUT )); then
    warn "超时 ${WAIT_TIMEOUT}s，请手动确认："
    warn "  tail -f ${RESULTS_DIR}/ray_latest.log"
    break
  fi
  sleep 5
  ELAPSED=$(( ELAPSED + 5 ))
  echo -ne "  已等待 ${ELAPSED}s...\r"
done

echo ""
echo -e "${GREEN}${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   ✅  OpenClaw-Combined 已就绪，可以开始实验！      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"
API_URL="${API_PORT_URL:-http://localhost:${API_PORT}}"
echo -e "  模式      : ${CYAN}${MODE_LABEL}${RESET}"
echo -e "  API       : ${CYAN}${API_URL}${RESET}"
echo -e "  记录文件  : ${CYAN}${RECORD_FILE}${RESET}"
echo -e "  PRM 记录  : ${CYAN}${RECORD_PRM_FILE}${RESET}"
echo -e "  实时日志  : ${CYAN}tail -f ${RESULTS_DIR}/ray_latest.log${RESET}"
echo -e "  OPD hint  : ${CYAN}grep 'OpenClaw-OPD' ${RESULTS_DIR}/ray_latest.log${RESET}"
echo -e "  停止      : ${CYAN}bash ${SCRIPT_DIR}/stop_combine.sh${RESET}"
echo ""
echo -e "  实验脚本  :"
echo -e "    conda activate openclaw-rl"
echo -e "    cd ${REPO_ROOT}/experiment"
echo -e "    python3 evaluate.py --tag baseline          # 先跑基准"
echo -e "    python3 train.py${DEMO_MODE:+ --demo}                     # 训练（含中间评估）"
echo ""
