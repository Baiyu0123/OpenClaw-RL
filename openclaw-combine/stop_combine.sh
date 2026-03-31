#!/usr/bin/env bash
# OpenClaw-Combined 一键停止脚本
set -euo pipefail

RED='\033[31m'; GREEN='\033[32m'; CYAN='\033[36m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
CONDA_ENV="${CONDA_ENV:-openclaw-rl}"

JOB_ID_FILE="${RESULTS_DIR}/.last_job_id"
if [[ -f "${JOB_ID_FILE}" ]]; then
  JOB_ID=$(cat "${JOB_ID_FILE}")
  info "停止 Ray job: ${JOB_ID}"
  conda run -n "${CONDA_ENV}" ray job stop "${JOB_ID}" --no-wait 2>/dev/null || true
  rm -f "${JOB_ID_FILE}"
else
  info "未找到 job id 文件，跳过 job stop"
fi

LOG_TAIL_PID_FILE="${RESULTS_DIR}/.log_tail_pid"
if [[ -f "${LOG_TAIL_PID_FILE}" ]]; then
  LOG_PID=$(cat "${LOG_TAIL_PID_FILE}")
  kill "${LOG_PID}" 2>/dev/null || true
  rm -f "${LOG_TAIL_PID_FILE}"
fi

info "停止 Ray 集群..."
conda run -n "${CONDA_ENV}" ray stop --force 2>/dev/null || true
sleep 3

info "清理残留进程..."
pkill -9 sglang  2>/dev/null || true
pkill -9 ray     2>/dev/null || true
pkill -9 python  2>/dev/null || true
sleep 2
pkill -9 ray     2>/dev/null || true
pkill -9 python  2>/dev/null || true

success "所有进程已停止"

echo ""
info "当前 GPU 显存占用："
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader \
  | awk -F',' '{printf "  GPU %s %-24s used=%-10s free=%s\n", $1, $2, $3, $4}'
echo ""
