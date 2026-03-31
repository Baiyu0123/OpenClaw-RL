#!/usr/bin/env bash
# =============================================================
#  实验清理脚本：归档旧日志 + 清理 checkpoint
#  用法:  bash experiment/cleanup.sh <新实验ID>
#  例如:  bash experiment/cleanup.sh exp26
# =============================================================
set -euo pipefail

RED='\033[31m'; GREEN='\033[32m'; CYAN='\033[36m'; YELLOW='\033[33m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
die()     { echo -e "${RED}[ERR]${RESET}  $*" >&2; exit 1; }

# ── 参数检查 ──────────────────────────────────────────────────
NEW_EXP_ID="${1:-}"
if [[ -z "$NEW_EXP_ID" ]]; then
  die "用法: bash experiment/cleanup.sh <新实验ID>  (例如: exp26)"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
COMBINE_RESULTS="${SCRIPT_DIR}/../openclaw-combine/results"
CKPT_DIR="/data/openclaw-rl/ckpt/qwen3-4b-openclaw-combine-full"

echo ""
echo -e "${CYAN}════════════════════════════════════════════${RESET}"
echo -e "${CYAN}  实验清理: 准备 ${NEW_EXP_ID}${RESET}"
echo -e "${CYAN}════════════════════════════════════════════${RESET}"
echo ""

# ── 1. 推断上一个实验 ID ──────────────────────────────────────
PREV_EXP_ID=""
if [[ -f "${RESULTS_DIR}/experiment_log.jsonl" ]] && [[ -s "${RESULTS_DIR}/experiment_log.jsonl" ]]; then
  # 从 eval_log 中找最近的 tag
  if [[ -f "${RESULTS_DIR}/eval_log.jsonl" ]] && [[ -s "${RESULTS_DIR}/eval_log.jsonl" ]]; then
    PREV_EXP_ID=$(grep -oP '"tag":\s*"\K(exp\d+)' "${RESULTS_DIR}/eval_log.jsonl" | tail -1 || true)
  fi
  if [[ -z "$PREV_EXP_ID" ]]; then
    PREV_EXP_ID="unknown_$(date +%Y%m%d_%H%M%S)"
  fi
fi

# ── 2. 归档日志 ──────────────────────────────────────────────
if [[ -n "$PREV_EXP_ID" ]]; then
  info "归档上一轮实验: ${PREV_EXP_ID}"

  for f in experiment_log.jsonl eval_log.jsonl; do
    src="${RESULTS_DIR}/${f}"
    if [[ -f "$src" ]] && [[ -s "$src" ]]; then
      dst="${RESULTS_DIR}/${PREV_EXP_ID}_${f}"
      if [[ -f "$dst" ]]; then
        warn "归档目标已存在: ${dst}，跳过"
      else
        cp "$src" "$dst"
        success "归档: ${f} → ${PREV_EXP_ID}_${f}"
      fi
    fi
  done
else
  info "没有找到需要归档的日志"
fi

# 清空当前日志（新实验从空文件开始）
> "${RESULTS_DIR}/experiment_log.jsonl" 2>/dev/null || true
> "${RESULTS_DIR}/eval_log.jsonl" 2>/dev/null || true
success "日志已清空"

# ── 3. 清空 PRM record ──────────────────────────────────────
PRM_FILE="${COMBINE_RESULTS}/qwen3_4b_combine_record_prm.jsonl"
if [[ -f "$PRM_FILE" ]]; then
  > "$PRM_FILE"
  success "PRM record 已清空"
fi

RECORD_FILE="${COMBINE_RESULTS}/qwen3_4b_combine_record.jsonl"
if [[ -f "$RECORD_FILE" ]]; then
  > "$RECORD_FILE"
  success "Record 文件已清空"
fi

# ── 4. 清理 checkpoint ──────────────────────────────────────
if [[ -d "$CKPT_DIR" ]]; then
  ITER_COUNT=$(find "$CKPT_DIR" -maxdepth 1 -name "iter_*" -type d | wc -l)
  if [[ $ITER_COUNT -gt 0 ]]; then
    info "删除 ${ITER_COUNT} 个旧 checkpoint..."
    rm -rf "${CKPT_DIR}"/iter_*
    rm -f "${CKPT_DIR}/latest_checkpointed_iteration.txt"
    success "checkpoint 已清理"
  else
    info "没有旧 checkpoint 需要清理"
  fi
else
  info "checkpoint 目录不存在，跳过"
fi

# ── 5. 汇总 ──────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════${RESET}"
echo -e "${GREEN}  清理完成，可以启动 ${NEW_EXP_ID}${RESET}"
echo -e "${GREEN}════════════════════════════════════════════${RESET}"
echo ""
echo "  下一步:"
echo "    bash openclaw-combine/start_combine.sh"
echo "    conda activate openclaw-rl"
echo "    python experiment/train.py --exp-id ${NEW_EXP_ID}"
echo ""
