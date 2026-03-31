#!/usr/bin/env bash
# =============================================================
#  Demo 一键运行脚本
#  完整流程：启动服务 → 基准评估 → 训练（含中间评估）→ 汇报结果
#
#  预计运行时间：约 15-25 分钟（取决于 GPU 和 API 延迟）
#  Demo 参数：batch=4, 2 RL steps, 15 训练题, 5 评估题
#
#  前置条件：
#    1. 填写 experiment/.env 中的 QWEN72B_API_KEY
#    2. 确认模型路径: HF_CKPT=/data/openclaw-rl/models/Qwen3-4B
# =============================================================
set -euo pipefail

CYAN='\033[36m'; GREEN='\033[32m'; YELLOW='\033[33m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[DEMO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

echo -e "${CYAN}${BOLD}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║   OpenClaw 5.1.1 Student Experiment — DEMO RUN       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── 检查 API Key ──────────────────────────────────────────────
if [[ ! -f "${SCRIPT_DIR}/.env" ]]; then
  warn ".env 文件不存在，正在从模板创建..."
  cp "${SCRIPT_DIR}/.env.example" "${SCRIPT_DIR}/.env"
  warn "请编辑 experiment/.env，填入 QWEN72B_API_KEY 后重新运行"
  exit 1
fi
source "${SCRIPT_DIR}/.env" 2>/dev/null || true
if [[ -z "${QWEN72B_API_KEY:-}" ]]; then
  warn "QWEN72B_API_KEY 未设置，请在 experiment/.env 中填写"
  exit 1
fi
success "API Key 已配置"

# ── Step 1: 启动 Combined 服务（Demo 模式，batch=4）──────────
info "Step 1: 启动 OpenClaw-Combined 服务（demo mode）..."
bash "${REPO_ROOT}/openclaw-combine/start_combine.sh" --demo

# ── 激活 conda 环境 ───────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-openclaw-rl}"
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate "${CONDA_ENV}" 2>/dev/null || true
cd "${SCRIPT_DIR}"

# ── Step 2: 基准评估（训练前）────────────────────────────────
info "Step 2: 基准评估（应接近论文基准 0.17）..."
python3 evaluate.py --num-questions 5 --tag baseline
success "基准评估完成，查看: experiment/results/eval_log.jsonl"

# ── Step 3: 训练（含中间评估）────────────────────────────────
info "Step 3: 运行训练（15 道题，期望 2 次 RL 更新）..."
info "        训练将在每个 RL step 完成后自动触发中间评估"
python3 train.py --demo

# ── Step 4: 汇报结果 ──────────────────────────────────────────
info "Step 4: 汇报实验结果..."
echo ""
echo -e "${GREEN}${BOLD}═══════════ 实验结果摘要 ═══════════${RESET}"
python3 - << 'PYEOF'
import json
from pathlib import Path

log_file = Path(__file__).parent / "results" / "eval_log.jsonl" if False else Path("results/eval_log.jsonl")
if not log_file.exists():
    print("  eval_log.jsonl 不存在，请检查评估是否成功运行")
    exit(0)

summaries = []
with open(log_file) as f:
    for line in f:
        entry = json.loads(line.strip())
        if entry.get("eval_index") == "summary":
            summaries.append(entry)

if not summaries:
    print("  未找到 summary 记录")
else:
    print(f"  {'Tag':<25} {'n':>3}  {'Avg Score':>10}")
    print(f"  {'-'*25} {'-'*3}  {'-'*10}")
    for s in summaries:
        print(f"  {s['tag']:<25} {s['num_questions']:>3}  {s['average_score']:>10.4f}")
    print()
    print("  论文 Table 3 参考值: baseline≈0.17, step8≈0.76, step16≈0.81")
PYEOF

echo ""
echo -e "${GREEN}${BOLD}═══════════ 文件路径 ═══════════${RESET}"
echo -e "  训练日志: ${CYAN}experiment/results/experiment_log.jsonl${RESET}"
echo -e "  评估日志: ${CYAN}experiment/results/eval_log.jsonl${RESET}"
echo -e "  OPD hint: ${CYAN}grep 'OpenClaw-OPD' \$(cat openclaw-combine/results/.driver_log_path)${RESET}"
echo ""

success "Demo 运行完成！"
