# Tool-Call RL: Local Setup for 8xA100-40GB

## Prerequisites

- Conda environment `openclaw-rl` with all dependencies installed
- Models at `/data/openclaw-rl/models/`:
  - `qwen3-4b-retool-sft` (SFT checkpoint, used as policy + reference)
  - `Qwen3-4B` (base model, used as PRM judge)
- Data at `/data/openclaw-rl/data/`:
  - `dapo_math_17k.jsonl` (17k math training problems)
  - `aime-2024.jsonl` (30 AIME 2024 eval problems)

## Key Adaptations for A100-40GB

Compared to the upstream scripts (which target 80GB GPUs):

1. `--megatron-to-hf-mode bridge` — loads HF checkpoint directly, no torch_dist conversion needed
2. `--ref-load` points to the same HF checkpoint (bridge mode handles this)
3. `--max-tokens-per-gpu 8192` (halved from 16384 for 40GB VRAM)
4. No `pkill` at script start — safe for shared machines
5. GPU busy check at startup with 10s grace period
6. All paths on `/data` (root disk only has 16GB free)
7. Wandb disabled by default (set `WANDB_KEY` to enable)

## Scripts

| Script | Mode | GPU Layout | Duration |
|--------|------|------------|----------|
| `retool_qwen3_4b_rl_demo.sh` | Quick demo | 4A + 4R = 8 | ~45 min |
| `retool_qwen3_4b_rl_local.sh` | Full RL | 4A + 4R = 8 | ~24-48h |
| `retool_qwen3_4b_prm_rl_local.sh` | PRM + RL | 2A + 4R + 2P = 8 | ~24-48h |

(A = Actor/Training, R = Rollout/Inference, P = PRM Judge)

## How to Run

```bash
conda activate openclaw-rl
cd /home/test6/OpenClaw-RL/slime

# Demo (~45 min, 50 rollout steps, small batches)
bash ../toolcall-rl/retool_qwen3_4b_rl_demo.sh

# Full training (~24-48h, 3000 rollout steps)
bash ../toolcall-rl/retool_qwen3_4b_rl_local.sh

# PRM + RL with step-wise rewards
bash ../toolcall-rl/retool_qwen3_4b_prm_rl_local.sh
```

## Monitoring

- Ray Dashboard: `http://<server-ip>:8265`
- Logs: stdout (or tee to a file)
- Checkpoints: `/data/openclaw-rl/ckpt/<run-name>/`

## Stopping

```bash
ray stop --force
```

## Demo Results (2026-04-01)

- Job: `SUCCEEDED` in ~45 minutes
- 50 rollout steps completed, 2 AIME evaluations
- eval/aime: 0.0167 (1/60 correct — normal for 4B model on AIME)
- Training sample success rate: ~8.8% (9/102 correct on DAPO-Math-17k)
- Checkpoint saved: `/data/openclaw-rl/ckpt/qwen3-4b-retool-rl-demo/iter_0000049`
