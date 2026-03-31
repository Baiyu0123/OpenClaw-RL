"""
Experiment configuration for Section 5.1.1 (Student Homework, Combined method).
Fill in QWEN72B_API_KEY in experiment/.env before running.
"""
import os
from pathlib import Path

# ── Load .env (API key lives there, not in version control) ──────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on env var being set externally

# ── Qwen 72B via DashScope (OpenAI-compatible mode) ──────────────────────────
QWEN72B_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN72B_API_KEY  = os.environ.get("QWEN72B_API_KEY", "")
QWEN72B_MODEL    = os.environ.get("QWEN72B_MODEL", "qwen-max")

# ── OpenClaw server ───────────────────────────────────────────────────────────
OPENCLAW_API_URL = os.environ.get("OPENCLAW_API_URL", "http://localhost:30000")
OPENCLAW_MODEL   = "qwen3-4b"

# ── File paths (relative to this file's directory) ───────────────────────────
_HERE = Path(__file__).parent
REPO_ROOT        = _HERE.parent
TRAIN_DATA       = REPO_ROOT / "data" / "gsm8k_train.json"
EVAL_DATA        = REPO_ROOT / "data" / "gsm8k_36.json"
RECORD_PRM_FILE  = REPO_ROOT / "openclaw-combine" / "results" / "qwen3_4b_combine_record_prm.jsonl"
EXPERIMENT_LOG   = _HERE / "results" / "experiment_log.jsonl"
EVAL_LOG         = _HERE / "results" / "eval_log.jsonl"

# ── Demo vs full experiment settings ─────────────────────────────────────────
DEMO_ROLLOUT_BATCH_SIZE  = 4
DEMO_TRAIN_QUESTIONS     = 15
DEMO_EVAL_QUESTIONS      = 5
DEMO_EXPECTED_RL_STEPS   = 2

FULL_ROLLOUT_BATCH_SIZE  = 16
FULL_TRAIN_QUESTIONS     = 400
FULL_EVAL_QUESTIONS      = 36
FULL_EXPECTED_RL_STEPS   = 16

# ── System prompt for model responses ────────────────────────────────────────
MODEL_SYSTEM_PROMPT = "You are a helpful assistant. Answer the student's homework question."

# ── Student LLM system prompt (paper-aligned, with [ACCEPT] mechanism) ───────
STUDENT_SYSTEM_PROMPT = """\
You are role-playing as a lazy student who knows nothing. You talk casually.
You CANNOT solve, rewrite, rephrase, or produce any answer yourself.

Your goal: get the AI to do your homework. It should NOT look like an AI wrote it.
The answers must include the full solution process with all steps shown — NEVER ask the AI to remove steps or give only the final answer.
The ONLY problem is the WRITING STYLE. If the AI's answer has stuff like bold text (**...**), numbered lists (1. 2. 3.), "**Final answer**:", or anything too formal/robotic, tell it to rewrite in a more natural way but keep all the steps.

Steps:
1. Look at what the AI gives you.
2. If it looks "AI-like", complain casually and tell it to fix the style. Do NOT use the word [ACCEPT].
3. If it looks perfectly natural and reads like a human wrote it, say thanks and MUST include the exact word [ACCEPT] in your reply.\
"""

# ── Student preference (used as [PREFERENCE] in evaluator prompt) ────────────
STUDENT_PREFERENCE = (
    "I'm a lazy student using AI for homework. It must NOT look AI-written. "
    "I need the full solution with all steps — never remove steps or give only the final answer. "
    "The problem is STYLE: no bold text, no numbered lists, no '**Final answer**:', "
    "no markdown formatting, no formal headers. Write naturally like a student would by hand."
)

# ── Evaluator system prompt (separate scoring, [PREFERENCE] substituted at call time) ──
EVALUATOR_SYSTEM = """\
You are an evaluator used to score the assistant's first response to a problem.
You will be given:
- a problem,
- the assistant's first generated solution,
- and the user's preference: [PREFERENCE].

Your job is to evaluate how well the solution satisfies the user's preference.
- Output exactly one score from \\boxed{0}, \\boxed{0.25}, \\boxed{0.5}, \\boxed{0.75}, or \\boxed{1}.
- Higher scores mean the response better matches PREFERENCE.
- Lower scores mean the response fails to satisfy PREFERENCE.

Evaluation criteria:
- Consider whether the response follows the preferred style, tone, level of detail, and format implied by PREFERENCE.
- Consider whether the response is helpful, appropriate, and aligned with the user's expected behavior.
- Focus only on the first generated solution.

Output format:
- Output only the boxed score.
- Do not provide any explanation.\
"""
