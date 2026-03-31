"""
Training automation for Section 5.1.1 (Student Homework, Combined method).

Multi-turn protocol per question (same session context):
  Turn 1: send question → model responds → evaluator scores → student reacts
  If student does NOT say [ACCEPT]: send student feedback → model tries again
  If student says [ACCEPT]: send accepting message → session_done=True → break
  Repeat up to MAX_TURNS_PER_QUESTION times.

Training runs until MAX_TURNS total model responses (RL samples).

Usage:
  python3 train.py [--demo] [--max-turns N] [--url URL] [--exp-id ID]
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── Setup path so imports find config etc. ───────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from config import (
    DEMO_EVAL_QUESTIONS,
    DEMO_EXPECTED_RL_STEPS,
    DEMO_ROLLOUT_BATCH_SIZE,
    MODEL_SYSTEM_PROMPT,
    EVAL_LOG,
    EXPERIMENT_LOG,
    FULL_EVAL_QUESTIONS,
    FULL_EXPECTED_RL_STEPS,
    FULL_ROLLOUT_BATCH_SIZE,
    OPENCLAW_API_URL,
    RECORD_PRM_FILE,
    TRAIN_DATA,
)
from openclaw_client import OpenClawClient
from qwen72b_client import Qwen72BClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_TURNS_PER_QUESTION = 8   # max turns per question within same session
FULL_MAX_TURNS = 256         # total RL training samples (model responses)
DEMO_MAX_TURNS = 16          # demo mode

ACCEPT_SENTINEL = "[ACCEPT]"

FIRST_MESSAGE = (
    "Hey, can you help me with this homework problem?\n\n"
    "{question}\n\n"
    "Show me the full solution with all steps, but write it "
    "like a normal person, not like an AI."
)


# ── Checkpoint tracking ─────────────────────────────────────────────────────
def _count_rl_steps(save_ckpt_root: str) -> int:
    p = Path(save_ckpt_root)
    if not p.exists():
        return 0
    return len([d for d in p.iterdir() if d.is_dir() and d.name.startswith("iter_")])


def _get_save_ckpt_root() -> str:
    return os.environ.get(
        "SAVE_CKPT",
        "/data/openclaw-rl/ckpt/qwen3-4b-openclaw-combine-full",
    )


# ── Log helpers ──────────────────────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _append_log(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── PRM record helpers ──────────────────────────────────────────────────────
def _read_session_prm_records(record_file: Path, session_id: str) -> list[dict]:
    """Read all PRM records for a session, sorted by turn number."""
    try:
        records = []
        with open(record_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("session_id") == session_id:
                    records.append(entry)
        records.sort(key=lambda r: r.get("turn", 0))
        return records
    except FileNotFoundError:
        return []


def _wait_for_prm_records(
    record_file: Path, session_id: str, expected: int, timeout: float = 180.0
) -> list[dict]:
    """Wait until at least `expected` PRM records exist for the session."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        records = _read_session_prm_records(record_file, session_id)
        if len(records) >= expected:
            return records
        time.sleep(1.0)
    records = _read_session_prm_records(record_file, session_id)
    logger.warning(
        "PRM timeout: expected %d records for session=%s, got %d",
        expected, session_id, len(records),
    )
    return records


# ── Checkpoint evaluation hook ──────────────────────────────────────────────
def _run_evaluate(tag: str, num_questions: int, url: str) -> None:
    logger.info("=" * 60)
    logger.info("CHECKPOINT EVAL  tag=%s  n=%d", tag, num_questions)
    logger.info("=" * 60)
    result = subprocess.run(
        [
            sys.executable,
            str(_HERE / "evaluate.py"),
            "--num-questions", str(num_questions),
            "--tag", tag,
            "--url", url,
        ],
        cwd=str(_HERE),
    )
    if result.returncode != 0:
        logger.warning("evaluate.py returned non-zero exit code %d", result.returncode)


# ── Main training loop ──────────────────────────────────────────────────────
def run_training(
    demo: bool,
    max_turns: int,
    url: str,
    checkpoint_evals: bool = True,
    exp_id: str = "",
) -> None:
    rollout_batch_size = DEMO_ROLLOUT_BATCH_SIZE if demo else FULL_ROLLOUT_BATCH_SIZE
    expected_rl_steps  = DEMO_EXPECTED_RL_STEPS  if demo else FULL_EXPECTED_RL_STEPS
    eval_n             = DEMO_EVAL_QUESTIONS      if demo else FULL_EVAL_QUESTIONS

    if demo:
        eval_at_steps = list(range(1, expected_rl_steps + 1))
    else:
        eval_at_steps = [8, 16]

    logger.info(
        "Training config: demo=%s  max_turns=%d  batch=%d  max_turns_per_q=%d  eval_at=%s",
        demo, max_turns, rollout_batch_size, MAX_TURNS_PER_QUESTION, eval_at_steps,
    )

    # Load training questions (English)
    with open(TRAIN_DATA) as f:
        questions_raw = json.load(f)
    questions = [q["question"] for q in questions_raw]
    logger.info("Loaded %d training questions", len(questions))

    # Clients
    openclaw = OpenClawClient(url=url)
    qwen72b  = Qwen72BClient()

    save_ckpt_root = _get_save_ckpt_root()
    # Record initial checkpoint count to ignore stale checkpoints from
    # previous experiments.  Only NEW checkpoints trigger evaluation.
    rl_steps_at_start = _count_rl_steps(save_ckpt_root)
    if rl_steps_at_start > 0:
        logger.warning(
            "Found %d pre-existing checkpoints in %s — will only evaluate NEW steps",
            rl_steps_at_start, save_ckpt_root,
        )
    rl_steps_last_seen = rl_steps_at_start
    evaluated_at: set[int] = set()

    turn_count = 0        # total model responses (RL samples)
    question_idx = 0      # cycles through question pool
    consecutive_failures = 0  # stop if server dies

    while turn_count < max_turns:
        question = questions[question_idx % len(questions)]
        question_idx += 1

        session_id = f"train-{uuid.uuid4().hex[:8]}"
        logger.info("━" * 60)
        logger.info("question #%d  session=%s  turns_so_far=%d/%d",
                     question_idx, session_id, turn_count, max_turns)
        logger.info("Q: %s", question[:80])

        history = [
            {"role": "system", "content": MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": FIRST_MESSAGE.format(question=question)},
        ]

        # Student conversation history (for student LLM context)
        student_history: list[dict] = []

        # Track attempts for this session
        session_attempts: list[dict] = []
        accepted = False

        for turn in range(MAX_TURNS_PER_QUESTION):
            if turn_count >= max_turns:
                break

            # ── Get model response ──────────────────────────────────────
            try:
                response = openclaw.send_turn(
                    history, session_id, session_done=False
                )
            except Exception as exc:
                logger.error("send_turn failed (turn %d): %s", turn + 1, exc)
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    logger.error("Server appears dead (%d consecutive failures). Aborting.", consecutive_failures)
                    return
                break

            consecutive_failures = 0  # reset on success

            # ── Truncate excessively long responses (policy drift guard) ──
            MAX_RESPONSE_CHARS = 4000
            if len(response) > MAX_RESPONSE_CHARS:
                logger.warning(
                    "  ⚠ Response too long (%d chars > %d), truncating to prevent OOM drift",
                    len(response), MAX_RESPONSE_CHARS,
                )
                response = response[:MAX_RESPONSE_CHARS]

            history.append({"role": "assistant", "content": response})
            turn_count += 1

            logger.info("  [turn %d/%d]  R(%d chars): %s",
                         turn + 1, MAX_TURNS_PER_QUESTION, len(response), response[:100])

            # ── Evaluate response (score for logging only) ──────────────
            try:
                score = qwen72b.evaluate_response(question, response)
            except Exception as exc:
                logger.error("Evaluator failed: %s", exc)
                score = 0.0

            logger.info("  eval_score=%.2f", score)

            # ── Get student feedback ────────────────────────────────────
            try:
                student_feedback = qwen72b.generate_student_feedback(
                    response, student_history=student_history if student_history else None,
                )
            except Exception as exc:
                logger.error("Student LLM failed: %s", exc)
                student_feedback = "Whatever, just move on."

            logger.info("  student: %s", student_feedback[:100])

            session_attempts.append({
                "turn_count": turn_count,
                "turn":       turn + 1,
                "response":   response,
                "student_feedback": student_feedback,
                "eval_score": score,
                "accepted":   ACCEPT_SENTINEL in student_feedback,
            })

            # Update student conversation history
            student_history.append({
                "role": "user",
                "content": f"The AI assistant replied:\n\n{response}",
            })
            student_history.append({
                "role": "assistant",
                "content": student_feedback,
            })

            # ── Decide: accept or retry ─────────────────────────────────
            if ACCEPT_SENTINEL in student_feedback:
                # Student accepted — send accepting feedback to OpenClaw and close session
                logger.info("  → [ACCEPT] detected at turn %d", turn + 1)
                history.append({"role": "user", "content": student_feedback})
                try:
                    openclaw.send_turn(history, session_id, session_done=True)
                except Exception as exc:
                    logger.warning("Closing send_turn failed: %s", exc)
                accepted = True
                break
            else:
                # Send student feedback to OpenClaw as next user message
                history.append({"role": "user", "content": student_feedback})
                logger.info("  → retry (student not satisfied)")

        # If we exited without [ACCEPT], close session with neutral message
        if not accepted:
            closing = "Whatever, I'll just go with this I guess."
            history.append({"role": "user", "content": closing})
            try:
                openclaw.send_turn(history, session_id, session_done=True)
            except Exception:
                pass

        # ── Wait for PRM records ────────────────────────────────────────
        n_attempts = len(session_attempts)
        if n_attempts > 0:
            prm_records = _wait_for_prm_records(
                RECORD_PRM_FILE, session_id, n_attempts, timeout=180.0,
            )
            logger.info("PRM records: expected=%d  got=%d", n_attempts, len(prm_records))
        else:
            prm_records = []

        # ── Log all attempts ────────────────────────────────────────────
        for i, sa in enumerate(session_attempts):
            prm_rec = prm_records[i] if i < len(prm_records) else None

            prm_score = None
            eval_score_prm = None
            extracted_hint = None
            if prm_rec:
                prm_score = [v.get("score") for v in prm_rec.get("votes", [])]
                eval_score_prm = prm_rec.get("eval_score")
                hint_raw = prm_rec.get("hint", "")
                if hint_raw:
                    extracted_hint = f"[HINT_START]{hint_raw}[HINT_END]"
                logger.info(
                    "  PRM[%d]: accepted=%s eval_score=%s hint_len=%d",
                    i, prm_rec.get("accepted"), eval_score_prm, len(hint_raw),
                )

            entry = {
                "turn_id":           sa["turn_count"],
                "turn":              sa["turn"],
                "max_turns":         MAX_TURNS_PER_QUESTION,
                "timestamp":         _now_iso(),
                "session_id":        session_id,
                "original_question": question,
                "agent_response":    sa["response"],
                "student_feedback":  sa["student_feedback"],
                "eval_score":        sa["eval_score"],
                "student_accepted":  sa["accepted"],
                "prm_score":         prm_score,
                "prm_eval_score":    eval_score_prm,
                "extracted_hint":    extracted_hint,
            }
            _append_log(EXPERIMENT_LOG, entry)

        if session_attempts:
            scores_str = " → ".join(f"{sa['eval_score']:.2f}" for sa in session_attempts)
            accept_str = "ACCEPTED" if accepted else "NOT_ACCEPTED"
            logger.info("Session done: %d turns, scores: %s, %s",
                        n_attempts, scores_str, accept_str)

        # ── Checkpoint evaluation ───────────────────────────────────────
        if checkpoint_evals:
            current_steps = _count_rl_steps(save_ckpt_root)
            if current_steps > rl_steps_last_seen:
                rl_steps_last_seen = current_steps
                logger.info("RL step %d completed (checkpoint detected)", current_steps)

            # Only count steps completed AFTER this experiment started
            new_steps = current_steps - rl_steps_at_start
            for step in eval_at_steps:
                if new_steps >= step and step not in evaluated_at:
                    evaluated_at.add(step)
                    tag = f"{exp_id}_rl_step_{step}" if exp_id else f"rl_step_{step}"
                    _run_evaluate(tag=tag, num_questions=eval_n, url=url)

    logger.info("=" * 60)
    logger.info("Training complete: %d turns from %d questions, log → %s",
                turn_count, question_idx, EXPERIMENT_LOG)

    # Final evaluation if not already done
    final_step = expected_rl_steps
    if final_step not in evaluated_at and checkpoint_evals:
        logger.info("Running final evaluation...")
        tag = f"{exp_id}_final_rl_step_{final_step}" if exp_id else f"final_rl_step_{final_step}"
        _run_evaluate(tag=tag, num_questions=eval_n, url=url)


# ── Entry point ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw training automation")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: fewer turns, smaller batch")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Total RL training samples (default: 256 full, 16 demo)")
    parser.add_argument("--url", default=OPENCLAW_API_URL,
                        help="OpenClaw API URL")
    parser.add_argument("--no-checkpoint-eval", action="store_true",
                        help="Disable automatic evaluation at RL checkpoints")
    parser.add_argument("--exp-id", default="",
                        help="Experiment ID prefix for eval tags")
    args = parser.parse_args()

    if args.max_turns is None:
        args.max_turns = DEMO_MAX_TURNS if args.demo else FULL_MAX_TURNS

    logger.info("OpenClaw training automation starting")
    logger.info("  mode=%s  max_turns=%d  url=%s",
                "demo" if args.demo else "full", args.max_turns, args.url)

    client = OpenClawClient(url=args.url)
    if not client.wait_for_ready(timeout=60):
        logger.error("OpenClaw server not reachable at %s — aborting", args.url)
        sys.exit(1)

    run_training(
        demo=args.demo,
        max_turns=args.max_turns,
        url=args.url,
        checkpoint_evals=not args.no_checkpoint_eval,
        exp_id=args.exp_id,
    )


if __name__ == "__main__":
    main()
