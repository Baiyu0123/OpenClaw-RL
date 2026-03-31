"""
Personalization evaluation for Section 5.1.1 (Student Homework, Combined method).

For each eval question:
  1. Send to OpenClaw in a fresh, isolated session (X-Session-Done: true immediately)
  2. Score the first response using Qwen72B evaluator (Appendix C.3)
  3. Log: question, response, score, session_id, timestamp, tag

Reports average score (to compare with Table 3: baseline≈0.17, step8≈0.76, step16≈0.81).

Usage:
  python3 evaluate.py [--num-questions N] [--tag TAG] [--url URL]
"""
import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from config import (
    EVAL_DATA,
    EVAL_LOG,
    FULL_EVAL_QUESTIONS,
    MODEL_SYSTEM_PROMPT,
    OPENCLAW_API_URL,
    STUDENT_PREFERENCE,
)
from openclaw_client import OpenClawClient
from qwen72b_client import Qwen72BClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _append_log(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_evaluation(
    num_questions: int,
    url: str,
    tag: str = "eval",
) -> float:
    """
    Evaluate the current model on the first `num_questions` eval questions.
    Returns the average personalization score.
    """
    # Load eval questions (use English)
    with open(EVAL_DATA) as f:
        eval_raw = json.load(f)
    eval_questions = eval_raw[:num_questions]

    logger.info("Evaluation: tag=%s  n=%d  url=%s", tag, len(eval_questions), url)

    openclaw = OpenClawClient(url=url)
    qwen72b  = Qwen72BClient()

    scores = []
    for i, item in enumerate(eval_questions, start=1):
        # Use English question for evaluation
        question = item.get("question_en") or item.get("question", "")
        expected_answer = item.get("answer", "")

        session_id = f"eval-{uuid.uuid4().hex[:8]}"
        logger.info("Eval %d/%d  session=%s  Q: %s", i, len(eval_questions),
                    session_id, question[:60])

        # Single turn, immediately mark session done
        # This does NOT submit a training sample (session_done prevents next_state eval)
        first_msg = (
            "Hey, can you help me with this homework problem?\n\n"
            f"{question}\n\n"
            "Show me the full solution with all steps, but write it "
            "like a normal person, not like an AI."
        )
        try:
            response = openclaw.send_turn(
                messages=[
                    {"role": "system", "content": MODEL_SYSTEM_PROMPT},
                    {"role": "user", "content": first_msg},
                ],
                session_id=session_id,
                session_done=True,
                max_tokens=2048,
            )
        except Exception as exc:
            logger.error("OpenClaw request failed: %s", exc)
            score = 0.0
            response = ""
        else:
            # Score with Qwen72B evaluator (Appendix C.3)
            try:
                score = qwen72b.evaluate_response(
                    question=question,
                    agent_response=response,
                    preference=STUDENT_PREFERENCE,
                )
            except Exception as exc:
                logger.error("Evaluator failed: %s", exc)
                score = 0.0

        scores.append(score)
        logger.info("  → score=%.2f  ans_expected=%s  resp_preview=%s",
                    score, expected_answer, response[:80])

        entry = {
            "tag":              tag,
            "eval_index":       i,
            "session_id":       session_id,
            "timestamp":        _now_iso(),
            "question_id":      item.get("id"),
            "original_question": question,
            "expected_answer":  expected_answer,
            "agent_response":   response,
            "score":            score,
        }
        _append_log(EVAL_LOG, entry)

    avg = sum(scores) / len(scores) if scores else 0.0

    # Summary
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULT  tag={tag}")
    print(f"  Questions scored : {len(scores)}")
    print(f"  Average score    : {avg:.4f}")
    print(f"  Individual scores: {[round(s, 2) for s in scores]}")
    print(f"  Log written to   : {EVAL_LOG}")
    print("=" * 60 + "\n")

    # Append a summary record for easy comparison
    _append_log(EVAL_LOG, {
        "tag":           tag,
        "eval_index":    "summary",
        "timestamp":     _now_iso(),
        "num_questions": len(scores),
        "average_score": avg,
        "all_scores":    scores,
    })

    return avg


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw personalization evaluation")
    parser.add_argument("--num-questions", type=int, default=FULL_EVAL_QUESTIONS,
                        help="Number of eval questions (default 36)")
    parser.add_argument("--tag",  default="eval",
                        help="Tag for this evaluation run (e.g. baseline, rl_step_8)")
    parser.add_argument("--url",  default=OPENCLAW_API_URL,
                        help="OpenClaw API URL")
    args = parser.parse_args()

    client = OpenClawClient(url=args.url)
    if not client.wait_for_ready(timeout=30):
        logger.error("OpenClaw server not reachable at %s", args.url)
        sys.exit(1)

    run_evaluation(
        num_questions=args.num_questions,
        url=args.url,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
