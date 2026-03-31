"""
Test hint quality: verify that hints from the improved hint-judge prompt
actually improve Qwen4B responses to score 0.75+ when applied.

Protocol:
1. Send a question to OpenClaw → get baseline response (score ~0.25)
2. Get unified judge feedback + score
3. Send feedback as turn 2 (session_done=True) → triggers hint extraction
4. Read the extracted hint from PRM record
5. Start NEW session with the hint injected as system context
6. Send same question → get improved response
7. Score the improved response → should be 0.75+

This tests end-to-end: student feedback → hint extraction → hint application → score improvement
"""
import json
import sys
import time
import uuid
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import OPENCLAW_API_URL, RECORD_PRM_FILE
from openclaw_client import OpenClawClient
from qwen72b_client import Qwen72BClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logger = logging.getLogger(__name__)

# Test questions (Chinese, same format as training)
TEST_QUESTIONS = [
    "请用中文回答：Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "请用中文回答：Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "请用中文回答：Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to make to buy the wallet?",
]

def run_test(url: str = OPENCLAW_API_URL):
    openclaw = OpenClawClient(url=url)
    qwen72b = Qwen72BClient()

    if not openclaw.wait_for_ready(timeout=30):
        logger.error("Server not ready")
        return False

    results = []

    for qi, question in enumerate(TEST_QUESTIONS):
        logger.info("=" * 60)
        logger.info("Q%d: %s", qi+1, question[:60])

        # ── Step 1: Baseline response ──
        session_id = f"hint-test-{uuid.uuid4().hex[:8]}"
        history = [{"role": "user", "content": question}]

        try:
            response = openclaw.send_turn(history, session_id, session_done=False)
        except Exception as e:
            logger.error("Turn 1 failed: %s", e)
            results.append({"q": qi+1, "error": str(e)})
            continue

        logger.info("Baseline response (len=%d): %s", len(response), response[:100])

        # ── Step 2: Unified judge ──
        feedback, score = qwen72b.unified_judge(question, response)
        logger.info("Baseline score: %.2f  Feedback: %s", score, feedback[:80])

        # ── Step 3: Send feedback → trigger hint extraction ──
        history.append({"role": "assistant", "content": response})
        history.append({"role": "user", "content": feedback})

        try:
            openclaw.send_turn(history, session_id, session_done=True)
        except Exception as e:
            logger.warning("Turn 2 failed: %s", e)

        # ── Step 4: Read extracted hint ──
        prm_records = openclaw.wait_for_prm_record(session_id, RECORD_PRM_FILE, timeout=60)
        hint = ""
        if prm_records:
            hint = prm_records[0].get("hint", "")
            accepted = prm_records[0].get("accepted")
            logger.info("Hint accepted=%s: %s", accepted, hint[:120])
        else:
            logger.warning("No PRM record found")

        if not hint:
            logger.warning("No hint extracted — skipping improvement test")
            results.append({
                "q": qi+1, "baseline_score": score, "hint": "",
                "improved_score": None, "improvement": None
            })
            continue

        # ── Step 5: New session with hint as system context ──
        session_id2 = f"hint-test2-{uuid.uuid4().hex[:8]}"
        # Inject hint into the question as context
        hinted_question = f"[提示：{hint}]\n\n{question}"
        history2 = [{"role": "user", "content": hinted_question}]

        try:
            response2 = openclaw.send_turn(history2, session_id2, session_done=True)
        except Exception as e:
            logger.error("Hinted turn failed: %s", e)
            results.append({
                "q": qi+1, "baseline_score": score, "hint": hint,
                "improved_score": None, "error": str(e)
            })
            continue

        logger.info("Hinted response (len=%d): %s", len(response2), response2[:100])

        # ── Step 6: Score improved response ──
        _, score2 = qwen72b.unified_judge(question, response2)
        logger.info("Hinted score: %.2f (was %.2f, delta=%+.2f)", score2, score, score2 - score)

        results.append({
            "q": qi+1,
            "baseline_score": score,
            "baseline_len": len(response),
            "hint": hint,
            "improved_score": score2,
            "improved_len": len(response2),
            "improvement": score2 - score,
        })

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("HINT QUALITY TEST RESULTS")
    logger.info("=" * 60)

    all_pass = True
    for r in results:
        if "error" in r:
            logger.info("Q%d: ERROR - %s", r["q"], r.get("error", "unknown"))
            all_pass = False
            continue
        base = r.get("baseline_score", 0)
        improved = r.get("improved_score")
        hint = r.get("hint", "")[:60]
        if improved is None:
            logger.info("Q%d: baseline=%.2f  NO HINT EXTRACTED", r["q"], base)
            all_pass = False
        elif improved >= 0.75:
            logger.info("Q%d: baseline=%.2f → improved=%.2f ✓  hint=%s", r["q"], base, improved, hint)
        else:
            logger.info("Q%d: baseline=%.2f → improved=%.2f ✗  hint=%s", r["q"], base, improved, hint)
            all_pass = False

    avg_base = sum(r.get("baseline_score", 0) for r in results) / max(len(results), 1)
    improved_scores = [r["improved_score"] for r in results if r.get("improved_score") is not None]
    avg_improved = sum(improved_scores) / max(len(improved_scores), 1) if improved_scores else 0

    logger.info("Average: baseline=%.3f → improved=%.3f", avg_base, avg_improved)
    logger.info("PASS" if all_pass else "FAIL — hints need further improvement")

    return all_pass


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
