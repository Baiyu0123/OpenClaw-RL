"""
Qwen 72B client via DashScope (OpenAI-compatible API).

Two separate components:
  1. Student LLM — generates natural feedback (sent to OpenClaw as next_state)
  2. Evaluator — scores the response independently (for logging/monitoring)
"""
import re
import logging
from typing import Optional

import openai

from config import (
    EVALUATOR_SYSTEM,
    QWEN72B_API_BASE,
    QWEN72B_API_KEY,
    QWEN72B_MODEL,
    STUDENT_PREFERENCE,
    STUDENT_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Regex to extract \boxed{N} from output
_BOXED_RE = re.compile(r"\\boxed\{([0-9.]+)\}")
_VALID_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}

# Regex to strip <think>...</think> blocks
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*")


def _make_client() -> openai.OpenAI:
    if not QWEN72B_API_KEY:
        raise ValueError(
            "QWEN72B_API_KEY is not set. "
            "Create experiment/.env with QWEN72B_API_KEY=sk-... "
            "or export the variable before running."
        )
    return openai.OpenAI(api_key=QWEN72B_API_KEY, base_url=QWEN72B_API_BASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return _THINK_RE.sub("", text).strip()


def _parse_boxed_score(text: str) -> float:
    """Extract the last \\boxed{N} value and snap to the nearest valid score."""
    matches = _BOXED_RE.findall(text)
    if matches:
        try:
            value = float(matches[-1])
            return min(_VALID_SCORES, key=lambda v: abs(v - value))
        except ValueError:
            pass
    # Fallback: look for standalone score-like numbers
    fallback = re.findall(r'\b(0(?:\.(?:25|5|75))?|1(?:\.0)?)\b', text)
    if fallback:
        try:
            value = float(fallback[-1])
            result = min(_VALID_SCORES, key=lambda v: abs(v - value))
            logger.info("Fallback parse: found score %.2f in text: %r", result, text[:80])
            return result
        except ValueError:
            pass
    logger.warning("No score found in evaluator output: %r", text[:120])
    return 0.0


class Qwen72BClient:
    """Thin wrapper around the DashScope-hosted Qwen 72B model."""

    def __init__(self):
        self._client = _make_client()

    def _chat(self, messages: list[dict], temperature: float = 0.6,
              max_tokens: int = 512) -> str:
        """Fire a chat request and return the assistant's reply."""
        try:
            resp = self._client.chat.completions.create(
                model=QWEN72B_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Qwen72B API error: %s", exc)
            raise

    # ── Student LLM: generate feedback ─────────────────────────────────────

    def generate_student_feedback(
        self,
        openclaw_response: str,
        student_history: list[dict] | None = None,
    ) -> str:
        """
        Have the student LLM react to OpenClaw's response.

        Args:
            openclaw_response: The latest response from OpenClaw.
            student_history: Prior turns as alternating user/assistant messages.
                user = "The AI assistant replied:\n\n{response}"
                assistant = "{previous student feedback}"

        Returns:
            The student's feedback string (may contain [ACCEPT]).
        """
        messages = [{"role": "system", "content": STUDENT_SYSTEM_PROMPT}]
        if student_history:
            messages.extend(student_history)
        messages.append({
            "role": "user",
            "content": f"The AI assistant replied:\n\n{openclaw_response}",
        })

        raw = self._chat(messages, temperature=0.6)
        feedback = _strip_thinking(raw)
        logger.info("Student feedback: %r", feedback[:100])
        return feedback

    # ── Evaluator: score the response ──────────────────────────────────────

    def evaluate_response(
        self,
        question: str,
        agent_response: str,
        preference: Optional[str] = None,
    ) -> float:
        """
        Score the response using the evaluator prompt.

        Args:
            question: The original problem.
            agent_response: The model's response to score.
            preference: The student preference string (substituted into [PREFERENCE]).

        Returns:
            Score in {0, 0.25, 0.5, 0.75, 1.0}.
        """
        pref = preference or STUDENT_PREFERENCE
        system = EVALUATOR_SYSTEM.replace("[PREFERENCE]", pref)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": (
                f"Problem:\n{question}\n\n"
                f"Assistant's solution:\n{agent_response}"
            )},
        ]

        raw = self._chat(messages, temperature=0.6)
        raw = _strip_thinking(raw)
        score = _parse_boxed_score(raw)
        logger.info(
            "Evaluator: score=%.2f  raw=%r  resp_len=%d",
            score, raw[:60], len(agent_response),
        )
        return score
