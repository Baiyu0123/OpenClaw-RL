"""
OpenClaw API client.
Handles:
  - Sending chat turns with the correct X-Session-Id / X-Turn-Type headers
  - Polling until the server is ready
  - Reading PRM records for a given session (with retry/timeout)
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from config import OPENCLAW_API_URL, OPENCLAW_MODEL, RECORD_PRM_FILE

logger = logging.getLogger(__name__)


class OpenClawClient:
    def __init__(self, url: str = OPENCLAW_API_URL, timeout: float = 120.0):
        self._url = url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    # ── Readiness probe ───────────────────────────────────────────────────

    def wait_for_ready(self, poll_interval: float = 5.0, timeout: float = 360.0) -> bool:
        """
        Poll GET /v1/chat/completions until HTTP 405 (or 200) is returned.
        Returns True on success, False on timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = self._http.get(f"{self._url}/v1/chat/completions", timeout=5)
                if r.status_code in (200, 405):
                    logger.info("OpenClaw server ready at %s", self._url)
                    return True
            except Exception:
                pass
            time.sleep(poll_interval)
        logger.warning("OpenClaw server not ready after %.0fs", timeout)
        return False

    # ── Chat turn ─────────────────────────────────────────────────────────

    def send_turn(
        self,
        messages: list[dict],
        session_id: str,
        session_done: bool = False,
        enable_thinking: bool = False,
        max_tokens: int = 2048,
        temperature: float = 0.6,
    ) -> str:
        """
        POST /v1/chat/completions with OpenClaw-specific headers.
        Returns the assistant's text response (reasoning stripped).

        The 'messages' list is the full conversation history including the
        latest user turn. The server uses the latest user message as the
        "next_state" signal for evaluating the previous assistant turn.
        """
        payload = {
            "model": OPENCLAW_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if not enable_thinking:
            payload["extra_body"] = {"enable_thinking": False}

        headers = {
            "X-Session-Id":   session_id,
            "X-Turn-Type":    "main",
            "X-Session-Done": "true" if session_done else "false",
        }

        # Retry on 503 (server busy during RL weight update, typically 2-5s)
        max_retries = 12          # up to 60s of retries
        retry_interval = 5.0
        last_exc = None
        for attempt in range(max_retries):
            try:
                r = self._http.post(
                    f"{self._url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                r.raise_for_status()
                data = r.json()
                break  # success
            except Exception as exc:
                last_exc = exc
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                if status == 503 and attempt < max_retries - 1:
                    logger.warning("503 Service Unavailable (attempt %d/%d), retrying in %.0fs...",
                                   attempt + 1, max_retries, retry_interval)
                    import time; time.sleep(retry_interval)
                    continue
                logger.error("OpenClaw request failed: %s", exc)
                raise
        else:
            logger.error("OpenClaw request failed after %d retries: %s", max_retries, last_exc)
            raise last_exc

        choice = data["choices"][0]["message"]
        # Qwen3's reasoning parser may put thinking in reasoning_content
        content = choice.get("content") or choice.get("reasoning_content") or ""
        return content.strip()

    # ── PRM record reading (with polling) ─────────────────────────────────

    def wait_for_prm_record(
        self,
        session_id: str,
        record_file: Optional[Path] = None,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
    ) -> Optional[list[dict]]:
        """
        Poll the PRM record JSONL file until an entry for this session_id appears.
        Returns the list of matching records, or None on timeout.

        This avoids the fragile hard-coded sleep(5) approach.
        The record file is written asynchronously by the PRM eval task;
        we simply poll until the entry arrives.
        """
        path = record_file or RECORD_PRM_FILE
        deadline = time.time() + timeout
        while time.time() < deadline:
            records = _read_prm_records(path, session_id)
            if records:
                return records
            time.sleep(poll_interval)
        logger.warning(
            "PRM record for session=%s not found after %.0fs (file=%s)",
            session_id, timeout, path,
        )
        return None


def _read_prm_records(path: Path, session_id: str) -> list[dict]:
    """Read all PRM record entries matching the given session_id."""
    try:
        records = []
        with open(path) as f:
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
        return records
    except FileNotFoundError:
        return []
    except Exception as exc:
        logger.debug("Error reading PRM record file: %s", exc)
        return []
