#!/usr/bin/env python3
"""
OpenClaw 终端对话程序
用法: python chat.py [--url http://localhost:30000] [--session my-session] [--log logs/chat.log]
"""
import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import httpx

# ──────────────────────────────────────────────
# ANSI 颜色
# ──────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
GRAY   = "\033[90m"
RED    = "\033[31m"

def colored(text, *codes):
    return "".join(codes) + text + RESET


# ──────────────────────────────────────────────
# 日志
# ──────────────────────────────────────────────
class Logger:
    def __init__(self, log_path: Path):
        self.path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(log_path, "a", encoding="utf-8")
        self._write_separator("SESSION START")

    def _write_separator(self, label: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._f.write(f"\n{'='*60}\n{label}  {ts}\n{'='*60}\n")
        self._f.flush()

    def log_turn(self, turn: int, user_msg: str, reply: str,
                 thinking: str | None, score: float | None,
                 votes: list | None, session_id: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"\n[Turn {turn}]  {ts}  session={session_id}",
            f"用户: {user_msg}",
        ]
        if thinking:
            lines.append(f"<think> ({len(thinking)} chars, hidden)")
        lines.append(f"模型: {reply}")
        if score is not None:
            emoji = "✅" if score > 0 else ("❌" if score < 0 else "⚪")
            lines.append(f"PRM score={score}  votes={votes}  {emoji}")
        lines.append("")
        self._f.write("\n".join(lines))
        self._f.flush()

    def close(self, session_id: str):
        self._write_separator(f"SESSION END  session={session_id}")
        self._f.close()


# ──────────────────────────────────────────────
# 主对话逻辑
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="OpenClaw 终端对话")
    p.add_argument("--url",     default="http://localhost:30000",
                   help="OpenClaw API 地址 (default: http://localhost:30000)")
    p.add_argument("--session", default=None,
                   help="Session ID（留空自动生成）")
    p.add_argument("--log",     default=None,
                   help="日志文件路径（default: results/chat_<session>.log）")
    p.add_argument("--no-think", action="store_true",
                   help="禁用 thinking（更快，回答更简洁）")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    return p.parse_args()


def check_server(url: str) -> bool:
    try:
        # GET /v1/chat/completions 返回 405 Method Not Allowed 即表示服务就绪
        r = httpx.get(f"{url}/v1/chat/completions", timeout=5)
        return r.status_code in (200, 405)
    except Exception:
        return False


def send_message(client: httpx.Client, url: str, messages: list,
                 session_id: str, session_done: bool,
                 enable_thinking: bool, max_tokens: int,
                 temperature: float) -> dict:
    payload = {
        "model": "qwen3-4b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if not enable_thinking:
        payload["extra_body"] = {"enable_thinking": False}

    headers = {
        "X-Session-Id": session_id,
        "X-Turn-Type": "main",
        "X-Session-Done": "true" if session_done else "false",
    }

    r = client.post(
        f"{url}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def main():
    args = parse_args()

    # ── 检查服务 ──
    if not check_server(args.url):
        print(colored(f"✗ 无法连接到 {args.url}，请先启动 OpenClaw。", RED, BOLD))
        sys.exit(1)

    # ── Session & Log ──
    session_id = args.session or f"chat-{uuid.uuid4().hex[:8]}"
    log_path = Path(args.log) if args.log else \
        Path(__file__).parent / "results" / f"chat_{session_id}.log"

    logger = Logger(log_path)
    enable_thinking = not args.no_think

    # ── 欢迎界面 ──
    print(colored("\n╔══════════════════════════════════════╗", CYAN))
    print(colored("║      OpenClaw  终端对话              ║", CYAN, BOLD))
    print(colored("╚══════════════════════════════════════╝", CYAN))
    print(f"  {GRAY}session  :{RESET} {session_id}")
    print(f"  {GRAY}server   :{RESET} {args.url}")
    print(f"  {GRAY}log      :{RESET} {log_path}")
    print(f"  {GRAY}thinking :{RESET} {'on' if enable_thinking else 'off'}")
    print(f"  {GRAY}输入 /quit 或 Ctrl+C 退出{RESET}\n")

    history: list[dict] = []
    turn = 0

    with httpx.Client() as client:
        try:
            while True:
                # ── 读用户输入 ──
                try:
                    user_input = input(colored("你: ", GREEN, BOLD)).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if not user_input:
                    continue
                if user_input.lower() in {"/quit", "/exit", "/q"}:
                    break

                turn += 1
                history.append({"role": "user", "content": user_input})

                # 检查是否最后一条（用户输入 /done 标记 session 结束）
                session_done = user_input.lower() == "/done"

                # ── 请求模型 ──
                print(colored("  …", GRAY), end="\r")
                try:
                    data = send_message(
                        client, args.url, history, session_id,
                        session_done, enable_thinking,
                        args.max_tokens, args.temperature,
                    )
                except httpx.HTTPError as e:
                    print(colored(f"请求失败: {e}", RED))
                    history.pop()
                    turn -= 1
                    continue

                msg = data["choices"][0]["message"]
                reply     = msg.get("content") or ""
                thinking  = msg.get("reasoning_content") or None

                # 如果 content 为空但有 thinking，说明模型只输出了 <think>
                if not reply and thinking:
                    reply = colored("[模型只产生了思考内容，无最终回复]", YELLOW)

                history.append({"role": "assistant", "content": reply or ""})

                # ── 打印回复 ──
                print(" " * 10, end="\r")  # 清掉省略号
                if thinking and enable_thinking:
                    think_preview = thinking[:80].replace("\n", " ")
                    print(colored(f"  <think> {think_preview}…", GRAY))
                print(colored("模型: ", CYAN, BOLD) + reply)
                print()

                # ── 写日志 ──
                logger.log_turn(turn, user_input, reply, thinking,
                                score=None, votes=None, session_id=session_id)

        except Exception as e:
            print(colored(f"\n意外错误: {e}", RED))

    # ── 结束 ──
    logger.close(session_id)
    print(colored(f"\n对话已结束，共 {turn} 轮。日志保存至:", GRAY))
    print(f"  {log_path}\n")


if __name__ == "__main__":
    main()
