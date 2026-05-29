# Copyright 2025, LLM-PySC2 Contributors. All Rights Reserved.
"""Optional stdout dump of LLM prompts/responses for live demos (graphic window + terminal).

Enable with env LLM_PYSC2_DEMO_IO=1 or config LLM_DEMO_PRINT_IO.
Full text without truncation: LLM_PYSC2_DEMO_IO_FULL=1
Max chars per input/output side: LLM_DEMO_IO_MAX_CHARS or env LLM_PYSC2_DEMO_IO_MAX_CHARS.

Structured recording for the HTML viewer: env LLM_PYSC2_IO_JSONL=1 (writes llm_layers.jsonl under the run log dir).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Optional


def demo_io_enabled(config: Optional[Any] = None) -> bool:
  e = str(os.environ.get("LLM_PYSC2_DEMO_IO", "")).strip().lower()
  if e in ("1", "true", "yes", "on"):
    return True
  if config is not None and getattr(config, "LLM_DEMO_PRINT_IO", False):
    return True
  return False


def _max_chars(config: Optional[Any]) -> Optional[int]:
  if str(os.environ.get("LLM_PYSC2_DEMO_IO_FULL", "")).strip().lower() in (
      "1", "true", "yes", "on"):
    return None
  env_n = os.environ.get("LLM_PYSC2_DEMO_IO_MAX_CHARS", "").strip()
  if env_n.isdigit():
    return int(env_n)
  n = getattr(config, "LLM_DEMO_IO_MAX_CHARS", 16000) if config is not None else 16000
  try:
    return int(n)
  except (TypeError, ValueError):
    return 16000


def _trunc(text: str, max_chars: Optional[int]) -> str:
  if text is None:
    return ""
  if max_chars is None or len(text) <= max_chars:
    return text
  return (
      text[:max_chars]
      + f"\n\n... <<< truncated, total {len(text)} chars; set LLM_PYSC2_DEMO_IO_FULL=1 for full >>>\n"
  )


def _append_llm_io_jsonl(
    config: Optional[Any],
    title: str,
    agent_name: str,
    log_id: int,
    text_in: str,
    text_out: str,
) -> None:
  if config is None or not getattr(config, "LLM_IO_JSONL_RECORD", False):
    return
  path = getattr(config, "LLM_IO_JSONL_PATH", "") or ""
  if not path:
    return
  rec = {
      "ts": time.time(),
      "title": title,
      "agent": agent_name,
      "log_id": log_id,
      "input": text_in or "",
      "output": text_out or "",
  }
  try:
    with open(path, "a", encoding="utf-8") as jf:
      jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
  except OSError:
    pass


def demo_print_llm_exchange(
    config: Optional[Any],
    title: str,
    agent_name: str,
    log_id: int,
    text_in: str,
    text_out: str,
) -> None:
  _append_llm_io_jsonl(config, title, agent_name, log_id, text_in, text_out)
  if not demo_io_enabled(config):
    return
  mc = _max_chars(config)
  sep = "#" * 72
  block = (
      f"\n{sep}\n"
      f"# LLM DEMO | {title} | agent={agent_name} | log_id={log_id}\n"
      f"{sep}\n"
      f"--- INPUT ---\n{_trunc(text_in, mc)}\n"
      f"--- OUTPUT ---\n{_trunc(text_out, mc)}\n"
      f"{sep}\n"
  )
  print(block, flush=True, file=sys.stdout)


def demo_serialize_chat_messages(messages: list) -> str:
  """Flatten OpenAI-style messages for console (hide huge base64)."""
  parts = []
  for m in messages:
    role = m.get("role", "?")
    c = m.get("content")
    if isinstance(c, str):
      parts.append(f"===== {role.upper()} =====\n{c}")
    elif isinstance(c, list):
      for chunk in c:
        if not isinstance(chunk, dict):
          parts.append(f"===== {role.upper()} =====\n{chunk!r}")
          continue
        t = chunk.get("type")
        if t == "text":
          parts.append(f"===== {role.upper()} (text) =====\n{chunk.get('text', '')}")
        elif t == "image_url":
          url = (chunk.get("image_url") or {}).get("url") or ""
          if url.startswith("data:") and len(url) > 120:
            parts.append(
                f"===== {role.upper()} (image) =====\n<data URL, {len(url)} chars>"
            )
          else:
            parts.append(f"===== {role.upper()} (image) =====\n{url!r}")
        else:
          parts.append(f"===== {role.upper()} ({t}) =====\n{chunk!r}")
    else:
      parts.append(f"===== {role.upper()} =====\n{c!r}")
  return "\n\n".join(parts)
