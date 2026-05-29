# Copyright 2025, LLM-PySC2 Contributors. All Rights Reserved.
#
# Shared SPC (策脉) state: StrategicPulseBrain (MainAgent) publishes after query_strategic;
# Commander / Developer / Combat 等只读快照，不调用顶层宏观 LLM。

from __future__ import annotations

from typing import Any, Dict, Tuple

Key = Tuple[int, str]

_REGISTRY: Dict[Key, Dict[str, Any]] = {}


def _key(log_id: int, start_time: str) -> Key:
  return (log_id, start_time)


def publish(
    log_id: int,
    start_time: str,
    pulse: str,
    pivot: str,
    inertia: int,
    last_action_feedback: str = "",
    commander_intent: str = "",
    developer_intent: str = "",
) -> None:
  _REGISTRY[_key(log_id, start_time)] = {
      "pulse": pulse,
      "pivot": pivot,
      "inertia": int(inertia),
      "last_action_feedback": last_action_feedback or "",
      "commander_intent": commander_intent or "",
      "developer_intent": developer_intent or "",
  }


def get_snapshot(log_id: int, start_time: str) -> Dict[str, Any]:
  """Return last published SPC state, or defaults matching LLMAgent initial SPC."""
  k = _key(log_id, start_time)
  if k in _REGISTRY:
    return dict(_REGISTRY[k])
  return {
      "pulse": "DEFEND",
      "pivot": "Situation unclear; keep scouting and defensive posture.",
      "inertia": 0,
      "last_action_feedback": "",
      "commander_intent": "Stabilize, scout, and avoid over-extension.",
      "developer_intent": "Keep macro spending balanced: probes, production, and key tech.",
  }


def clear(log_id: int, start_time: str) -> None:
  """Remove published SPC for this run so a new SC2 episode does not read the previous game's brain."""
  _REGISTRY.pop(_key(log_id, start_time), None)
