# Copyright 2025, LLM-PySC2 Contributors. All Rights Reserved.
#
# 顶层「策脉链」：独立于 Commander，仅负责 query_strategic、SPM（惯性记忆）与写入 spc_shared。
# Commander 只通过本模块发布的快照 + 下行反馈字符串与顶层通信，并继续统领 Developer/Builder/战斗组等子代理。

from __future__ import annotations

import json
import os
import re
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger
from pysc2.lib import features

from llm_pysc2.lib import utils

PULSE_ALIASES = {
    "进攻": "ATTACK",
    "防御": "DEFEND",
    "扩张": "EXPAND",
    "撤退": "RETREAT",
    "ATTACK": "ATTACK",
    "DEFEND": "DEFEND",
    "EXPAND": "EXPAND",
    "RETREAT": "RETREAT",
}


def _normalize_pulse(text: str, fallback: str) -> str:
  t = (text or "").replace("（", "(").replace("）", ")").strip()
  token = t.split("(")[0].strip().upper()
  if token in PULSE_ALIASES:
    return PULSE_ALIASES[token]
  for k, v in PULSE_ALIASES.items():
    if k in t:
      return v
  return fallback


def _scalar_game_loop(o) -> int:
  """PySC2 转换后的 observation 里 game_loop 常为 length-1 的 ndarray。"""
  gl = o.game_loop
  try:
    return int(np.asarray(gl, dtype=np.int64).reshape(-1)[0])
  except Exception:
    return int(gl)


def parse_strategic_reply(
    text: str,
    current_pulse: str,
    current_pivot: str,
    current_commander_intent: str,
    current_developer_intent: str,
) -> Tuple[str, str, str, int, str, str]:
  """Parse [Memory] / [Pulse] / [Pivot] / [Horizon] / [CommanderIntent] / [DeveloperIntent]."""
  if not text or not str(text).strip():
    return (
        current_pulse,
        current_pivot,
        "",
        0,
        current_commander_intent,
        current_developer_intent,
    )
  mem_m = re.search(
      r"\[Memory\]\s*(.*?)\s*\[/Memory\]",
      text, re.DOTALL | re.IGNORECASE)
  pulse_rows = re.findall(r'\[Pulse\]\s*([^\n\[]+)', text, re.IGNORECASE)
  pivot_rows = re.findall(r'\[Pivot\]\s*([^\n\[]+)', text, re.IGNORECASE)
  horizon_rows = re.findall(r'\[Horizon\]\s*([^\n\[]+)', text, re.IGNORECASE)
  commander_intent_rows = re.findall(r'\[CommanderIntent\]\s*([^\n\[]+)', text, re.IGNORECASE)
  developer_intent_rows = re.findall(r'\[DeveloperIntent\]\s*([^\n\[]+)', text, re.IGNORECASE)
  new_mem = mem_m.group(1).strip() if mem_m else ""
  new_pulse = pulse_rows[-1].strip() if pulse_rows else current_pulse
  new_pivot = pivot_rows[-1].strip() if pivot_rows else current_pivot
  new_commander_intent = (
      commander_intent_rows[-1].strip()
      if commander_intent_rows else current_commander_intent
  )
  new_developer_intent = (
      developer_intent_rows[-1].strip()
      if developer_intent_rows else current_developer_intent
  )
  try:
    horizon = int(float(horizon_rows[-1].strip())) if horizon_rows else 0
  except Exception:
    horizon = 0
  horizon = max(0, min(600, horizon))
  np_clean = _normalize_pulse(new_pulse, current_pulse)
  return np_clean, new_pivot, new_mem, horizon, new_commander_intent, new_developer_intent


class StrategicPulseBrain:
  """独立于 Commander 的宏观策脉模块：慢时钟调用 query_strategic，维护 SPM 并 publish。
  与 Commander 共用同一套 API 配置（传入 Commander 的 llm_client），但逻辑与指挥官战术 query 分离。"""

  def __init__(self, config, log_id: int, start_time: str, llm_client, log_dir_path: Optional[str] = None):
    self.config = config
    self.log_id = log_id
    self.start_time = start_time
    self.llm_client = llm_client
    self.log_dir_path = log_dir_path
    self._spc_log_initialized = False
    self.spc_clock_interval = int(getattr(config, 'SPC_STRATEGIC_STEP_INTERVAL', 8))
    self.last_strategic_game_loop = -1
    self.current_pulse = "DEFEND"
    self.current_pivot = "Situation unclear; keep scouting and defensive posture."
    self.current_commander_intent = "Stabilize, scout, and avoid over-extension."
    self.current_developer_intent = "Keep macro spending balanced: probes, production, and key tech."
    self.pulse_inertia = 0
    self.long_term_memory: Dict[str, Dict[str, object]] = {}
    self.last_decision_horizon = 0
    self.memory_slot_ttl_default = int(getattr(config, 'SPC_MEMORY_SLOT_TTL_DEFAULT', 6))
    self.memory_slot_ttl_max = int(getattr(config, 'SPC_MEMORY_SLOT_TTL_MAX', 18))
    self.self_profile_map_name = str(getattr(config, 'SPC_SELF_PROFILE_MAP_NAME', 'Simple64'))
    self.self_profile_race = str(getattr(config, 'SPC_SELF_PROFILE_RACE', 'Protoss'))
    self.self_profile_opponent = str(getattr(config, 'SPC_SELF_PROFILE_OPPONENT', 'Built-in AI'))
    self.scipt_enabled = bool(getattr(config, 'scipt', False))
    self.reference_playbook_name = "PvT_Colossus_DefenseCounter"

  def _spc_log_root(self) -> Optional[str]:
    if not self.log_dir_path:
      return None
    return os.path.join(self.log_dir_path, "SPC")

  def _ensure_spc_log_dir(self) -> Optional[str]:
    root = self._spc_log_root()
    if not root:
      return None
    if not self._spc_log_initialized:
      if not os.path.isdir(root):
        os.mkdir(root)
      lib_dir = os.path.dirname(os.path.abspath(__file__))
      template_show = os.path.normpath(os.path.join(lib_dir, "..", "..", "llm_log", "log_show.py"))
      if os.path.isfile(template_show):
        dest_show = os.path.join(root, "log_show.py")
        if not os.path.exists(dest_show):
          try:
            copyfile(template_show, dest_show)
          except OSError:
            pass
      prompt_path = os.path.join(root, "prompt.txt")
      if not os.path.exists(prompt_path):
        utils.write_to_file(
            "StrategicPulseBrain (SPC): top-level macro brain via query_strategic.\n"
            "Logged I/O mirrors other agents: o.txt = user macro prompt (observation text bundle); "
            "a_raw.txt = raw LLM reply; a_pro.txt = parsed Pulse/Pivot/Intents/Horizon.\n"
            "System role and map/race/opponent framing are fixed in lib/llm_client.query_strategic.\n",
            prompt_path,
        )
      for fn in ("o.txt", "a_raw.txt", "a_pro.txt", "a_his.txt", "cost.txt"):
        path = os.path.join(root, fn)
        if not os.path.exists(path):
          utils.write_to_file("", path)
      self._spc_log_initialized = True
    return root

  def _append_spc_io(
      self,
      log_step: int,
      text_o: str,
      raw_a: str,
      pro_a: str,
  ) -> None:
    root = self._ensure_spc_log_dir()
    if not root:
      return
    utils.write_to_file(json.dumps({log_step: text_o}), os.path.join(root, "o.txt"))
    utils.write_to_file(json.dumps({log_step: raw_a}), os.path.join(root, "a_raw.txt"))
    utils.write_to_file(json.dumps({log_step: pro_a}), os.path.join(root, "a_pro.txt"))
    c = self.llm_client
    cost = (
        f"time={getattr(c, 'query_time', 0):.2f}, ave_time={getattr(c, 'ave_query_time', 0):.2f}, "
        f"token_in={getattr(c, 'query_token_in', 0)}, ave_token_in={getattr(c, 'ave_query_token_in', 0):.2f}, "
        f"token_out={getattr(c, 'query_token_out', 0)}, ave_token_out={getattr(c, 'ave_query_token_out', 0):.2f}"
    )
    utils.write_to_file(json.dumps({log_step: cost}), os.path.join(root, "cost.txt"))

  def reset_episode(self) -> None:
    self.last_strategic_game_loop = -1
    self.current_pulse = "DEFEND"
    self.current_pivot = "Situation unclear; keep scouting and defensive posture."
    self.current_commander_intent = "Stabilize, scout, and avoid over-extension."
    self.current_developer_intent = "Keep macro spending balanced: probes, production, and key tech."
    self.pulse_inertia = 0
    self.long_term_memory = {}
    self.last_decision_horizon = 0

  def _render_memory_for_prompt(self) -> str:
    if not self.long_term_memory:
      return "(none)"
    rows = []
    for k in sorted(self.long_term_memory.keys()):
      rec = self.long_term_memory.get(k) or {}
      v = str(rec.get("value", "")).strip()
      if not v:
        continue
      c = float(rec.get("confidence", 0.5))
      ttl = int(rec.get("ttl", 0))
      rows.append(f"- {k}: {v} (c={c:.2f}, ttl={ttl})")
    text = "\n".join(rows).strip() or "(none)"
    return text

  def _decay_memory(self) -> None:
    if not self.long_term_memory:
      return
    to_del = []
    for k, rec in self.long_term_memory.items():
      ttl = int(rec.get("ttl", 0))
      if ttl <= 1:
        to_del.append(k)
      else:
        rec["ttl"] = ttl - 1
    for k in to_del:
      self.long_term_memory.pop(k, None)

  def _ingest_memory_block(self, raw_mem: str) -> None:
    raw = (raw_mem or "").strip()
    if not raw:
      return
    parsed = None
    try:
      parsed = json.loads(raw)
    except Exception:
      parsed = None
    updates: Dict[str, Dict[str, object]] = {}
    if isinstance(parsed, dict):
      for k, v in parsed.items():
        key = str(k).strip()
        if not key:
          continue
        if isinstance(v, dict):
          value = str(v.get("value", "")).strip()
          conf = float(v.get("confidence", 0.6))
          ttl = int(v.get("ttl", self.memory_slot_ttl_default))
        else:
          value = str(v).strip()
          conf = 0.6
          ttl = self.memory_slot_ttl_default
        if not value:
          continue
        updates[key] = {
            "value": value,
            "confidence": max(0.0, min(1.0, conf)),
            "ttl": max(1, min(self.memory_slot_ttl_max, ttl)),
            "source": "llm",
        }
    else:
      updates["notes"] = {
          "value": raw,
          "confidence": 0.5,
          "ttl": max(1, min(self.memory_slot_ttl_max, self.memory_slot_ttl_default)),
          "source": "llm",
      }
    for k, rec in updates.items():
      self.long_term_memory[k] = rec

  def _upsert_memory_slot(self, key: str, value: Any, confidence: float = 0.95) -> None:
    self.long_term_memory[str(key)] = {
        "value": str(value),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "ttl": self.memory_slot_ttl_max,
        "source": "scipt",
    }

  def _memory_text(self, key: str) -> str:
    rec = self.long_term_memory.get(key) or {}
    return str(rec.get("value", "")).strip()

  def _resolve_scipt_playbook_name(self) -> str:
    for k in ("playbook_name", "selected_playbook", "active_playbook"):
      v = self._memory_text(k)
      if v:
        return v
    return "AUTO_SELECT_FROM_SC2_KNOWLEDGE"

  def _build_scipt_playbook_state(self, obs, playbook_name: str) -> Dict[str, Any]:
    o = obs.observation
    gl = _scalar_game_loop(o)
    loops_per_s = float(getattr(self.config, 'SC2_GAME_LOOPS_PER_SECOND', 22.4))
    game_time_s = gl / loops_per_s

    if game_time_s < 300:
      stage = 1
      deadline_s = 300
      goal = "Establish economy and core tech for the selected playbook."
      stage_intent = "EXPAND"
    elif game_time_s < 420:
      stage = 2
      deadline_s = 420
      goal = "Unlock key tech structure and reinforce defenses around chokepoints."
      stage_intent = "DEFEND"
    elif game_time_s < 540:
      stage = 3
      deadline_s = 540
      goal = "Reach first major power spike unit mix and stabilize."
      stage_intent = "DEFEND"
    else:
      stage = 4
      deadline_s = 9999
      goal = "Counter-attack and convert defensive lead into map control."
      stage_intent = "ATTACK"

    food_army = -1
    if 'player' in o:
      try:
        food_army = int(o['player'][features.Player.food_army])
      except Exception:
        food_army = -1
    elif 'player_common' in o:
      try:
        food_army = int(getattr(o['player_common'], 'food_army', -1))
      except Exception:
        food_army = -1

    ru = o.get('raw_units') if hasattr(o, 'get') else getattr(o, 'raw_units', None)
    enemy_visible = 0
    if ru is not None and len(ru) > 0:
      enemy_visible = sum(1 for u in ru if u.alliance == features.PlayerRelative.ENEMY)

    interrupted = False
    status = "on_track"
    if enemy_visible >= 12 and food_army >= 0 and enemy_visible > (food_army + 6):
      interrupted = True
      status = "interrupted"
    elif stage < 4 and game_time_s > (deadline_s + 30):
      status = "delayed"

    return {
        "name": playbook_name,
        "stage": stage,
        "goal": goal,
        "deadline_s": deadline_s,
        "intent": stage_intent,
        "status": status,
        "interrupted": interrupted,
        "enemy_visible": enemy_visible,
        "food_army": food_army,
        "game_time_s": game_time_s,
    }

  def _sync_scipt_playbook_memory(self, pb: Dict[str, Any]) -> None:
    self._upsert_memory_slot("playbook_stage", pb["stage"])
    self._upsert_memory_slot("playbook_stage_goal", pb["goal"])
    self._upsert_memory_slot("playbook_stage_deadline_s", pb["deadline_s"])
    self._upsert_memory_slot("playbook_stage_intent", pb["intent"])
    self._upsert_memory_slot("playbook_status", pb["status"])

  def _render_scipt_playbook_block(self, pb: Dict[str, Any]) -> str:
    return (
        "[Playbook]\n"
        f"name={pb['name']}; stage={pb['stage']}; stage_status={pb['status']}; "
        f"stage_deadline_s={pb['deadline_s']}; default_intent={pb['intent']}.\n"
        f"stage_goal: {pb['goal']}\n"
        "selection_policy:\n"
        "- Select the playbook yourself using your SC2 knowledge and persist it in [Memory] with key playbook_name.\n"
        "- Keep continuity of the selected playbook unless there is strong evidence to switch.\n"
        "- If uncertain, use the reference playbook below.\n"
        "reference_playbook:\n"
        f"- name: {self.reference_playbook_name}\n"
        "reference_stage_table:\n"
        "- Stage 1 (0-300s): Double expand; build Robotics Facility.\n"
        "- Stage 2 (300-420s): Must build Robotics Bay; add choke Photon Cannons.\n"
        "- Stage 3 (420-540s): Produce Colossus; hold defensive posture.\n"
        "- Stage 4 (540s+): Counter-attack.\n"
        "rules:\n"
        "- Keep plan continuity by default.\n"
        "- If a stage objective is unfinished, you may delay this stage briefly and continue the same objective.\n"
        "- If sudden pressure spikes, issue temporary defensive commands, then return to playbook progression.\n\n"
    )

  def _should_run_strategic_tick(self, obs, num_step: int) -> bool:
    """按局内秒（game_loop）或按步数决定是否调用 query_strategic。"""
    game_sec = float(getattr(self.config, 'SPC_STRATEGIC_INTERVAL_GAME_SECONDS', 0.0))
    if game_sec > 0:
      gl = _scalar_game_loop(obs.observation)
      loops_per_s = float(getattr(self.config, 'SC2_GAME_LOOPS_PER_SECOND', 22.4))
      need = max(1, int(game_sec * loops_per_s))
      last = self.last_strategic_game_loop
      if last < 0:
        return gl >= need
      return (gl - last) >= need
    if self.spc_clock_interval <= 0 or num_step < 0:
      return False
    return num_step % self.spc_clock_interval == 0

  def maybe_run_tick(
      self,
      obs,
      num_step: int,
      *,
      main_loop_step: int = 0,  # reserved for correlating with tactical loops (logging uses num_step)
      commander_feedback: str = "",
      commander_observation_text: str = "",
  ) -> None:
    """Run in MainAgent loop; SPC owns long-term memory and may receive commander feedback."""
    if not getattr(self.config, 'ENABLE_SPC_STRATEGIC_LAYER', True):
      return
    if num_step < 0:
      return
    if not self._should_run_strategic_tick(obs, num_step):
      return
    if getattr(self.config, 'LLM_SIMULATION_TIME', 0) > 0:
      return
    if not hasattr(self.llm_client, 'query_strategic'):
      logger.warning(f"[ID {self.log_id}] StrategicBrain: no query_strategic, skip tick")
      return

    self._decay_memory()
    commander_obs_block = (commander_observation_text or "").strip()
    comm_block = ""
    if (commander_feedback or "").strip():
      comm_block += f"[CommanderExecutionFeedback]\n{commander_feedback.strip()[:2000]}\n\n"
    playbook_block = ""
    pb = None
    playbook_before = ""
    if self.scipt_enabled:
      playbook_before = self._resolve_scipt_playbook_name()
      pb = self._build_scipt_playbook_state(obs, playbook_before)
      self._sync_scipt_playbook_memory(pb)
      playbook_block = self._render_scipt_playbook_block(pb)

    doctrine = str(getattr(self.config, 'SPC_STRATEGIC_DOCTRINE', '') or '').strip()
    doctrine_block = ""
    if doctrine:
      doctrine_block = (
          "[StrategicDoctrine]\n"
          "Treat this as strategic guidance only (do not output executable game actions).\n"
          f"{doctrine}\n\n"
      )
    # Env step index: unique per SPC invocation (main_loop_step can stay flat across many steps).
    log_step = int(num_step)

    macro_prompt = (
        "Strategic guidance: Do not defend blindly. On Simple64 versus built-in AI, adapt tactics "
        "dynamically to map space, timings, and known bot tendencies.\n"
        "You are StrategicPulseBrain. Output ONLY these slots:\n"
        "[Pulse] ATTACK|DEFEND|EXPAND|RETREAT\n"
        "[Pivot] one concise sentence\n"
        "[CommanderIntent] one concise intent sentence for Commander\n"
        "[DeveloperIntent] one concise intent sentence for Developer\n"
        "[Horizon] integer seconds (0-600)\n"
        "[Memory] optional compact JSON object for stable strategic memory\n"
        "Never output executable tactical actions.\n"
        f"current_pulse: {self.current_pulse}; current_pivot: {self.current_pivot}; pulse_inertia_h_t: {self.pulse_inertia}.\n"
        f"current_commander_intent: {self.current_commander_intent}\n"
        f"current_developer_intent: {self.current_developer_intent}\n"
        f"last_decision_horizon_s: {self.last_decision_horizon}.\n"
        f"{playbook_block}"
        f"{doctrine_block}"
        f"[SPCLongTermMemoryPrev]\n{self._render_memory_for_prompt()}\n\n"
        f"{comm_block}"
        "=== CommanderObservationText (full, same source as Commander tactical observation) ===\n\n"
        f"{commander_obs_block if commander_obs_block else '(empty)'}\n"
    )
    try:
      raw = self.llm_client.query_strategic(macro_prompt)
      cand_pulse, cand_pivot, new_mem, horizon, cand_commander_intent, cand_developer_intent = parse_strategic_reply(
          raw,
          self.current_pulse,
          self.current_pivot,
          self.current_commander_intent,
          self.current_developer_intent,
      )
      old_pulse = self.current_pulse
      new_pulse = cand_pulse
      new_pivot = cand_pivot
      reason = "llm_direct"
      allow_switch = new_pulse != old_pulse
      self.last_decision_horizon = horizon
      self._ingest_memory_block(new_mem)
      if self.scipt_enabled and pb is not None:
        playbook_after = self._resolve_scipt_playbook_name()
        if playbook_after and playbook_after != playbook_before:
          logger.info(
              f"[ID {self.log_id}] StrategicBrain scipt playbook switched: "
              f"{playbook_before} -> {playbook_after}"
          )
          pb["name"] = playbook_after
        self._sync_scipt_playbook_memory(pb)
        if pb["interrupted"]:
          new_pulse = "DEFEND"
          new_pivot = (
              f"[PlaybookInterrupt] enemy_visible={pb['enemy_visible']}, army_supply={pb['food_army']}; "
              "stabilize defenses now, then return to playbook stage progression."
          )
          reason = "scipt_interrupt_override"

      if new_pulse == old_pulse:
        self.pulse_inertia += 1
      else:
        self.pulse_inertia = 0
      self.current_pulse = new_pulse
      self.current_pivot = new_pivot
      self.current_commander_intent = (cand_commander_intent or "").strip() or self.current_commander_intent
      self.current_developer_intent = (cand_developer_intent or "").strip() or self.current_developer_intent
      self.last_strategic_game_loop = _scalar_game_loop(obs.observation)
      logger.info(
          f"[ID {self.log_id}] StrategicBrain tick step={num_step} gl={self.last_strategic_game_loop} "
          f"pulse={self.current_pulse} pivot={self.current_pivot} inertia={self.pulse_inertia} "
          f"cand={cand_pulse} switch={allow_switch} reason={reason} "
          f"commander_intent={self.current_commander_intent} developer_intent={self.current_developer_intent}"
      )
      from llm_pysc2.lib.spc_shared import publish
      publish(
          self.log_id,
          self.start_time,
          self.current_pulse,
          self.current_pivot,
          self.pulse_inertia,
          (commander_feedback or "").strip(),
          self.current_commander_intent,
          self.current_developer_intent,
      )
      pro_summary = (
          f"main_loop_step={main_loop_step} env_step={num_step}\n"
          f"Pulse={self.current_pulse}\nPivot={self.current_pivot}\n"
          f"CommanderIntent={self.current_commander_intent}\n"
          f"DeveloperIntent={self.current_developer_intent}\n"
          f"Horizon={horizon}\nMemoryBlock={(new_mem or '')[:2000]}"
      )
      self._append_spc_io(log_step, macro_prompt, raw or "", pro_summary)
    except Exception as e:
      logger.error(f"[ID {self.log_id}] StrategicBrain tick failed: {e}")
      try:
        self._append_spc_io(log_step, macro_prompt, f"<error> {e}", "")
      except Exception:
        pass
