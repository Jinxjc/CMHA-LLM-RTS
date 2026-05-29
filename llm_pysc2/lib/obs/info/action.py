# Copyright 2025, LLM-PySC2 Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from llm_pysc2.lib.action import space as action_space
from llm_pysc2.lib.utils import *
from llm_pysc2.lib.action.condition import *


from loguru import logger
import re


# 高风险技能：能量不足或武器/技能 CD 未好时从合法动作中剔除（基于 feature_units）。
# 未列出的 Ability_* 默认按 DEFAULT_ABILITY_ENERGY 要求能量。
DEFAULT_ABILITY_ENERGY = 50
ABILITY_WEAPON_COOLDOWN_MAX = 0  # 帧；0 表示观测上必须完全就绪

ABILITY_ENERGY_REQUIRED = {
    'Ability_Blink_Screen': 50,
    'Ability_PsiStorm_Screen': 75,
    'Ability_PsiStorm_Attack_Unit': 75,
    'Ability_ForceField_Screen': 50,
    'Ability_TimeWarp_Screen': 50,
    'Ability_TimeWarp_Attack': 50,
    'Ability_GuardianShield': 50,
    'Ability_ChronoBoost_Unit': 50,
    'Ability_NexusMassRecall_Near': 50,
    'Ability_MothershipMassRecall_Near': 50,
    'Ability_ShadowStride_Unit': 50,
    'Ability_GravitonBeam_Unit': 0,
    'Ability_OracleRevelation_Screen': 25,
    'Ability_StasisTrap_Screen': 50,
    'Ability_PurificationNova_Screen': 100,
    'Ability_PurificationNova_Attack_Unit': 100,
    'Ability_AdeptPhaseShift_Screen': 0,
    'Ability_AdeptPhaseShift_Minimap': 0,
    'Ability_CancelPhaseShift': 0,
    'Ability_PulsarBeamOn': 0,
    'Ability_PrismaticAlignment': 0,
    'Select_Phoenix_Ability_GravitonBeam_Unit': 0,
    'Select_Unit_Ability_PulsarBeamOn': 0,
}


def _is_high_risk_ability_name(action_name: str) -> bool:
  return 'Ability_' in action_name


def _selected_team_feature_units(obs, team):
  """当前队伍在屏幕上的选中单位（与 get_valid_actions 中 ctrl 逻辑一致）。"""
  selected = []
  for unit in obs.observation.feature_units:
    if not getattr(unit, 'is_selected', False):
      continue
    if unit.tag not in team['unit_tags']:
      continue
    if unit.alliance != features.PlayerRelative.SELF:
      continue
    selected.append(unit)
  return selected


def _ability_physically_castable(action_name: str, selected_units) -> bool:
  """是否存在至少一个选中单位满足能量与 weapon_cooldown（技能 CD 代理）。"""
  if not selected_units:
    return False
  required = ABILITY_ENERGY_REQUIRED.get(action_name, DEFAULT_ABILITY_ENERGY)
  for u in selected_units:
    en = int(getattr(u, 'energy', 0) or 0)
    wd = int(getattr(u, 'weapon_cooldown', 0) or 0)
    if en >= required and wd <= ABILITY_WEAPON_COOLDOWN_MAX:
      return True
  return False


def _prune_valid_actions_by_ability_physics(valid_actions, obs, team, log_id: int):
  """在返回给 LLM 前裁剪：Ability_* 在能量不足或 CD 中则移除。"""
  selected = _selected_team_feature_units(obs, team)
  pruned = []
  for action in valid_actions:
    name = action.get('name', '')
    if not _is_high_risk_ability_name(name):
      pruned.append(action)
      continue
    if not selected:
      logger.debug(f"[ID {log_id}] Prune {name}: no selected feature_units to verify energy/CD")
      continue
    if _ability_physically_castable(name, selected):
      pruned.append(action)
    else:
      logger.debug(f"[ID {log_id}] Prune {name}: insufficient energy or ability on cooldown (weapon_cooldown)")
  return pruned


def extract_allowed_tactical_action_names(valid_actions_text: str):
  """从 Valid Actions 文本中抽取形如 <Name(...)> 的动作名集合，供战术输出门控。"""
  if not valid_actions_text:
    return set()
  return set(re.findall(r'<(\w+)\s*\(', valid_actions_text))


# 通讯模块解析的伪动作，不是 pysc2 available_actions；不得与「Valid actions」里的游戏动作混为一谈。
TACTICAL_GATE_COMMUNICATION_NAMES = frozenset({'MessageTo', 'ListenTo'})


def _tactical_bracket_action_names(raw_llm_text: str):
  """从模型输出中抽取 <Name(...)> 里的 Name（与 extract_allowed 使用同一正则）。"""
  if not raw_llm_text:
    return []
  return re.findall(r'<(\w+)\s*\(', str(raw_llm_text))


def tactical_gate_validate(
    raw_llm_text: str,
    allowed_names: set,
    *,
    allow_communication_only: bool = False,
) -> bool:
  """Require at least one bracket action and enforce allowed game action names.

  MessageTo / ListenTo are communication pseudo-actions and are excluded from
  game-action validation. If allow_communication_only=True (Commander),
  communication-only outputs are allowed.
  """
  if not raw_llm_text or not str(raw_llm_text).strip():
    return False
  if not re.search(r'<[^>]+>', raw_llm_text):
    return False
  if not allowed_names:
    return True
  found_all = _tactical_bracket_action_names(raw_llm_text)
  if not found_all:
    return False
  comm = TACTICAL_GATE_COMMUNICATION_NAMES
  game_names = [n for n in found_all if n not in comm]
  if not game_names:
    if allow_communication_only and all(n in comm for n in found_all):
      return True
    return False
  return all(name in allowed_names for name in game_names)


def tactical_gate_feedback(
    raw_llm_text: str,
    allowed_names: set,
    *,
    allow_communication_only: bool = False,
) -> str:
  """Detailed retry feedback for tactical sandbox; empty means pass."""
  if tactical_gate_validate(
      raw_llm_text, allowed_names, allow_communication_only=allow_communication_only
  ):
    return ""
  if not raw_llm_text or not str(raw_llm_text).strip():
    return "Output is empty. Provide at least one instruction like <Action_Name(...)>."
  if not re.search(r'<[^>]+>', raw_llm_text):
    return "No bracket action detected. Output only legal actions in format <Name(...)>."
  if not allowed_names:
    return "Action format is still invalid. Check angle brackets and parentheses."
  found_all = _tactical_bracket_action_names(raw_llm_text)
  if not found_all:
    return "Actions must be written as <Name(args...)> (Name adjacent to parentheses); avoid pure descriptive text."
  comm = TACTICAL_GATE_COMMUNICATION_NAMES
  game_names = [n for n in found_all if n not in comm]
  if not game_names:
    if allow_communication_only and all(n in comm for n in found_all):
      return ""
    return (
        "No legal game action detected. MessageTo/ListenTo are communication formats and do not count as Valid Actions. "
        "If your role includes build/train/research, output at least one exact <ActionName(...)> from the current list."
    )
  bad = [n for n in game_names if n not in allowed_names]
  if bad:
    return (
        "These game action names are not in the current Valid Actions list: "
        + ", ".join(bad)
        + ". Choose action names exactly from the list; unlisted names are currently unavailable."
    )
  return "Output did not pass validation. Fix and retry."


# 根据obs获取合法动作，以文本格式输出，这个需要作为input prompt的一个独立部分
def get_valid_actions(agent) -> str:

  text_valid_actions = "Valid Actions:"
  for team in agent.teams:
    team_obs_list = team['obs'] if (len(team['obs']) != 0 and len(team['unit_tags']) != 0) else None
    if team['select_type'] == 'select' and len(team['obs']) != len(team['unit_tags']) or team_obs_list is None:
      continue

    for i in range(len(team_obs_list)):

      # determine current controlled unit types
      ctrl_unit_type = []
      obs = team_obs_list[i]
      for unit in obs.observation.feature_units:
        if unit.is_on_screen and unit.is_selected and unit.tag in team['unit_tags']:
          ctrl_unit_type.append(unit.unit_type)
      ctrl_unit_type = list(set(ctrl_unit_type))

      # determine current team name
      team_name = f"Team {team['name']}-{i+1}" if team['select_type'] == 'select' else f"Team {team['name']}"
      text_valid_actions += f"\n\t{team_name} Valid Actions:"

      # reduce to team action space
      team_config = agent.config.AGENTS[agent.name]['team'][team['name']]
      team_action_space = []
      for unit_type in ctrl_unit_type:
        if unit_type in list(team_config['actions'].keys()):
          team_action_space += team_config['actions'][unit_type]
        else:
          logger.error(f"[ID {agent.log_id}] cannot get valid actions of unit_type {unit_type}")

      # reduce to obs.observation.available_actions
      valid_actions = []
      for action in team_action_space:
        valid = True
        # print(action)
        for func_triple in action['func']:
          if func_triple[0] not in obs.observation.available_actions:
            valid = False
        # if 'Attack' in action['name'] and 'Ability' not in action['name']:
        #   valid = llm_action.check_weapon_state(team['obs'][0], 'now', None)
        if valid:
          valid_actions.append(action)

      valid_actions = _prune_valid_actions_by_ability_physics(
          valid_actions, obs, team, agent.log_id)

      # TODO: special actions
      #  这些动作是执行时临时选择建筑的，因此无法在obs中查看合法性，需要根据资源/前置条件/闲置建筑另行判断
      for action in team_action_space:
        if 'Build_' in action['name'] and '_Easy' in action['name']:  # enough minerals and gas, easy mode, do not select worker
          pass
        if 'Train_' in action['name']:  # enough minerals and gas, exist relevant building active == 0, and in power
          pass
        if 'WarpTrain_' in action['name']:  # enough minerals and gas, exist relevant building active == 0, and in power
          pass
        if 'Research_' in  action['name']:  # enough minerals and gas, exist relevant building active == 0, and in power
          pass

      # record valid actions
      for action in valid_actions:
        arg = action['arg']
        if len(arg) == 0:
          text_valid_actions += f"\n\t\t<{action['name']}()>"
        if len(arg) == 1:
          text_valid_actions += f"\n\t\t<{action['name']}({arg[0]})>"
        if len(arg) == 2:
          text_valid_actions += f"\n\t\t<{action['name']}({arg[0]}, {arg[1]})>"
        if len(arg) == 3:
          text_valid_actions += f"\n\t\t<{action['name']}({arg[0]}, {arg[1]}, {arg[2]})>"
  text_valid_actions += "\n\n"
  return text_valid_actions


# record action arg explanation
def get_valid_action_args_explanation(agent):
  obs = agent.team_unit_obs_list[0]
  text_arg_explanation = ''
  size_screen = obs.observation.feature_screen.height_map.shape[0]
  size_minimap = obs.observation.feature_minimap.height_map.shape[0]
  ratio = size_screen / SCREEN_WORLD_GRID

  screen_edge = f"where x and y range from 0 to {int(size_screen/ratio)}."
  if len(agent.teams) > 0:
    team = agent.teams[0]
    if 'l' in team.keys():
      screen_edge = f"{int(team['l']/ratio)} < x < {int(team['r']/ratio)}, {int(team['u']/ratio)} < y < {int(team['b']/ratio)}"
    else:
      screen_edge = f"0 < x < {int(size_screen/ratio)}, 0 < y < {int(size_screen/ratio)}"

  text_arg_explanation += f"Action Args: "
  text_arg_explanation += f"\n\t(1) tag: tag refers to a hexadecimal number, shape as 0x000000000."
  text_arg_explanation += f"\n\t(2) screen: screen refers to a screen coordinate, shape as [x, y], where {screen_edge}."
  text_arg_explanation += f"\n\t(3) minimap: minimap refers to a minimap coordinate, shape as [x, y], where x and y range from 0 to {size_minimap}."
  text_arg_explanation += f"\nFor example, when you want to use an action like <Action_Name(tag, screen)>, you should output like <Action_Name(0x100580001, [12, 16])>; when you want to use an action like <Action_Name(screen)>, you should output like <Action_Name([20, 8])>. "
  text_arg_explanation += f"What's more, You need to see clearly whether an action is using screen coordinates or minimap coordinates, If an action name as XXXX_Screen, it uses screen coordinate; if an action name as XXXX_Minimap, it uses minimap coordinate."
  if len(agent.teams) > 0:
    team = agent.teams[0]
    if 'l' in team.keys():
      if (team['l'] != 0 or team['u'] != 0 or team['r'] != size_screen - 1 or team['b'] != size_screen - 1):
        text_arg_explanation += f"\nWarning! controlled team near the map edge! Pay attention to using coordinates within the boundary!"
  text_arg_explanation += "\n\n"
  return text_arg_explanation


def get_last_action_info(agent) -> str:
  text_last_action = ""
  if isinstance(agent.last_text_a_pro, str) and len(agent.last_text_a_pro) > 0:
    text_last_action += f"Last Step {agent.last_text_a_pro}"
    text_last_action += f"\nYou need to confirm whether the previous action finished executing, and based on this, determine whether to continue the old strategy or immediately take other actions."
    text_last_action += "\n\n"
  return text_last_action


def get_valid_actions_build(agent) -> (list, str):
  obs = agent.team_unit_obs_list[0]
  _, _, ba, _, _, bc, m, g, s, u, b = get_condition_elements(agent)
  easy_build = bool(getattr(agent.config, 'ENABLE_EASY_BUILD', False))
  if easy_build:
    try:
      standard_build_actions = list(action_space.PROTOSS_ACTION_BUILD)
    except Exception:
      standard_build_actions = []
    ba = list(ba) + standard_build_actions

  building_types, building_types_text, pylons_construct = [], [], []
  building_types += BUILDING_TYPE_DEFENSE
  for building_type in BUILDING_TYPE:
    building_types_text.append(str(units.get_unit_type(building_type)).split('.')[-1])
  for unit in obs.observation.raw_units:
    if unit.alliance == features.PlayerRelative.SELF and unit.build_progress == 100 and unit.unit_type in BUILDING_TYPE:
      building_types.append(unit.unit_type)
      if unit.unit_type == units.Protoss.WarpGate:
        building_types.append(units.Protoss.Gateway)
    if unit.alliance == features.PlayerRelative.SELF and unit.unit_type in BUILDING_TYPE:
      building_types_text.append(str(units.get_unit_type(unit.unit_type)).split('.')[-1])
      if unit.unit_type == units.Protoss.WarpGate:
        building_types.append('Gateway')
    if unit.alliance == features.PlayerRelative.SELF and unit.unit_type == units.Protoss.Pylon and unit.build_progress != 100:
      pylons_construct.append(unit)

  # valid_actions = []
  basic_actions_info = ''
  valid_actions_info = ''
  partial_valid_actions_info = ''

  for action in ba:
    func_id, valid = action['func'][-1][0], True

    # Show args based on action definition (not global mode), so merged lists render correctly.
    arg = action.get('arg', [])
    if len(arg) == 0:
      arg_to_show = ''
    elif len(arg) == 1:
      arg_to_show = arg[0]
    else:
      arg_to_show = ",".join([str(a) for a in arg])

    if easy_build:
      valid_actions_info += f"\n\t\t<{action['name']}({arg_to_show})> "
      continue

    building_name = action['name'].split('_')[1]

    if func_id in bc.keys():
      conditions = bc[func_id]
      # condition = {'m': 175, 'g': 175, 'b': units.Protoss.CyberneticsCore, 'u': u.ProtossAirArmorsLevel1, 't': 215},
      cs = conditions
      valid = False if ('b' in cs.keys() and not all_building_condition_reached(cs['b'], building_types)) else valid
      valid = False if ('u' in cs.keys() and cs['u'] not in u) else valid
      partial_valid = True if valid else False
      valid = False if ('m' in cs.keys() and m < cs['m']) else valid
      valid = False if ('g' in cs.keys() and g < cs['g']) else valid
      valid = False if ('s' in cs.keys() and s < cs['s']) else valid

      if func_id in [actions.FUNCTIONS.Build_Pylon_screen.id]:
        s_cap, s_used = obs.observation.player.food_cap, obs.observation.player.food_used
        supply = 7 * (len(pylons_construct) + 1) + s_cap - s_used
        if 0 < s_used and supply > 20 and obs.observation.player.minerals < 300:
          partial_valid, valid = False, False
        if 50 < s_used < 100 and supply > 25 and obs.observation.player.minerals < 500:
          partial_valid, valid = False, False
        if 100 <= s_used < 150 and supply > 50 and obs.observation.player.minerals < 750:
          partial_valid, valid = False, False
        if obs.observation.player.food_cap == 200 and len(pylons_construct) > 0:
          partial_valid, valid = False, False

      _, _, ves_new_base_tags, ves_near_tags = get_ves_for_base_and_gas_building(obs)
      if func_id in [actions.FUNCTIONS.Build_Nexus_screen.id]:
        partial_valid, valid = (False, False) if len(ves_new_base_tags) == 0 else (partial_valid, valid)
      if func_id in [actions.FUNCTIONS.Build_Assimilator_screen.id]:
        partial_valid, valid = (False, False) if len(ves_near_tags) == 0 else (partial_valid, valid)

      if partial_valid and not valid:  # resource not enough
        cost = {'mineral': cs['m'], 'gas': cs['g']}  # 'time': cs['t']
        note = ", note: 'New Building! We do not have this building yet, it may unlock new buildings/technologies/units for us'" if building_name not in building_types_text else ''
        partial_valid_actions_info += f"\n\t\t<{action['name']}({arg_to_show})> \n\t\t\t cost: {cost}{note}, lack of resources, currently invalid"
      if valid:
        # valid_actions.append(action['name'])
        cost = {'mineral': cs['m'], 'gas': cs['g']}  # 'time': cs['t']
        note = ", note: 'New Building! We do not have this building yet, it may unlock new buildings/technologies/units for us'" if building_name not in building_types_text else ''
        valid_actions_info += f"\n\t\t<{action['name']}({arg_to_show})> \n\t\t\t cost: {cost}{note}"
        # valid_actions_info += f"{building_name} {building_types_text}"
      if func_id in [actions.FUNCTIONS.Build_Nexus_screen.id] and valid:
        valid_actions_info += ', note: Important Building! This is base building, you should build it to expand your economy.'
      if func_id in [actions.FUNCTIONS.Build_Nexus_screen.id] and partial_valid and not valid:
        partial_valid_actions_info += ', note: Important Building! This is base building, you should build it to expand your economy.'
    else:
      # valid_actions.append(action['name'])
      basic_actions_info += f"\n\t\t<{action['name']}({arg_to_show})> "

  # if valid_actions_info != '':
  #   valid_actions_info = "Valid Research Actions: " + valid_actions_info + "\n\n"

  # return valid_actions, basic_actions_info + valid_actions_info
  return basic_actions_info + valid_actions_info, partial_valid_actions_info


def get_valid_actions_research(agent) -> (list, str):
  obs = agent.team_unit_obs_list[0]
  ra, _, _, rc, _, _, m, g, s, u, b = get_condition_elements(agent)

  building_types = []
  for unit in obs.observation.raw_units:
    if unit.alliance == features.PlayerRelative.SELF and unit.build_progress == 100 and unit.unit_type in BUILDING_TYPE:
      building_types.append(unit.unit_type)

  # valid_actions = []
  valid_actions_info = ''
  partial_valid_actions_info = ''
  for action in ra:
    func_id = map_research_quick_to_level(action['func'][-1][0], u)
    if func_id == -1:
      continue
    conditions, valid = rc[func_id], True
    # condition = {'m': 175, 'g': 175, 'b': units.Protoss.CyberneticsCore, 'u': u.ProtossAirArmorsLevel1, 't': 215},
    cs = conditions
    valid = False if ('b' in cs.keys() and not all_building_condition_reached(cs['b'], building_types)) else valid
    valid = False if ('u' in cs.keys() and cs['u'] not in u) else valid
    partial_valid = True if valid else False
    valid = False if ('m' in cs.keys() and m < cs['m']) else valid
    valid = False if ('g' in cs.keys() and g < cs['g']) else valid
    valid = False if ('s' in cs.keys() and s < cs['s']) else valid
    if partial_valid and not valid:  # resource not enough
      cost = {'mineral': cs['m'], 'gas': cs['g']}  # 'time': cs['t']
      partial_valid_actions_info += f"\n\t\t<{action['name']}()> \n\t\t\t cost: {cost}, note:lack of resources, currently invalid)"
    if valid:
      # valid_actions.append(action['name'])
      cost = {'mineral': cs['m'], 'gas': cs['g']}  # 'time': cs['t']
      valid_actions_info += f"\n\t\t<{action['name']}()> \n\t\t\t cost: {cost}"

  # if valid_actions_info != '':
  #   valid_actions_info = "Valid Research Actions: " + valid_actions_info + "\n\n"

  # return valid_actions, valid_actions_info
  return valid_actions_info, partial_valid_actions_info


def get_valid_actions_train(agent) -> (list, str):
  obs = agent.team_unit_obs_list[0]
  _, ta, _, _, tc, _, m, g, s, u, b = get_condition_elements(agent)

  building_types = []
  for unit in obs.observation.raw_units:
    if unit.alliance == features.PlayerRelative.SELF and unit.build_progress == 100 and unit.unit_type in BUILDING_TYPE:
      building_types.append(unit.unit_type)

  # valid_actions = []
  valid_actions_info = ''
  partial_valid_actions_info = ''
  for action in ta:
    func_id = action['func'][-1][0]
    conditions, valid = tc[func_id], True
    # condition = {'m': 125, 'g': 50, 'b': units.Protoss.Gateway, 't': 42, 's': 2},
    cs = conditions
    valid = False if ('b' in cs.keys() and not all_building_condition_reached(cs['b'], building_types)) else valid
    valid = False if ('u' in cs.keys() and cs['u'] not in u) else valid
    partial_valid = True if valid else False
    valid = False if ('m' in cs.keys() and m < cs['m']) else valid
    valid = False if ('g' in cs.keys() and g < cs['g']) else valid
    valid = False if ('s' in cs.keys() and s < cs['s']) else valid
    max_number = min(round(m / cs['m'] if cs['m'] != 0 else 99), round(m / cs['g'] if cs['g'] != 0 else 99))
    if partial_valid and not valid:  # resource not enough
      cost = {'mineral': cs['m'], 'gas': cs['g'], 'supply': cs['s']}  # 'time': cs['t']
      partial_valid_actions_info += f"\n\t\t<{action['name']}()> \n\t\t\t cost: {cost}, note: lack of resources, currently invalid"
    if valid:
      # valid_actions.append(action['name'])
      cost = {'mineral': cs['m'], 'gas': cs['g'], 'supply': cs['s']}  # 'time': cs['t']
      valid_actions_info += f"\n\t\t<{action['name']}()> \n\t\t\t cost: {cost}, note: we can afford {max_number} at most"

  # if valid_actions_info != '':
  #   valid_actions_info = "Valid Unit Training Actions: " + valid_actions_info + "\n\n"

  # return valid_actions, valid_actions_info
  return valid_actions_info, partial_valid_actions_info


def get_valid_actions_chrono_boost(agent):
  obs = agent.team_unit_obs_list[0]

  source_unit_tag, target_unit_tag = None, None
  active_buildings_base, active_buildings_military, active_buildings_research = [], [], []
  for unit in obs.observation.raw_units:
    if unit.unit_type == units.Protoss.Nexus and unit.alliance == features.PlayerRelative.SELF and unit.build_progress == 100 and unit.energy > 50:
      source_unit_tag = unit.tag
    if unit.unit_type == units.Protoss.Nexus and unit.alliance == features.PlayerRelative.SELF and unit.build_progress == 100 and unit.active != 0 and unit.buff_id_0 == 0:
      active_buildings_base.append(unit.tag)
    if unit.unit_type in BUILDING_TYPE_MILITARY and unit.alliance == features.PlayerRelative.SELF and unit.build_progress == 100 and unit.active != 0 and unit.buff_id_0 == 0:
      active_buildings_military.append(unit.tag)
    if unit.unit_type in BUILDING_TYPE_RESEARCH and unit.alliance == features.PlayerRelative.SELF and unit.build_progress == 100 and unit.active != 0 and unit.buff_id_0 == 0:
      active_buildings_research.append(unit.tag)

  valid_actions_info = ''
  if source_unit_tag is not None and len(active_buildings_base) > 0 and len(active_buildings_military) == 0 and len(active_buildings_research) == 0:
    valid_actions_info += f"\n\t\t<ChronoBoost_Economy()>"
  if source_unit_tag is not None and len(active_buildings_military) > 0 and len(active_buildings_research) == 0:
    valid_actions_info += f"\n\t\t<ChronoBoost_Military()>"
  if source_unit_tag is not None and len(active_buildings_research) > 0:
    valid_actions_info += f"\n\t\t<ChronoBoost_Research()>"
  return valid_actions_info


def get_valid_actions_developer(agent):
  teams_valid_actions_info = ''

  if agent.name != 'Developer':
    logger.error(f"[ID {agent.log_id}] LLMAgent {agent.name}: use get_valid_actions_developer but agent name is not Developer")

  for team in agent.teams:

    if 'Workers' in team['name'] and not agent.config.ENABLE_EASY_BUILD:
      pass
    else:
      if agent.flag_enable_empty_unit_group and len(team['unit_type']) == 0:
        teams_valid_actions_info += f"\n\tTeam {team['name']}-1:"
      elif team['select_type'] == 'select':
        for i in range(len(team['obs'])):
          teams_valid_actions_info += f"\n\tTeam {team['name']}-{i + 1}:"
      else:
        teams_valid_actions_info += f"\n\tTeam {team['name']}:"

    valid_actions_info = ''
    partial_valid_actions_info = ''

    valid_actions_info_, partial_valid_actions_info_ = get_valid_actions_research(agent)
    valid_actions_info += valid_actions_info_
    partial_valid_actions_info += partial_valid_actions_info_
    valid_actions_info_, partial_valid_actions_info_ = get_valid_actions_train(agent)
    valid_actions_info += valid_actions_info_
    partial_valid_actions_info += partial_valid_actions_info_
    valid_actions_info_, partial_valid_actions_info_ = get_valid_actions_build(agent)
    valid_actions_info += valid_actions_info_
    partial_valid_actions_info += partial_valid_actions_info_
    valid_actions_info += get_valid_actions_chrono_boost(agent)
    if valid_actions_info == '':
      teams_valid_actions_info += '\n\t\t currently none, build buildings to unlock training/warping researching actions, and build building actions.'
    else:
      teams_valid_actions_info += valid_actions_info
    if partial_valid_actions_info != '':
      teams_valid_actions_info += '\n\t\t(Actions only lack of resources below, currently invalid)' + partial_valid_actions_info

    # if 'Buildings' in team['name']:
    #   valid_actions_info_, partial_valid_actions_info_ = get_valid_actions_research(agent)
    #   valid_actions_info += valid_actions_info_
    #   partial_valid_actions_info += partial_valid_actions_info_
    #   valid_actions_info_, partial_valid_actions_info_  = get_valid_actions_train(agent)
    #   valid_actions_info += valid_actions_info_
    #   partial_valid_actions_info += partial_valid_actions_info_
    #   valid_actions_info += get_valid_actions_chrono_boost(agent)
    #   if valid_actions_info == '':
    #     teams_valid_actions_info += '\n\t\t currently none, build buildings to unlock training/warping and researching actions.'
    #   else:
    #     teams_valid_actions_info += valid_actions_info
    #   if partial_valid_actions_info != '':
    #     teams_valid_actions_info += '\n\t\t(Actions only lack of resources below, currently invalid)' + partial_valid_actions_info
    #
    # if 'Workers' in team['name'] and agent.config.ENABLE_EASY_BUILD:
    #   valid_actions_info_, partial_valid_actions_info_  = get_valid_actions_build(agent)
    #   valid_actions_info += valid_actions_info_
    #   partial_valid_actions_info += partial_valid_actions_info_
    #   if valid_actions_info == '':
    #     teams_valid_actions_info += '\n\t\t currently none, waiting for more resource to build buildings.'
    #   else:
    #     teams_valid_actions_info += valid_actions_info
    #   if partial_valid_actions_info != '':
    #     teams_valid_actions_info += '\n\t\t(Actions only lack of resources below, currently invalid)' + partial_valid_actions_info

  teams_valid_actions_info = 'Valid actions:' + teams_valid_actions_info + '\n\n'
  return teams_valid_actions_info


def get_valid_actions_builder(agent):
  teams_valid_actions_info = ''

  if agent.name == 'Builder':
    for team in agent.teams:

      if agent.flag_enable_empty_unit_group and len(team['unit_type']) == 0:
        teams_valid_actions_info += f"\n\tTeam {team['name']}-1:"
      if team['select_type'] == 'select':
        for i in range(len(team['obs'])):
          teams_valid_actions_info += f"\n\tTeam {team['name']}-{i + 1}:"
      else:
        teams_valid_actions_info += f"\n\tTeam {team['name']}:"

      valid_actions_info, partial_valid_actions_info = get_valid_actions_build(agent)
      if valid_actions_info == '':
        teams_valid_actions_info += '\n\t\t currently none, waiting for more resource to unlock build actions.'
      else:
        teams_valid_actions_info += valid_actions_info
      if partial_valid_actions_info != '':
        teams_valid_actions_info += '\n\t\t(Actions only lack of resources below, currently invalid)' + partial_valid_actions_info

  else:
    teams_valid_actions_info += f"\n\tAgent Builder's probe's valid actions:"
    valid_actions_info, partial_valid_actions_info = get_valid_actions_build(agent)
    if valid_actions_info == '':
      teams_valid_actions_info += '\n\t\t currently none, waiting for more resource to unlock build actions.'
    else:
      teams_valid_actions_info += valid_actions_info
    if partial_valid_actions_info != '':
      teams_valid_actions_info += '\n\t\t(Actions only lack of resources below, currently invalid)' + partial_valid_actions_info

  if agent.name == 'Builder':
    teams_valid_actions_info = 'Valid actions:' + teams_valid_actions_info + '\n\n'
  else:
    teams_valid_actions_info = "Agent Builder's Valid actions:" + teams_valid_actions_info + '\n\n'
  return teams_valid_actions_info


def get_valid_actions_commander(agent):
  teams_valid_actions_info = ''

  if agent.name != 'Commander':
    logger.error(f"[ID {agent.log_id}] LLMAgent {agent.name}: use get_valid_actions_commander but agent name is not Commander")

  if agent.race == 'protoss':
    commander_actions = action_space.PROTOSS_ACTION_EASY_CONTROL
  elif agent.race == 'terran':
    commander_actions = []  # TODO: ADD
  elif agent.race == 'zerg':
    commander_actions = []  # TODO: ADD
  else:
    commander_actions = []

  for team in agent.teams:
    if agent.flag_enable_empty_unit_group and len(team['unit_type']) == 0:
      teams_valid_actions_info += f"\n\tTeam {team['name']}-1:"
    elif team['select_type'] == 'select':
      for i in range(len(team['obs'])):
        teams_valid_actions_info += f"\n\tTeam {team['name']}-{i + 1}:"
    else:
      teams_valid_actions_info += f"\n\tTeam {team['name']}:"

    obs = team['obs'][0]
    valid_actions_info, unit_types, unit_type_names = '', [], []
    for unit in obs.observation.raw_units:
      if unit.alliance == features.PlayerRelative.SELF and unit.build_progress == 100 and \
          unit.unit_type not in BUILDING_TYPE + WORKER_TYPE + unit_types:
        unit_types.append(unit.unit_type)
        unit_type_names.append(str(units.get_unit_type(unit.unit_type)).split('.')[-1])

    team_action_space = []
    for key in team['actions'].keys():
      team_action_space += team['actions'][key]

    for action in team_action_space:
      if 'All_Units_' in action['name']:
        valid_actions_info += f"\n\t\t <{action['name']}()>"
      if '_Scan' in action['name']:
        unit_name = action['name'].split('_')[0]
        arg = '' if len(action['arg']) == 0 else tuple(action['arg'])
        if action['name'] == 'Worker_Scan' or unit_name in unit_type_names:
          valid_actions_info += f"\n\t\t <{action['name']}({arg})>"

    if valid_actions_info != '':
      teams_valid_actions_info += valid_actions_info
    else:
      teams_valid_actions_info += 'none, currently'

  teams_valid_actions_info = 'Valid Actions' + teams_valid_actions_info + '\n\n'
  return teams_valid_actions_info

def get_action_error_info(agent):
  action_errors = agent.action_errors
  action_error_info = ''

  for key in action_errors.keys():
    action_error_info += f"\n\t<{key}(...)>:"
    for error in action_errors[key]:
      action_error_info += f"\n\t\t{error}"

  if action_error_info != '':
    action_error_info = "Last Step Action Errors:" + action_error_info + "\n\n"
  return action_error_info

