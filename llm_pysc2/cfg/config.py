# Copyright 2024, LLM-PySC2 Contributors. All Rights Reserved.
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

from llm_pysc2.lib.action.space import *
from llm_pysc2.lib.llm_client import vision_model_names  #, video_model_names
from llm_pysc2.cfg.llm_env import llm_credentials_from_env

from pysc2.lib import units
from loguru import logger
import os
import time


def wait(ignore, second, log_id, more_info=''):
  if ignore:
   return
  for i in range(second):
    logger.warning(f"[ID {log_id}] Experiment will start with UNSAFE settings in {second - i} seconds. {more_info}")
    time.sleep(1)


class AgentConfig:

  def __init__(self):

    self.race = 'protoss'
    self.model_name, self.api_base, self.api_key = llm_credentials_from_env()
    self.temperature = 0.1

    self.basic_prompt = 'default'
    self.translator_o = 'default'
    self.translator_a = 'default'
    self.communicator = 'default'

    # For debug
    self.ENABLE_MULTI_THREAD_QUERY = True
    self.IGNORE_INIT_WARNINGS = False
    self.SAFE_MODE = True

    # Game settings
    self.ENABLE_INIT_STEPS = True
    self.ENABLE_AUTO_WORKER_MANAGE = True
    self.ENABLE_AUTO_WORKER_TRAINING = True
    self.ENABLE_COMMUNICATION = False
    self.ENABLE_EASY_BUILD = True
    self.ENABLE_EASY_CONTROL = True
    self.ENABLE_EASY_WARP = True

    # True: broadcast Builder/team + last action text to in-game chat (shows in HUD; embedded in replay).
    # False: no chat spam (recommended when watching the window).
    self.ENABLE_IN_GAME_ACTION_CHAT = False

    # Only one SubAgent queries the LLM; others stay disabled (matches single-brain / SPC architecture).
    self.ENABLE_SINGLE_LLM_AGENT = False
    self.SINGLE_LLM_AGENT_NAME = 'Commander'

    # SPC：顶层策脉由 lib/strategic_pulse_brain.StrategicPulseBrain 独立维护（非 Commander）；与 Commander 经反馈/参谋摘录通信。
    self.ENABLE_SPC_STRATEGIC_LAYER = True
    # PySC2 常用换算：game_loop / SC2_GAME_LOOPS_PER_SECOND ≈ 游戏内秒（与 MAX_LLM_DECISION_FREQUENCY 文档一致）。
    self.SC2_GAME_LOOPS_PER_SECOND = 22.4
    # 顶层 query_strategic 最小间隔（游戏内秒）。>0 时按 game_loop 计时；0 则退回下面「步数间隔」。
    self.SPC_STRATEGIC_INTERVAL_GAME_SECONDS = 30.0
    # 宏观策脉 tick 间隔（游戏步）；仅当 SPC_STRATEGIC_INTERVAL_GAME_SECONDS == 0 时生效。
    self.SPC_STRATEGIC_STEP_INTERVAL = 8
    # Commander 战术 LLM（query_tactical）最小间隔（游戏内秒）。<=0 表示不额外节流（仍受主循环闸门等约束）。
    self.COMMANDER_LLM_INTERVAL_GAME_SECONDS = 15.0
    # Developer 战术 LLM 最小间隔（游戏内秒）。<=0 表示每轮主循环可照常询问（与 Commander 独立计时）。
    self.DEVELOPER_LLM_INTERVAL_GAME_SECONDS = 0.0
    # True：各代理战术前缀统一读 spc_shared（顶层发布）；False：用各 LLMAgent 本地 current_pulse（消融用）。
    self.SPC_SHARED_STRATEGIC_BRAIN = True
    # 保留字段：战术前缀里「由谁牵头执行」的文案；宏观 LLM 已迁至 StrategicBrain。
    self.SPC_STRATEGIC_OWNER_AGENT = 'Commander'
    # 是否把 Commander 的 last_action_feedback 传入顶层策脉（query_strategic 中「指挥官侧执行反馈」块及 publish 快照）。
    # False：最顶层宏观 LLM 不读指挥官执行反馈（参谋摘录不受此项影响）。
    self.SPC_INCLUDE_COMMANDER_FEEDBACK = False
    # Pulse 切换最少保持轮次（inertia 小于该值时，非紧急情况下不允许切换）。
    self.SPC_MIN_PULSE_HOLD_TICKS = 1
    # Pulse 切换最低置信度阈值（0~1）。
    self.SPC_MIN_PULSE_SWITCH_CONFIDENCE = 0.62
    # 进攻脉搏的最低军队人口（在可见敌情下低于该值会被硬拦截）。
    self.SPC_MIN_ATTACK_ARMY_SUPPLY = 10
    # 结构化长期记忆槽位默认 TTL（按 strategic tick 计数）与最大 TTL。
    self.SPC_MEMORY_SLOT_TTL_DEFAULT = 6
    self.SPC_MEMORY_SLOT_TTL_MAX = 18
    # 顶层策脉自我画像（用于 query_strategic prompt 上下文）。
    self.SPC_SELF_PROFILE_MAP_NAME = "Simple64"
    self.SPC_SELF_PROFILE_RACE = "Protoss"
    self.SPC_SELF_PROFILE_OPPONENT = "Built-in AI"
    # Optional strategic doctrine injected into SPC query_strategic prompt.
    # This is strategy-only guidance (no executable action format required).
    self.SPC_STRATEGIC_DOCTRINE = ""

    # Image settings
    self.ENABLE_IMAGE_RGB = False
    self.ENABLE_IMAGE_FEATURE = False
    self.ENABLE_SAVE_IMAGES = True

    self.LLM_SIMULATION_TIME = 0
    # True：各 LLM 调用的文本输入/输出打印到 stdout（演示用；亦可设环境变量 LLM_PYSC2_DEMO_IO=1）。
    self.LLM_DEMO_PRINT_IO = False
    # 演示模式下 INPUT/OUTPUT 单侧最大字符数（全文见 LLM_PYSC2_DEMO_IO_FULL=1）。
    self.LLM_DEMO_IO_MAX_CHARS = 16000
    # True：每次 LLM 调用把「层名称 / 代理 / 完整输入输出」追加到日志目录下的 llm_layers.jsonl（配合 llm_io_viewer.html）。
    self.LLM_IO_JSONL_RECORD = False
    self.LLM_IO_JSONL_PATH = ""
    # False: 战术 LLM 只调用一次 query_tactical，不做「Valid Actions」名检查与最多 3 次重试。
    self.ENABLE_TACTICAL_OUTPUT_GATE = True
    self.MAX_LLM_QUERY_TIMES = 5
    self.MAX_LLM_WAITING_TIME = 15
    self.MAX_LLM_RUNTIME_ERROR_TIME = 45
    self.MAX_LLM_DECISION_FREQUENCY = 1
    self.MAX_NUM_ACTIONS = 3
    # Unified experiment switch for scripted macro stack:
    # - Developer tech-starvation guard (force VB before gas spend drift)
    # - SPC playbook memory and stage-aware prompt shaping
    self.scipt = False
    # Developer 资源约束：未满人口时，矿过高需优先把钱转成兵/产能/升级/开矿。
    self.DEVELOPER_MINERAL_CAP_SOFT = 1000
    self.DEVELOPER_MINERAL_CAP_ADVANCED = 500

    self.AGENTS = {}
    self.AGENTS_ALWAYS_DISABLE = []

  def reset_llm(self, model_name=None, api_base=None, api_key=None, ENABLE_IMAGE_RGB=None, ENABLE_IMAGE_FEATURE=None):
    if model_name is not None and model_name != 'YOUR-MODEL-NAME':
      self.model_name = model_name
    if api_base is not None and api_base != 'YOUR-API-BASE':
      self.api_base = api_base
    if api_key is not None and api_key != 'YOUR-API-KEY':
      self.api_key = api_key
    if ENABLE_IMAGE_RGB is not None:
      self.ENABLE_IMAGE_RGB = ENABLE_IMAGE_RGB
    if ENABLE_IMAGE_FEATURE is not None:
      self.ENABLE_IMAGE_FEATURE = ENABLE_IMAGE_FEATURE
    if ENABLE_IMAGE_RGB is True and ENABLE_IMAGE_FEATURE is True:
      raise AssertionError("Do not support ENABLE_IMAGE_RGB and ENABLE_IMAGE_FEATURE at the same time, currently")
    for agent_name in self.AGENTS.keys():
      self.AGENTS[agent_name]['llm']['model_name'] = self.model_name
      self.AGENTS[agent_name]['llm']['api_base'] = self.api_base
      self.AGENTS[agent_name]['llm']['api_key'] = self.api_key
      if self.ENABLE_IMAGE_RGB:
        self.AGENTS[agent_name]['llm']['img_rgb'] = True
        self.AGENTS[agent_name]['llm']['img_fea'] = False
      elif self.ENABLE_IMAGE_FEATURE:
        self.AGENTS[agent_name]['llm']['img_rgb'] = False
        self.AGENTS[agent_name]['llm']['img_fea'] = True
      else:
        self.AGENTS[agent_name]['llm']['img_rgb'] = False
        self.AGENTS[agent_name]['llm']['img_fea'] = False

    if str(os.environ.get("LLM_PYSC2_DEMO_IO", "")).strip().lower() in (
        "1", "true", "yes", "on"):
      self.LLM_DEMO_PRINT_IO = True
    if str(os.environ.get("LLM_PYSC2_IO_JSONL", "")).strip().lower() in (
        "1", "true", "yes", "on"):
      self.LLM_IO_JSONL_RECORD = True
    env_demo_max = str(os.environ.get("LLM_PYSC2_DEMO_IO_MAX_CHARS", "")).strip()
    if env_demo_max.isdigit():
      self.LLM_DEMO_IO_MAX_CHARS = int(env_demo_max)

  def auto_check(self, log_id):
    error_in_llm_setting = False
    if self.model_name == '' or self.model_name == 'YOUR-MODEL-NAME':
      self.reset_llm(model_name='gpt-3.5-turbo')
      logger.error(f"[ID {log_id}] No model_name set, please specify model_name in the config.")
      error_in_llm_setting = True
    if self.api_key == '' or self.api_key == 'YOUR-API-KEY':
      logger.error(f"[ID {log_id}] No api_key set, please specify your api_key in the config.")
      error_in_llm_setting = True
    if self.model_name == '' or self.api_key == '':
      error_in_llm_setting = True

    if not isinstance(self.LLM_SIMULATION_TIME, (int, float)) or self.LLM_SIMULATION_TIME <= 0:
      if error_in_llm_setting:
        wait(self.IGNORE_INIT_WARNINGS, 5, log_id, "(in LLM SIMULATION MODE)")
      if error_in_llm_setting:
        self.LLM_SIMULATION_TIME = 5

    if self.ENABLE_IMAGE_RGB or self.ENABLE_IMAGE_FEATURE:
      if self.ENABLE_IMAGE_RGB and self.ENABLE_IMAGE_FEATURE:
        logger.error(f"[ID {log_id}] can not enable config.ENABLE_IMAGE_RGB and config.ENABLE_IMAGE_FEATURE together.")
        AssertionError(f"config.ENABLE_IMAGE_RGB and config.ENABLE_IMAGE_FEATURE can not be True together")
      if self.model_name not in vision_model_names:
        logger.error(f"[ID {log_id}] config.ENABLE_IMAGE_RGB/FEATURE with large models that do not support images.")
        wait(self.IGNORE_INIT_WARNINGS, 5, log_id)
      if self.model_name in vision_model_names:
        logger.warning(f"[ID {log_id}] You are using a vision model with image obs, this may cost a lot, be cautious.")
        wait(self.IGNORE_INIT_WARNINGS, 5, log_id)
    else:
      if self.model_name in vision_model_names:
        logger.warning(f"[ID {log_id}] You are using a vision avaliable model without using any image obs.")
        wait(self.IGNORE_INIT_WARNINGS, 5, log_id)

    if getattr(self, 'ENABLE_SINGLE_LLM_AGENT', False):
      sole = getattr(self, 'SINGLE_LLM_AGENT_NAME', 'Commander')
      if sole not in self.AGENTS:
        logger.error(
          f"[ID {log_id}] ENABLE_SINGLE_LLM_AGENT: SINGLE_LLM_AGENT_NAME={sole!r} not in config.AGENTS keys."
        )
        error_in_llm_setting = True

class ProtossAgentConfig(AgentConfig):

  def __init__(self):
    super(ProtossAgentConfig, self).__init__()

    # Program control parameters in class AgentConfig (above)

    self.AGENTS_ALWAYS_DISABLE = []
    self.AGENTS = {
      'Commander': {
        'describe': "Protoss military supreme commander. "
                    "Responsible for making macro decision through communication, and controls nexus for massrecall "
                    "for tactical objectives. When make deployment, describe the time, location, and objectives of the "
                    "mission as clearly as possible",
        'llm': {
          'basic_prompt': self.basic_prompt,
          'translator_o': 'commander',
          'translator_a': self.translator_a,
          'img_names': ['rgb_minimap'],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
          'img_fea': self.ENABLE_IMAGE_FEATURE,
          'img_rgb': self.ENABLE_IMAGE_RGB,
          'model_name': self.model_name,
          'api_base': self.api_base,
          'api_key': self.api_key,
        },
        'team': {
          'Protoss-Units': {
            'name': 'Protoss-Units', 'unit_type': [], 'game_group': -1, 'select_type': 'select',
            'actions': {'ALWAYS': PROTOSS_ACTION_EASY_CONTROL}
          },
        },
      },

      'Developer': {
        'describe': "Protoss logistics commander. "
                    "Responsible for unit trainning, unit warp trainning, technology upgrade and order the Builder "
                    "to build.",
        'llm': {
          'basic_prompt': self.basic_prompt,
          'translator_o': 'developer',
          'translator_a': self.translator_a,
          'img_names': ['rgb_minimap'],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
          'img_fea': self.ENABLE_IMAGE_FEATURE,
          'img_rgb': self.ENABLE_IMAGE_RGB,
          'model_name': self.model_name,
          'api_base': self.api_base,
          'api_key': self.api_key,
        },
        'team': {
          # 'Protoss-Buildings': {
          #   'name': 'Protoss-Buildings', 'unit_type': [], 'game_group': -1, 'select_type': 'select',
          #   'actions': {'ALWAYS': PROTOSS_ACTION_RESEARCH + PROTOSS_ACTION_TRAIN + PROTOSS_ACTION_WARPTRAIN + PROTOSS_ACTION_EASY_WARPTRAIN + PROTOSS_ACTION_EASY_CHRONO_BOOST}  #  + PROTOSS_ACTION_BUILD + PROTOSS_ACTION_EASY_BUILD
          # },
          # 'Protoss-Workers': {
          #   'name': 'Protoss-Workers', 'unit_type': [], 'game_group': -1, 'select_type': 'select',
          #   'actions': {
          #     'ALWAYS': PROTOSS_ACTION_BUILD + PROTOSS_ACTION_EASY_BUILD}
          # },
          'Protoss-Buildings': {
            'name': 'Protoss-Buildings', 'unit_type': [], 'game_group': -1, 'select_type': 'select',
            'actions': {
              'ALWAYS': PROTOSS_ACTION_RESEARCH + PROTOSS_ACTION_TRAIN + PROTOSS_ACTION_WARPTRAIN + PROTOSS_ACTION_EASY_WARPTRAIN + PROTOSS_ACTION_EASY_CHRONO_BOOST + PROTOSS_ACTION_BUILD + PROTOSS_ACTION_EASY_BUILD}
          },
        },
      },

      'Builder': {
        'describe': "Protoss builder, controls several Probe. Responsible for build buildings",
        'llm': {
          'basic_prompt': self.basic_prompt,
          'translator_o': 'builder',
          'translator_a': self.translator_a,
          'img_names': ['rgb_screen', 'rgb_minimap', 'buildable', 'power'],  # 'rgb_screen', , 'pathable'
          # ['power', 'pathable', 'buildable', 'height_map', 'player_relative']
          'img_fea': self.ENABLE_IMAGE_FEATURE,
          'img_rgb': self.ENABLE_IMAGE_RGB,
          'model_name': self.model_name,
          'api_base': self.api_base,
          'api_key': self.api_key,
        },
        'team': {
          'Builder-Worker': {
            'name': 'Builder-Worker', 'unit_type': [units.Protoss.Probe], 'game_group': -1, 'select_type': 'select',
            'actions': {units.Protoss.Probe: [MOVE_MINIMAP, MOVE_SCREEN] + PROTOSS_ACTION_BUILD}
            # 具体输出的valid actions 在 llm_observation.get_valid_actions_build 函数中
          },
        },
      },

      'CombatGroup0': {
        'describe': "Protoss frontline commander, controls several Zealots. "
                    "Responsible for providing cover for the main force and executing multi line combat.",
        'llm': {
          'basic_prompt': self.basic_prompt,
          'translator_o': self.translator_o,
          'translator_a': self.translator_a,
          'img_names': ['rgb_screen', 'rgb_minimap'],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
          'img_fea': self.ENABLE_IMAGE_FEATURE,
          'img_rgb': self.ENABLE_IMAGE_RGB,
          'model_name': self.model_name,
          'api_base': self.api_base,
          'api_key': self.api_key,
        },
        'team': {
          'Zealot-1': {
            'name': 'Zealot-1', 'unit_type': [units.Protoss.Zealot], 'game_group': 2, 'select_type': 'group',
            'actions': {units.Protoss.Zealot: PROTOSS_BASIC_ACTION_2}},
          # 'Zealot-2': {
          #   'name': 'Zealot-2', 'unit_type': [units.Protoss.Zealot], 'game_group': 3, 'select_type': 'group',
          #   'actions': {units.Protoss.Zealot: PROTOSS_BASIC_ACTION_2}},
        },
      },

      'CombatGroup1': {
        'describe': "Protoss frontline commander, controls several Stalkers. "
                    "Responsible for providing cover for the main force and restraining enemy forces.",
        'llm': {
          'basic_prompt': self.basic_prompt,
          'translator_o': self.translator_o,
          'translator_a': self.translator_a,
          'img_names': ['rgb_screen', 'rgb_minimap'],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
          'img_fea': self.ENABLE_IMAGE_FEATURE,
          'img_rgb': self.ENABLE_IMAGE_RGB,
          'model_name': self.model_name,
          'api_base': self.api_base,
          'api_key': self.api_key,
        },
        'team': {
          'Stalker-1': {
            'name': 'Stalker-1', 'unit_type': [units.Protoss.Stalker], 'game_group': 4, 'select_type': 'group',
            'actions': {units.Protoss.Stalker: STANDARD_ACTION_STALKER}},
          # 'Stalker-2': {
          #   'name': 'Stalker-2', 'unit_type': [units.Protoss.Stalker], 'game_group': 5, 'select_type': 'group',
          #   'actions': {units.Protoss.Stalker: STANDARD_ACTION_STALKER}},
          # 'Stalker-3': {
          #   'name': 'Stalker-3', 'unit_type': [units.Protoss.Stalker], 'game_group': 6, 'select_type': 'group',
          #   'actions': {units.Protoss.Stalker: STANDARD_ACTION_STALKER}},
        },
      },

      'CombatGroup2': {
        'describe': "Protoss frontline commander, controls ground main force such as Immortal, Colossus and Archon. "
                    "Responsible for frontal combat.",
        'llm': {
          'basic_prompt': self.basic_prompt,
          'translator_o': self.translator_o,
          'translator_a': self.translator_a,
          'img_names': ['rgb_screen', 'rgb_minimap'],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
          'img_fea': self.ENABLE_IMAGE_FEATURE,
          'img_rgb': self.ENABLE_IMAGE_RGB,
          'model_name': self.model_name,
          'api_base': self.api_base,
          'api_key': self.api_key,
        },
        'team': {
          'Immortal-1': {
            'name': 'Immortal-1', 'unit_type': [units.Protoss.Immortal], 'game_group': -1, 'select_type': 'select_all_type',
            'actions': {units.Protoss.Immortal: PROTOSS_BASIC_ACTION_2}},
          'Colossus-1': {
            'name': 'Colossus-1', 'unit_type': [units.Protoss.Colossus], 'game_group': -1, 'select_type': 'select_all_type',
            'actions': {units.Protoss.Colossus: PROTOSS_BASIC_ACTION_2}},
          'Archon-1': {
            'name': 'Archon-1', 'unit_type': [units.Protoss.Archon], 'game_group': -1, 'select_type': 'select_all_type',
            'actions': {units.Protoss.Archon: PROTOSS_BASIC_ACTION_2}},
        },
      },

      'CombatGroup3': {
        'describe': "Protoss frontline commander, controls air main force such as VoidRay, Carrier and Tempest. "
                    "Responsible for frontal combat.",
        'llm': {
          'basic_prompt': self.basic_prompt,
          'translator_o': self.translator_o,
          'translator_a': self.translator_a,
          'img_names': ['rgb_screen', 'rgb_minimap'],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
          'img_fea': self.ENABLE_IMAGE_FEATURE,
          'img_rgb': self.ENABLE_IMAGE_RGB,
          'model_name': self.model_name,
          'api_base': self.api_base,
          'api_key': self.api_key,
        },
        'team': {
          'VoidRay-1': {
            'name': 'VoidRay-1', 'unit_type': [units.Protoss.VoidRay], 'game_group': -1, 'select_type': 'select_all_type',
            'actions': {
              units.Protoss.VoidRay: PROTOSS_BASIC_ACTION_2 + [
                {'name': 'Ability_PrismaticAlignment', 'arg': [], 'func': [(244, F.Effect_VoidRayPrismaticAlignment_quick, ('queued'))]}]
              }
            },
          'Carrier-1': {
            'name': 'Carrier-1', 'unit_type': [units.Protoss.Carrier], 'game_group': -1, 'select_type': 'select_all_type',
            'actions': {units.Protoss.Carrier: PROTOSS_BASIC_ACTION_2}},
          'Tempest-1': {
            'name': 'Tempest-1', 'unit_type': [units.Protoss.Tempest], 'game_group': -1, 'select_type': 'select_all_type',
            'actions': {units.Protoss.Tempest: PROTOSS_BASIC_ACTION_2}},
        },
      },

      'CombatGroup4': {
        'describe': "Protoss reconnaissance commander, controls Observer and several Probe. "
                    "Responsible for providing reconnaissance infomation and detect cloak unit for main force",
        'llm': {
          'basic_prompt': self.basic_prompt,
          'translator_o': self.translator_o,
          'translator_a': self.translator_a,
          'img_names': ['rgb_screen', 'rgb_minimap'],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
          'img_fea': self.ENABLE_IMAGE_FEATURE,
          'img_rgb': self.ENABLE_IMAGE_RGB,
          'model_name': self.model_name,
          'api_base': self.api_base,
          'api_key': self.api_key,
        },
        'team': {
          # 'Probe': {
          #   'name': 'Probe', 'unit_type': [units.Protoss.Probe], 'game_group': -1, 'select_type': 'select',
          #   'actions': {units.Protoss.Probe: SCANNING_ACTION_PROBE}},
          'Observer': {'name': 'Observer', 'unit_type': [units.Protoss.Observer, units.Protoss.ObserverSurveillanceMode], 'game_group': -1, 'select_type': 'select',
            'actions': {
              units.Protoss.Observer: STANDARD_ACTION_OBSERVER1,
              units.Protoss.ObserverSurveillanceMode: STANDARD_ACTION_OBSERVER2,
            }},
        },
      },

      'CombatGroup5': {
        'describe': "Protoss AOE commander, controls HighTemplar and Disruptor. "
                    "Responsible for dealing high damage to clustered enemies",
        'llm': {
          'basic_prompt': self.basic_prompt,
          'translator_o': self.translator_o,
          'translator_a': self.translator_a,
          'img_names': ['rgb_screen', 'rgb_minimap'],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
          'img_fea': self.ENABLE_IMAGE_FEATURE,
          'img_rgb': self.ENABLE_IMAGE_RGB,
          'model_name': self.model_name,
          'api_base': self.api_base,
          'api_key': self.api_key,
        },
        'team': {
          'HighTemplar-1': {
            'name': 'HighTemplar-1', 'unit_type': [units.Protoss.HighTemplar], 'game_group': 7, 'select_type': 'group',
            'actions': {units.Protoss.HighTemplar: STANDARD_ACTION_HIGHTEMPLAR}},
          'Disruptor-1': {
            'name': 'Disruptor-1', 'unit_type': [units.Protoss.Disruptor], 'game_group': 8, 'select_type': 'group',
            'actions': {units.Protoss.Disruptor: STANDARD_ACTION_DISRUPTOR}},
        },
      },

      'CombatGroup6': {
        'describe': "Protoss tactical support commander, controls Sentry and Mothership. "
                      "Responsible for providing tactical support by using skills",
        'llm': {
          'basic_prompt': self.basic_prompt,
          'translator_o': self.translator_o,
          'translator_a': self.translator_a,
          'img_names': ['rgb_screen', 'rgb_minimap'],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
          'img_fea': self.ENABLE_IMAGE_FEATURE,
          'img_rgb': self.ENABLE_IMAGE_RGB,
          'model_name': self.model_name,
          'api_base': self.api_base,
          'api_key': self.api_key,
        },
        'team': {
          'Sentry-1': {'name': 'Sentry-1', 'unit_type': [units.Protoss.Sentry], 'game_group': 9, 'select_type': 'group',
            'actions': {units.Protoss.Sentry: STANDARD_ACTION_SENTRY}},
          'Mothership': {'name': 'Mothership', 'unit_type': [units.Protoss.Mothership], 'game_group': -1, 'select_type': 'select',
           'actions': {units.Protoss.Mothership: STANDARD_ACTION_MOTHERSHIP}},
        },
      },

      'CombatGroup7': {
        'describe': "Protoss special force commander, controls Adept and DarkTemplar. "
                    "Responsible for infiltrating the enemy's rear and disrupt economic production, sometimes "
                    "collecting reconnaissance infomation, participating in frontline combat.",
        'llm': {
          'basic_prompt': self.basic_prompt,
          'translator_o': self.translator_o,
          'translator_a': self.translator_a,
          'img_names': ['rgb_screen', 'rgb_minimap'],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
          'img_fea': self.ENABLE_IMAGE_FEATURE,
          'img_rgb': self.ENABLE_IMAGE_RGB,
          'model_name': self.model_name,
          'api_base': self.api_base,
          'api_key': self.api_key,
        },
        'team': {
          'Adept-1': {'name': 'Adept-1', 'unit_type': [units.Protoss.Adept], 'game_group': -1, 'select_type': 'select_all_type',
           'actions': {units.Protoss.Adept: STANDARD_ACTION_ADEPT}},
          'AdeptPhase-1': {'name': 'AdeptPhase-1', 'unit_type': [units.Protoss.AdeptPhaseShift], 'game_group': -1, 'select_type': 'select_all_type',
           'actions': {units.Protoss.AdeptPhaseShift: PROTOSS_BASIC_ACTION_3}},
          'DarkTemplar-1': {'name': 'DarkTemplar-1', 'unit_type': [units.Protoss.DarkTemplar], 'game_group': -1, 'select_type': 'select_all_type',
           'actions': {units.Protoss.DarkTemplar: STANDARD_ACTION_DARKTEMPLAR}},
        },
      },

      'CombatGroup8': {
        'describe': "Protoss air special force commander, controls Oracle and Phoenix. "
                    "Responsible for infiltrating the enemy's rear and disrupt economic production, sometimes "
                    "collecting reconnaissance infomation, participating in frontline combat, or build StasisTrap "
                    "to block the enemy's main force.",
        'llm': {
          'basic_prompt': self.basic_prompt,
          'translator_o': self.translator_o,
          'translator_a': self.translator_a,
          'img_names': ['rgb_screen', 'rgb_minimap'],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
          'img_fea': self.ENABLE_IMAGE_FEATURE,
          'img_rgb': self.ENABLE_IMAGE_RGB,
          'model_name': self.model_name,
          'api_base': self.api_base,
          'api_key': self.api_key,
        },
        'team': {
          'Oracle-1': {
            'name': 'Oracle-1', 'unit_type': [units.Protoss.Oracle], 'game_group': -1, 'select_type': 'select_all_type',
            'actions': {units.Protoss.Oracle: STANDARD_ACTION_ORACLE}},
          'Phoenix-1': {
            'name': 'Phoenix-1', 'unit_type': [units.Protoss.Phoenix], 'game_group': -1, 'select_type': 'select_all_type',
            'actions': {units.Protoss.Phoenix: STANDARD_ACTION_PHOENIX}},
        },
      },

      'CombatGroup9': {
        'describe': "Protoss airborne commander, controls WarpPrism and airborne units like Zealots, Stalkers."
                    "Responsible for supplement troops on the front line, or executing multi line combat. "
                    "Keep stability as much as possible in WarpRismPhashing mode to provide stable power field for "
                    "unit warpping.",
        'llm': {
          'basic_prompt': self.basic_prompt,
          'translator_o': self.translator_o,
          'translator_a': self.translator_a,
          'img_names': ['rgb_screen', 'rgb_minimap'],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
          'img_fea': self.ENABLE_IMAGE_FEATURE,
          'img_rgb': self.ENABLE_IMAGE_RGB,
          'model_name': self.model_name,
          'api_base': self.api_base,
          'api_key': self.api_key,
        },
        'team': {
          'WarpPrism': {'name': 'WarpPrism', 'unit_type': [units.Protoss.WarpPrism, units.Protoss.WarpPrismPhasing], 'game_group': -1, 'select_type': 'select',
            'actions': {units.Protoss.WarpPrism: STANDARD_ACTION_WARPPRISM,
                        units.Protoss.WarpPrismPhasing : STANDARD_ACTION_WARPPRISMPHASING}},
        },
      },

      # 'Airborne': {
      #   'describe': "Protoss airborne commander, controls units airborne/warptrain from WarpPrism. "
      #               "Responsible for quick reinforcing nearby units or executing multiline combat.",
      #   'llm': {
      #     'basic_prompt': self.basic_prompt,
      #     'translator_o': self.translator_o,
      #     'translator_a': self.translator_a,
      #     'img_names': [],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
      #     'img_fea': self.ENABLE_IMAGE_FEATURE,
      #     'img_rgb': self.ENABLE_IMAGE_RGB,
      #     'model_name': self.model_name,
      #     'api_base': self.api_base,
      #     'api_key': self.api_key,
      #   },
      #   'team': {
      #     'Airborne-Zealot-1': {
      #       'name': 'Airborne-Zealot-1', 'unit_type': [units.Protoss.Zealot], 'game_group': -1, 'select_type': 'select_all_type',
      #       'actions': {units.Protoss.Zealot: PROTOSS_BASIC_ACTION_2}
      #     },
      #   },
      # },

      # 'Defender': {
      #   'describe': "Protoss garrison troops commander, controls several Stalkers. "
      #               "Responsible for intercepting enemy infiltrating forces.",
      #   'llm': {
      #     'basic_prompt': self.basic_prompt,
      #     'translator_o': self.translator_o,
      #     'translator_a': self.translator_a,
      #     'img_names': [],  # ['power', 'pathable', 'buildable','height_map', 'player_relative']
      #     'img_fea': self.ENABLE_IMAGE_FEATURE,
      #     'img_rgb': self.ENABLE_IMAGE_RGB,
      #     'model_name': self.model_name,
      #     'api_base': self.api_base,
      #     'api_key': self.api_key,
      #   },
      #   'team': {
      #     'Stalker-1': {
      #       'name': 'Stalker-1', 'unit_type': [units.Protoss.Stalker], 'game_group': 1, 'select_type': 'group',
      #       'actions': {units.Protoss.Stalker: STANDARD_ACTION_STALKER}},
      #   },
      # },
    }


# TerranAgentConfig part undergoing
class TerranAgentConfig(AgentConfig):
  def __init__(self):
    super(TerranAgentConfig, self).__init__()


# ZergAgentConfig part undergoing
class ZergAgentConfig(AgentConfig):
  def __init__(self):
    super(ZergAgentConfig, self).__init__()
