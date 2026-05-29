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


from llm_pysc2.cfg import ProtossAgentConfig
from llm_pysc2.cfg.llm_env import llm_credentials_from_env
from llm_pysc2.agents import *
from llm_pysc2.bin.run_env import ensure_sc2path, python_exe
import os

map_name = f"Simple64"
# map_name = f"debug_map"
difficult_level = int(os.getenv('SC2_DIFFICULTY_LEVEL', '3'))
difficulties = ['very_easy', 'easy', 'medium',
                'medium_hard', 'hard', 'harder', 'very_hard',
                'cheat_vision', 'cheat_money', 'cheat_insane']
if difficult_level < 1 or difficult_level > len(difficulties):
  difficult_level = 3
difficulty = difficulties[difficult_level - 1]  # from sc2_env.Difficulty

num_episodes = int(os.getenv('MAX_EPISODES', '5'))

# 打开 RGB 小地图/屏幕图像送入 LLM（gpt-4o-mini 在 vision_model_names 中；费 token / 延迟更高）
enable_image_rgb, enable_image_feature = False, False
# enable_image_rgb, enable_image_feature = False, True


def _mode_flags_from_env():
  """Return (ENABLE_EASY_CONTROL, ENABLE_EASY_BUILD) from SC2_MODE.

  Naming (do not confuse ECSB with SCSB):
  - ECEB: easy control + easy build  -> (True, True)
  - ECSB: easy control + standard build -> (True, False)
  - SCSB: standard control + standard build -> (False, False), not ECSB
  If SC2_MODE is unset, default matches ECSB: (True, False).
  """
  mode = str(os.getenv("SC2_MODE", "")).strip().upper()
  if mode == "ECEB":
    return True, True
  if mode == "ECSB":
    return True, False
  if mode == "SCSB":
    return False, False
  return True, False


class MainAgentLLMPysc2(MainAgent):
  def __init__(self):
    config = ProtossAgentConfig()

    model_name, api_base, api_key = llm_credentials_from_env()
    config.reset_llm(model_name, api_base, api_key, enable_image_rgb, enable_image_feature)

    # ECEB + SPC：chat 模式响应通常快于 reasoner；仍保留宽松超时以防网络抖动。
    config.MAX_LLM_RUNTIME_ERROR_TIME = 120
    # 多 agent 轮询时主循环等待窗口（秒）；略增大以减少多 LLM 排队时误触发超时。
    config.MAX_LLM_WAITING_TIME = 90

    # for name in config.AGENTS.keys():
    #   if name not in ['Builder', 'Commander', 'Developer', 'CombatGroup0', 'CombatGroup1']:  # , 'Developer'
    #     config.AGENTS_ALWAYS_DISABLE.append(name)
    #   if name not in ['Builder', 'Commander', 'Developer']:  # For CombatGroups
    #     config.AGENTS[name]['llm']['model_name'] = 'YOUR-MODEL-NAME'  # another
    #     config.AGENTS[name]['llm']['api_base'] = 'YOUR-API-BASE'  # another
    #     config.AGENTS[name]['llm']['api_key'] = 'YOUR-API-KEY'  # another
    #     config.AGENTS[name]['llm']['img_names'] = []  # disable all images
    #     config.AGENTS[name]['llm']['img_rgb'] = False
    #     config.AGENTS[name]['llm']['img_fea'] = False

    # config.SAFE_MODE = False
    # config.LLM_SIMULATION_TIME = 0.5
    # config.IGNORE_INIT_WARNINGS = True
    # config.ENABLE_MULTI_THREAD_QUERY = False

    # config.MAX_LLM_DECISION_FREQUENCY = 0.2
    # ECEB + 论文/LLM-PySC2 默认：Builder/Combat 不问 LLM，Commander + Developer 均 query（经济产兵在 Developer）。
    config.ENABLE_SINGLE_LLM_AGENT = False
    # 顶层策脉：StrategicPulseBrain（MainAgent 驱动，与 Commander.query 解耦）；共享快照供 Developer 等对齐全局脉搏。
    config.ENABLE_SPC_STRATEGIC_LAYER = True
    config.SPC_SHARED_STRATEGIC_BRAIN = True
    config.SPC_STRATEGIC_OWNER_AGENT = 'Commander'
    # 顶层 query_strategic（策脉）≈15s；Commander 战术 query_tactical ≈5s（游戏内秒）
    config.SPC_STRATEGIC_INTERVAL_GAME_SECONDS = 15.0
    config.COMMANDER_LLM_INTERVAL_GAME_SECONDS = 5.0
    # 最顶层 query_strategic 不使用 Commander 执行反馈（与 config 默认一致，可改为 True 做消融）。
    config.SPC_INCLUDE_COMMANDER_FEEDBACK = False
    config.SPC_STRATEGIC_DOCTRINE = """
You are an AI specialized in structured StarCraft II strategic planning for Protoss.
Map is Simple64 (standard 1v1): max 3 base locations and max 4 gas buildings.

Race reminder:
- For Protoss, monitor Nexus energy and use Chrono Boost to speed up key structures.

Tactic candidates (choose one as current tactic based on game time and scouting):
1) Zealot and Stalker Tactic (strong all-in within 8 minutes)
- Buildings:
  * Military: Gateway/Warp Gate (about 4 is enough).
  * Tech: Cybernetics Core (1), Twilight Council (1), Forge (1).
- Technologies:
  * Required: Warp Gate, Charge.
  * Optimal: Ground Weapons 1-2, Ground Armor 1-2, Shields 1-2.
- Force ratio: Zealot:Stalker = 1:1 or 2:1.
- Timing anchors:
  * 2:00 -> 2 Nexuses + 2 Gateways.
  * 4:00 -> 3 to 4 Warp Gates + Cybernetics Core + start Warp Gate research.
  * 8:00 -> 4 to 5 Warp Gates + Cybernetics Core + Twilight Council + key tech online.
- Applicable window: first 8 minutes.

2) Carrier Tactic (late-game stable)
- Buildings:
  * Military: Stargate (max 4).
  * Tech: Cybernetics Core, Fleet Beacon (1), Forge.
- Technologies:
  * Required: Air Weapons 1-2.
  * Optimal: Air Armor, Shields, Ground Armor.
- Force structure:
  * Before 6:00 -> defend with Zealots + Stalkers.
  * After 6:00 -> Carrier core with a small Stalker support group.
- Timing anchors:
  * 2:00 -> 2 Nexuses + 2 Gateways.
  * 6:00 -> 2 Stargates + Cybernetics Core + Fleet Beacon + first Carrier in production.
  * 10:00 -> 4 Stargates + 3 to 4 Carriers.
- Applicable window: after 6 minutes, or when enemy has air/heavy units.

Priority construction analysis:
- Nexus rules (Simple64 max 3):
  * Workers > 16 -> need 2 Nexuses.
  * Workers > 35 -> need 3 Nexuses (max).
  * Beyond this, do not prioritize additional Nexuses.
- Gas rules (Simple64 max 4 Assimilators):
  * Workers < 25 -> no Assimilator priority.
  * Workers > 25 -> need 2 Assimilators.
  * Workers > 45 -> need 4 Assimilators (map max).
- Extract data first, then determine whether a real priority exists.

Standard planning guidance:
- Tech: build required tech structures and complete missing technologies according to the chosen tactic.
- Economy: keep producing Probes while workers < 80; build Pylon when supply left < 4.
- Military: build enough production and army according to the chosen tactic.
- Scouting: send Probe scout when worker/economy state is comfortable.
- Chrono Boost priority: Stargate (during Carrier plan), then key tech, then Nexuses.
- Attack thresholds:
  * Army supply < 60 -> do not attack.
  * Army supply >= 60 -> can attack.
  * Army supply >= 80 -> should attack.

Output style constraints:
- Output strategic slots only (Pulse/Pivot/CommanderIntent/DeveloperIntent/Horizon/Memory).
- Do not output executable action names.
"""
    config.ENABLE_COMMUNICATION = True
    easy_control, easy_build = _mode_flags_from_env()
    config.ENABLE_EASY_CONTROL = easy_control
    config.ENABLE_EASY_BUILD = easy_build
    # 用户要求：主循环频控固定为 1（不再按时间段动态调节）。
    config.MAX_LLM_DECISION_FREQUENCY = 1
    # False: 关闭剧本记忆/PlaybookInterrupt 与 Developer 的 scipt 科技兜底（VB 强制等）。
    # True: 启用 llm_pysc2/cfg/config.py 注释里描述的 scipt 整包行为。
    config.scipt = False

    # config.ENABLE_AUTO_WORKER_MANAGE = False
    # config.ENABLE_EASY_BUILD = False

    super(MainAgentLLMPysc2, self).__init__(config, LLMAgent)

  def step(self, obs):
    self.config.MAX_LLM_DECISION_FREQUENCY = 1
    return super().step(obs)


if __name__ == "__main__":

  ensure_sc2path()
  ep_flag = f"--max_episodes {num_episodes}"
  # Built-in AI race: random | terran | protoss | zerg (pysc2 --agent2_race)
  _a2 = str(os.getenv("SC2_AGENT2_RACE", "random")).strip().lower()
  bot_race_flag = f"--agent2_race {_a2}"
  # 子进程无缓冲（Unix 的 env 前缀在 Windows 不可用，故改环境变量 + 同解释器）
  os.environ['PYTHONUNBUFFERED'] = '1'
  py = python_exe()
  # 默认弹出 PySC2 pygame 观测窗；仅 SSH 无 DISPLAY 的节点设 LLM_PYSC2_NORENDER=1 关闭。
  _norender_env = str(os.getenv("LLM_PYSC2_NORENDER", "")).strip() in (
      "1", "true", "TRUE", "yes", "YES")
  _render_suffix = " --norender" if _norender_env else " --render"
  if not (enable_image_rgb or enable_image_feature):
    os.system(f"{py} -m pysc2.bin.agent --map {map_name} --difficulty {difficulty} --agent_race protoss {bot_race_flag} --parallel 1 "
              f"{ep_flag} "
              f"--agent llm_pysc2.bin.experiment_full_game.MainAgentLLMPysc2"
              f"{_render_suffix}")
  elif enable_image_rgb:
    os.system(f"{py} -m pysc2.bin.agent --map {map_name} --difficulty {difficulty} --agent_race protoss {bot_race_flag} --parallel 1 "
              f"{ep_flag} "
              f"--agent llm_pysc2.bin.experiment_full_game.MainAgentLLMPysc2 "
              f"--feature_screen_size 256 --feature_minimap_size 64 "
              f"--rgb_screen_size 256 --rgb_minimap_size 64 "
              f"--action_space RGB"
              f"{_render_suffix}")
  elif enable_image_feature:  # parallel experiments with feature map obs do not available currently, set --parallel 1
    os.system(f"{py} -m pysc2.bin.agent --map {map_name} --difficulty {difficulty} --agent_race protoss {bot_race_flag} --parallel 1 "
              f"{ep_flag} "
              f"--agent llm_pysc2.bin.experiment_full_game.MainAgentLLMPysc2 "
              f"--feature_screen_size 256 --feature_minimap_size 64 "
              f"--rgb_screen_size 0 --rgb_minimap_size 0 "
              f"{_render_suffix}")
  else:
    print("Can not enable_image_rgb and enable_image_feature at the same time")
