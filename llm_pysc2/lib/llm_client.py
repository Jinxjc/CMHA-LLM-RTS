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


from llamaapi import LlamaAPI
from zhipuai import ZhipuAI
from openai import OpenAI


from loguru import logger
from threading import Event
import threading
import random
import time

from llm_pysc2.lib.demo_llm_io import (
    demo_print_llm_exchange,
    demo_serialize_chat_messages,
)


def gpt_query_runtime(self, event):
  llm_response = self.client.chat.completions.create(
  # llm_response = self.client.ChatCompletion.create(
    model=self.model_name,
    messages=self.messages,
    temperature=self.temperature
  )
  if event.is_set():
    return
  self.query_token_in = llm_response.usage.prompt_tokens
  self.query_token_out = llm_response.usage.completion_tokens
  msg = llm_response.choices[0].message
  text = msg.content
  if text is None or (isinstance(text, str) and not text.strip()):
    text = getattr(msg, 'reasoning_content', None) or ''
  self.llm_response = text
  # print(self.query_token_in)
  # print(self.query_token_out)
  # print(self.llm_response)


class GptClient:

  def __init__(self, name, log_id, config):

    self.model_name = config.AGENTS[name]['llm']['model_name']
    self.api_base = config.AGENTS[name]['llm']['api_base']
    self.api_key = config.AGENTS[name]['llm']['api_key']
    self.temperature = config.temperature

    self.client = OpenAI(
      api_key=self.api_key,
      base_url=self.api_base,
    )
    self.client.api_base = self.api_base
    self.client.api_key = self.api_key

    self.agent_name = name
    self.log_id = log_id
    self.config = config
    self.system_prompt = ''
    self.example_i_prompt = ''
    self.example_o_prompt = ''
    self.messages = []
    self.llm_response = None
    self.query_runtime = gpt_query_runtime
    # if 'gpt' in self.model_name or self.model_name == 'default':
    #   logger.info(f"[ID {self.log_id}] {self.agent_name} {self.model_name} GptClient initialized")
    logger.info(f"[ID {self.log_id}] {self.agent_name} {self.model_name} GptClient initialized")

    self.num_query = 0
    self.query_time = 0
    self.query_token_in = 0
    self.query_token_out = 0
    self.total_query_time = 0
    self.total_query_token_in = 0
    self.total_query_token_out = 0
    self.ave_query_time = 0
    self.ave_query_token_in = 0
    self.ave_query_token_out = 0

  def set_prompt(self, prompt):
    self.system_prompt = prompt.sp
    self.example_i_prompt = prompt.eip
    self.example_o_prompt = prompt.eop

  def wrap_message(self, obs_prompt, base64_images):

    if (base64_images is not None) and (self.model_name not in vision_model_names):
      logger.warning(f"[ID {self.log_id}] {self.agent_name} {self.model_name}: Model may not accept img, img discarded. vision_model_names: \n {vision_model_names}")
    if (base64_images is None) and (self.model_name in vision_model_names):
      logger.warning(f"[ID {self.log_id}] {self.agent_name} {self.model_name}: Vision available but img disabled")
    self.messages = [
      {"role": "system", "content": self.system_prompt},
      {"role": "user", "content": self.example_i_prompt},
      {"role": "assistant", "content": self.example_o_prompt},
    ]
    if (base64_images is not None and self.model_name in vision_model_names):
      for key in base64_images:
        if base64_images[key] is None:
          continue
        img_name = f'feature_map_{key}_screen' if key not in ['rgb_minimap', 'rgb_screen'] else key
        self.messages.append({"role": "user", "content": [
          {"type": "text", "text": f'This is the {img_name} image:'},  # obs_prompt
          {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_images[key]}"}}]}
        )
      logger.warning(f"[ID {self.log_id}] {self.agent_name} {self.model_name}: You are using image input, be care of the cost")
    self.messages.append({"role": "user", "content": obs_prompt})

  def query(self, obs_prompt, base64_images: "Dict or None"=None):

    # 重置 messages 列表
    self.wrap_message(obs_prompt, base64_images)

    # 尝试发送请求并获取回复
    max_retries = self.config.MAX_LLM_QUERY_TIMES
    events = [Event() for _ in range(max_retries)]
    for retries in range(max_retries):
      try:
        # tracemalloc.start()

        self.llm_response = None
        logger.success(f"[ID {self.log_id}] {self.agent_name} Start calling llm api!")
        logger.debug(f"[ID {self.log_id}] {self.agent_name} input prompt: \n{obs_prompt}")

        self.thread = threading.Thread(target=self.query_runtime, args=(self, events[retries]))
        self.thread.start()

        # 超时错误
        query_start_time = float(time.time())
        while not isinstance(self.llm_response, str):
          time.sleep(0.1)
          if float(time.time()) - query_start_time > self.config.MAX_LLM_RUNTIME_ERROR_TIME:
            events[retries].is_set()
            logger.error(f"[ID {self.log_id}] {self.agent_name} LLM query runtime error")
            raise RuntimeError(f"{self.agent_name} LLM query runtime error")

        if isinstance(self.llm_response, str):
          self.num_query += 1
          self.query_time = float(time.time()) - query_start_time
          self.total_query_time += self.query_time
          self.total_query_token_in += self.query_token_in
          self.total_query_token_out += self.query_token_out
          self.ave_query_time = self.total_query_time / self.num_query
          self.ave_query_token_in = self.total_query_token_in / self.num_query
          self.ave_query_token_out = self.total_query_token_out / self.num_query
          # current_dir = os.path.dirname(os.path.abspath(__file__))
          # self.log_dir_path = f"{current_dir}/../../llm_log/temp-{self.log_id}"
          # if not os.path.exists(self.log_dir_path):
          #   os.mkdir(self.log_dir_path)
          # if not os.path.exists(self.log_dir_path + f"/{self.agent_name}"):
          #   os.mkdir(self.log_dir_path + f"/{self.agent_name}")
          # path = self.log_dir_path + f"/{self.agent_name}/cost_temp.txt"
          # client_cost = f"time={self.query_time:.2f}, ave_time={self.ave_query_time:.2f}, " \
          #               f"token_in={self.query_token_in}, ave_token_in={self.ave_query_token_in:.2f}, " \
          #               f"token_out={self.query_token_out}, ave_token_out = {self.ave_query_token_out:.2f}"
          # utils.write_to_file(json.dumps({self.num_query: client_cost}), path)

        answer = self.llm_response
        logger.success(f"[ID {self.log_id}] {self.agent_name} Get llm response!")
        logger.debug(f"[ID {self.log_id}] {self.agent_name} llm response: \n{answer}")
        demo_print_llm_exchange(
            self.config,
            "query / query_tactical",
            self.agent_name,
            self.log_id,
            demo_serialize_chat_messages(self.messages),
            answer,
        )
        self.llm_response = None

        return answer
      except Exception as e:
        # 输出错误信息
        logger.error(f"[ID {self.log_id}] {self.agent_name} Error when calling the OpenAI API: {e}")
        # print(f"Error when calling the OpenAI API: {e}")

        # 如果达到最大尝试次数，返回一个特定的回复
        if retries >= max_retries - 1:
          logger.error \
            (f"[ID {self.log_id}] {self.agent_name} Maximum number of retries reached. The OpenAI API is not responding.")
          return "I'm sorry, but I am unable to provide a response at this time due to technical difficulties."

        # 重试前等待一段时间，使用 exponential backoff 策略
        sleep_time = min((2 ** retries) + random.random(), 8 + random.random())
        logger.info(f"[ID {self.log_id}] {self.agent_name} Waiting for {sleep_time} seconds before retrying...")
        time.sleep(sleep_time)

    logger.error(f"[ID {self.log_id}] {self.agent_name} Can not get llm response after try {max_retries} times!")
    return f'[ID {self.log_id}] {self.agent_name} Can not get llm response after try {max_retries} times!'

  def query_strategic(self, user_macro_prompt: str) -> str:
    """Standalone macro / SPC call: does not touch self.messages used by query()."""
    map_name = str(getattr(self.config, "SPC_SELF_PROFILE_MAP_NAME", "Simple64"))
    our_race = str(getattr(self.config, "SPC_SELF_PROFILE_RACE", "Protoss"))
    opponent = str(getattr(self.config, "SPC_SELF_PROFILE_OPPONENT", "Built-in AI"))
    system = (
        "You are the top-level macro strategy module for StarCraft II. "
        f"You are playing on the map **{map_name}** as **{our_race}** against **{opponent}**. "
        "Use your knowledge of StarCraft II tactics, build orders, and meta-game strategies to make decisions. "
        "Recall well-known Protoss strategies from your training data: "
        "4-Gate rush, Stalker-Sentry pressure, Immortal all-in, Chargelot-Archon timing, fast expand into macro, "
        "Colossus deathball, DT harassment, Oracle opening, Void Ray rush, etc. "
        "Choose the best strategy based on:\n"
        "- The map layout and expansion geometry\n"
        "- The opponent race (once scouted: PvZ / PvT / PvP each have different optimal builds)\n"
        "- The current game phase (early / mid / late) and resource situation\n"
        "- Known tendencies of SC2 built-in AI (it tends to macro up and attack at specific timing windows; "
        "it does not cheese or rush early; it can be punished by early aggression or timing attacks)\n\n"
        "**Critical rules:**\n"
        "- Do NOT defend blindly. If you have army and the opponent is still building up, ATTACK or apply pressure.\n"
        "- If minerals are floating high, prioritize spending: train units, build production, expand, or upgrade.\n"
        "- Adapt dynamically: if your initial plan fails, switch to a new one.\n\n"
        "Reason about the situation first, then end your response with structured lines using exact tags:\n"
        "[Memory] A JSON object (flat or nested; recommended keys: enemy_race_guess / enemy_tech_guess / our_plan / risks / open_tasks; "
        "prefer fields value/confidence/ttl per key) [/Memory]\n"
        "[Pulse] MUST be one of: ATTACK, DEFEND, EXPAND, RETREAT\n"
        "[Pivot] One concise tactical focus sentence in English\n"
        "[Horizon] Expected duration of this suggestion in in-game seconds (integer)\n"
        "You may include free-form reasoning before the structured lines, but do not omit any required tag."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_macro_prompt},
    ]
    max_retries = self.config.MAX_LLM_QUERY_TIMES
    for retries in range(max_retries):
      try:
        query_start_time = float(time.time())
        llm_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        usage = getattr(llm_response, "usage", None)
        if usage is not None:
          self.query_token_in = int(getattr(usage, "prompt_tokens", 0) or 0)
          self.query_token_out = int(getattr(usage, "completion_tokens", 0) or 0)
        else:
          self.query_token_in = 0
          self.query_token_out = 0
        self.num_query += 1
        self.query_time = float(time.time()) - query_start_time
        self.total_query_time += self.query_time
        self.total_query_token_in += self.query_token_in
        self.total_query_token_out += self.query_token_out
        self.ave_query_time = self.total_query_time / self.num_query
        self.ave_query_token_in = self.total_query_token_in / self.num_query
        self.ave_query_token_out = self.total_query_token_out / self.num_query
        msg = llm_response.choices[0].message
        text = msg.content
        if text is None or (isinstance(text, str) and not text.strip()):
          text = getattr(msg, 'reasoning_content', None) or ''
        logger.success(f"[ID {self.log_id}] {self.agent_name} SPC strategic LLM response received")
        logger.debug(f"[ID {self.log_id}] {self.agent_name} SPC raw: \n{text}")
        demo_print_llm_exchange(
            self.config,
            "query_strategic (SPC / 顶层策脉)",
            self.agent_name,
            self.log_id,
            "SYSTEM:\n" + system + "\n\nUSER (macro_prompt):\n" + user_macro_prompt,
            text,
        )
        return text
      except Exception as e:
        logger.error(f"[ID {self.log_id}] {self.agent_name} query_strategic API error: {e}")
        if retries >= max_retries - 1:
          return ""
        sleep_time = min((2 ** retries) + random.random(), 8 + random.random())
        time.sleep(sleep_time)
    return ""

  def query_tactical(self, tactical_prompt: str, base64_images=None) -> str:
    """与 query 相同通道；单独命名便于战术沙盒重试，不改动 messages 拼装语义。"""
    return self.query(tactical_prompt, base64_images=base64_images)

# for config's auto check
vision_model_names = [
  'gpt-4o', 'gpt-4o-all', 'gpt-4o-mini',
]
video_model_names = [

]

FACTORY = {
  'default': GptClient,
  'gpt-3.5-turbo': GptClient,

  'gpt-4o': GptClient,
  'gpt-4o-all': GptClient,
  'gpt-4o-mini': GptClient,

  'deepseek-chat': GptClient,
  'deepseek-reasoner': GptClient,
  'deepseek-v3': GptClient,
  'deepseek-r1': GptClient,
}

if __name__ == "__main__":
  from llm_pysc2.cfg.config import ProtossAgentConfig
  config = ProtossAgentConfig()
  model_name = 'gpt-3.5-turbo'
  api_base = 'https://api.xty.app/v1'
  api_key = ''
  config.reset_llm(model_name, api_base, api_key)
  c = GptClient('CombatGroup0', 0, config)
  response = c.query('hello')
  print(response)