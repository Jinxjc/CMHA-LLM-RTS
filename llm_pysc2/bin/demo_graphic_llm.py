# Copyright 2025, LLM-PySC2 Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# 图形界面（PySC2 pygame 观测窗）+ 终端实时打印各 LLM 的输入/输出，便于演示「一边看对局一边看模型在干什么」。
#
# 推荐在有桌面 / DISPLAY 的机器上运行：
#   PYTHONUNBUFFERED=1 python -m llm_pysc2.bin.demo_graphic_llm
#
# 等价于开启 LLM_PYSC2_DEMO_IO 后跑 experiment_full_game，并去掉 LLM_PYSC2_NORENDER（保证带 --render）。
#
# 仅打印、不截断长文本：
#   LLM_PYSC2_DEMO_IO_FULL=1 python -m llm_pysc2.bin.demo_graphic_llm
#
# 将终端 I/O 另存为文件：
#   PYTHONUNBUFFERED=1 python -m llm_pysc2.bin.demo_graphic_llm 2>&1 | tee demo_llm_io.log

from __future__ import annotations

import os
import runpy


def main():
  os.environ.setdefault("PYTHONUNBUFFERED", "1")
  os.environ.setdefault("LLM_PYSC2_DEMO_IO", "1")
  os.environ.setdefault("MAX_EPISODES", "1")
  # 演示入口默认要窗口；服务器纯终端请用 experiment_full_game + LLM_PYSC2_NORENDER=1
  os.environ.pop("LLM_PYSC2_NORENDER", None)
  here = os.path.dirname(os.path.abspath(__file__))
  runpy.run_path(os.path.join(here, "experiment_full_game.py"), run_name="__main__")


if __name__ == "__main__":
  main()
