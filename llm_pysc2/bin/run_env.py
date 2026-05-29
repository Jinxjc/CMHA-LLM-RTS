# Copyright 2025, LLM-PySC2 Contributors. All Rights Reserved.
#
# Subprocess uses the same interpreter as the parent by default (correct conda env on any OS).
# Override: set LLM_PYSC2_PYTHON=/path/to/python

import os
import sys


def python_exe():
  return os.environ.get('LLM_PYSC2_PYTHON', sys.executable)


def _registry_sc2path_windows():
  if os.name != 'nt':
    return None
  try:
    import winreg
  except ImportError:
    return None
  for hive, subkey in (
      (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\WOW6432Node\Blizzard Entertainment\StarCraft II'),
      (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Blizzard Entertainment\StarCraft II'),
  ):
    try:
      with winreg.OpenKey(hive, subkey) as key:
        val, _ = winreg.QueryValueEx(key, 'InstallPath')
        if val and os.path.isdir(val):
          return os.path.abspath(val)
    except OSError:
      pass
  return None


def ensure_sc2path():
  """If SC2PATH is unset, point pysc2 at a local StarCraft II install.

  Override anytime with: export SC2PATH=/path/to/StarCraftII
  """
  if os.environ.get('SC2PATH'):
    return
  reg = _registry_sc2path_windows()
  if reg:
    os.environ['SC2PATH'] = reg
    return
  candidates = (
      os.path.expanduser('~/StarCraftII'),
      os.path.expanduser('~/StarCraft II'),
      r'C:\Program Files (x86)\StarCraft II',
      r'C:\Program Files\StarCraft II',
  )
  for base in candidates:
    if os.path.isdir(os.path.join(base, 'Versions')):
      os.environ['SC2PATH'] = os.path.abspath(base)
      return
