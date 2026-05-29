# Copyright 2025, LLM-PySC2 Contributors. All Rights Reserved.
"""Read LLM API settings from environment variables (safe for public repos)."""

import os

PLACEHOLDER_API_BASE = 'YOUR-API-BASE'
PLACEHOLDER_API_KEY = 'YOUR-API-KEY'


def llm_credentials_from_env(default_model='gpt-4o-mini'):
  """Return (model_name, api_base, api_key) from env with safe placeholders.

  Supported variables (first match wins per field):
    model: LLM_MODEL_NAME, OPENAI_MODEL
    base:  OPENAI_API_BASE, LLM_API_BASE
    key:   OPENAI_API_KEY, LLM_API_KEY
  """
  model = (
      os.getenv('LLM_MODEL_NAME')
      or os.getenv('OPENAI_MODEL')
      or default_model
  )
  api_base = (
      os.getenv('OPENAI_API_BASE')
      or os.getenv('LLM_API_BASE')
      or PLACEHOLDER_API_BASE
  )
  api_key = (
      os.getenv('OPENAI_API_KEY')
      or os.getenv('LLM_API_KEY')
      or PLACEHOLDER_API_KEY
  )
  return model, api_base, api_key
