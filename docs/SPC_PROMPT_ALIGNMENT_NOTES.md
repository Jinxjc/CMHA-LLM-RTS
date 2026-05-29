# SPC prompts and environment inputs — implementation notes

Notes for aligning the paper description with this codebase. Update when prompts or call paths change.

## 1. `query_strategic` vs. tactical baseline prompts

**Observation**

- Macro calls attach the same full `text_o` as the baseline (`translator_o.translate`).
- `query_strategic` uses a dedicated macro `system` + single `user` (`macro_prompt`) and does **not** reuse each agent’s `basic_prompt` (`system_prompt` / `example_i_prompt` / `example_o_prompt`).
- Tactical path `query` → `wrap_message` uses **system + few-shot examples + final user (SPC prefix + text_o)**.

**Impact**

- Macro and tactical layers may differ in instruction style, constraints, and few-shot examples for the same game state. A fully unified prompt stack across layers is **not** implemented.

**Possible extensions (not implemented)**

- Prepend or merge `basic_prompt` (sp/eip/eop) into `query_strategic`; or maintain macro-specific few-shot aligned with tactical prompts.
- Macro-only `text_o` compression if token budget is tight (trade-off with using full `text_o`).

## 2. Tactical side (`query_tactical` → `query`)

Uses the original **system + examples + text_o**, with a short SPC prefix (and sandbox retry hints) prepended to `text_o`. Matches the baseline aside from the prefix.

## 3. Simulation mode (`LLM_SIMULATION_TIME > 0`)

Skips sandbox and remote tactical calls; input still flows through `get_text_a(text_o, ...)`. Differs from the live tactical + sandbox path by design.

---

## 4. Full-game runs (Simple64)

**Entry:** `llm_pysc2/bin/experiment_full_game.py` (`SC2_DIFFICULTY_LEVEL` 1–10 maps to built-in AI names).

**API:** After `reset_llm`, all `AGENTS[*]['llm']` share one `model_name / api_base / api_key`. Each `LLMAgent` uses the same `GptClient` for **`query_strategic`** and **`query` / `query_tactical`**.

**SC2PATH** must point to the StarCraft II root (`Versions/`, `Maps/`):

```bash
export SC2PATH=/path/to/StarCraftII
```

```powershell
$env:SC2PATH='C:\Program Files (x86)\StarCraft II'
```

If unset, `llm_pysc2.bin.run_env.ensure_sc2path()` checks the Windows registry and common install paths.

**Latency:** For slow reasoner models, increase `MAX_LLM_RUNTIME_ERROR_TIME` in `experiment_full_game.py`; adjust `SPC_STRATEGIC_INTERVAL_GAME_SECONDS` to reduce dual-call frequency if needed.

---

*Internal alignment doc for CMHA / SPC implementation.*
