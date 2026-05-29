# CMHA — StarCraft II LLM Environment


This repository provides a StarCraft II environment where LLM agents interact with the game via text observations and actions. For full setup and reproduction details, see [**docs/REPRODUCE.md**](docs/REPRODUCE.md).

---

## Requirements

- **StarCraft II** with game API support (**≥ 3.16.1**); Starter Edition is sufficient
- **Python 3.9**
- OpenAI-compatible API (or set `LLM_SIMULATION_TIME` in config for dry runs without API calls)
- Maps from this repo plus the Blizzard [Melee map pack](https://github.com/Blizzard/s2client-proto#map-packs) for full-game maps (e.g. Simple64)

---

## Installation

```bash
conda create -n cmha-env python=3.9
conda activate cmha-env
cd CMHA-LLM-RTS
pip install -e .
cp .env.example .env   # local only; never commit
```

Set `SC2PATH` to your StarCraft II install root (must contain `Versions/` and `Maps/`):

```bash
export SC2PATH=/path/to/StarCraft\ II
```

```powershell
$env:SC2PATH='C:\Program Files (x86)\StarCraft II'
```

Copy maps into the game directory:

```
llm_pysc2/maps/llm_pysc2  →  <SC2>/Maps/llm_pysc2
llm_pysc2/maps/llm_smac   →  <SC2>/Maps/llm_smac
Melee/Simple64.SC2Map     →  <SC2>/Maps/Melee/
```

API settings (see also `llm_pysc2/cfg/llm_env.py` and `.env.example`):

```bash
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=your-key
LLM_MODEL_NAME=gpt-4o-mini
```

---

## Run

### Full-game (Simple64)

```bash
conda activate cmha-env
export SC2PATH=/path/to/StarCraft\ II
export SC2_MODE=ECSB
export SC2_DIFFICULTY_LEVEL=3
export MAX_EPISODES=1
export SC2_AGENT2_RACE=protoss
export LLM_PYSC2_NORENDER=1

python -m llm_pysc2.bin.experiment_full_game
```

Common environment variables:

| Variable | Description |
|----------|-------------|
| `SC2PATH` | StarCraft II install root |
| `SC2_MODE` | Control mode, e.g. `ECSB`, `ECEB`, `SCSB` |
| `SC2_DIFFICULTY_LEVEL` | Built-in AI level (1–7) |
| `MAX_EPISODES` | Number of games to run |
| `SC2_AGENT2_RACE` | Opponent race: `protoss`, `terran`, `zerg` |
| `LLM_PYSC2_NORENDER` | Set `1` to disable the PySC2 render window |

Run logs are written to `llm_log/` (gitignored).

### Other entry points

```bash
python -m llm_pysc2.bin.experiment_llm_pysc2   # LLM-PySC2 PvZ micro tasks
python -m llm_pysc2.bin.experiment_llm_smac    # SMAC micro tasks
```

Task descriptions: [`docs/experiments.md`](docs/experiments.md). Troubleshooting: [`docs/problems.md`](docs/problems.md).

---

## Project layout

```
llm_pysc2/
  bin/          # experiment entry scripts
  cfg/          # agent & environment config
  lib/          # LLM client, translators, SPC module
  maps/         # custom maps (copy to SC2 Maps/)
docs/           # setup & experiment docs
paper/          # anonymous paper draft (LaTeX)
```

---
