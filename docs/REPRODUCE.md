# Reproduction Guide (CMHA)

Anonymous supplementary material for **CMHA**. Extends [LLM-PySC2](https://github.com/NKAI-Decision-Team/LLM-PySC2) with **SPC (Strategy Pulse Chain Brain)** and full-game experiments. Overview: [`README.md`](../README.md). Paper draft: [`paper/cmha_en.tex`](../paper/cmha_en.tex).

---

## 1. StarCraft II

CMHA requires a **full StarCraft II install** with the **game API (≥ 3.16.1)**. **Starter Edition** is enough for built-in AI and PySC2.

### Linux

Install from [Blizzard s2client-proto](https://github.com/Blizzard/s2client-proto#downloads). Default expected path:

```text
~/StarCraftII/
```

Override with:

```bash
export SC2PATH=/path/to/StarCraftII
```

### Windows / macOS

Install via [Battle.net](https://battle.net). Default paths are usually detected automatically.

Common Windows path:

```text
C:\Program Files (x86)\StarCraft II
```

Custom install:

```powershell
$env:SC2PATH='C:\Program Files (x86)\StarCraft II'
# persistent:
[Environment]::SetEnvironmentVariable('SC2PATH', 'C:\Program Files (x86)\StarCraft II', 'User')
```

Restart the terminal after setting `SC2PATH`.

> **Regional clients:** Some non-international Battle.net builds may not support PySC2 API mode reliably. If you see `ConnectError` on launch, use an international Starter Edition install and point `SC2PATH` to it.

### Verify

```text
StarCraft II/
├── Versions/     # Base***** / SC2_x64.exe
├── Maps/
└── StarCraft II.exe
```

---

## 2. Python environment

```bash
conda create -n cmha-env python=3.9
conda activate cmha-env
cd /path/to/repo
pip install -e .
```

Optional PyPI mirror for faster downloads in some regions:

```bash
pip install -e . -i https://pypi.org/simple
```

Python 3.8+ is supported upstream; paper experiments used **3.9**. On Windows, set `LLM_PYSC2_NORENDER=1` to reduce display-related SC2 issues.

---

## 3. Maps

### Task maps (llm_pysc2 / llm_smac)

```text
llm_pysc2/maps/llm_pysc2  →  <SC2>/Maps/llm_pysc2
llm_pysc2/maps/llm_smac   →  <SC2>/Maps/llm_smac
```

### Full-game maps (Simple64)

Download the [Melee map pack](https://github.com/Blizzard/s2client-proto#map-packs):

```text
<SC2>/Maps/Melee/Simple64.SC2Map
```

---

## 4. LLM API

No API keys are stored in this repository.

**Option A — environment variables:**

```bash
export OPENAI_API_BASE=https://api.openai.com/v1
export OPENAI_API_KEY=your-key
export LLM_MODEL_NAME=gpt-4o-mini
```

**Option B — local `.env` (gitignored):**

```bash
cp .env.example .env
# edit OPENAI_API_KEY, SC2PATH, etc.
```

Implementation: `llm_pysc2/cfg/llm_env.py`

**Dry run without API:**

```python
config.LLM_SIMULATION_TIME = 5
```

---

## 5. Experiments

### 5.1 Full-game (Table 1)

Entry: `llm_pysc2/bin/experiment_full_game.py`

Settings: **Simple64**, **ECSB**, built-in AI **Levels 1–7**, **12** episodes per level, Protoss.

| Variable | Paper setting |
|----------|----------------|
| `SC2_MODE` | `ECSB` |
| `SC2_DIFFICULTY_LEVEL` | `1`–`7` |
| `MAX_EPISODES` | `12` |
| `SC2_AGENT2_RACE` | e.g. `protoss` |
| `LLM_PYSC2_NORENDER` | `1` |

```bash
conda activate cmha-env
export SC2PATH=/path/to/StarCraft\ II
export SC2_MODE=ECSB
export SC2_DIFFICULTY_LEVEL=3
export MAX_EPISODES=12
export SC2_AGENT2_RACE=protoss
export LLM_PYSC2_NORENDER=1
export OPENAI_API_KEY=your-key

python -m llm_pysc2.bin.experiment_full_game
```

### 5.2 llm_pysc2 PvZ tasks

```bash
python -m llm_pysc2.bin.experiment_llm_pysc2
```

Edit `task` / `level` in the script. See `docs/experiments.md`.

### 5.3 llm_smac

```bash
python -m llm_pysc2.bin.experiment_llm_smac
```

Scripts: `llm_pysc2/bin/llm_smac/`.

---

## 6. Logs

Runs write to `llm_log/<timestamp-id>/` (gitignored). Remove old folders to free disk space.

---

## 7. Citation

See `CITATION.bib`. Cite CMHA (anonymous entry until acceptance) and LLM-PySC2 / PySC2 as appropriate.

Prior work: https://arxiv.org/abs/2411.05348
