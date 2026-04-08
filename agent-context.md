# Agent Context: Cost-Aware FinQA Environment

## Project Overview

This is a **Cost-Aware FinQA Deep Research Agent** for the **Meta PyTorch Hackathon x Scaler School of Technology, Round 1**.

- **GitHub**: https://github.com/nsharan2000/cost-aware-finqa
- **HF Space**: https://huggingface.co/spaces/Teachafy/cost-aware-finqa
- **Deadline**: Round 1 closes 8 April 2026, 11:59 PM IST

## Architecture

### What Runs Where

| Component | Where it runs | Python version | Key files |
|-----------|--------------|----------------|-----------|
| **Environment Server** | HF Space (Docker) | 3.11-slim | `server/app.py`, `server/cost_aware_finqa_environment.py`, `server/tools.py`, `server/gradio_ui.py` |
| **Inference Script** | Validator's machine | Unknown (not 3.14) | `inference.py` |
| **Package** | Installed via `pip install .` | Both | `__init__.py`, `models.py`, `client.py` |

### Repo Structure (Non-Standard!)

```
repo_root/                  # This IS the cost_aware_finqa package
├── __init__.py             # Package init
├── models.py               # CostAwareFinqaAction, CostAwareFinqaObservation (Pydantic models)
├── client.py               # CostAwareFinqaEnv (EnvClient subclass)
├── inference.py             # Validator runs this directly
├── pyproject.toml           # Package config
├── Dockerfile               # For HF Space environment server
├── openenv.yaml             # OpenEnv framework config
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI app (create_app + Gradio UI)
│   ├── cost_aware_finqa_environment.py  # Environment logic
│   ├── tools.py             # Tool implementations (sql_query, web_search, upgrade_llm)
│   └── gradio_ui.py         # Tabbed Gradio UI
├── data/
│   ├── financial_data.db    # SQLite with financial tables
│   └── finqa_200.json       # 200 financial QA questions
```

**CRITICAL**: The `package-dir` mapping in `pyproject.toml` maps `.` (repo root) as `cost_aware_finqa`:
```toml
package-dir = { "cost_aware_finqa" = ".", "cost_aware_finqa.server" = "server" }
```
This is non-standard and causes issues with some setuptools versions (see Known Issues below).

### Dependency Chain

```
inference.py
  → cost_aware_finqa (package)
    → models.py → openenv.core.env_server.types (Action, Observation)  # just Pydantic BaseModel subclasses
    → client.py → openenv.core (EnvClient, StepResult, State)
  → openai (for LLM calls)
```

`openenv-core[core]>=0.2.2` is declared as a dependency in `pyproject.toml`. The `Action` and `Observation` base classes are simple Pydantic `BaseModel` subclasses with `extra="forbid"` config.

## Validator Behavior

The competition validator:
1. Clones the GitHub repo to `/tmp/workspace/`
2. Runs `pip install .` (installs package + dependencies including `openenv-core`)
3. Injects env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, `IMAGE_NAME`
4. Runs `python inference.py`
5. All LLM calls must go through their `API_BASE_URL` (LiteLLM proxy) with `HF_TOKEN` as the API key

**Phase 2 checks are fail-fast** — one failure stops the pipeline.

### Injected Environment Variables

| Variable | Purpose | Used in inference.py |
|----------|---------|---------------------|
| `API_BASE_URL` | LLM proxy endpoint | `OpenAI(base_url=API_BASE_URL)` |
| `MODEL_NAME` | Model identifier | `client.chat.completions.create(model=MODEL_NAME)` |
| `HF_TOKEN` | API key for LLM proxy | `OpenAI(api_key=API_KEY)` where `API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")` |
| `IMAGE_NAME` | Docker image for environment | `CostAwareFinqaEnv.from_docker_image(IMAGE_NAME)` |

## Known Issues & Solutions

### Issue #1: ImportError — `(unknown location)` (Submissions #3-5)

**Error**:
```
ImportError: cannot import name 'CostAwareFinqaAction' from 'cost_aware_finqa' (unknown location)
```

**Root Cause**: The `(unknown location)` message means Python found the `cost_aware_finqa` module but as a **namespace package** (without `__init__.py`). This happens because the non-standard `package-dir = { "cost_aware_finqa" = "." }` mapping doesn't always include `__init__.py` in the installed package on certain setuptools versions.

Previously, `__init__.py` had a `try/except ImportError` that silently caught failures and set `__all__ = []`, masking the real error.

**How to reproduce**:
```python
# Install package, then delete __init__.py from site-packages
import cost_aware_finqa, os
os.remove(os.path.join(os.path.dirname(cost_aware_finqa.__file__), '__init__.py'))
# Clear cache and re-import
import sys; del sys.modules['cost_aware_finqa']
from cost_aware_finqa import CostAwareFinqaAction  # → (unknown location) error
```

**Fix Applied** (commit ec8bcd5):
1. `__init__.py`: Removed try/except — imports now propagate errors clearly
2. `inference.py`: Added fallback direct imports from same directory:
   ```python
   try:
       from cost_aware_finqa import CostAwareFinqaAction, CostAwareFinqaEnv
   except ImportError:
       sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
       from models import CostAwareFinqaAction
       from client import CostAwareFinqaEnv
   ```

**Tested scenarios** (all pass):
- Package installed correctly → package import works
- Package installed but `__init__.py` missing → fallback works
- Package not installed → fallback works

### Issue #2: "No API calls through LLM proxy" (Submission #4)

**Root cause**: The import crash in `inference.py` happened before any LLM calls were made.

**Fix**: Same as Issue #1 — once the import works, LLM calls go through the validator's proxy via `OpenAI(base_url=API_BASE_URL, api_key=API_KEY)`.

### Issue #3: Easy sample question scoring 0

**Root cause**: Agent queried wrong tables and computed wrong answers.

**Fix applied**: 
- `server/gradio_ui.py` loads a specific GS question (id: `ded692383bf6`) for the easy demo
- Gold answer: 13588.0 (simple subtraction: 35764 - 22176)
- Improved agent system prompt to use exact table names from schema

## Key Files to Understand

### inference.py (what the validator runs)
- Uses `OpenAI` client with validator-injected `API_BASE_URL` and `HF_TOKEN`
- Creates environment via `CostAwareFinqaEnv.from_docker_image(IMAGE_NAME)`
- Runs 3 tasks × 3 questions each: basic_retrieval, analytical_reasoning, strategic_research
- Agent uses LLM to decide tools at each step (max 8 steps)
- `SYSTEM_PROMPT` guides the agent on tool selection strategy

### server/tools.py (environment internals)
- `sql_query`: $0.001/call, runs SQL on SQLite DB. Error penalty: -0.15
- `web_search`: $0.02/call, uses Serper API (with simulated fallback)
- `upgrade_llm`: $1.00/call, provides reasoning guidance (1000x SQL cost)
- `submit_answer`: Free, grades via fuzzy numerical matching
- **Reward formula**: `R_total = R_correctness × max(0.1, 1 - cost/budget) × (1 - min(0.5, |negative_rewards|))`

### server/gradio_ui.py (HF Space UI)
- Tabbed interface: Agent Chat + Playground
- Sample questions with specific question IDs for demo reliability
- Custom agent system prompt for the chat interface

## Debugging Checklist

If the import error recurs:
1. Check if `openenv-core` is actually installed: `pip show openenv-core`
2. Check if our package is installed: `pip show openenv-cost_aware_finqa`
3. Check if `__init__.py` exists in installed package: `python -c "import cost_aware_finqa; print(cost_aware_finqa.__file__)"`
4. If `__file__` is None → namespace package issue → the fallback in inference.py should handle this

If new errors appear:
1. The fallback import path needs `openenv-core` installed for `models.py` and `client.py` to work
2. If `openenv-core` itself fails, check Python version compatibility
3. The Dockerfile uses Python 3.11-slim; validator's Python version is unknown

## HF Space Deployment

- Push code changes via `HfApi.upload_file()` (not git push — binary files cause issues with Xet storage)
- Always restart after upload: `HfApi().restart_space('Teachafy/cost-aware-finqa')`
- Verify status: `HfApi().get_space_runtime('Teachafy/cost-aware-finqa')`
- The Space uses Docker SDK runtime, builds from Dockerfile in repo root

## Submission History

| # | Date | Result | Issue |
|---|------|--------|-------|
| 3 | Apr 8 | Failed Phase 2 | ImportError: `(unknown location)` — try/except masked real error |
| 4 | Apr 8 | Failed Phase 2 | Same ImportError + "No API calls through LLM proxy" |
| 5 | Apr 8 | Failed Phase 2 | Same ImportError — stale build or packaging issue |
| 6 | Apr 8 | Pending | Fixed: removed try/except, added fallback imports |
