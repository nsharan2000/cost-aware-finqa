---
title: Cost-Aware FinQA Deep Research agent
sdk: docker
app_port: 8000
tags:
  - openenv
---
# Cost-Aware FinQA Deep research agent

RL environment for cost-aware tool selection in financial question answering. See [DESIGN.md](DESIGN.md) for the full rationale and competition notes.

## Repo Structure

```
cost_aware_finqa/
  models.py                 # Pydantic Action/Observation models
  client.py                 # WebSocket client (EnvClient subclass)
  __init__.py               # Package exports
  openenv.yaml              # OpenEnv spec metadata
  pyproject.toml            # Dependencies
  inference.py              # Baseline inference script (competition format)
  DESIGN.md                 # Full design doc, motivation, examples
  data/
    financial_data.db       # SQLite datastore (82 companies, 200 tables, 4600+ docs)
    finqa_200.json          # Question index (200 questions, 3 tasks)
    curate_dataset.py       # Script that built the above from FinQA
  server/
    app.py                  # FastAPI entry point
    cost_aware_finqa_environment.py  # Environment (reset/step/state)
    tools.py                # Tool implementations (SQL, vector, web, LLM)
    gradio_ui.py            # Interactive demo UI
    Dockerfile              # Container build
```

## Quick Start

```bash
uv venv .venv && uv sync
uvicorn server.app:app --port 8000
# Visit http://localhost:8000 (redirects to Gradio UI)
```

Docker:

```bash
docker build -t cost-aware-finqa -f server/Dockerfile .
docker run -p 8000:8000 cost-aware-finqa
```

## API

**POST /reset** — Start a new episode. Returns question, table schema, budget.

**POST /step** — Execute a tool:

```json
{"action": {"tool": "sql_query", "query": "SELECT * FROM table_catalog LIMIT 5", "answer": ""}}
```

**Tools**: `sql_query` (free), `vector_search` ($0.50), `web_search` ($3.00), `upgrade_llm` ($3.00), `submit_answer` (free)

**GET /state** — Current episode state.

**WS /ws** — WebSocket for persistent sessions (used by inference.py via EnvClient).

## Datastore

SQLite database with:

- `table_catalog` — index of all financial tables (use this to discover data)
- `financials_<company>_<n>` — company financial data tables
- `documents` — SEC filing text passages (company, year, section, content)
- `questions` — question metadata (id, question, category, difficulty, task)

82 companies from S&P 500. Data sourced from FinQA (EMNLP 2021).

## Tasks

| Task                     | Questions | Budget | Difficulty             |
| ------------------------ | --------- | ------ | ---------------------- |
| `basic_retrieval`      | 70        | $10    | Easy — mostly SQL     |
| `analytical_reasoning` | 70        | $15    | Medium — SQL + vector |
| `strategic_research`   | 60        | $12    | Hard — all tools      |

Set via `FINQA_TASK` env var.

## Scoring

```
score = correctness * max(0.1, 1 - cost/budget) * (1 - error_penalty)
```

- Correctness: fuzzy numerical match (1% = exact, 5% = close, 20% = rough)
- Step rewards: +0.05 valid SQL, +0.05 vector hit, -0.15 bad SQL, -0.05 redundant call
- Score range: [0.0, 1.0]

## Inference

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=your_token \
python inference.py
```

Outputs `[START]`/`[STEP]`/`[END]` format per competition spec.

## Env Vars

| Variable                 | Required | Default                              |
| ------------------------ | -------- | ------------------------------------ |
| `API_BASE_URL`         | Yes      | `https://router.huggingface.co/v1` |
| `MODEL_NAME`           | Yes      | `Qwen/Qwen2.5-72B-Instruct`        |
| `HF_TOKEN`             | Yes      | —                                   |
| `SERPER_API_KEY`       | No       | Falls back to simulated search       |
| `FINQA_TASK`           | No       | `basic_retrieval`                  |

## Validation

```bash
openenv validate                    # Structure check
openenv validate --url http://localhost:8000  # Running server check
```
