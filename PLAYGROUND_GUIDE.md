# Cost-Aware FinQA — Playground Guide

Use this Playground to interact directly with the environment's API (Reset, Step, State).

## How to Use

1. **Click Reset** to start a new episode. This gives you a financial question and budget.
2. **Fill in Tool, Query, and Answer** fields, then click **Step** to execute an action.
3. **Click Get State** to see the current episode state at any time.

## Tool Reference

| Tool | Cost | What to Enter |
|------|------|---------------|
| `sql_query` | $0.001 | Enter `sql_query` in **Tool**, your SQL in **Query**, leave **Answer** blank |
| `web_search` | $0.020 | Enter `web_search` in **Tool**, your search terms in **Query**, leave **Answer** blank |
| `upgrade_llm` | $1.000 | Enter `upgrade_llm` in **Tool**, your reasoning question in **Query**, leave **Answer** blank |
| `submit_answer` | FREE | Enter `submit_answer` in **Tool**, leave **Query** blank, put your answer in **Answer** |

## Example Walkthrough

### Step 1 — Discover tables (cheap SQL)
- **Tool:** `sql_query`
- **Query:** `SELECT * FROM table_catalog LIMIT 10`
- **Answer:** *(leave blank)*

### Step 2 — Query specific data
- **Tool:** `sql_query`
- **Query:** `SELECT * FROM financials_intc_0 LIMIT 5`
- **Answer:** *(leave blank)*

### Step 3 — Submit your answer
- **Tool:** `submit_answer`
- **Query:** *(leave blank)*
- **Answer:** `53.23`

## Scoring

```
score = correctness x max(0.1, 1 - cost/budget) x (1 - error_penalty)
```

- Correctness: ≤1% error = 1.0, ≤5% = 0.6, >5% = 0.0
- Cost efficiency: lower spending = higher score
- Error penalty: accumulated from bad SQL (-0.15), redundant calls (-0.05), etc.
