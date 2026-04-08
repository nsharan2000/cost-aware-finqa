# Cost-Aware FinQA Deep Research Agent: Why This Environment Matters

## The Principle

When it comes to Deep Research Agents: AI is capable of performing massive-scale deep research (involving compiling information from multiple public pages and custom databases). Given enough time, current agents can find the answer to a query, but the key limiting factor is **cost**. The same answer can be obtained through multiple tools (more easily with costlier tools and with greater difficulty using custom ones), while some answers can only be reached through the right tool. This is especially true for market analysis in the fintech space, where even the custom data sources are highly complex (deeply nested tables, dense schemas) and detailed. For better accuracy and cost efficiency across large-traffic deep research agents over custom data sources, the AI needs to learn to discover the most cost-effective path to a solution.

Hence we built an **RLVE environment using OpenEnv** that can improve the model without affecting its overall knowledge and general-purpose ability, to arrive at a solution in the cheapest way possible and teach it to construct better SQL queries specific to the database (i.e., minimizing the number of invalid SQL calls, more fruitful search query strings).

## The Problem

Financial AI agents in production today have access to multiple tools — web search, database queries, APIs, stronger models — but they treat them all as interchangeable. A question like "What was Apple's revenue last year?" gets routed through a $0.02 web search when a $0.001 SQL query on the local database returns the same answer instantly.

At scale, this adds up fast. A research session can burn $50+ in unnecessary API calls. The missing piece: agents don't learn *when a cheaper tool is sufficient*.

This environment fixes that. The agent gets a financial question, a budget, and 3 tools at different price points. It has to learn which tool fits which question type — and the scoring formula rewards efficiency alongside correctness.

## How It Works

The agent gets a financial question (from real SEC filings — 82 S&P 500 companies) and a budget. It has 3 tools:

- **SQL Query** — $0.001 per call. The financial data is sitting right there in a SQLite database. Income statements, balance sheets, cash flow data. If the answer is a number in a table, this is the tool to use. Extremely cheap but the agent must write correct queries — bad SQL is penalized.
- **Web Search** — $0.02 per call. Hits the Serper API for external data. Industry benchmarks, market averages, peer comparisons. 20x the cost of SQL but sometimes you genuinely need data that isn't in the company's filing.
- **Upgrade LLM** — $1.00 per call. Switches to a stronger model for complex multi-step calculations (3-year CAGR, compound ratios). The base model often gets the arithmetic wrong on these. **1000x the cost of SQL — absolute last resort only.** Carries a -0.10 reward penalty on top of the dollar cost.

The scoring formula is: **correctness x cost_efficiency x (1 - error_penalty)**

So a correct answer using minimal SQL gets a high score. A correct answer that burned the whole budget gets ~0.1. And if you wrote bad SQL queries along the way, that error penalty eats into your score too.

## Why Each Tool Actually Matters

Each tool has a varying cost/compute that the agent must account for when deciding its search strategy:

**SQL is cheap but you have to write correct queries.** The data is real financial tables with messy column names like `percent_of_totaloperating_expenses`. Write wrong SQL and you get -0.15 per bad query, plus each call costs $0.001. The agent has to learn the schema and write efficient queries — not spam SQL calls.

**Web search is moderately expensive but sometimes necessary.** When a question asks "How does this company's debt ratio compare to the industry average?" — that industry average isn't in any company's filing. You need external data. But 80% of questions DON'T need this. The agent has to learn which 20% do.

**LLM upgrade catches arithmetic errors but is extremely expensive.** Multi-step financial calculations — compound growth rates, chained ratios — are where base models fail. The upgrade gives you better accuracy but costs $1.00 (1000x SQL). Only worth it for the hardest questions where all other approaches have failed. The -0.10 reward penalty ensures the agent learns this is a last resort.

## The Reward Signal

Dense rewards at every step, not just binary right/wrong at the end:

- Valid SQL that returns data: +0.03
- Web search data: +0.02
- Bad SQL (syntax error or empty): -0.15
- SQL call cost penalty: -0.01 per call (discourages excessive queries)
- Upgrade LLM penalty: -0.10 per call (strong disincentive — last resort only)
- Redundant tool call: -0.05

Final answer grading (numerical matching):
- Within 1% error: 1.0 (exact)
- Within 5% error: 0.6 (close)
- Over 5% error: 0.0 (wrong)

The final score is multiplied by cost efficiency, so the agent gets signal throughout the episode. It can learn "that SQL query structure works" or "web search was wasteful for this type of question" from intermediate rewards.

## Three Difficulty Levels

- **basic_retrieval** (70 questions, $10 budget): Most answers are in the tables. SQL-heavy. An agent that learns "just use SQL" for these will score well.
- **analytical_reasoning** (70 questions, $15 budget): Mix of structured data and calculations. Need SQL for numbers + possibly web search for context.
- **strategic_research** (60 questions, $12 budget): Tight budget, complex questions. Some genuinely need web search or LLM upgrade. The agent has to prioritize.

## The Dataset

**Table 1 (Questions):** 200 questions curated from FinQA (EMNLP 2021), each with a gold answer and calculation program. These are the queries the agent must answer.

**Table 2 (Financial Data):** A subset of real SEC filing data from 82 S&P 500 companies loaded into a SQLite database. This is the sample database the environment uses for SQL queries. It contains:
- `table_catalog` — index of all financial tables
- `financials_<company>_<n>` — company financial data tables (200 tables total)
- `questions` — question metadata

**Note:** This environment runs on a subset of the FinQA dataset exported from [snorkelai/finqa-data](https://huggingface.co/datasets/snorkelai/finqa-data) (EMNLP 2021). If you wish to run it with a different or larger database, you should:
1. Replace the contents in `/data/financial_data.db` with your target database
2. Modify `/server/tools.py` to match your schema
3. Update the questions in `/data/finqa_200.json` to match your data

## What Makes This Different

Most financial QA benchmarks treat tool selection as an afterthought. They test whether the model can answer the question, period. This environment tests whether it can answer the question *efficiently*. That's the real-world skill that matters when you're deploying deep research agents at scale.

The cost pressure creates a genuine optimization problem. The agent can't just throw everything at every question — it has to develop intuitions about question types and tool utility. That's a learned behavior, not a prompt-engineered one, which makes it a good fit for RL training.

## Training Approach: GRPO with Safeguards

We use **Group Relative Policy Optimization (GRPO)** as the primary training methodology, implemented via Unsloth for efficient 4-bit LoRA training.

Key design decisions to preserve general-purpose ability:
- **Low LoRA rank (r=16)** with small alpha — the model learns cost-aware tool selection without overwriting its general knowledge
- **Reward shaping focuses on process, not memorization** — the model learns *when* to use SQL vs. web search, not specific answers
- **LLM-as-judge evaluation** — uses a separate model to evaluate process quality alongside programmatic grading

The trained agent learns to prefer cheap SQL queries over expensive tools, improving scores while reducing costs — without losing its ability to handle general tasks.

## Sample Interactions

### Example 1: SQL is the smart choice

**Question:** "What percentage of total cash and investments was available-for-sale investments in Dec 2012?"
**Company:** INTC (Intel)

| Strategy | Tools Used | Cost | Correctness | Final Score |
| -------- | --------- | ---- | ----------- | ----------- |
| Smart | sql_query -> submit_answer | $0.001 | 1.0 | ~0.99 |
| Wasteful | web_search -> submit_answer | $0.02 | 1.0 | ~0.98 |
| Overkill | upgrade_llm -> submit_answer | $1.00 | 1.0 | ~0.90 |

The data is right there in the table: 14,001 / 26,302 = 0.532. No need to pay for anything more than SQL.

### Example 2: Don't overspend on easy math

**Question:** "What is the growth rate in net revenue in 2008?" (Entergy)

| Strategy | Tools Used | Cost | Correctness | Final Score |
| -------- | --------- | ---- | ----------- | ----------- |
| Smart | sql_query -> submit_answer | $0.001 | 1.0 | ~0.99 |
| Wasteful | sql_query + upgrade_llm -> submit_answer | $1.001 | 1.0 | ~0.90 |

Simple subtraction doesn't need a $1.00 LLM upgrade.

### Example 3: Web search is worth the money

**Question:** "How does this debt-to-equity compare to the industry average?" (JPMorgan)

| Strategy | Tools Used | Cost | Correctness | Final Score |
| -------- | --------- | ---- | ----------- | ----------- |
| Smart | sql_query -> web_search -> submit_answer | $0.021 | 0.9 | ~0.85 |
| Cheap but wrong | sql_query -> submit_answer | $0.001 | 0.2 | ~0.15 |

You need external industry data that isn't in JPMorgan's filing. Sometimes paying is the right call.
