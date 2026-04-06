# Cost-Aware FinQA Research: Why This Environment Matters

## The Idea

Financial AI agents (in production) today have access to multiple tools — web search, database queries, APIs, stronger models — but they treat them all as interchangeable. A question like "What was Apple's revenue last year?" gets routed through a $3 web search when a free SQL query on the local database returns the same answer instantly.

At scale, this adds up fast. A research session can burn $50+ in unnecessary API calls. The missing piece: agents don't learn *when a cheaper tool is sufficient*.

This environment fixes that. The agent gets a financial question, a budget, and 4 tools at different price points. It has to learn which tool fits which question type — and the scoring formula rewards efficiency alongside correctness.

## How It Works

The agent gets a financial question (from real SEC filings — 82 S&P 500 companies) and a budget. It has 4 tools:

- **SQL Query** — FREE. The financial data is sitting right there in a SQLite database. Income statements, balance sheets, cash flow data. If the answer is a number in a table, this is the tool to use.
- **Vector Search** — $0.50 per call. Searches through 4,600+ text passages from 10-K filings. Good for "why did margins decline?" type questions where you need the narrative, not just numbers.
- **Web Search** — $3.00 per call. Hits the Serper API for external data. Industry benchmarks, market averages, peer comparisons. Expensive but sometimes you genuinely need data that isn't in the company's filing.
- **Upgrade LLM** — $3.00 per call. Switches to a stronger model for complex multi-step calculations (3-year CAGR, compound ratios). The base model often gets the arithmetic wrong on these.

The scoring formula is: **correctness x cost_efficiency x (1 - error_penalty)**

So a perfect answer using only free tools gets ~1.0. A perfect answer that burned the whole budget gets ~0.1. And if you wrote bad SQL queries along the way, that error penalty eats into your score too.

## Why Each Tool Actually Matters

Each Tool has a varying amount of cost/compute that the agent has to account for when deciding its search strategy:

**SQL is free but you have to write correct queries.** The data is real financial tables with messy column names like `percent_of_totaloperating_expenses`. Write wrong SQL and you get -0.15 per bad query. The agent has to learn the schema.

**Vector search finds things SQL can't.** "What factors drove the decline in operating income?" — that answer lives in the MD&A section of a 10-K, not in a table column. The agent needs to recognize when a question is about narrative context vs. numerical data.

**Web search is expensive but sometimes necessary.** When a question asks "How does this company's debt ratio compare to the industry average?" — that industry average isn't in any company's filing. You need external data. But 80% of questions DON'T need this. The agent has to learn which 20% do.

**LLM upgrade catches arithmetic errors.** Multi-step financial calculations — compound growth rates, chained ratios — are where base models fail. The upgrade gives you better accuracy but costs $3. Only worth it for the hard questions.

## The Reward Signal

I wanted dense rewards, not just a binary right/wrong at the end. At every step:

- Valid SQL that returns data: +0.05
- Relevant vector search results: +0.05
- Bad SQL (syntax error or empty): -0.15
- Redundant tool call: -0.05
- Web search data: +0.02

Then the final answer gets graded on a 0-1 scale (fuzzy numerical matching — within 1% is perfect, within 5% is good, etc.) and multiplied by cost efficiency.

This means the agent gets signal throughout the episode. It can learn "oh, that SQL query structure works" or "vector search was wasteful for this type of question" from intermediate rewards, not just from whether the final answer was right.

## Three Difficulty Levels

- **basic_retrieval** (70 questions, $10 budget): Most answers are in the tables. SQL-heavy. An agent that learns "just use SQL" for these will score well.
- **analytical_reasoning** (70 questions, $15 budget): Mix of structured and unstructured data. Need to combine SQL for numbers + vector search for context.
- **strategic_research** (60 questions, $12 budget): Tight budget, complex questions. Some genuinely need web search or LLM upgrade. The agent has to prioritize.

## The Dataset

200 questions curated from FinQA (EMNLP 2021), sourced from real SEC filings of 82 S&P 500 companies. Each question has:

- A gold answer (195 numerical, 5 text)
- A calculation program showing the reasoning steps
- Associated financial tables loaded into SQLite
- Context paragraphs from the actual 10-K filing

The datastore is small (2.1 MB) — deliberately kept lean so it runs comfortably on HF Spaces (2 vCPU, 8 GB RAM, under 20 min inference).

## What Makes This Different

Most financial QA benchmarks treat tool selection as an afterthought. They test whether the model can answer the question, period. This environment tests whether it can answer the question *efficiently*. That's the real-world skill that matters when you're deploying these systems at scale.

The cost pressure creates a genuine optimization problem. The agent can't just throw everything at every question — it has to develop intuitions about question types and tool utility. That's a learned behavior, not a prompt-engineered one, which makes it a good fit for RL training.

## A Note on Training Approaches

I'm keeping the environment fully interpretable with programmatic grading, as the competition rules require. But I want to flag where I think this could go next: LLM-as-a-judge within a GRPO-like training loop, similar to what OpenPipe does with RULER. The idea is that the judge model uses its own outputs as training data to explore diverse tool-selection strategies. The judge evaluates not just answer correctness but *process quality* — did the agent use the cheapest sufficient tool? Did it avoid unnecessary calls?

I'm not implementing that here because the rules require an interpretable environment, but it's the natural next step for training cost-aware agents at scale.

## Sample Interactions

### Example 1: SQL is the smart choice

**Question:** "What percentage of total cash and investments was available-for-sale investments in Dec 2012?"
**Company:** INTC (Intel)

| Strategy | Tools Used                  | Cost | Correctness | Final Score |
| -------- | --------------------------- | ---- | ----------- | ----------- |
| Smart    | sql_query -> submit_answer  | $0   | 1.0         | ~0.95       |
| Wasteful | web_search -> submit_answer | $3   | 1.0         | ~0.65       |

The data is right there in the table: 14,001 / 26,302 = 0.532. No need to pay for a web search.

### Example 2: Vector search earns its cost

**Question:** "What factors contributed to the change in operating expenses?"
**Company:** ETR (Entergy)

| Strategy   | Tools Used                     | Cost  | Correctness | Final Score |
| ---------- | ------------------------------ | ----- | ----------- | ----------- |
| Smart      | vector_search -> submit_answer | $0.50 | 0.8         | ~0.75       |
| Wrong tool | sql_query -> submit_answer     | $0    | 0.0         | ~0.0        |

The answer lives in the narrative text, not the numbers. SQL returns nothing useful.

### Example 3: Web search is worth the money

**Question:** "How does this debt-to-equity compare to the industry average?"
**Company:** JPM (JPMorgan)

| Strategy        | Tools Used                               | Cost | Correctness | Final Score |
| --------------- | ---------------------------------------- | ---- | ----------- | ----------- |
| Smart           | sql_query -> web_search -> submit_answer | $3   | 0.9         | ~0.60       |
| Cheap but wrong | sql_query -> submit_answer               | $0   | 0.2         | ~0.15       |

You need external industry data that isn't in JPMorgan's filing. Sometimes paying is the right call.
