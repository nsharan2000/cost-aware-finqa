"""
Gradio frontend for the Cost-Aware FinQA Environment.

Presentation-ready UI with runnable walkthrough examples
that demonstrate cost-aware tool selection.
"""

import os

import gradio as gr

try:
    from .cost_aware_finqa_environment import CostAwareFinqaEnvironment, TASK_CONFIG
except (ImportError, SystemError):
    from server.cost_aware_finqa_environment import CostAwareFinqaEnvironment, TASK_CONFIG

try:
    from ..models import CostAwareFinqaAction
except (ImportError, SystemError):
    from models import CostAwareFinqaAction


def _load_design_doc():
    """Load DESIGN.md content."""
    paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DESIGN.md"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DESIGN.md"),
        "DESIGN.md",
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return f.read()
    return "DESIGN.md not found."


# Pre-configured walkthrough examples with actual SQL queries that work against the datastore
WALKTHROUGH_EXAMPLES = [
    {
        "id": "intc_smart",
        "title": "1. SQL is free and sufficient (INTC)",
        "subtitle": "Score: 1.000 vs 0.700 — saving $3 by using the right tool",
        "question_match": "percentage of total cash",
        "task": "basic_retrieval",
        "description": (
            "**Question:** What percentage of total cash and investments as of Dec 29, 2012 "
            "was comprised of available-for-sale investments? (Intel)\n\n"
            "The data is right there in the SQL table. No need to pay for a web search."
        ),
        "strategies": [
            {
                "name": "Smart: SQL only",
                "steps": [
                    {"tool": "sql_query", "query": "SELECT in_millions, dec_292012 FROM financials_intc_0", "answer": ""},
                    {"tool": "submit_answer", "query": "", "answer": "0.53232"},
                ],
                "expected": "Score ~1.000 | Cost: $0 | 14,001 / 26,302 = 0.532",
            },
            {
                "name": "Wasteful: Web search",
                "steps": [
                    {"tool": "web_search", "query": "Intel available-for-sale investments percentage 2012", "answer": ""},
                    {"tool": "submit_answer", "query": "", "answer": "0.53232"},
                ],
                "expected": "Score ~0.700 | Cost: $3 | Same answer, 30% lower score",
            },
        ],
    },
    {
        "id": "etr_growth",
        "title": "2. Don't overspend on easy math (ETR)",
        "subtitle": "Score: 1.000 vs 0.800 — LLM upgrade wastes $3 on simple arithmetic",
        "question_match": "growth rate in net revenue",
        "task": "analytical_reasoning",
        "description": (
            "**Question:** What is the growth rate in net revenue in 2008? (Entergy)\n\n"
            "This is a two-step calculation: (959.2 - 991.1) / 991.1 = -0.032. "
            "Simple enough for SQL + mental math. The $3 LLM upgrade is overkill."
        ),
        "strategies": [
            {
                "name": "Smart: SQL only",
                "steps": [
                    {"tool": "sql_query", "query": "SELECT metric, amount_in_millions FROM financials_etr_3 WHERE metric LIKE '%net revenue%'", "answer": ""},
                    {"tool": "submit_answer", "query": "", "answer": "-0.03219"},
                ],
                "expected": "Score ~1.000 | Cost: $0 | Data right in the table",
            },
            {
                "name": "Wasteful: SQL + LLM upgrade",
                "steps": [
                    {"tool": "sql_query", "query": "SELECT metric, amount_in_millions FROM financials_etr_3 WHERE metric LIKE '%net revenue%'", "answer": ""},
                    {"tool": "upgrade_llm", "query": "Calculate growth rate: (959.2 - 991.1) / 991.1", "answer": ""},
                    {"tool": "submit_answer", "query": "", "answer": "-0.03219"},
                ],
                "expected": "Score ~0.800 | Cost: $3 | Same answer, 20% lower score",
            },
        ],
    },
    {
        "id": "intc_bad_sql",
        "title": "3. Bad SQL queries destroy your score (INTC)",
        "subtitle": "Score: 1.000 vs 0.700 — error penalties from failed queries",
        "question_match": "percentage of total cash",
        "task": "basic_retrieval",
        "description": (
            "**Question:** Same INTC question as Example 1, but with bad SQL attempts first.\n\n"
            "Each failed SQL query adds a -0.15 penalty. Two bad queries = 0.30 error penalty. "
            "Even with the correct final answer, your score drops from 1.0 to 0.7."
        ),
        "strategies": [
            {
                "name": "Clean: Correct SQL on first try",
                "steps": [
                    {"tool": "sql_query", "query": "SELECT in_millions, dec_292012 FROM financials_intc_0", "answer": ""},
                    {"tool": "submit_answer", "query": "", "answer": "0.53232"},
                ],
                "expected": "Score ~1.000 | No penalties",
            },
            {
                "name": "Sloppy: Two failed queries first",
                "steps": [
                    {"tool": "sql_query", "query": "SELECT * FROM intc_financials", "answer": ""},
                    {"tool": "sql_query", "query": "SELECT revenue FROM financials_intc_0", "answer": ""},
                    {"tool": "sql_query", "query": "SELECT in_millions, dec_292012 FROM financials_intc_0", "answer": ""},
                    {"tool": "submit_answer", "query": "", "answer": "0.53232"},
                ],
                "expected": "Score ~0.700 | Two -0.15 penalties = 0.30 error penalty",
            },
        ],
    },
    {
        "id": "pnc_sum",
        "title": "4. Explore the schema, then query (PNC)",
        "subtitle": "Score: 1.000 — using table_catalog to discover the right table",
        "question_match": "total of home equity",
        "task": "basic_retrieval",
        "description": (
            "**Question:** In millions, what is the total of home equity lines of credit? (PNC)\n\n"
            "A smart agent starts with `SELECT * FROM table_catalog` to discover available tables, "
            "then queries the right one. This mirrors how a real analyst would work."
        ),
        "strategies": [
            {
                "name": "Smart: Discover schema, then query",
                "steps": [
                    {"tool": "sql_query", "query": "SELECT * FROM table_catalog WHERE company = 'PNC'", "answer": ""},
                    {"tool": "sql_query", "query": "SELECT * FROM financials_pnc_0", "answer": ""},
                    {"tool": "submit_answer", "query": "", "answer": "22929"},
                ],
                "expected": "Score ~1.000 | Cost: $0 | 15,553 + 7,376 = 22,929",
            },
        ],
    },
    {
        "id": "aal_web",
        "title": "5. When web search is justified (AAL)",
        "subtitle": "Industry comparison needs external data — but the core answer is in SQL",
        "question_match": "labor-related deemed claim",
        "task": "basic_retrieval",
        "description": (
            "**Question:** What was the percent of labor-related deemed claim to total reorganization costs? "
            "How does this compare to the market capitalization benchmark? (American Airlines)\n\n"
            "The ratio (1733/2655 = 0.653) is in SQL. The benchmark comparison part needs web search. "
            "But the gold answer is the ratio — so SQL alone scores perfectly here."
        ),
        "strategies": [
            {
                "name": "Efficient: SQL is enough for the core answer",
                "steps": [
                    {"tool": "sql_query", "query": "SELECT metric, col_2013 FROM financials_aal_5", "answer": ""},
                    {"tool": "submit_answer", "query": "", "answer": "0.65273"},
                ],
                "expected": "Score ~1.000 | Cost: $0 | 1,733 / 2,655 = 0.653",
            },
            {
                "name": "Cautious: SQL + web search for benchmark context",
                "steps": [
                    {"tool": "sql_query", "query": "SELECT metric, col_2013 FROM financials_aal_5", "answer": ""},
                    {"tool": "web_search", "query": "airline reorganization costs industry benchmark", "answer": ""},
                    {"tool": "submit_answer", "query": "", "answer": "0.65273"},
                ],
                "expected": "Score ~0.700 | Cost: $3 | Benchmark context but lower score",
            },
        ],
    },
]


def create_gradio_app():
    """Create the Gradio interface."""

    env = CostAwareFinqaEnvironment()
    current_obs = {"obs": None}
    tool_log = {"entries": []}

    def format_tool_log(entries):
        if not entries:
            return "No tool calls yet. Reset the environment to start."
        lines = []
        for i, e in enumerate(entries, 1):
            cost_str = f"${e['cost']:.2f}" if e['cost'] > 0 else "FREE"
            reward_str = f"{e['reward']:+.3f}" if e['reward'] != 0 else "0.000"
            lines.append(
                f"**Step {i}** | Tool: `{e['tool']}` | Cost: {cost_str} | Reward: {reward_str}\n"
                f"Query: {e['query'][:120]}\n"
                f"Result: {e['result'][:250]}"
            )
            if e.get('error'):
                lines.append(f"Error: {e['error']}")
            lines.append("---")
        return "\n".join(lines)

    def run_walkthrough(example_idx, strategy_idx):
        """Run a pre-configured walkthrough example end-to-end."""
        if example_idx is None or strategy_idx is None:
            return "Select an example and strategy to run."

        example_idx = int(example_idx)
        strategy_idx = int(strategy_idx)

        if example_idx >= len(WALKTHROUGH_EXAMPLES):
            return "Invalid example."

        ex = WALKTHROUGH_EXAMPLES[example_idx]
        if strategy_idx >= len(ex["strategies"]):
            return "Invalid strategy."

        strategy = ex["strategies"][strategy_idx]
        os.environ['FINQA_TASK'] = ex["task"]

        run_env = CostAwareFinqaEnvironment()
        obs = run_env.reset()

        # Find the target question
        attempts = 0
        while ex["question_match"].lower() not in obs.question.lower() and attempts < 250:
            obs = run_env.reset()
            attempts += 1

        if attempts >= 250:
            return f"Could not find question matching '{ex['question_match']}'. Try resetting."

        lines = []
        lines.append(f"## {ex['title']}")
        lines.append(f"**Strategy: {strategy['name']}**\n")
        lines.append(f"**Question:** {obs.question}\n")
        lines.append(f"**Budget:** ${obs.budget_total:.2f} | **Task:** {ex['task']}\n")
        lines.append("---\n")

        for i, step in enumerate(strategy["steps"], 1):
            action = CostAwareFinqaAction(
                tool=step["tool"],
                query=step["query"],
                answer=step["answer"],
            )
            obs = run_env.step(action)

            cost_str = f"${obs.tool_cost:.2f}" if obs.tool_cost > 0 else "FREE"
            lines.append(f"### Step {i}: `{step['tool']}` ({cost_str})")

            if step["tool"] == "submit_answer":
                lines.append(f"**Submitted answer:** {step['answer']}\n")
            elif step["query"]:
                lines.append(f"```sql\n{step['query']}\n```\n")

            # Show result
            result_preview = obs.tool_result[:500]
            if obs.error:
                lines.append(f"**Error:** {obs.error}\n")
            lines.append(f"```\n{result_preview}\n```\n")

            lines.append(f"Budget remaining: ${obs.budget_remaining:.2f} | "
                         f"Reward: {obs.reward:+.3f}\n")
            lines.append("---\n")

        # Final summary
        if obs.done:
            lines.append(f"## Final Score: **{obs.score:.3f}**\n")
            lines.append(f"Total cost: ${obs.cost_so_far:.2f} / ${obs.budget_total:.2f}")

        return "\n".join(lines)

    def get_strategy_choices(example_idx):
        """Get strategy names for the selected example."""
        if example_idx is None:
            return gr.update(choices=[], value=None)
        idx = int(example_idx)
        if idx >= len(WALKTHROUGH_EXAMPLES):
            return gr.update(choices=[], value=None)
        ex = WALKTHROUGH_EXAMPLES[idx]
        choices = [(s["name"], str(i)) for i, s in enumerate(ex["strategies"])]
        return gr.update(choices=choices, value="0")

    def get_example_description(example_idx):
        """Get description for the selected example."""
        if example_idx is None:
            return ""
        idx = int(example_idx)
        if idx >= len(WALKTHROUGH_EXAMPLES):
            return ""
        ex = WALKTHROUGH_EXAMPLES[idx]
        return f"### {ex['title']}\n{ex['subtitle']}\n\n{ex['description']}"

    def reset_env(task_name):
        env._task_name = task_name
        obs = env.reset()
        current_obs["obs"] = obs
        tool_log["entries"] = []

        status = (
            f"**Task:** {obs.task_name}\n"
            f"**Budget:** ${obs.budget_total:.2f}\n"
            f"**Max Steps:** {obs.max_steps}\n"
            f"**Question:** {obs.question}"
        )

        return (
            status, obs.question, obs.table_schema,
            format_tool_log([]),
            f"${obs.budget_remaining:.2f} / ${obs.budget_total:.2f}",
            "0", "0.000",
        )

    def execute_tool(tool, query, answer):
        if current_obs["obs"] is None:
            return ("Reset the environment first.", "", "", "", "", "", "")

        action = CostAwareFinqaAction(tool=tool, query=query, answer=answer)
        obs = env.step(action)
        current_obs["obs"] = obs

        entry = {
            "tool": tool,
            "query": query if tool != "submit_answer" else f"Answer: {answer}",
            "result": obs.tool_result,
            "cost": obs.tool_cost,
            "reward": obs.reward,
            "error": obs.error,
        }
        tool_log["entries"].append(entry)

        status = ""
        if obs.done:
            status = f"**EPISODE COMPLETE** | Final Score: **{obs.score:.3f}**\n\n"
            if obs.score >= 0.7:
                status += "Great result - correct and cost-efficient!"
            elif obs.score >= 0.3:
                status += "Decent - room to improve on cost or accuracy."
            else:
                status += "Low score - try different tools or fix SQL queries."

        return (
            status if status else f"Step {obs.step_number}/{obs.max_steps}",
            obs.question, obs.tool_result[:500],
            format_tool_log(tool_log["entries"]),
            f"${obs.budget_remaining:.2f} / ${obs.budget_total:.2f}",
            str(obs.step_number), f"{obs.score:.3f}",
        )

    with gr.Blocks(title="Cost-Aware FinQA") as demo:

        gr.Markdown(
            "# Cost-Aware FinQA Environment\n"
            "An RL environment where agents learn to answer financial questions using the cheapest sufficient tool.\n"
            "**Tools:** `sql_query` (FREE) | `vector_search` ($0.50) | "
            "`web_search` ($3.00) | `upgrade_llm` ($3.00) | `submit_answer` (FREE)\n\n"
            "**Scoring:** `correctness x cost_efficiency x (1 - error_penalty)` — "
            "a correct answer with $0 cost scores 1.0; the same answer after spending $3 scores 0.7."
        )

        with gr.Tabs():
            # ===================== TAB 1: WALKTHROUGH EXAMPLES =====================
            with gr.Tab("Walkthrough Examples"):
                gr.Markdown(
                    "### Run pre-configured examples to see how tool choice affects scores\n"
                    "Each example shows the same question answered with different strategies. "
                    "Select an example, pick a strategy, and click **Run** to see the full execution trace."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        example_dropdown = gr.Dropdown(
                            choices=[(ex["title"], str(i)) for i, ex in enumerate(WALKTHROUGH_EXAMPLES)],
                            value="0",
                            label="Select Example",
                        )
                        strategy_dropdown = gr.Dropdown(
                            choices=[(s["name"], str(i)) for i, s in enumerate(WALKTHROUGH_EXAMPLES[0]["strategies"])],
                            value="0",
                            label="Select Strategy",
                        )
                        run_btn = gr.Button("Run Walkthrough", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        example_desc = gr.Markdown(
                            value=f"### {WALKTHROUGH_EXAMPLES[0]['title']}\n"
                                  f"{WALKTHROUGH_EXAMPLES[0]['subtitle']}\n\n"
                                  f"{WALKTHROUGH_EXAMPLES[0]['description']}"
                        )

                walkthrough_output = gr.Markdown(
                    value="Click **Run Walkthrough** to execute the selected example.",
                    label="Execution Trace",
                )

                # Wire example selection
                example_dropdown.change(
                    fn=get_strategy_choices,
                    inputs=[example_dropdown],
                    outputs=[strategy_dropdown],
                )
                example_dropdown.change(
                    fn=get_example_description,
                    inputs=[example_dropdown],
                    outputs=[example_desc],
                )
                run_btn.click(
                    fn=run_walkthrough,
                    inputs=[example_dropdown, strategy_dropdown],
                    outputs=[walkthrough_output],
                )

                gr.Markdown(
                    "---\n"
                    "### Summary of Cost-Accuracy Trade-offs\n\n"
                    "| Example | Smart Strategy | Score | Wasteful Strategy | Score | Lesson |\n"
                    "|---------|---------------|-------|-------------------|-------|--------|\n"
                    "| INTC: % of cash | SQL only | **1.000** | Web search | 0.700 | Free data in DB, don't pay for it |\n"
                    "| ETR: Growth rate | SQL only | **1.000** | SQL + LLM upgrade | 0.800 | Simple math doesn't need $3 LLM |\n"
                    "| INTC: Bad SQL | Clean SQL | **1.000** | 2 failed + 1 good | 0.700 | Error penalties stack up fast |\n"
                    "| PNC: Home equity | Schema discovery + SQL | **1.000** | — | — | Use table_catalog to find tables |\n"
                    "| AAL: Reorg costs | SQL only | **1.000** | SQL + web search | 0.700 | Core answer is in SQL |\n"
                )

            # ===================== TAB 2: INTERACTIVE PLAYGROUND =====================
            with gr.Tab("Interactive Playground"):
                gr.Markdown(
                    "### Try it yourself\n"
                    "Reset the environment, explore the schema, query data, and submit answers. "
                    "Start with `SELECT * FROM table_catalog` to discover available tables."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        task_dropdown = gr.Dropdown(
                            choices=list(TASK_CONFIG.keys()),
                            value="basic_retrieval",
                            label="Task",
                        )
                        reset_btn = gr.Button("Reset Environment", variant="primary")
                        budget_display = gr.Textbox(label="Budget", interactive=False)
                        step_display = gr.Textbox(label="Step", interactive=False)
                        score_display = gr.Textbox(label="Score", interactive=False)

                    with gr.Column(scale=2):
                        status_display = gr.Markdown(label="Status")
                        question_display = gr.Textbox(label="Current Question", lines=3, interactive=False)
                        schema_display = gr.Textbox(label="Table Schema / Tool Result", lines=6, interactive=False)

                gr.Markdown("### Execute Tool")
                with gr.Row():
                    tool_dropdown = gr.Dropdown(
                        choices=["sql_query", "vector_search", "web_search", "upgrade_llm", "submit_answer"],
                        value="sql_query",
                        label="Tool", scale=1,
                    )
                    query_input = gr.Textbox(
                        label="Query",
                        placeholder='SELECT * FROM table_catalog LIMIT 5',
                        scale=3,
                    )
                    answer_input = gr.Textbox(
                        label="Answer (for submit_answer)",
                        placeholder="e.g., 0.532",
                        scale=1,
                    )
                execute_btn = gr.Button("Execute Tool", variant="secondary")

                gr.Markdown("### Tool Call History")
                tool_log_display = gr.Markdown(
                    value="No tool calls yet. Reset the environment to start.",
                )

                # Wire events
                reset_btn.click(
                    fn=reset_env, inputs=[task_dropdown],
                    outputs=[status_display, question_display, schema_display,
                             tool_log_display, budget_display, step_display, score_display],
                )
                execute_btn.click(
                    fn=execute_tool, inputs=[tool_dropdown, query_input, answer_input],
                    outputs=[status_display, question_display, schema_display,
                             tool_log_display, budget_display, step_display, score_display],
                )

            # ===================== TAB 3: DESIGN DOC =====================
            with gr.Tab("Design & Motivation"):
                design_content = _load_design_doc()
                gr.Markdown(design_content)

    return demo


def mount_gradio(app):
    """Mount Gradio app onto FastAPI. API routes take precedence."""
    demo = create_gradio_app()
    gr.mount_gradio_app(app, demo, path="/")
    return demo
