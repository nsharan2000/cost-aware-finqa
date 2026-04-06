"""
Gradio frontend for the Cost-Aware FinQA Environment.

Clean, minimal UI. Shows tool-use decisions clearly for evaluators.
Includes a Design tab with the full project rationale.
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


SAMPLE_EXAMPLES = [
    {
        "title": "Example 1: SQL is the smart choice ($0 vs $3)",
        "description": (
            "**Q:** 'What percentage of total cash was available-for-sale investments in Dec 2012?' (INTC)\n\n"
            "| Strategy | Tools | Cost | Score |\n"
            "|----------|-------|------|-------|\n"
            "| Smart | sql_query -> submit | $0 | ~0.95 |\n"
            "| Wasteful | web_search -> submit | $3 | ~0.65 |\n\n"
            "The numbers (14,001 and 26,302) are right in the SQL table. Paying $3 for a web search gives the same answer but kills your score."
        ),
    },
    {
        "title": "Example 2: Vector search earns its $0.50",
        "description": (
            "**Q:** 'What factors drove the decrease in the valuation allowance?' (STT)\n\n"
            "| Strategy | Tools | Cost | Score |\n"
            "|----------|-------|------|-------|\n"
            "| Smart | vector_search -> submit | $0.50 | ~0.75 |\n"
            "| Wrong tool | sql_query -> submit | $0 | ~0.0 |\n\n"
            "SQL can't find narrative explanations. Vector search pulls relevant paragraphs from the 10-K filing."
        ),
    },
    {
        "title": "Example 3: Web search is worth $3 here",
        "description": (
            "**Q:** 'How does this debt ratio compare to the industry average?' (JPM)\n\n"
            "| Strategy | Tools | Cost | Score |\n"
            "|----------|-------|------|-------|\n"
            "| Smart | sql_query + web_search -> submit | $3 | ~0.60 |\n"
            "| Cheap but wrong | sql_query -> submit | $0 | ~0.15 |\n\n"
            "Industry benchmarks aren't in company filings. You need external data, and $3 is worth it for a correct answer."
        ),
    },
    {
        "title": "Example 4: LLM upgrade catches arithmetic errors ($3)",
        "description": (
            "**Q:** 'What is the fluctuation of the credit spread in 2008 and 2009, in basis points?' (JPM)\n"
            "**Program:** divide(39, 37), subtract(#0, 1), multiply(#1, 100)\n\n"
            "| Strategy | Tools | Cost | Score |\n"
            "|----------|-------|------|-------|\n"
            "| Smart | sql_query + upgrade_llm -> submit | $3 | ~0.65 |\n"
            "| Risky | sql_query -> submit | $0 | ~0.20 |\n\n"
            "Multi-step arithmetic. Base model often gets the chained division/multiplication wrong."
        ),
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

        with gr.Tabs():
            # ===================== TAB 1: INTERACTIVE DEMO =====================
            with gr.Tab("Interactive Demo"):
                gr.Markdown(
                    "# Cost-Aware FinQA\n"
                    "Answer financial questions by choosing tools strategically. "
                    "Each tool has a different cost. Maximize correctness, minimize spending.\n\n"
                    "**Tools:** `sql_query` (FREE) | `vector_search` ($0.50) | "
                    "`web_search` ($3.00) | `upgrade_llm` ($3.00) | `submit_answer` (FREE)"
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

            # ===================== TAB 2: EXAMPLES =====================
            with gr.Tab("Tool Trade-off Examples"):
                gr.Markdown(
                    "# When to Use Each Tool\n"
                    "These examples show the cost-accuracy trade-offs the agent needs to learn."
                )
                for ex in SAMPLE_EXAMPLES:
                    with gr.Accordion(ex["title"], open=True):
                        gr.Markdown(ex["description"])

            # ===================== TAB 3: DESIGN DOC =====================
            with gr.Tab("Design & Motivation"):
                design_content = _load_design_doc()
                gr.Markdown(design_content)

    return demo


def mount_gradio(app):
    """Mount Gradio app onto FastAPI at /web path."""
    if os.environ.get("ENABLE_WEB_INTERFACE", "").lower() in ("true", "1", "yes"):
        demo = create_gradio_app()
        gr.mount_gradio_app(app, demo, path="/web")
        return demo
    return None
