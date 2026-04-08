"""
Gradio frontend for the Cost-Aware FinQA Environment.

Two-column layout:
- Left: Agent Chat UI with tool-use display and thinking
- Right: DESIGN.md rendered permanently
"""

import json
import os
import textwrap
import urllib.request

import gradio as gr

try:
    from .cost_aware_finqa_environment import CostAwareFinqaEnvironment, TASK_CONFIG
except (ImportError, SystemError):
    from server.cost_aware_finqa_environment import CostAwareFinqaEnvironment, TASK_CONFIG

try:
    from ..models import CostAwareFinqaAction
except (ImportError, SystemError):
    from models import CostAwareFinqaAction

# Display-friendly names for the task dropdown
TASK_DISPLAY_NAMES = {
    "Easy Task (Basic Retrieval)": "basic_retrieval",
    "Medium Task (Analytical Reasoning)": "analytical_reasoning",
    "Hard Task (Strategic Research)": "strategic_research",
}
TASK_DISPLAY_REVERSE = {v: k for k, v in TASK_DISPLAY_NAMES.items()}


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


# 3 sample questions for the chat UI — one per difficulty level
SAMPLE_QUESTIONS = [
    "(Easy) What percentage of total cash and investments as of Dec 29, 2012 was comprised of available-for-sale investments? (Intel)",
    "(Medium) What is the growth rate in net revenue in 2008 for Entergy?",
    "(Hard) What percentage decrease occurred from 2011-2012 for deferred acquisition payments at IPG?",
]


def _get_hf_token() -> str:
    """Get HF token from environment, .env file, or HF Space secrets."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    # Fallback: try .env file in project root
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def _call_hf_inference(messages: list, hf_token: str) -> str:
    """Call HF Inference API for the agent chat."""
    api_url = "https://router.huggingface.co/v1/chat/completions"
    model = os.environ.get("CHAT_MODEL", "Qwen/Qwen2.5-72B-Instruct")

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 800,
    })
    req = urllib.request.Request(
        api_url,
        data=payload.encode("utf-8"),
        headers={
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling HF API: {e}"


AGENT_SYSTEM_PROMPT = textwrap.dedent("""\
You are a cost-aware financial research agent. You answer financial questions using
tools strategically to minimize cost while maximizing accuracy.

Available tools:
- sql_query ($0.001): Run SQL on financial tables. The data is already in a local SQLite database. Use this FIRST.
- web_search ($0.02): Search the internet. Only needed for industry comparisons or external benchmarks.
- upgrade_llm ($1.00): Stronger model for complex reasoning. EXTREMELY EXPENSIVE — 1000x SQL cost. Last resort only.
- submit_answer (FREE): Submit your final answer. Answer must be a single number (e.g. "53.23"), not a sentence.

Strategy:
- ALWAYS use sql_query first. The table name is shown in the Table Schema — query it directly.
- Only use web_search when the question asks about industry averages or peer comparisons.
- NEVER use upgrade_llm unless all other tools have failed.
- Submit a single number as your answer, not a sentence.

Respond with JSON:
{"thinking": "<your reasoning>", "tool": "<tool_name>", "query": "<your query>", "answer": "<if submitting>"}
""").strip()


def create_gradio_app():
    """Create the Gradio interface with Agent Chat + permanent DESIGN.md."""

    env = CostAwareFinqaEnvironment()
    session = {
        "obs": None,
        "history": [],
        "chat_messages": [],
        "tool_log": [],
        "step_count": 0,
        "done": False,
    }

    def format_tool_log_html(tool_log):
        if not tool_log:
            return "<i>No tool calls yet.</i>"
        lines = []
        for i, e in enumerate(tool_log, 1):
            cost_str = f"${e['cost']:.4f}" if e['cost'] > 0 else "FREE"
            reward_str = f"{e['reward']:+.3f}" if e['reward'] != 0 else "0.000"
            color = "#22c55e" if e['reward'] > 0 else "#ef4444" if e['reward'] < 0 else "#6b7280"
            lines.append(
                f"<div style='margin:4px 0;padding:6px 10px;background:#1e1e2e;border-left:3px solid {color};border-radius:4px;font-size:13px;color:#e0e0e0;'>"
                f"<b>Step {i}</b> &mdash; <code style='color:#ccc;'>{e['tool']}</code> ({cost_str}) "
                f"<span style='color:{color};font-weight:600;'>[{reward_str}]</span><br>"
                f"<span style='color:#aaa;'>{e['query'][:120]}</span>"
                f"</div>"
            )
        return "".join(lines)

    def reset_session(task_name):
        # Map display name to internal task name
        task_name = TASK_DISPLAY_NAMES.get(task_name, task_name)
        env._task_name = task_name
        obs = env.reset()
        session["obs"] = obs
        session["history"] = []
        session["chat_messages"] = []
        session["tool_log"] = []
        session["step_count"] = 0
        session["done"] = False

        welcome = (
            f"**New episode started!**\n\n"
            f"**Task:** {obs.task_name} | **Budget:** ${obs.budget_total:.2f} | "
            f"**Max Steps:** {obs.max_steps}\n\n"
            f"**Question:** {obs.question}\n\n"
            f"**Table Schema:**\n```\n{obs.table_schema[:400]}\n```\n\n"
            f"Ask me to solve this, or type your own financial question!"
        )
        return (
            [{"role": "assistant", "content": welcome}],
            format_tool_log_html([]),
            f"Budget: ${obs.budget_remaining:.2f} / ${obs.budget_total:.2f} | Steps: 0/{obs.max_steps} | Score: 0.000",
        )

    def _parse_agent_response(text):
        """Parse agent JSON response, extracting thinking and tool call."""
        text = text.strip()
        thinking = ""
        tool_call = None

        # Try to extract JSON
        if "{" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            try:
                parsed = json.loads(text[start:end])
                thinking = parsed.get("thinking", "")
                tool_call = {
                    "tool": parsed.get("tool", "submit_answer"),
                    "query": parsed.get("query", ""),
                    "answer": parsed.get("answer", ""),
                }
            except json.JSONDecodeError:
                pass

        if not tool_call:
            tool_call = {"tool": "submit_answer", "query": "", "answer": text}

        return thinking, tool_call

    def agent_step(user_message, chat_history, task_name):
        """Run one agent interaction: user asks, agent thinks + uses tools + responds."""
        # Map display name to internal task name
        task_name = TASK_DISPLAY_NAMES.get(task_name, task_name)
        if session["obs"] is None or session["done"]:
            # Auto-reset if needed
            env._task_name = task_name
            obs = env.reset()
            session["obs"] = obs
            session["history"] = []
            session["tool_log"] = []
            session["step_count"] = 0
            session["done"] = False

        if chat_history is None:
            chat_history = []

        obs = session["obs"]
        hf_token = _get_hf_token()

        if not hf_token:
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": (
                "**HF_TOKEN not configured.** The Agent Chat requires a Hugging Face API token.\n\n"
                "- **HF Space:** Set `HF_TOKEN` as a Space secret in Settings.\n"
                "- **Local:** `export HF_TOKEN=hf_...` or add `HF_TOKEN=hf_...` to `.env`\n\n"
                "You can still use the **Playground** tab to interact with the environment directly (Reset/Step/State)."
            )})
            return (
                chat_history,
                format_tool_log_html(session["tool_log"]),
                f"Budget: ${obs.budget_remaining:.2f} / ${obs.budget_total:.2f} | Steps: {session['step_count']}/{obs.max_steps} | Score: {obs.score:.3f}",
            )

        # Build context for the agent
        history_text = ""
        if session["history"]:
            history_text = "\n".join([
                f"Step {h['step']}: {h['tool']} -> {h['result'][:200]}"
                for h in session["history"][-4:]
            ])

        user_prompt = (
            f"Question: {obs.question}\n\n"
            f"Table Schema:\n{obs.table_schema[:800]}\n\n"
            f"Budget remaining: ${obs.budget_remaining:.2f}\n"
            f"Steps: {session['step_count']}/{obs.max_steps}\n\n"
            f"Previous tool results:\n{history_text if history_text else 'None yet'}\n\n"
            f"Choose a tool. Your final answer must be a single number, not a sentence.\n"
            f"Respond with JSON: {{\"thinking\": \"...\", \"tool\": \"...\", \"query\": \"...\", \"answer\": \"...\"}}"
        )

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Run the agent loop (up to remaining steps)
        full_response_parts = []
        max_agent_steps = min(6, obs.max_steps - session["step_count"])

        for _ in range(max_agent_steps):
            if session["done"]:
                break

            llm_response = _call_hf_inference(messages, hf_token)
            thinking, tool_call = _parse_agent_response(llm_response)

            # Show thinking
            if thinking:
                full_response_parts.append(f"**Thinking:** {thinking}\n")

            tool = tool_call["tool"]
            query = tool_call["query"]
            answer = tool_call["answer"]

            full_response_parts.append(
                f"**Tool:** `{tool}` | "
                f"{'Query: `' + query[:80] + '`' if query else 'Answer: `' + answer[:80] + '`'}\n"
            )

            # Execute the tool
            action = CostAwareFinqaAction(tool=tool, query=query, answer=answer)
            obs = env.step(action)
            session["obs"] = obs
            session["step_count"] += 1

            # Log the tool call
            session["tool_log"].append({
                "tool": tool,
                "query": query if tool != "submit_answer" else f"Answer: {answer}",
                "result": obs.tool_result,
                "cost": obs.tool_cost,
                "reward": obs.reward,
                "error": obs.error,
            })
            session["history"].append({
                "step": session["step_count"],
                "tool": tool,
                "result": obs.tool_result[:200],
            })

            # Show result
            result_preview = obs.tool_result[:300]
            if obs.error:
                full_response_parts.append(f"**Error:** {obs.error}\n")
            full_response_parts.append(f"```\n{result_preview}\n```\n")

            if obs.done:
                session["done"] = True
                full_response_parts.append(
                    f"\n**Episode Complete!** Final Score: **{obs.score:.3f}** | "
                    f"Total Cost: ${obs.cost_so_far:.2f}\n"
                )
                break

            # Add result to messages for next iteration
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "user", "content": (
                f"Tool result:\n{obs.tool_result[:400]}\n\n"
                f"Budget remaining: ${obs.budget_remaining:.2f} | Step {session['step_count']}/{obs.max_steps}\n"
                f"Continue. Respond with JSON: {{\"thinking\": \"...\", \"tool\": \"...\", \"query\": \"...\", \"answer\": \"...\"}}"
            )})

        response = "\n".join(full_response_parts)
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": response})

        status = (
            f"Budget: ${obs.budget_remaining:.2f} / ${obs.budget_total:.2f} | "
            f"Steps: {session['step_count']}/{obs.max_steps} | "
            f"Score: {obs.score:.3f}"
        )

        return (
            chat_history,
            format_tool_log_html(session["tool_log"]),
            status,
        )

    def use_sample_question(idx, chat_history, task_name):
        """Handle clicking a sample question button."""
        q = SAMPLE_QUESTIONS[idx]
        return agent_step(q, chat_history, task_name)

    with gr.Blocks(
        title="Cost-Aware FinQA",
    ) as demo:

        gr.Markdown(
            "# Cost-Aware FinQA Research Agent\n"
            "An RL environment where agents learn to answer financial questions using the cheapest sufficient tool.\n"
            "**Scoring:** `correctness x cost_efficiency x (1 - error_penalty)`"
        )

        with gr.Row():
            # =================== LEFT COLUMN: Agent Chat ===================
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Agent Chat"):
                        with gr.Row():
                            task_dropdown = gr.Dropdown(
                                choices=list(TASK_DISPLAY_NAMES.keys()),
                                value="Easy Task (Basic Retrieval)",
                                label="Task",
                                scale=2,
                            )
                            reset_btn = gr.Button("New Episode", variant="primary", scale=1)

                        status_bar = gr.Markdown(
                            value="Click **New Episode** to start.",
                            elem_classes=["status-bar"],
                        )

                        chatbot = gr.Chatbot(
                            value=[],
                            height=420,
                            label="Agent Chat",
                        )

                        gr.Markdown("**Sample questions:**")
                        with gr.Row():
                            sample_btn_0 = gr.Button(
                                SAMPLE_QUESTIONS[0][:70] + "...",
                                size="sm", variant="secondary",
                            )
                            sample_btn_1 = gr.Button(
                                SAMPLE_QUESTIONS[1][:70] + "...",
                                size="sm", variant="secondary",
                            )
                            sample_btn_2 = gr.Button(
                                SAMPLE_QUESTIONS[2][:70] + "...",
                                size="sm", variant="secondary",
                            )

                        with gr.Row():
                            user_input = gr.Textbox(
                                placeholder="Ask a financial question or instruct the agent...",
                                label="Your Message",
                                scale=4,
                                lines=1,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)

                        gr.Markdown("### Tool Call Log")
                        tool_log_display = gr.HTML(
                            value="<i>No tool calls yet.</i>",
                            elem_classes=["tool-log"],
                        )

                # Wire events
                reset_btn.click(
                    fn=reset_session,
                    inputs=[task_dropdown],
                    outputs=[chatbot, tool_log_display, status_bar],
                )
                send_btn.click(
                    fn=agent_step,
                    inputs=[user_input, chatbot, task_dropdown],
                    outputs=[chatbot, tool_log_display, status_bar],
                ).then(fn=lambda: "", outputs=[user_input])
                user_input.submit(
                    fn=agent_step,
                    inputs=[user_input, chatbot, task_dropdown],
                    outputs=[chatbot, tool_log_display, status_bar],
                ).then(fn=lambda: "", outputs=[user_input])

                sample_btn_0.click(
                    fn=lambda ch, t: use_sample_question(0, ch, t),
                    inputs=[chatbot, task_dropdown],
                    outputs=[chatbot, tool_log_display, status_bar],
                )
                sample_btn_1.click(
                    fn=lambda ch, t: use_sample_question(1, ch, t),
                    inputs=[chatbot, task_dropdown],
                    outputs=[chatbot, tool_log_display, status_bar],
                )
                sample_btn_2.click(
                    fn=lambda ch, t: use_sample_question(2, ch, t),
                    inputs=[chatbot, task_dropdown],
                    outputs=[chatbot, tool_log_display, status_bar],
                )

            # =================== RIGHT COLUMN: DESIGN.md ===================
            with gr.Column(scale=2):
                gr.Markdown("## Design & Motivation")
                gr.Markdown(
                    value=_load_design_doc(),
                    elem_classes=["design-doc"],
                )

    return demo


def _build_playground_tab(env_cls, action_cls, obs_cls):
    """Build a Playground tab with Reset/Step/State — mirrors OpenEnv's default UI."""
    from openenv.core.env_server.web_interface import (
        WebInterfaceManager,
        load_environment_metadata,
        _extract_action_fields,
        _is_chat_env,
        get_quick_start_markdown,
    )
    from openenv.core.env_server.gradio_ui import build_gradio_app

    metadata = load_environment_metadata(env_cls, "cost_aware_finqa")
    web_manager = WebInterfaceManager(env_cls, action_cls, obs_cls, metadata)
    action_fields = _extract_action_fields(action_cls)
    is_chat = _is_chat_env(action_cls)
    quick_start = get_quick_start_markdown(metadata, action_cls, obs_cls)

    return build_gradio_app(
        web_manager, action_fields, metadata, is_chat, metadata.name, quick_start
    )


def mount_tabbed_gradio(app, env_cls, action_cls, obs_cls):
    """Mount tabbed Gradio UI: Agent Chat (default) + Playground."""
    agent_chat = create_gradio_app()
    playground = _build_playground_tab(env_cls, action_cls, obs_cls)

    tabbed = gr.TabbedInterface(
        [agent_chat, playground],
        tab_names=["Agent Chat", "Playground"],
        title="Cost-Aware FinQA Environment",
    )
    gr.mount_gradio_app(app, tabbed, path="/")
