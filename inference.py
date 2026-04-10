"""
Inference Script for Cost-Aware FinQA Environment
==================================================
Runs a baseline agent across all 3 tasks (basic_retrieval, analytical_reasoning,
strategic_research). The agent uses an LLM to decide which tool to use at each step.

MANDATORY env vars (injected by validator):
    API_BASE_URL       The API endpoint for the LLM
    MODEL_NAME         The model identifier
    HF_TOKEN           Your HuggingFace / API key
    IMAGE_NAME         Docker image name for the environment
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

try:
    from cost_aware_finqa import CostAwareFinqaAction, CostAwareFinqaEnv
except ImportError:
    # Fallback: when package-dir mapping doesn't install __init__.py correctly,
    # import directly from the same directory as this script.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import CostAwareFinqaAction  # noqa: F811
    from client import CostAwareFinqaEnv  # noqa: F811

IMAGE_NAME = os.getenv("IMAGE_NAME")
# Validator injects API_KEY — prioritize it over HF_TOKEN
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASKS = ["basic_retrieval", "analytical_reasoning", "strategic_research"]
BENCHMARK = "cost_aware_finqa"
MAX_STEPS = 8
QUESTIONS_PER_TASK = 3
TEMPERATURE = 0.3
MAX_TOKENS = 500

SYSTEM_PROMPT = textwrap.dedent("""
You are a financial research agent. You answer financial questions by choosing tools strategically.

Available tools:
- sql_query: Run SQL on financial tables. Costs $0.001. Penalized for bad queries.
- web_search: Search the internet for benchmarks/comparisons. Costs $0.02.
- upgrade_llm: Use a stronger model for complex reasoning. Costs $1.00. EXTREMELY EXPENSIVE (1000x SQL) — absolute last resort only.
- submit_answer: Submit your final answer. FREE.

Strategy:
- ALWAYS start with sql_query. The table name is given in the schema — use it directly.
- First query: SELECT * FROM "<table_name>" LIMIT 5 (use EXACT table name from schema).
- Only use web_search when the question asks about industry comparisons or external benchmarks.
- Only use upgrade_llm for complex multi-step calculations after SQL data is retrieved.
- Submit your answer as a single number or short phrase.
- Minimize costs to maximize your score. Avoid redundant SQL calls.

Respond with a JSON object: {"tool": "<tool_name>", "query": "<your query>", "answer": "<if submitting>"}
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def parse_llm_response(text: str) -> dict:
    """Parse LLM response into tool call dict."""
    text = text.strip()
    if "{" in text:
        start = text.index("{")
        end = text.rindex("}") + 1
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"tool": "submit_answer", "query": "", "answer": text}


def get_agent_action(client: OpenAI, question: str, table_schema: str,
                     history: List[dict], budget_remaining: float) -> dict:
    """Ask the LLM which tool to use next."""
    history_text = ""
    if history:
        history_text = "\n".join([
            f"Step {h['step']}: Used {h['tool']} -> {h['result'][:150]}"
            for h in history[-4:]
        ])

    user_prompt = textwrap.dedent(f"""
    Question: {question}

    Table Schema:
    {table_schema[:500]}

    Budget remaining: ${budget_remaining:.2f}

    Previous steps:
    {history_text if history_text else "None yet"}

    What tool should I use next? Respond with JSON: {{"tool": "...", "query": "...", "answer": "..."}}
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_llm_response(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"tool": "submit_answer", "query": "", "answer": "0"}


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await CostAwareFinqaEnv.from_docker_image(IMAGE_NAME)

    all_scores = {}
    for task in TASKS:
        scores = []
        for q_idx in range(QUESTIONS_PER_TASK):
            log_start(task=f"{task}_q{q_idx}", env=BENCHMARK, model=MODEL_NAME)

            history: List[dict] = []
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False

            try:
                os.environ["FINQA_TASK"] = task
                result = await env.reset()
                obs = result.observation
                question = obs.question
                table_schema = obs.table_schema

                for step in range(1, MAX_STEPS + 1):
                    if result.done:
                        break

                    action_dict = get_agent_action(
                        client, question, table_schema,
                        history, obs.budget_remaining
                    )

                    tool = action_dict.get("tool", "submit_answer")
                    query = action_dict.get("query", "")
                    answer = action_dict.get("answer", "")

                    action = CostAwareFinqaAction(tool=tool, query=query, answer=answer)
                    result = await env.step(action)
                    obs = result.observation

                    reward = result.reward or 0.0
                    done = result.done
                    error = obs.error if obs.error else None

                    rewards.append(reward)
                    steps_taken = step

                    action_str = f"{tool}({query[:50]})" if tool != "submit_answer" else f"submit({answer[:30]})"
                    log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                    history.append({
                        "step": step,
                        "tool": tool,
                        "result": obs.tool_result[:200],
                    })

                    if obs.tool_result and tool != "submit_answer":
                        table_schema = obs.tool_result[:300]

                    if done:
                        break

                score = obs.score if obs.score else 0.0
                score = min(max(score, 0.0), 1.0)
                success = score >= 0.3

            except Exception as e:
                print(f"[DEBUG] Error in task {task} q{q_idx}: {e}", flush=True)
            finally:
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

            scores.append(score)

        all_scores[task] = scores
        avg = sum(scores) / len(scores) if scores else 0
        print(f"[INFO] Task {task}: avg_score={avg:.3f} scores={scores}", flush=True)

    try:
        await env.close()
    except Exception as e:
        print(f"[DEBUG] env.close() error: {e}", flush=True)

    all_vals = [s for scores in all_scores.values() for s in scores]
    overall = sum(all_vals) / len(all_vals) if all_vals else 0
    print(f"[INFO] Overall baseline score: {overall:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
