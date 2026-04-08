"""
Inference Script for Cost-Aware FinQA Environment
==================================================
Runs a baseline agent across all 3 tasks (basic_retrieval, analytical_reasoning,
strategic_research). The agent uses an LLM to decide which tool to use at each step.

Required env vars:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier
    HF_TOKEN       Your HuggingFace / API key
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from cost_aware_finqa import CostAwareFinqaAction, CostAwareFinqaEnv

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASKS = ["basic_retrieval", "analytical_reasoning", "strategic_research"]
BENCHMARK = "cost_aware_finqa"
MAX_STEPS = 8
QUESTIONS_PER_TASK = 3  # Run 3 questions per task for baseline
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
- Use sql_query first for numerical lookups (it's cheap).
- Only use web_search when the question asks about industry comparisons or external benchmarks.
- Only use upgrade_llm for complex multi-step calculations.
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
    # Try to extract JSON from the response
    if "{" in text:
        start = text.index("{")
        end = text.rindex("}") + 1
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Fallback: try to parse as simple tool call
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


async def run_task(client: OpenAI, task_name: str, num_questions: int) -> List[float]:
    """Run a task and return scores for each question."""
    scores = []

    for q_idx in range(num_questions):
        log_start(task=f"{task_name}_q{q_idx}", env=BENCHMARK, model=MODEL_NAME)

        if IMAGE_NAME:
            env = await CostAwareFinqaEnv.from_docker_image(IMAGE_NAME)
        else:
            env = await CostAwareFinqaEnv.from_space(
                "Teachafy/cost-aware-finqa",
                hf_token=os.getenv("HF_TOKEN"),
            )

        history = []
        rewards = []
        steps_taken = 0
        score = 0.0
        success = False

        try:
            # Set task via environment variable
            os.environ["FINQA_TASK"] = task_name
            result = await env.reset()
            obs = result.observation
            question = obs.question
            table_schema = obs.table_schema

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                # Ask LLM what to do
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

                # Update table schema from tool results
                if obs.tool_result and tool != "submit_answer":
                    table_schema = obs.tool_result[:300]

                if done:
                    break

            score = obs.score if obs.score else 0.0
            score = min(max(score, 0.0), 1.0)
            success = score >= 0.3

        except Exception as e:
            print(f"[DEBUG] Error in task {task_name} q{q_idx}: {e}", flush=True)
        finally:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

        scores.append(score)

    return scores


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = {}
    for task in TASKS:
        scores = await run_task(client, task, QUESTIONS_PER_TASK)
        all_scores[task] = scores
        avg = sum(scores) / len(scores) if scores else 0
        print(f"[INFO] Task {task}: avg_score={avg:.3f} scores={scores}", flush=True)

    # Overall summary
    all_vals = [s for scores in all_scores.values() for s in scores]
    overall = sum(all_vals) / len(all_vals) if all_vals else 0
    print(f"[INFO] Overall baseline score: {overall:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
