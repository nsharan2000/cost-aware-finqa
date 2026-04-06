"""
Cost-Aware FinQA Environment.

An RL environment where agents answer financial questions by choosing tools
strategically. Each tool has a different cost. The agent is rewarded for
correct answers and penalized for waste.

Datastore:
- financials_*: Company financial tables (income statements, balance sheets, etc.)
- documents: SEC filing text passages organized by company/year
- table_catalog: Index of all available financial tables

Tools:
- sql_query ($0): Query the datastore. Penalized for bad SQL.
- vector_search ($0.50): Semantic search over filing text.
- web_search ($3.00): External search via Serper API.
- upgrade_llm ($3.00): Use a stronger model for reasoning.
- submit_answer ($0): Submit final answer.
"""

import json
import os
import sqlite3
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CostAwareFinqaAction, CostAwareFinqaObservation
except (ImportError, SystemError):
    from models import CostAwareFinqaAction, CostAwareFinqaObservation

try:
    from .tools import (
        TOOL_COSTS, REDUNDANT_CALL_PENALTY,
        execute_sql_query, execute_vector_search, execute_web_search,
        execute_upgrade_llm, get_table_schema, grade_answer,
    )
except (ImportError, SystemError):
    from server.tools import (
        TOOL_COSTS, REDUNDANT_CALL_PENALTY,
        execute_sql_query, execute_vector_search, execute_web_search,
        execute_upgrade_llm, get_table_schema, grade_answer,
    )


TASK_CONFIG = {
    "basic_retrieval": {"budget": 10.0, "max_steps": 8, "difficulty": "easy"},
    "analytical_reasoning": {"budget": 15.0, "max_steps": 10, "difficulty": "medium"},
    "strategic_research": {"budget": 12.0, "max_steps": 10, "difficulty": "hard"},
}

DEFAULT_TASK = "basic_retrieval"


def _get_data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load_questions_for_task(task_name):
    """Load questions for a specific task from the datastore."""
    db_path = os.path.join(_get_data_dir(), "financial_data.db")
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, question, gold_answer, category, difficulty, task, program, company, fiscal_year, financial_table "
        "FROM questions WHERE task = ?",
        (task_name,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"id": r[0], "question": r[1], "answer": r[2], "category": r[3],
         "difficulty": r[4], "task": r[5], "program": r[6], "company": r[7],
         "fiscal_year": r[8], "financial_table": r[9]}
        for r in rows
    ]


class CostAwareFinqaEnvironment(Environment):
    """
    Cost-Aware Financial QA Environment.

    Score = answer_correctness * cost_efficiency * (1 - error_penalty)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name = os.environ.get("FINQA_TASK", DEFAULT_TASK)
        self._current_question = None
        self._budget_total = 0.0
        self._budget_remaining = 0.0
        self._cost_spent = 0.0
        self._step_rewards = []
        self._tool_history = []
        self._max_steps = 8
        self._answered = False
        self._final_score = 0.0
        self._question_index = 0
        self._task_questions = []

    def reset(self) -> CostAwareFinqaObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        config = TASK_CONFIG.get(self._task_name, TASK_CONFIG[DEFAULT_TASK])
        self._budget_total = config["budget"]
        self._budget_remaining = config["budget"]
        self._max_steps = config["max_steps"]

        self._task_questions = _load_questions_for_task(self._task_name)
        if not self._task_questions:
            # Fallback: load all questions
            for t in TASK_CONFIG:
                self._task_questions.extend(_load_questions_for_task(t))

        if self._question_index >= len(self._task_questions):
            self._question_index = 0
        self._current_question = self._task_questions[self._question_index]
        self._question_index += 1

        self._cost_spent = 0.0
        self._step_rewards = []
        self._tool_history = []
        self._answered = False
        self._final_score = 0.0

        q_id = self._current_question["id"]
        table_schema = get_table_schema(q_id)

        return CostAwareFinqaObservation(
            question=self._current_question["question"],
            task_name=self._task_name,
            tool_result=(
                "Environment ready. You have access to a financial datastore with company "
                "financial tables and SEC filing documents.\n\n"
                "Available tools: sql_query (FREE), vector_search ($0.50), "
                "web_search ($3.00), upgrade_llm ($3.00), submit_answer (FREE)\n\n"
                "Tip: Start with sql_query to explore the data. "
                "Use 'SELECT * FROM table_catalog' to discover available tables."
            ),
            tool_used="",
            tool_cost=0.0,
            budget_remaining=self._budget_remaining,
            budget_total=self._budget_total,
            step_number=0,
            max_steps=self._max_steps,
            error="",
            available_tools=["sql_query", "vector_search", "web_search", "upgrade_llm", "submit_answer"],
            table_schema=table_schema,
            score=0.0,
            cost_so_far=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: CostAwareFinqaAction) -> CostAwareFinqaObservation:
        # Auto-reset if step is called on a fresh instance (HTTP endpoints are stateless)
        if self._current_question is None:
            self.reset()

        self._state.step_count += 1
        step_num = self._state.step_count

        tool = action.tool.strip().lower()
        query = action.query.strip()
        answer = action.answer.strip()
        error = ""
        step_reward = 0.0
        tool_result = ""
        tool_cost = 0.0
        done = False

        valid_tools = ["sql_query", "vector_search", "web_search", "upgrade_llm", "submit_answer"]

        if tool not in valid_tools:
            error = f"Invalid tool '{tool}'. Use: {', '.join(valid_tools)}"
            step_reward = -0.05
            tool_result = error
        elif self._answered:
            error = "Episode already ended."
            done = True
            tool_result = error
        elif tool != "submit_answer" and self._budget_remaining <= 0:
            error = "Budget exhausted. Submit your answer now."
            tool_result = error
        else:
            # Redundancy check
            call_key = (tool, query)
            if tool != "submit_answer" and call_key in self._tool_history:
                step_reward += REDUNDANT_CALL_PENALTY
                error = "Redundant call (same tool+query). Penalty applied."
            self._tool_history.append(call_key)

            tool_cost = TOOL_COSTS.get(tool, 0.0)

            if tool_cost > self._budget_remaining and tool != "submit_answer":
                error = f"Not enough budget for {tool} (${tool_cost}, remaining ${self._budget_remaining:.2f})"
                tool_result = error
                tool_cost = 0.0
            else:
                company = self._current_question.get("company", "")
                q_id = self._current_question["id"]
                fin_table = self._current_question.get("financial_table", "")

                if tool == "sql_query":
                    tool_result, bonus = execute_sql_query(query, fin_table)
                    step_reward += bonus

                elif tool == "vector_search":
                    tool_result, bonus = execute_vector_search(query, company=company, question_id=q_id)
                    step_reward += bonus

                elif tool == "web_search":
                    tool_result, bonus = execute_web_search(query)
                    step_reward += bonus

                elif tool == "upgrade_llm":
                    prev_context = "\n".join([
                        f"[{h[0]}]: {h[1][:200]}" for h in self._tool_history[:-1]
                    ])
                    tool_result, bonus = execute_upgrade_llm(query, prev_context)
                    step_reward += bonus

                elif tool == "submit_answer":
                    self._answered = True
                    done = True

                    gold = self._current_question.get("answer", "")
                    correctness, explanation = grade_answer(answer, gold, q_id)

                    cost_eff = max(0.1, 1.0 - self._cost_spent / self._budget_total) if self._budget_total > 0 else 1.0
                    neg = sum(r for r in self._step_rewards if r < 0)
                    err_penalty = min(0.5, abs(neg))

                    self._final_score = max(0.0, min(1.0, correctness * cost_eff * (1.0 - err_penalty)))
                    step_reward = self._final_score

                    tool_result = (
                        f"Answer graded.\n"
                        f"  Correctness: {correctness:.2f} ({explanation})\n"
                        f"  Cost efficiency: {cost_eff:.2f} (spent ${self._cost_spent:.2f}/{self._budget_total:.2f})\n"
                        f"  Error penalty: {err_penalty:.2f}\n"
                        f"  Final score: {self._final_score:.3f}"
                    )

                self._budget_remaining -= tool_cost
                self._cost_spent += tool_cost

        if step_num >= self._max_steps and not self._answered:
            done = True
            self._final_score = 0.0
            step_reward = 0.0
            tool_result += "\nMax steps reached. Score: 0.0"

        self._step_rewards.append(step_reward)

        return CostAwareFinqaObservation(
            question=self._current_question["question"] if self._current_question else "",
            task_name=self._task_name,
            tool_result=tool_result,
            tool_used=tool,
            tool_cost=tool_cost,
            budget_remaining=self._budget_remaining,
            budget_total=self._budget_total,
            step_number=step_num,
            max_steps=self._max_steps,
            error=error,
            available_tools=valid_tools,
            table_schema="",
            score=self._final_score if done else sum(max(0, r) for r in self._step_rewards),
            cost_so_far=self._cost_spent,
            done=done,
            reward=step_reward,
            metadata={
                "tool": tool, "cost": tool_cost,
                "budget_remaining": self._budget_remaining,
                "final_score": self._final_score if done else None,
            },
        )

    @property
    def state(self) -> State:
        return self._state
