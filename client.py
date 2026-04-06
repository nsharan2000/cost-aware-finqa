"""Cost-Aware FinQA Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CostAwareFinqaAction, CostAwareFinqaObservation


class CostAwareFinqaEnv(
    EnvClient[CostAwareFinqaAction, CostAwareFinqaObservation, State]
):
    """
    Client for the Cost-Aware FinQA Environment.

    Example:
        >>> with CostAwareFinqaEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.question)
        ...     result = client.step(CostAwareFinqaAction(tool="sql_query", query="SELECT * FROM ..."))
        ...     print(result.observation.tool_result)
    """

    def _step_payload(self, action: CostAwareFinqaAction) -> Dict:
        return {
            "tool": action.tool,
            "query": action.query,
            "answer": action.answer,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CostAwareFinqaObservation]:
        obs_data = payload.get("observation", {})
        observation = CostAwareFinqaObservation(
            question=obs_data.get("question", ""),
            task_name=obs_data.get("task_name", ""),
            tool_result=obs_data.get("tool_result", ""),
            tool_used=obs_data.get("tool_used", ""),
            tool_cost=obs_data.get("tool_cost", 0.0),
            budget_remaining=obs_data.get("budget_remaining", 0.0),
            budget_total=obs_data.get("budget_total", 0.0),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 8),
            error=obs_data.get("error", ""),
            available_tools=obs_data.get("available_tools", []),
            table_schema=obs_data.get("table_schema", ""),
            score=obs_data.get("score", 0.0),
            cost_so_far=obs_data.get("cost_so_far", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
