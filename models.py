"""
Data models for the Cost-Aware FinQA Environment.

The agent interacts with financial data using 4 tools (sql_query, vector_search,
web_search, upgrade_llm) and submits answers. Each tool has a different cost.
"""

from typing import Optional, List, Dict, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CostAwareFinqaAction(Action):
    """Action the agent can take in the environment.

    Tools:
        - sql_query: Run SQL against financial tables ($0, penalized for errors)
        - vector_search: Semantic search over filing text ($0.50)
        - web_search: External search via Serper API ($3.00)
        - upgrade_llm: Use a stronger model for reasoning ($3.00)
        - submit_answer: Submit final answer for grading
    """

    tool: str = Field(
        ...,
        description="Tool to use: sql_query | vector_search | web_search | upgrade_llm | submit_answer"
    )
    query: str = Field(
        default="",
        description="SQL query string (for sql_query), search query (for vector_search/web_search), or reasoning prompt (for upgrade_llm)"
    )
    answer: str = Field(
        default="",
        description="Final answer to submit (only used with submit_answer tool)"
    )


class CostAwareFinqaObservation(Observation):
    """Observation returned by the environment after each step."""

    question: str = Field(default="", description="The current financial question")
    task_name: str = Field(default="", description="Current task: basic_retrieval | analytical_reasoning | strategic_research")
    tool_result: str = Field(default="", description="Result from the tool call")
    tool_used: str = Field(default="", description="Which tool was executed")
    tool_cost: float = Field(default=0.0, description="Cost of this tool call")
    budget_remaining: float = Field(default=0.0, description="Remaining budget for this episode")
    budget_total: float = Field(default=0.0, description="Total budget for this episode")
    step_number: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=8, description="Maximum steps allowed")
    error: str = Field(default="", description="Error message if tool call failed")
    available_tools: List[str] = Field(
        default_factory=lambda: ["sql_query", "vector_search", "web_search", "upgrade_llm", "submit_answer"],
        description="Tools available to the agent"
    )
    table_schema: str = Field(default="", description="SQL table schema hint for the current question")
    score: float = Field(default=0.0, description="Current cumulative score")
    cost_so_far: float = Field(default=0.0, description="Total cost spent so far")
