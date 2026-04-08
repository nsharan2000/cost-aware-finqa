"""
FastAPI application for the Cost-Aware FinQA Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
    - /: Tabbed Gradio UI (Agent Chat + Playground)
"""

import os

# Load .env file (HF_TOKEN, SERPER_API_KEY, etc.) before anything else
from dotenv import load_dotenv
# Try project-level .env first, then parent directory .env
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_dir, ".env"), override=False)
load_dotenv(os.path.join(os.path.dirname(_project_dir), ".env"), override=False)

# Disable OpenEnv's built-in web interface — we build our own tabbed UI
os.environ.setdefault("ENABLE_WEB_INTERFACE", "false")

# Point OpenEnv's Playground README to our guide
_guide_path = os.path.join(_project_dir, "PLAYGROUND_GUIDE.md")
if os.path.exists(_guide_path):
    os.environ.setdefault("ENV_README_PATH", _guide_path)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import CostAwareFinqaAction, CostAwareFinqaObservation
    from .cost_aware_finqa_environment import CostAwareFinqaEnvironment
    from .gradio_ui import mount_tabbed_gradio
except (ImportError, SystemError):
    from models import CostAwareFinqaAction, CostAwareFinqaObservation
    from server.cost_aware_finqa_environment import CostAwareFinqaEnvironment
    from server.gradio_ui import mount_tabbed_gradio

# Create the base API app (no built-in web interface)
app = create_app(
    CostAwareFinqaEnvironment,
    CostAwareFinqaAction,
    CostAwareFinqaObservation,
    env_name="cost_aware_finqa",
    max_concurrent_envs=3,
)

# Mount our custom tabbed UI: Agent Chat (default) + Playground
mount_tabbed_gradio(app, CostAwareFinqaEnvironment, CostAwareFinqaAction, CostAwareFinqaObservation)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
