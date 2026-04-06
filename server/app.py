"""
FastAPI application for the Cost-Aware FinQA Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
    - /: Custom Gradio UI with walkthrough examples
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import CostAwareFinqaAction, CostAwareFinqaObservation
    from .cost_aware_finqa_environment import CostAwareFinqaEnvironment
except (ImportError, SystemError):
    from models import CostAwareFinqaAction, CostAwareFinqaObservation
    from server.cost_aware_finqa_environment import CostAwareFinqaEnvironment

# Create the base API app (ENABLE_WEB_INTERFACE=false disables default playground)
app = create_app(
    CostAwareFinqaEnvironment,
    CostAwareFinqaAction,
    CostAwareFinqaObservation,
    env_name="cost_aware_finqa",
    max_concurrent_envs=3,
)

# Mount custom Gradio UI at root (API routes take precedence)
try:
    from .gradio_ui import mount_gradio
except (ImportError, SystemError):
    try:
        from server.gradio_ui import mount_gradio
    except (ImportError, SystemError):
        mount_gradio = None

if mount_gradio:
    mount_gradio(app)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
