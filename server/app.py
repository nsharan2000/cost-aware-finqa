"""
FastAPI application for the Cost-Aware FinQA Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
    - /web: Gradio interactive demo (when ENABLE_WEB_INTERFACE=true)
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

# Create the app
app = create_app(
    CostAwareFinqaEnvironment,
    CostAwareFinqaAction,
    CostAwareFinqaObservation,
    env_name="cost_aware_finqa",
    max_concurrent_envs=3,
)

# Mount Gradio web interface if enabled
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
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m cost_aware_finqa.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn cost_aware_finqa.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
