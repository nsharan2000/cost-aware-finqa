FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# Copy dependency file first for cache-friendly builds
COPY pyproject.toml /app/env/pyproject.toml

WORKDIR /app/env
RUN mkdir -p server && \
    touch __init__.py server/__init__.py && \
    uv venv .venv && \
    uv pip install -e "." --python .venv/bin/python 2>/dev/null || \
    uv pip install pydantic fastapi uvicorn gradio openenv-core --python .venv/bin/python

# Copy code (separate layer — changes don't invalidate dep install)
COPY . /app/env/

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
