# syntax=docker/dockerfile:1.6
FROM ghcr.io/astral-sh/uv:python3.14-bookworm

WORKDIR /app

# Copy shared core utilities followed by the datasets service.
COPY core /app/core
COPY datasets /app/datasets

# Install runtime dependencies (excluding dev extras).
RUN --mount=type=cache,target=/root/.cache \
    cd /app/datasets && uv sync --frozen --no-dev --link-mode=copy

ENV PATH="/app/datasets/.venv/bin:${PATH}"
WORKDIR /app/datasets

EXPOSE 8100

ENTRYPOINT ["./deploy/entrypoint.sh"]
