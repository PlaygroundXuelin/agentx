# syntax=docker/dockerfile:1.6
FROM ghcr.io/astral-sh/uv:python3.14-bookworm

WORKDIR /app

# Copy shared core utilities followed by the service sources to leverage Docker layer caching.
COPY core /app/core
COPY catalog /app/catalog

# Install runtime dependencies using uv without pulling in dev extras.
RUN --mount=type=cache,target=/root/.cache \
    cd /app/catalog && uv sync --frozen --no-dev --link-mode=copy

ENV PATH="/app/catalog/.venv/bin:${PATH}"
WORKDIR /app/catalog

EXPOSE 9000

ENTRYPOINT ["./deploy/entrypoint.sh"]
