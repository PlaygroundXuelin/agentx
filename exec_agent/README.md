# App Template

This subproject is a lightweight scaffold for building new rest services. It demonstrates
how to load configuration from YAML and optional env files, bootstrap application-level logging, and
expose a FastAPI application with a health check and sample router using the shared `core` utilities.

## Quickstart

```shell
uv run --active -m app_template.app --config src/app_template/configs/local.yaml
```

To provide additional environment variables, supply an env file:

```shell
uv run --active -m app_template.app \
  --config src/app_template/configs/local.yaml \
  --env src/app_template/configs/local.env
  
# health check
curl http://localhost:9000/ 


curl http://localhost:9000/v1/ping 
```

## Development

1. `uv sync --active --extra dev`
2. `uv run --active pytest`
3. `uvx ruff check --fix .`
4. `uvx ruff format .`

Adjust the `AppSettings` dataclass, extend `register_routes`, and enrich `create_app` in
`app_template/app.py` as your service grows. The included test shows how to exercise the shared settings loader with custom
arguments.
