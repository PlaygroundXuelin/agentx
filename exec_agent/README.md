## Quickstart

```shell
uv run --active -m exec_agent.app --config src/exec_agent/configs/local.yaml
```

To provide additional environment variables, supply an env file:

```shell
uv run --active -m exec_agent.app \
  --config src/exec_agent/configs/local.yaml \
  # --env src/exec_agent/configs/local.env # if there is an env file
  
# health check
curl http://localhost:9000/ # exec_agent health check


curl http://localhost:9000/v1/ping 
```

## Development

1. `uv sync --active --extra dev`
2. `uv run --active pytest`
3. `uvx ruff check --fix .`
4. `uvx ruff format .`

Adjust the `AppSettings` dataclass, extend `register_routes`, and enrich `create_app` in
`exec_agent/app.py` as your service grows. The included test shows how to exercise the shared settings loader with custom
arguments.
