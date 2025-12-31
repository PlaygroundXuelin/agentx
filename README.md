# Agentic RAG Workspace

Polyrepo that hosts several focused subprojects working together to deliver an agentic retrieval-augmented system.

## Todo
- synthetic data generation
generate questions/expected answers from a document. and store it using datasets api.

- a simple rag agent using llamaindex library. Must be able to load a pdf file, have summaries for each chunk, 
and when search, must combine BM25 that searches the summaries and vector search to get best matches of relevant
chunks for a query
  - file/pdf processing and search api in documents subproject: summary search
  - hook up documents search into rag agent.
  - dummy data analysis agent: process a hard coded csv file and return a plot and some stats
  - workflow: from query, determine whether call rag agent and/or data analysis agent
- A datasets agent that search categories of existing datasets to find relevant datasets for a query.
- A simple data analytics agent that takes a dataset and drive stats / plots for a query.
## Layout
- `app_api/` – FastAPI service that exposes the orchestration layer
- `core/` – Shared utilities (`agent_core`) reused by downstream agents
- `agents/rag_agent/` – Retrieval-augmented generation agent built on LlamaIndex
- `agents/analytics_agent/` – Analytics agent for tabular insights

Each subproject is an isolated Python package with its own `pyproject.toml` so you can install, test, and release them independently.

## Development Workflow
- Python 3.14 with `uv` for dependency management (see `AGENTS.md` for conventions)
- Format & lint with Ruff: `uvx ruff check --fix .` then `uvx ruff format .`
- Run subproject-specific tasks from within each directory or orchestrate using your preferred tooling

## Next Steps
- Configure your preferred LLM + embedding providers through `agent_core.settings.apply_llama_settings`
- Start wiring agents together via `agent_core.registry.AgentRegistry`
- Extend the agents with datastore connectors, tools, and application-specific logic

## run LLM locally
vllm qwen AWQ model
```shell
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve Qwen/Qwen3-4B-AWQ \
  --max-model-len 8192 \
  --max-num-batched-tokens 768 \
  --max-num-seqs 3 \
  --gpu-memory-utilization 0.72 \
  --kv-cache-dtype fp8_e4m3 \
  --enforce-eager
```