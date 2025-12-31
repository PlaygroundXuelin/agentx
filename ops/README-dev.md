
# Dev Environment Quickstart

These notes capture the workflow for running the Agentic RAG services locally with Nomad + Consul
Connect sidecars.

## 1. Prerequisites

- Docker (or a compatible container runtime)
- HashiCorp Nomad CLI
- HashiCorp Consul CLI (for managing intentions in production-like setups)

## 2. Start the dev agent

Run a single-box Nomad agent that embeds Consul and enables Consul Connect:

```bash
sudo NOMAD_BIND_ADDR=0.0.0.0 nomad agent \
  -dev-consul \
  -dev-connect \
  -bind=0.0.0.0 \
  -config=deploy/nomad/dev.hcl
```

## 3. Build + publish images (optional)

From the repo root:

```bash
./deploy/scripts/build-images.sh
docker push agentic-rag/catalog:dev
docker push agentic-rag/datasets:dev
```

## 4. Deploy the services

```bash
tag=dev
nomad job run \
  -var environment=dev \
  -var catalog_image=agentic-rag/catalog:${tag} \
  -var datasets_image=agentic-rag/datasets:${tag} \
  deploy/nomad/agentic-rag.nomad.hcl
```

Submit Traefik afterwards if you want HTTP ingress during development:

```bash
nomad job run -var environment=dev deploy/nomad/traefik.nomad.hcl
```

## 5. Smoke test

```bash
# check injected upstream env vars
nomad job status agentic-rag
nomad alloc exec -task datasets <alloc-id> env | grep SVC_CATALOG_URL

# verify the sidecar upstream
nomad alloc exec -task datasets <alloc-id> curl "$SVC_CATALOG_URL/v1/ping"

# optional: via Traefik ingress
curl -H 'Host: datasets.dev.nomad' http://127.0.0.1:8080/v1/ping-catalog
```

## 6. Intentions (prod posture)

In dev the default allow policy is usually fine. For stricter environments:

```bash
consul intention create -allow datasets-dev catalog-dev
consul intention list
```

Repeat for every permitted service pair. Blocking is implicit once the Consul intention policy is
set to `deny`.

## 7. Operational notes

- Each Nomad client should run a Consul client agent (`consul { address = "127.0.0.1:8500" }`).
- Sidecars expose upstreams on `127.0.0.1:<local_bind_port>` inside the allocation. Applications
  must read the injected `SVC_<DEPENDENCY>_URL` values instead of hard-coding host/container IPs.
- Traefik is not part of east–west traffic. Reserve it for inbound requests until a Consul API
  Gateway is introduced.

## 8. Steps
consul agent -dev \
-ui \
-client=0.0.0.0 \
-hcl='connect { enabled = true }' \
-hcl='ports { grpc = 8502 }'

sudo nomad agent -dev -bind=0.0.0.0

nomad job run   -var environment=dev   -var catalog_image=agentic-rag/catalog:${tag}   -var datasets_image=agentic-rag/datasets:${tag}   deploy/nomad/agentic-rag.nomad.hcl

consul intention create -allow datasets catalog