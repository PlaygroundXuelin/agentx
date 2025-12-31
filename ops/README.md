# Nomad Deployment

This directory contains a minimal HashiCorp Nomad deployment targeting the `catalog` and `datasets`
services. It provides build scripts for container images plus Nomad job specifications that can be
parameterised for development or production environments. Service discovery now uses Consul, and
internal traffic flows through Consul Connect sidecars. Traefik remains available exclusively for
public ingress.

## Prerequisites

- Docker (or another container runtime compatible with the Nomad Docker driver)
- HashiCorp Nomad CLI configured to talk to your cluster
- Optional: A container registry to publish the built images
- Optional: A Traefik image variant if you prefer something other than the default `traefik:v3.1`

## Layout

- `scripts/build-images.sh` – convenience script that builds the service images from the repository root.
- `nomad/agentic-rag.nomad.hcl` – Nomad job specification for the catalog and datasets services.
  Supply variables at submit time to pick the environment, datacenters, and image tags.
- `nomad/traefik.nomad.hcl` – Traefik job wired to the Nomad service registry and exposing a single
  `web` entrypoint on port 8080 for inbound traffic.

## Building Images

```bash
# if necessary
source .venv/bin/activate
cd <updated service such as datasets>
uv sync --active
cd ..
./deploy/scripts/build-images.sh
```

The script builds `agentic-rag/catalog:<tag>` and `agentic-rag/datasets:<tag>` images (defaults to
`dev`). Override the tag by exporting `IMAGE_TAG` before running the script. The build script
passes `--network host` to `docker build` by default to ensure the build container has outbound
network access. Adjust via `DOCKER_BUILD_NETWORK` if your environment requires a different setting.

Push the images to your registry of choice once built:

```bash
docker push agentic-rag/catalog:<tag>
docker push agentic-rag/datasets:<tag>
```

## Running the Job

For local development, launch a Nomad dev agent with Consul Connect enabled (requires sudo on Linux):

```bash
# start nomad server/client with embedded Consul + Connect
sudo NOMAD_BIND_ADDR=0.0.0.0 nomad agent -dev-consul -dev-connect -bind=0.0.0.0 -config=deploy/nomad/dev.hcl
```

In a separate terminal, deploy the application services, overriding variables as needed:

```bash
tag=dev
nomad job run \
  -var environment=dev \
  -var catalog_image=agentic-rag/catalog:${tag} \
  -var datasets_image=agentic-rag/datasets:${tag} \
  deploy/nomad/agentic-rag.nomad.hcl
```

Then submit Traefik if you want HTTP ingress during development:

```bash
nomad job run \
  -var environment=dev \
  deploy/nomad/traefik.nomad.hcl
```

### Smoke test

```bash
# inside the datasets allocation, the Consul Connect upstream is bound locally
nomad alloc exec -task datasets <alloc-id> env | grep SVC_CATALOG_URL
nomad alloc exec -task datasets <alloc-id> curl "$SVC_CATALOG_URL/v1/ping"

# from outside (Traefik ingress)
curl -H 'Host: datasets.dev.nomad' http://127.0.0.1:8080/v1/ping
curl -H 'Host: datasets.dev.nomad' http://127.0.0.1:8080/v1/ping-catalog
```

### debug
ui: http://127.0.0.1:4646/ui/jobs

```shell
# allocations
nomad operator api /v1/allocations?namespace=default | jq '.[].ID'
# check alloc status
nomad alloc statu <alloc id>
```

### Useful Variables

- `environment` – stored in Nomad job metadata and propagated to service registrations.
- `datacenters` – list of datacenters to target (`["dc1"]` by default).
- `catalog_image` / `datasets_image` – Docker images that Nomad should run.
- `catalog_count` / `datasets_count` – optional scaling knobs for each task group.
- `catalog_config_path` / `datasets_config_path` – configuration file paths inside the container,
  defaulting to the Nomad-specific YAML files committed in each service.
- `FS__ROOT` – injected automatically for the datasets task (`{{ env "NOMAD_ALLOC_DIR" }}/data/datasets`)
  so the store lives on the allocation's writable volume.
- Services run in bridge networking mode. Use `nomad alloc exec` (for example,
  `nomad alloc exec -task datasets <alloc-id> curl http://127.0.0.1:8100/v1/ping`) or launch the Traefik job
  to reach them from the host. Consul Connect upstreams listen on `0.0.0.0` inside allocations so dependents
  can reach peers via `http://127.0.0.1:<port>`.

The Nomad job name is fixed to `agentic-rag`; stop the job with `nomad job stop agentic-rag`.
With Consul Connect, each service registers in Consul (`nomad service list`) and exposes a local
loopback address for dependent workloads via sidecar proxies. The datasets task receives
`SVC_CATALOG_URL=http://127.0.0.1:9191` so internal calls never rely on container or host IPs.
Traefik is no longer part of the east–west traffic path; it should only be used for public ingress
during development or production.

## install tools
See https://developer.hashicorp.com/nomad/docs/deploy
### install nomad
```shell
# prereqs
sudo apt-get update && sudo apt-get install -y wget gpg coreutils

# add official HashiCorp key (keyring, not apt-key)
wget -O- https://apt.releases.hashicorp.com/gpg \
| sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg

# add repo (auto-detects your Ubuntu codename)
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(grep -oP '(?<=UBUNTU_CODENAME=).*' /etc/os-release || lsb_release -cs) main" \
| sudo tee /etc/apt/sources.list.d/hashicorp.list

# install
sudo apt-get update && sudo apt-get install -y nomad
```

May need post installation steps for cni etc, please consult the hashicorp page.
