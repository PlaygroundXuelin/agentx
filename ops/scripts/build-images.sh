#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-dev}"
CATALOG_IMAGE="${CATALOG_IMAGE:-agentic-rag/catalog:${IMAGE_TAG}}"
DATASETS_IMAGE="${DATASETS_IMAGE:-agentic-rag/datasets:${IMAGE_TAG}}"
DOCKER_BUILD_NETWORK="${DOCKER_BUILD_NETWORK:-host}"

echo "Building catalog image -> ${CATALOG_IMAGE}"
docker build \
  --network "${DOCKER_BUILD_NETWORK}" \
  --file "${REPO_ROOT}/catalog/deploy/Dockerfile" \
  --tag "${CATALOG_IMAGE}" \
  "${REPO_ROOT}"

echo "Building datasets image -> ${DATASETS_IMAGE}"
docker build \
  --network "${DOCKER_BUILD_NETWORK}" \
  --file "${REPO_ROOT}/datasets/deploy/Dockerfile" \
  --tag "${DATASETS_IMAGE}" \
  "${REPO_ROOT}"

echo "Images ready:"
echo "  ${CATALOG_IMAGE}"
echo "  ${DATASETS_IMAGE}"
