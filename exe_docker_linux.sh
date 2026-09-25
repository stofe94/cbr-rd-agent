#!/usr/bin/env bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_USER="stofe94"
IMAGE_REMOTE="ghcr.io/${GITHUB_USER}/local_cbr-rdagent:latest"
IMAGE_LOCAL="local_cbr-rdagent:latest"
LINUX_WS="${HOME}/cbr-rdagent"
LINUX_TMP="${HOME}/tmp"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Competition prompt ────────────────────────────────────────────────────────
read -rp "Which competition to run? (e.g. spaceship-titanic): " RD_COMPETITION

# ── [0/3] Ensure folders exist ────────────────────────────────────────────────
echo "[0/3] Ensuring folders exist..."
mkdir -p "${LINUX_WS}/workspace/knowledge_base"
mkdir -p "${LINUX_WS}/log"
touch "${LINUX_WS}/log/terminal.log"

# Source secrets (expects KEY=VALUE lines, no export needed)
set -a
source "${LINUX_TMP}/secrets.env"
set +a

# ── [1/3] Image ───────────────────────────────────────────────────────────────
# The pre-built image on ghcr.io is private (maintainer access via GHCR_TOKEN).
# Without access, the image is built locally from this repository.
echo "[1/3] Getting the image..."
if [[ -n "${GHCR_TOKEN:-}" ]] \
    && echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GITHUB_USER}" --password-stdin >/dev/null 2>&1 \
    && docker pull "${IMAGE_REMOTE}"; then
    docker tag "${IMAGE_REMOTE}" "${IMAGE_LOCAL}"
else
    echo "No access to ${IMAGE_REMOTE}; building ${IMAGE_LOCAL} from ${REPO_DIR}..."
    VERSION=$(git -C "${REPO_DIR}" describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || true)
    docker build \
        --build-arg SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION:-0.0.0}" \
        -t "${IMAGE_LOCAL}" \
        "${REPO_DIR}"
fi

# ── [2/3] Remove a previous container ─────────────────────────────────────────
echo "[2/3] Removing a previous container..."
docker rm -f cbr-rdagent 2>/dev/null || true

# ── [3/3] Start container ─────────────────────────────────────────────────────
echo "[3/3] Starting container..."
docker run -it \
    -p 19899:19899 \
    -v "${LINUX_WS}/workspace:/root/workspace" \
    -v "${LINUX_WS}/log:/root/log" \
    -v "${LINUX_WS}/log/terminal.log:/root/terminal.log" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "${LINUX_TMP}/config.env:/root/config.env:ro" \
    -v "${LINUX_TMP}/secrets.env:/root/secrets.env:ro" \
    -e DOCKER_HOST=unix:///var/run/docker.sock \
    -e DS_LOCAL_DATA_PATH=/root/workspace \
    -e HOST_WORKSPACE="${LINUX_WS}/workspace" \
    --entrypoint bash \
    --name cbr-rdagent \
    "${IMAGE_LOCAL}" \
    -c "sed 's/--competition [^ ]*/--competition ${RD_COMPETITION}/' /root/run_linux.sh | bash"
