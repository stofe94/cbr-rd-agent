#!/usr/bin/env bash
set -euo pipefail
export PYTHONFAULTHANDLER=0

# chrome paths
export CHROME_BIN=/usr/bin/chromium
export CHROMEDRIVER_PATH=/usr/bin/chromedriver

# load env defaults if present (before defaults so explicit envs take priority)
if [ -f ./config.env ]; then
  set -a
  source ./config.env
  set +a
fi

# HOST_WORKSPACE: default to ./workspace if not set (abs path)
HOST_WORKSPACE="${HOST_WORKSPACE:-$(pwd)/workspace}"
export HOST_WORKSPACE
echo "[INFO] HOST_WORKSPACE set to: ${HOST_WORKSPACE}"

# data paths (relative, or abs if worksapce is not set before this script execution (local tool execution without docker))
export DS_LOCAL_DATA_PATH="${DS_LOCAL_DATA_PATH:-$(pwd)/workspace}"
export CBR_KB_DIR="${CBR_KB_DIR:-$(pwd)/workspace/knowledge_base}"
export CoSTEER_KNOWLEDGE_BASE_PATH="${CoSTEER_KNOWLEDGE_BASE_PATH:-$(pwd)/workspace/knowledge_base/costeer.pkl}"
export CoSTEER_NEW_KNOWLEDGE_BASE_PATH="${CoSTEER_NEW_KNOWLEDGE_BASE_PATH:-$(pwd)/workspace/knowledge_base/costeer_new.pkl}"

# Clean up
rm -rf ./pickle_cache
: > ./terminal.log  # truncate, not delete: the Docker scripts bind-mount this file
rm -rf ./prompt_cache.db

rdagent data_science --competition nomad2018-predict-transparent-conductors