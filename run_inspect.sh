#!/usr/bin/env bash
set -euo pipefail

# data paths
export DS_LOCAL_DATA_PATH="${DS_LOCAL_DATA_PATH:-$(pwd)/workspace}"

# Clean up pickle cache before run to ensure fresh execution
rm -rf ./pickle_cache

rdagent grade_summary ./log

rdagent ui --data-science --log-dir=./log 2> >(grep -v "gio:" >&2)