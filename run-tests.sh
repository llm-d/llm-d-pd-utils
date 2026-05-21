#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Require uv
if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

uv run "$SCRIPT_DIR/skills/llm-d-networking-tests/scripts/run-tests.py" "$@"
