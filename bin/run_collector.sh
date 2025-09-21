#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR=${VENV_DIR:-"$ROOT_DIR/.venv"}
PYTHON=${PYTHON:-"$VENV_DIR/bin/python"}
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=$(command -v python3)
fi

OUTPUT_PATH=${OUTPUT_PATH:-"$ROOT_DIR/output/top_picks.html"}
LIMIT=${LIMIT:-20}
THREAD_URL=${THREAD_URL:-}
USER_AGENT=${USER_AGENT:-"sportsbook-picks-bot/0.1 (by u/your_username)"}

ARGS=("$ROOT_DIR/src/pick_collector.py" "--output" "$OUTPUT_PATH" "--limit" "$LIMIT" "--user-agent" "$USER_AGENT")
if [[ -n "$THREAD_URL" ]]; then
  ARGS+=("--thread-url" "$THREAD_URL")
fi

exec "$PYTHON" "${ARGS[@]}"
