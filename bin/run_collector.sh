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
SUBREDDITS_ENV=${SUBREDDITS:-}

ARGS=("$ROOT_DIR/src/pick_collector.py" "--output" "$OUTPUT_PATH" "--limit" "$LIMIT" "--user-agent" "$USER_AGENT")
if [[ -n "$THREAD_URL" ]]; then
  ARGS+=("--thread-url" "$THREAD_URL")
fi
if [[ -n "$SUBREDDITS_ENV" ]]; then
  IFS=',' read -ra SUB_SPEC <<< "$SUBREDDITS_ENV"
  for raw_spec in "${SUB_SPEC[@]}"; do
    spec="${raw_spec#${raw_spec%%[![:space:]]*}}"
    spec="${spec%${spec##*[![:space:]]}}"
    if [[ -n "$spec" ]]; then
      ARGS+=("--subreddit" "$spec")
    fi
  done
fi

exec "$PYTHON" "${ARGS[@]}"
