# Repository Guidelines

## Project Structure & Module Organization
- `src/pick_collector.py` houses the scraping workflow, CLI entrypoint, and HTML writer; add helpers here rather than spawning new top-level scripts.
- `bin/run_collector.sh` wraps the collector with environment overrides (`OUTPUT_PATH`, `THREAD_URL`, `SUBREDDITS`, `LIMIT`); update it alongside new flags.
- `tests/` currently holds `test_parsing.py`, exercising record and pick parsing; mirror the layout when introducing new suites.
- `output/` is the default artifact directory, while `comments_raw.json` and `debug.json` capture sample Reddit payloads for regression hunting.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: provision the local environment expected by tooling and scripts.
- `pip install -r requirements.txt`: install runtime plus test dependencies.
- `python src/pick_collector.py --output output/top_picks.html --limit 20 --user-agent "sportsbook-picks-bot/0.1 (by u/your_username)"`: direct execution for quick smoke checks.
- `bin/run_collector.sh`: opinionated launcher; respects environment variables and fails fast via `set -euo pipefail`.
- `python -m pytest`: run the unittest-backed suite; append `-k pattern` during focused parsing iterations.

## Coding Style & Naming Conventions
- Stick to PEP 8 with 4-space indents; run Black (or equivalent) before opening a PR to avoid formatting churn.
- Preserve type hints, dataclasses, and `extract_*` / `parse_*` helper naming so new logic reads like existing heuristics.
- Extend CLI options through `argparse` in `pick_collector.py` and document the change in both `README.md` and the shell wrapper.
- Use the module-level `logger` for diagnostics instead of `print` so verbosity flags remain effective.

## Testing Guidelines
- Place tests under `tests/` with filenames starting `test_` and `unittest.TestCase` subclasses grouped by behavior.
- Model fixtures after real Reddit comments; inline sample bodies keep expectations obvious.
- Run `python -m pytest` before every push; expand coverage when tweaking parsing, ranking, or output formatting paths.
- Store any reusable payloads under `tests/data/` (create as needed) to avoid cluttering the repo root.

## Commit & Pull Request Guidelines
- Follow the short, imperative commit style shown in history (`Improve parsing`); add a body when the change needs rationale.
- PRs should outline scope, list validation commands, and mention new configuration knobs or artifacts.
- Link related issues or discussions and highlight risky areas (e.g., parsing heuristics) that deserve extra review.
- Attach updated HTML/JSON samples when output formatting changes so reviewers can validate the delta quickly.
