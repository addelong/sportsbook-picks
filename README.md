# Sportsbook Pick Collector

A small utility that grabs daily pick threads from Reddit betting communities (e.g. `r/sportsbook`’s **Pick of the Day**, `r/sportsbetting`’s **Best Bets**), extracts user picks with posted records, ranks them by win percentage, and writes the result to a simple HTML report.

> ⚠️ Reddit recently tightened unauthenticated API access. Supply a descriptive `User-Agent` and consider a `thread_url` override if you hit rate limits.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python src/pick_collector.py \
  --output output/top_picks.html \
  --subreddit sportsbook \
  --subreddit sportsbetting=title:"Best Bets" \
  --limit 20
```

Key flags:

- `--thread-url` – manually point at a specific Pick of the Day permalink (handy if the search API falls behind).
- `--user-agent` – pass a Reddit-friendly UA string to reduce `429` responses.
- `--limit` – cap the number of picks in the generated report (default 10).
- `--subreddit` – repeatable; accepts `name`, `name=Pick of the Day`, or a full query such as `name=title:"Best Bets"`. Defaults to `r/sportsbook`.
- `--verbose` – print INFO-level logging (handy when monitoring long fetches).

Outputs land in `output/top_picks.html` by default.

## Scheduling for 5 AM Eastern

Cron does not understand time zones directly. Set the `TZ` variable to Eastern time and schedule for 5 AM:

```cron
TZ=America/New_York
0 5 * * * /home/alan-delong/repos/sportsbook-picks/.venv/bin/python \
  /home/alan-delong/repos/sportsbook-picks/src/pick_collector.py \
  --output /home/alan-delong/repos/sportsbook-picks/output/top_picks.html \
  --limit 20 >> /home/alan-delong/repos/sportsbook-picks/log.txt 2>&1
```

Adjust the paths if you place the repository elsewhere. The redirect keeps a rolling log of each run.

## Development Notes

- The parser looks for lines that start with `Pick:`, `Sport:`, `Time:` and `Wager:` (case insensitive).
- Leading unit sizes on the `Pick:` line (e.g. `1.5U`) backfill the `Recommended Wager` column when no explicit wager field is present.
- Heuristics tease apart the pick text into a `Game / Match` and a `Bet` column (splitting on phrases like "over", "under", "to win", spreads, etc.); edge cases fall back to leaving `Game / Match` blank.
- `Game:`/`Event:` style lines populate the matchup column, and bare `Team vs Team` lines are detected as matchups as well; trailing kickoff times are redirected into the `Time` column automatically.
- Totals/props without a clear matchup stay in the `Bet` column—the parser avoids treating lines that start with bet keywords (`over`, `under`, `ML`, etc.) as games.
- Ranking uses a Bayesian-smoothed win percentage (Beta(5,5) prior) so large samples outrank tiny perfect records; both raw and adjusted rates are displayed.
- Records formatted as `12-5` or `18-9-2` are used to compute win percentage; comments without a record or pick fields are skipped.
- Three-value records are treated as Wins–Pushes–Losses by default, with support for explicit `W-L-P`/`W-L-T` labels.
- The script only touches the public JSON endpoints. To add authenticated API usage later, wrap `RedditClient` with a PRAW-based implementation.
- Set `SUBREDDITS="sportsbook,sportsbetting=title:\"Best Bets\""` in `bin/run_collector.sh` (or the environment) to mirror the CLI example above.

## Output Format

The generated HTML includes: author, record, raw win percentage, adjusted win percentage, subreddit, thread title, game/match, bet detail, sport, event time, suggested wager, and a permalink back to the Reddit comment.
