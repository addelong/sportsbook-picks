#!/usr/bin/env python3
"""Collect top picks from Reddit betting threads (Pick of the Day, Best Bets, etc.)."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import sys
import urllib.parse
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

REDDIT_BASE = "https://www.reddit.com"
SEARCH_URL_TEMPLATE = (
    "{base}/r/{sub}/search.json?q={query}&restrict_sr=1&sort=new&limit=1"
)
COMMENTS_URL = "{base}/comments/{post_id}.json?limit=500"
USER_AGENT = "sportsbook-picks-bot/0.1 (by u/your_username)"

RE_RECORD = re.compile(r"\b(\d{1,3})-(\d{1,3})(?:-(\d{1,3}))?\b")
BETA_ALPHA = 5.0
BETA_BETA = 5.0
DEFAULT_TITLE_QUERIES = {
    "sportsbook": 'title:"Pick of the Day"',
    "sportsbetting": 'title:"Best Bets"',
}
FIELD_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    "pick": [
        re.compile(
            r"^\s*(?:[^\w\s]+|\d+\.)*\s*(?:pick|play|potd|best\s+bets?|today'?s\s+pick|todays\s+pick|selection|bet(?:\s+on)?)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
            re.I,
        ),
    ],
    "game": [
        re.compile(
            r"^\s*[-•*>\u2022\u2013\u2014]*\s*(?:game|event|match(?:up)?|fixture)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
            re.I,
        ),
    ],
    "sport": [
        re.compile(
            r"^\s*[-•*>\u2022\u2013\u2014]*\s*(?:sport(?:\s*\|\s*league)?|league)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
            re.I,
        ),
    ],
    "time": [
        re.compile(
            r"^\s*[-•*>\u2022\u2013\u2014]*\s*(?:date/?time|date\s*&\s*time|time|kick(?:-?off)?(?:\s*time)?|start(?:\s*time)?|event\s*time)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
            re.I,
        ),
    ],
    "recommended_wager": [
        re.compile(
            r"^\s*[-•*>\u2022\u2013\u2014]*\s*(?:units?(?:\s+played)?|unit\s*size|stake|risk|(?:recommended\s*)?wager|bet\s*size|investment|units?\s*risked)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
            re.I,
        ),
    ],
}

AUX_FIELD_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    "odds": [re.compile(r"^\s*(?:odds?|line|price)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$", re.I)],
    "book": [re.compile(r"^\s*(?:book(?:ie)?|sportsbook)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$", re.I)],
}

STAKE_PREFIX = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(u|units?|unit)\b[:\s-]*", re.I)
TRAILING_STAKE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:u|units?|unit)\b(?:\s*to\s*win\s*\d+(?:\.\d+)?\s*(?:u|units?|unit)\b)?\s*$",
    re.I,
)
MARKDOWN_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\([^\)]+\)")
SPLIT_MARKERS = [
    re.compile(r"\b(over|under)\b", re.I),
    re.compile(r"\b(to\s+win)\b", re.I),
    re.compile(r"\b(tt\s+over|tt\s+under)\b", re.I),
    re.compile(r"\b(ml|moneyline)\b", re.I),
    re.compile(r"\b(btts)\b", re.I),
    re.compile(r"\b([-+]\s?\d+(?:\.\d+)?)"),
]
TIME_IN_TEXT = re.compile(
    r"(\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)\s*(?:[A-Z]{2,5})?)",
    re.I,
)
LIKELY_GAME_TEXT = re.compile(r"(?:\bvs?\.?\b|\bversus\b|\bat\b|@)", re.I)
BET_PREFIXES = (
    "over",
    "under",
    "total",
    "team total",
    "tt",
    "ml",
    "moneyline",
    "btts",
    "spread",
    "line",
    "runline",
    "puckline",
    "draw",
    "double chance",
    "parlay",
    "both teams",
    "o",
    "u",
)


@dataclass
class SourceConfig:
    subreddit: str
    query: str


@dataclass
class RecordStats:
    wins: int
    losses: int
    pushes: int
    display: str


@dataclass
class PickEntry:
    author: str
    wins: int
    losses: int
    pushes: int
    win_pct: float
    adjusted_pct: float
    source: str
    thread_title: str
    record_display: str
    game: Optional[str]
    pick: Optional[str]
    sport: Optional[str]
    time: Optional[str]
    recommended_wager: Optional[str]
    permalink: str

    def record_text(self) -> str:
        return self.record_display

    def as_row(self) -> List[str]:
        return [
            self.author,
            self.record_text(),
            f"{self.win_pct:.3f}",
            f"{self.adjusted_pct:.3f}",
            self.source,
            self.thread_title,
            self.game or "",
            self.pick or "",
            self.sport or "",
            self.time or "",
            self.recommended_wager or "",
            self.permalink,
        ]


class RedditClient:
    """Minimal reddit client that only hits public JSON endpoints."""

    def __init__(self, user_agent: str = USER_AGENT, timeout: int = 10) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def fetch_latest_thread(self, subreddit: str, query: str, base: str = REDDIT_BASE) -> dict:
        encoded_query = urllib.parse.quote(query)
        url = SEARCH_URL_TEMPLATE.format(base=base, sub=subreddit, query=encoded_query)
        response = self.session.get(url, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch search results: {response.status_code}")
        payload = response.json()
        children = payload.get("data", {}).get("children", [])
        if not children:
            raise RuntimeError("No Pick of the Day thread found")
        return children[0]["data"]

    def fetch_comments(self, post_id: str, base: str = REDDIT_BASE) -> List[dict]:
        url = COMMENTS_URL.format(base=base, post_id=post_id)
        response = self.session.get(url, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch comments: {response.status_code}")
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError("Unexpected comments payload")
        return payload[1]["data"]["children"]


def parse_record(text: str) -> Optional[RecordStats]:
    match = RE_RECORD.search(text)
    if not match:
        return None

    first = int(match.group(1))
    second = int(match.group(2))
    third_raw = match.group(3)
    third = int(third_raw) if third_raw is not None else None

    if first + second + (third or 0) == 0:
        return None

    window_after = text[match.end() : match.end() + 30].lower()
    window_before = text[max(0, match.start() - 30) : match.start()].lower()
    descriptor = f"{window_before} {window_after}"
    display = match.group(0)

    # Default convention: Wins – Pushes – Losses
    if third is not None:
        wins = first
        pushes = second
        losses = third

        if any(tag in descriptor for tag in ("w-l-p", "w l p", "w-l-t", "w l t", "w-l-d", "w l d")):
            pushes = third
            losses = second

        return RecordStats(wins=wins, losses=losses, pushes=pushes, display=display)

    # Two-value records: wins-losses
    wins = first
    losses = second
    pushes = 0
    return RecordStats(wins=wins, losses=losses, pushes=pushes, display=display)


def compute_win_pct(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 0.0
    return wins / total


def compute_adjusted_pct(wins: int, losses: int, alpha: float = BETA_ALPHA, beta: float = BETA_BETA) -> float:
    total = wins + losses
    return (wins + alpha) / (total + alpha + beta) if total >= 0 else 0.0


def clean_pick_text(text: str) -> str:
    text = MARKDOWN_LINK.sub(lambda m: m.group("label"), text)
    text = text.replace("**", "").replace("*", "")
    text = re.sub(r"[`_]+", "", text)
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def peel_trailing_parenthetical(detail: str) -> Tuple[str, Optional[str]]:
    stripped = detail.rstrip()
    match = re.search(r"\(([^)]*\b\d+(?:\.\d+)?\s*(?:u|units?|unit)[^)]*)\)\s*$", stripped, re.I)
    if not match:
        return detail, None
    stake = match.group(1).strip()
    trimmed = stripped[: match.start()].rstrip(",; -")
    return trimmed, stake


def peel_trailing_stake(detail: str) -> Tuple[str, Optional[str]]:
    detail, paren_stake = peel_trailing_parenthetical(detail)
    if paren_stake:
        return detail, paren_stake
    stripped = detail.rstrip()
    match = TRAILING_STAKE.search(stripped)
    if not match:
        return detail, None
    stake = stripped[match.start() :].strip(" ,;-")
    trimmed = stripped[: match.start()].rstrip(" ,;-")
    return trimmed, stake


def looks_like_bet_prefix(text: str) -> bool:
    lowered = text.strip().lower().lstrip("'\"")
    if not lowered:
        return False
    if lowered[0].isdigit() or lowered[0] in "+-":
        return True
    return any(lowered.startswith(prefix) for prefix in BET_PREFIXES)


def looks_like_plain_matchup(text: str) -> Optional[str]:
    candidate = text.strip()
    if not candidate:
        return None
    if not re.search(r"[a-zA-Z]", candidate):
        return None
    lowered = candidate.lower()
    if any(word in lowered for word in ("record", "analysis", "units", "bet", "odds", "stake", "roi", "notes")):
        return None
    if LIKELY_GAME_TEXT.search(candidate):
        return re.sub(r"\s+", " ", candidate)
    return None


def looks_like_sport_line(text: str) -> Optional[str]:
    candidate = text.strip()
    if not candidate:
        return None
    if "|" in candidate and len(candidate) <= 80:
        return re.sub(r"\s+", " ", candidate)
    letters = re.sub(r"[^A-Z]", "", candidate.upper())
    if candidate.upper() == candidate and 3 <= len(candidate) <= 40 and len(letters) >= 4:
        return candidate.title()
    return None


def normalize_query(subreddit: str, query: Optional[str]) -> SourceConfig:
    base_query = DEFAULT_TITLE_QUERIES.get(subreddit.lower(), 'title:"Pick of the Day"')
    trimmed = (query or "").strip()
    if not trimmed:
        normalized = base_query
    else:
        normalized = trimmed if trimmed.lower().startswith("title:") else f'title:"{trimmed}"'
    return SourceConfig(subreddit=subreddit, query=normalized)


def parse_subreddit_specs(specs: Optional[List[str]]) -> List[SourceConfig]:
    if not specs:
        return [normalize_query("sportsbook", None)]
    configs: List[SourceConfig] = []
    for spec in specs:
        if "=" in spec:
            name, raw_query = spec.split("=", 1)
        else:
            name, raw_query = spec, ""
        name = name.strip()
        if not name:
            raise ValueError("Subreddit name cannot be empty")
        configs.append(normalize_query(name, raw_query))
    return configs


def split_game_and_detail(text: str) -> Tuple[Optional[str], Optional[str]]:
    for pattern in SPLIT_MARKERS:
        match = pattern.search(text)
        if match:
            idx = match.start()
            game = text[:idx].strip(" ,;-@/\t")
            detail = text[idx:].strip()
            if game:
                if looks_like_bet_prefix(game):
                    return None, text.strip() or None
                return game, detail or None
    ou_match = re.search(r"\s[ou][0-9]", text, re.I)
    if ou_match:
        idx = ou_match.start()
        game = text[:idx].strip(" ,;-@/\t")
        detail = text[idx + 1 :].strip()
        if game:
            if looks_like_bet_prefix(game):
                return None, text.strip() or None
            return game, detail or None
    if "(" in text and LIKELY_GAME_TEXT.search(text):
        idx = text.index("(")
        game = text[:idx].strip(" ,;-@/\t")
        if game:
            if looks_like_bet_prefix(game):
                return None, text.strip() or None
            return game, text[idx:].strip() or None
    if "," in text:
        head, tail = text.split(",", 1)
        head = head.strip()
        tail = tail.strip()
        if head and tail:
            if looks_like_bet_prefix(head):
                return None, text.strip() or None
            return head, tail
    return None, text.strip() or None


def split_game_and_time(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    match = TIME_IN_TEXT.search(text)
    if match:
        start = match.start()
        if start >= len(text) // 3:
            time_text = text[start:].strip()
            game_text = text[:start].strip(" ,;-@/\t")
            if game_text:
                return game_text, time_text
    return text, None


def parse_pick_text(raw: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not raw:
        return None, None, None
    cleaned = clean_pick_text(raw)
    if not cleaned:
        return None, None, None
    stake = None
    stake_match = STAKE_PREFIX.match(cleaned)
    if stake_match:
        stake = stake_match.group(0).strip()
        cleaned = cleaned[stake_match.end() :].strip()
    cleaned = cleaned.lstrip("-: ")
    game, detail = split_game_and_detail(cleaned)
    if detail:
        detail, trailing_stake = peel_trailing_stake(detail)
        if trailing_stake and not stake:
            stake = trailing_stake
    return game, detail, stake


def _is_field_line(line: str) -> bool:
    for patterns in FIELD_PATTERNS.values():
        for pattern in patterns:
            if pattern.match(line):
                return True
    for patterns in AUX_FIELD_PATTERNS.values():
        for pattern in patterns:
            if pattern.match(line):
                return True
    return False


def _next_non_empty(lines: List[str], start_index: int, skip_field_lines: bool = True) -> Optional[str]:
    for idx in range(start_index, len(lines)):
        candidate_line = lines[idx]
        candidate = candidate_line.strip()
        if not candidate:
            continue
        if skip_field_lines and _is_field_line(candidate_line):
            continue
        return candidate
    return None


def extract_pick_fields(lines: Iterable[str]) -> dict:
    material = list(lines)
    result = {
        "pick": None,
        "game": None,
        "sport": None,
        "time": None,
        "recommended_wager": None,
    }
    aux: Dict[str, Optional[str]] = {"odds": None, "book": None}

    for idx, line in enumerate(material):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("last pick"):
            continue
        captured = False
        for key, patterns in FIELD_PATTERNS.items():
            if result[key]:
                continue
            for pattern in patterns:
                match = pattern.match(line)
                if match:
                    value = match.group(1).strip()
                    if not value:
                        value = _next_non_empty(material, idx + 1) or ""
                        if value and _is_field_line(value):
                            value = ""
                        value = value.strip()
                    if value:
                        result[key] = value
                    captured = True
                    break
            if captured:
                break
        if captured:
            continue
        if not result["time"] and stripped.lower().startswith("date"):
            remainder = stripped.split(":", 1)[1] if ":" in stripped else stripped[4:]
            remainder = remainder.strip(" -:\t")
            if remainder:
                result["time"] = remainder
                continue
        if not result["game"]:
            matchup = looks_like_plain_matchup(stripped)
            if matchup:
                result["game"] = matchup
                continue
        if not result["sport"]:
            sport_line = looks_like_sport_line(stripped)
            if sport_line:
                result["sport"] = sport_line
                continue
        for key, patterns in AUX_FIELD_PATTERNS.items():
            if aux.get(key):
                continue
            for pattern in patterns:
                match = pattern.match(line)
                if match:
                    value = match.group(1).strip()
                    if not value:
                        value = _next_non_empty(material, idx + 1) or ""
                        if value and _is_field_line(value):
                            value = ""
                        value = value.strip()
                    if value:
                        aux[key] = value
                    break

    if result["pick"]:
        game, detail, stake = parse_pick_text(result["pick"])
        if game:
            if result["game"]:
                if detail:
                    detail = f"{game} {detail}" if not detail.lower().startswith(game.lower()) else detail
                else:
                    detail = game
            else:
                if detail:
                    detail = f"{game} {detail}" if not detail.lower().startswith(game.lower()) else detail
                else:
                    detail = game
        if detail:
            result["pick"] = detail
        if stake and not result["recommended_wager"]:
            result["recommended_wager"] = stake

    if result["game"]:
        game_text, inferred_time = split_game_and_time(result["game"])
        sport_from_game = None
        if game_text and "(" in game_text and ")" in game_text:
            match = re.search(r"\(([^)]+)\)\s*$", game_text)
            if match:
                maybe_sport = match.group(1).strip()
                cleaned_sport = looks_like_sport_line(maybe_sport)
                sport_from_game = cleaned_sport or (
                    maybe_sport.title()
                    if maybe_sport.isupper() and 2 <= len(maybe_sport) <= 40
                    else None
                )
                game_text = game_text[: match.start()].rstrip(" ,;-@/\t")
        result["game"] = game_text
        if inferred_time and not result["time"]:
            result["time"] = inferred_time
        if sport_from_game and not result["sport"]:
            result["sport"] = sport_from_game

    odds = aux.get("odds")
    if result["pick"] and odds and odds.lower() not in result["pick"].lower():
        result["pick"] = f"{result['pick']} @ {odds}"
    book = aux.get("book")
    if result["pick"] and book and book.lower() not in result["pick"].lower():
        result["pick"] = f"{result['pick']} ({book})"
    if result["pick"] and result["sport"] and result["pick"].strip().lower() == result["sport"].strip().lower():
        result["pick"] = None

    return result


def flatten_comments(comments: Iterable[dict]) -> Iterable[dict]:
    for comment in comments:
        data = comment.get("data", {})
        if data.get("body"):
            yield data
        for reply in data.get("replies", {}).get("data", {}).get("children", []) if isinstance(data.get("replies"), dict) else []:
            yield from flatten_comments([reply])


def collect_picks(
    comments: Iterable[dict],
    base_permalink: str,
    source: str,
    thread_title: str,
) -> List[PickEntry]:
    picks: List[PickEntry] = []
    for comment in comments:
        body = comment.get("body", "")
        record = parse_record(body)
        if not record:
            continue
        wins = record.wins
        losses = record.losses
        pushes = record.pushes
        fields = extract_pick_fields(body.splitlines())
        if not any(value for key, value in fields.items() if key != "game"):
            continue
        win_pct = compute_win_pct(wins, losses)
        adjusted_pct = compute_adjusted_pct(wins, losses)
        permalink = f"{REDDIT_BASE}{comment.get('permalink', base_permalink)}"
        picks.append(
            PickEntry(
                author=comment.get("author", "unknown"),
                wins=wins,
                losses=losses,
                pushes=pushes,
                win_pct=win_pct,
                adjusted_pct=adjusted_pct,
                source=source,
                thread_title=thread_title,
                record_display=record.display,
                game=fields["game"],
                pick=fields["pick"],
                sport=fields["sport"],
                time=fields["time"],
                recommended_wager=fields["recommended_wager"],
                permalink=permalink,
            )
        )
    picks.sort(
        key=lambda p: (
            p.adjusted_pct,
            p.wins + p.losses,
            p.win_pct,
        ),
        reverse=True,
    )
    return picks


def _esc(value: Optional[str]) -> str:
    return escape(value, quote=True) if value else ""


def to_html(picks: List[PickEntry], title: str) -> str:
    now = dt.datetime.now(dt.timezone.utc).astimezone()
    safe_title = escape(title, quote=True)
    rows = "".join(
        "<tr>"
        f"<td>{_esc(p.author)}</td>"
        f"<td>{_esc(p.record_text())}</td>"
        f"<td>{p.win_pct:.3f}</td>"
        f"<td>{p.adjusted_pct:.3f}</td>"
        f"<td>{_esc(p.source)}</td>"
        f"<td>{_esc(p.thread_title)}</td>"
        f"<td>{_esc(p.game)}</td>"
        f"<td>{_esc(p.pick)}</td>"
        f"<td>{_esc(p.sport)}</td>"
        f"<td>{_esc(p.time)}</td>"
        f"<td>{_esc(p.recommended_wager)}</td>"
        f"<td><a href='{escape(p.permalink, quote=True)}'>link</a></td>"
        "</tr>"
        for p in picks
    )
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>{safe_title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
th {{ background-color: #f5f5f5; }}
caption {{ margin-bottom: 1rem; font-size: 1.1rem; font-weight: bold; }}
</style>
</head>
<body>
<table>
<caption>{safe_title} — Generated {now.strftime('%Y-%m-%d %H:%M %Z')}</caption>
<thead>
<tr><th>Author</th><th>Record</th><th>Win %</th><th>Adj Win %</th><th>Subreddit</th><th>Thread</th><th>Game / Match</th><th>Bet</th><th>Sport</th><th>Time</th><th>Recommended Wager</th><th>Permalink</th></tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
"""


def write_output(picks: List[PickEntry], output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(to_html(picks, title), encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subreddit",
        dest="subreddits",
        action="append",
        help=(
            "Subreddit to search; repeatable."
            " Use the form 'name' or 'name=Pick of the Day' or 'name=title:\"Best Bets\"'."
        ),
    )
    parser.add_argument("--output", default="output/top_picks.html", help="Output HTML file path")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of picks to include")
    parser.add_argument(
        "--base-url",
        default=REDDIT_BASE,
        help="Base Reddit URL (allowing corporate proxies / mirrors)",
    )
    parser.add_argument(
        "--user-agent",
        default=USER_AGENT,
        help="Custom User-Agent string to avoid 429s",
    )
    parser.add_argument(
        "--thread-url",
        default=None,
        help="Optional explicit thread permalink (.json will be fetched)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    return parser.parse_args(argv)


def thread_from_url(url: str) -> tuple[str, str, Optional[str]]:
    id_match = re.search(r"comments/([a-z0-9]+)/", url)
    if not id_match:
        raise ValueError("Could not extract thread id from URL")
    subreddit_match = re.search(r"/r/([^/]+)/comments/", url)
    subreddit = subreddit_match.group(1) if subreddit_match else None
    post_id = id_match.group(1)
    return post_id, url, subreddit


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.debug("Parsed arguments: %s", args)
    client = RedditClient(user_agent=args.user_agent)

    all_picks: List[PickEntry] = []
    thread_titles: List[str] = []

    if args.thread_url:
        post_id, permalink, subreddit = thread_from_url(args.thread_url)
        if not post_id:
            raise RuntimeError("Could not determine post id from thread URL")
        try:
            logger.info("Fetching comments for provided thread %s", post_id)
            comments_json = client.fetch_comments(post_id, base=args.base_url)
        except RuntimeError as exc:
            print(f"Failed to fetch comments for provided thread URL: {exc}", file=sys.stderr)
            return 1
        flattened = list(flatten_comments(comments_json))
        logger.info("Flattened %d top-level comments from custom thread", len(flattened))
        parsed = urllib.parse.urlparse(permalink)
        base_permalink = parsed.path or "/"
        thread_title = f"Custom thread ({subreddit or 'reddit'})"
        source_label = subreddit or "custom"
        picks = collect_picks(
            flattened,
            base_permalink=base_permalink,
            source=source_label,
            thread_title=thread_title,
        )
        if picks:
            all_picks.extend(picks)
            thread_titles.append(thread_title)
            logger.info("Collected %d picks from provided thread", len(picks))
        else:
            logger.info("No picks collected from provided thread")
    else:
        try:
            source_configs = parse_subreddit_specs(args.subreddits)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        for config in source_configs:
            logger.info("Searching r/%s with query %s", config.subreddit, config.query)
            try:
                thread_data = client.fetch_latest_thread(
                    config.subreddit, config.query, base=args.base_url
                )
            except RuntimeError as exc:
                print(
                    f"Warning: failed to locate thread for r/{config.subreddit} using query '{config.query}': {exc}",
                    file=sys.stderr,
                )
                logger.info("No thread found for r/%s", config.subreddit)
                continue

            post_id = thread_data.get("id")
            if not post_id:
                print(
                    f"Warning: thread missing id for r/{config.subreddit}; skipping",
                    file=sys.stderr,
                )
                logger.info("Skipping r/%s because thread missing id", config.subreddit)
                continue

            try:
                logger.info("Fetching comments for r/%s post %s", config.subreddit, post_id)
                comments_json = client.fetch_comments(post_id, base=args.base_url)
            except RuntimeError as exc:
                print(
                    f"Warning: failed to fetch comments for r/{config.subreddit}: {exc}",
                    file=sys.stderr,
                )
                logger.info("Failed fetching comments for r/%s", config.subreddit)
                continue

            flattened = list(flatten_comments(comments_json))
            logger.info("Flattened %d comment nodes for r/%s", len(flattened), config.subreddit)
            thread_title = thread_data.get("title", f"r/{config.subreddit} thread")
            picks = collect_picks(
                flattened,
                base_permalink=thread_data.get("permalink", "/"),
                source=config.subreddit,
                thread_title=thread_title,
            )
            if picks:
                all_picks.extend(picks)
                thread_titles.append(thread_title)
                logger.info("Collected %d picks from r/%s", len(picks), config.subreddit)
            else:
                logger.info("No picks collected from r/%s thread", config.subreddit)

    if not all_picks:
        print("No picks found with record + pick information", file=sys.stderr)
        return 1

    logger.info("Collected a total of %d picks before limiting", len(all_picks))

    all_picks.sort(
        key=lambda p: (
            p.adjusted_pct,
            p.wins + p.losses,
            p.win_pct,
        ),
        reverse=True,
    )

    if args.limit:
        all_picks = all_picks[: args.limit]

    if thread_titles and len(set(thread_titles)) == 1:
        report_title = thread_titles[0]
    elif thread_titles:
        report_title = "Reddit Top Picks"
    else:
        report_title = "Top Picks"

    write_output(all_picks, Path(args.output), report_title)
    logger.info("Wrote %d picks to %s with title %s", len(all_picks), args.output, report_title)
    print(f"Wrote {len(all_picks)} picks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
