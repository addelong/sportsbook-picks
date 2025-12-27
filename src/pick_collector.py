#!/usr/bin/env python3
"""Collect top picks from Reddit betting threads (Pick of the Day, Best Bets, etc.)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
import unicodedata
import urllib.parse
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

REDDIT_BASE = "https://www.reddit.com"
SEARCH_URL_TEMPLATE = (
    "{base}/r/{sub}/search.json?q={query}&restrict_sr=1&sort=new&limit=1"
)
COMMENTS_URL = "{base}/comments/{post_id}.json?limit=500"
USER_AGENT = "sportsbook-picks-bot/0.1 (by u/your_username)"

RE_RECORD = re.compile(
    r"\b(\d{1,3})\s*[-\u2013\u2014\u2212]\s*(\d{1,3})(?:\s*[-\u2013\u2014\u2212]\s*(\d{1,3}))?\b"
)
RE_RECORD_LINE = re.compile(r"\brecord\b", re.I)
RE_PARENTHESES_PUSH = re.compile(
    r"^\s*\(\s*(\d{1,3})\s*(?:push(?:es)?|ties?|draws?)\s*\)"
    r"|^\s*\(\s*(\d{1,3})\s*\)"
    r"|^\s*[\-–—]\s*(\d{1,3})\s*(?:push(?:es)?|ties?|draws?)\b",
    re.I,
)
RE_LETTER_RECORD_TOKEN = re.compile(
    r"(\d{1,3})\s*(w(?:ins?)?|l(?:oss(?:es)?)?|p(?:ush(?:es)?)?|d(?:raws?)?|t(?:ies?|ie)?)",
    re.I,
)
BETA_ALPHA = 5.0
BETA_BETA = 5.0
DEFAULT_TITLE_QUERIES = {
    "sportsbook": 'title:"Pick of the Day"',
    "sportsbetting": 'title:("Best Bet" OR "Best Bets" OR "Best Bet Thread")',
}
PICK_KEY_TOKENS = {
    "pick",
    "play",
    "potd",
    "today's pick",
    "todays pick",
    "selection",
    "bet",
}
FIELD_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    "pick": [
        re.compile(
            r"^\s*(?:[\W_]+|\d+\.)*\s*(?:pick(?:\s+of\s+the\s+day)?|play|potd|best\s+bets?|today'?s\s+pick|todays\s+pick|selection|bet(?:\s+on)?)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
            re.I,
        ),
        re.compile(
            r"^\s*today'?s\s+potd\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
            re.I,
        ),
        re.compile(
            r"^\s*(?:parlay|straight|single)\s+bet\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
            re.I,
        ),
        re.compile(
            r"^\s*(?:today'?s|todays)\s+(?:bet|play)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
            re.I,
        ),
        re.compile(
            r"^\s*next\s+(?:bet|play|pick)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
            re.I,
        ),
    ],
    "game": [
        re.compile(
            r"^\s*[-•*>\u2022\u2013\u2014]*\s*(?:game|event(?:/s)?|match(?:up)?|fixture)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
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
            r"^\s*[-•*>\u2022\u2013\u2014]*\s*(?:date/?time|date\s*&\s*time|time|kick(?:-?off)?(?:\s*time)?|start(?:\s*time)?|event\s*time)\b\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
            re.I,
        ),
    ],
    "recommended_wager": [
        re.compile(
            r"^\s*[-•*>\u2022\u2013\u2014]*\s*(?:units?(?:\s+played)?|unit\s*size|stake|risk|(?:recommended\s*)?wager(?:\s+amount)?|bet\s*size|investment|units?\s*risked)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
            re.I,
        ),
    ],
}

AUX_FIELD_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    "odds": [re.compile(r"^\s*(?:odds?|line|price)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$", re.I)],
    "book": [re.compile(r"^\s*(?:book(?:ie)?|sportsbook)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$", re.I)],
}

FIELD_KEYWORD_HINTS: Dict[str, Tuple[str, ...]] = {
    "pick": (
        "pick",
        "play",
        "potd",
        "best bet",
        "best bets",
        "parlay",
        "bet",
        "selection",
        "today's pick",
        "todays pick",
        "next pick",
        "next play",
    ),
    "game": ("game", "event", "match", "fixture"),
    "sport": ("sport", "league"),
    "time": ("time", "kick", "start", "date", "event time"),
    "recommended_wager": (
        "unit",
        "units",
        "stake",
        "risk",
        "wager",
        "bet size",
        "investment",
    ),
}

AUX_FIELD_KEYWORD_HINTS: Dict[str, Tuple[str, ...]] = {
    "odds": ("odds", "line", "price"),
    "book": ("book", "sportsbook"),
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
    re.compile(r"[-+]\s?\d+(?:\.\d+)?"),
]
TIME_IN_TEXT = re.compile(
    r"(\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)\s*(?:[A-Z]{2,5})?)",
    re.I,
)
LIKELY_GAME_TEXT = re.compile(r"(?:\bvs?\.?\b|\bversus\b|\bat\b|@|\bv\b)", re.I)


def _has_matchup_hint(text: Optional[str]) -> bool:
    if not text:
        return False
    normalized = _normalize_ascii(text)
    lowered = normalized.lower()
    if re.search(r"\bvs?\.?\b", lowered) or "versus" in lowered:
        return True
    if re.search(r"\bv\s+[a-z]", lowered):
        return True
    if re.search(r"\bat\s+[a-z]", lowered):
        return True
    if re.search(r"@\s*[A-Za-z]", normalized):
        return True
    return False
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

EMOJI_SPORT_MAP = {
    "\U0001F3C8": "Football",
    "\U0001F3C0": "Basketball",
    "\u26BE": "Baseball",
    "\u26BD": "Soccer",
    "\u26BD\uFE0F": "Soccer",
    "\U0001F3D2": "Hockey",
    "\U0001F3CF": "Cricket",
    "\U0001F3C9": "Rugby",
    "\U0001F3BE": "Tennis",
    "\U0001F3CE": "Motorsport",
    "\U0001F3C1": "Motorsport",
    "\U0001F94A": "Boxing",
    "\U0001F94B": "Martial Arts",
}

COMMON_SPORT_TOKENS = {
    "afl",
    "atp",
    "boxing",
    "bundesliga",
    "cfl",
    "champions league",
    "college football",
    "college basketball",
    "cricket",
    "cs2",
    "csgo",
    "dota",
    "epl",
    "esports",
    "formula 1",
    "f1",
    "golf",
    "la liga",
    "laliga",
    "league of legends",
    "ligue 1",
    "lol",
    "mlb",
    "mlr",
    "mls",
    "mma",
    "motogp",
    "nba",
    "ncaab",
    "ncaaf",
    "ncaa",
    "nhl",
    "nfl",
    "nrl",
    "pga",
    "premier league",
    "rugby",
    "serie a",
    "soccer",
    "tennis",
    "ufc",
    "valorant",
    "wnba",
}

# Team name to sport mapping for automatic sport detection
TEAM_NAME_SPORTS = {
    # NBA teams
    "lakers": "NBA", "warriors": "NBA", "celtics": "NBA", "nets": "NBA",
    "knicks": "NBA", "76ers": "NBA", "sixers": "NBA", "bulls": "NBA",
    "heat": "NBA", "bucks": "NBA", "nuggets": "NBA", "clippers": "NBA",
    "suns": "NBA", "mavericks": "NBA", "mavs": "NBA", "rockets": "NBA",
    "spurs": "NBA", "thunder": "NBA", "jazz": "NBA", "blazers": "NBA",
    "timberwolves": "NBA", "pelicans": "NBA", "grizzlies": "NBA", "kings": "NBA",
    "hawks": "NBA", "hornets": "NBA", "wizards": "NBA", "pacers": "NBA",
    "pistons": "NBA", "cavaliers": "NBA", "cavs": "NBA", "raptors": "NBA",
    "magic": "NBA",

    # NFL teams
    "49ers": "NFL", "bears": "NFL", "bengals": "NFL", "bills": "NFL",
    "broncos": "NFL", "browns": "NFL", "buccaneers": "NFL", "bucs": "NFL",
    "cardinals": "NFL", "chargers": "NFL", "chiefs": "NFL", "colts": "NFL",
    "cowboys": "NFL", "dolphins": "NFL", "eagles": "NFL", "falcons": "NFL",
    "giants": "NFL", "jaguars": "NFL", "jets": "NFL", "lions": "NFL",
    "packers": "NFL", "panthers": "NFL", "patriots": "NFL", "raiders": "NFL",
    "rams": "NFL", "ravens": "NFL", "saints": "NFL", "seahawks": "NFL",
    "steelers": "NFL", "texans": "NFL", "titans": "NFL", "vikings": "NFL",
    "commanders": "NFL",

    # English Premier League
    "arsenal": "Soccer", "liverpool": "Soccer", "chelsea": "Soccer", "manchester": "Soccer",
    "tottenham": "Soccer", "spurs": "Soccer", "villa": "Soccer", "wolves": "Soccer",
    "brighton": "Soccer", "newcastle": "Soccer", "fulham": "Soccer", "brentford": "Soccer",
    "everton": "Soccer", "leicester": "Soccer", "leeds": "Soccer", "southampton": "Soccer",
    "burnley": "Soccer", "watford": "Soccer",

    # Scottish Football
    "hibernian": "Soccer", "hearts": "Soccer", "celtic": "Soccer", "rangers": "Soccer",
}


def _detect_sport_from_team_names(text: str) -> Optional[str]:
    """Detect sport by looking for known team names in the text."""
    if not text:
        return None
    lowered = text.lower()
    for team_name, sport in TEAM_NAME_SPORTS.items():
        if team_name in lowered:
            return sport
    return None


def _normalize_ascii(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.replace("–", "-").replace("—", "-").replace("−", "-")
    return normalized.encode("ascii", "ignore").decode("ascii")


def _should_replace_game(existing: Optional[str], candidate: Optional[str]) -> bool:
    if not candidate:
        return False
    candidate_clean = candidate.strip()
    if not candidate_clean:
        return False
    lowered_candidate = candidate_clean.lower()
    if lowered_candidate.startswith(
        (
            "he ",
            "she ",
            "i ",
            "we ",
            "they ",
            "last ",
            "record",
            "profit",
            "bonus",
            "today ",
            "todays ",
            "write up",
            "write-up",
            "note ",
        )
    ):
        return False
    if not existing or not existing.strip():
        return True
    existing_clean = existing.strip()
    if candidate_clean.lower() == existing_clean.lower():
        return False
    candidate_has = _has_matchup_hint(candidate_clean)
    existing_has = _has_matchup_hint(existing_clean)
    if candidate_has and not existing_has:
        return True
    if candidate_has and existing_has:
        if candidate_clean.lower().startswith(("pick", "potd")) and not existing_clean.lower().startswith(
            ("pick", "potd")
        ):
            return False
        if not candidate_clean.lower().startswith(("pick", "potd")) and existing_clean.lower().startswith(
            ("pick", "potd")
        ):
            return True
        if len(candidate_clean) > len(existing_clean) + 3:
            return True
    if not candidate_has and existing_has:
        return False
    if candidate_clean.lower().startswith(("pick", "potd")):
        return False
    return len(candidate_clean) > len(existing_clean)


def _should_replace_sport(existing: Optional[str], candidate: Optional[str]) -> bool:
    if not candidate:
        return False
    cleaned_candidate = _normalize_ascii(candidate).strip().lstrip("*_ ")
    lowered_candidate = cleaned_candidate.lower()
    if lowered_candidate.startswith(
        (
            "bonus",
            "record",
            "last pick",
            "write up",
            "write-up",
            "today",
            "todays",
            "units",
        )
    ):
        return False
    if not existing or not existing.strip():
        return True
    existing_clean = existing.strip().lower()
    candidate_clean = cleaned_candidate.lower()
    if existing_clean == candidate_clean:
        return False
    low_quality = {"esports", "parlay", "unknown"}
    if existing_clean in low_quality:
        return True
    if candidate_clean in {"ncaaf", "college football"} and existing_clean in {"nfl", "football"}:
        return True
    if candidate_clean == "soccer" and existing_clean in {"football", "nfl"}:
        return True
    if candidate_clean.startswith("uefa") and existing_clean in {"football", "nfl"}:
        return True
    return False


def _is_summary_units_value(value: str, context: str) -> bool:
    lowered_context = context.lower()
    if any(token in lowered_context for token in ("units won", "net", "profit", "roi", "record", "balance")):
        return True
    if value.startswith(("+", "-")) and "to win" not in lowered_context:
        return True
    return False


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
    if match:
        window_after = text[match.end() : match.end() + 30].lower()
        window_before = text[max(0, match.start() - 30) : match.start()].lower()
        descriptor = f"{window_before} {window_after}"

        first = int(match.group(1))
        second = int(match.group(2))
        third_raw = match.group(3)
        third = int(third_raw) if third_raw is not None else None

        total = first + second + (third or 0)
        if total == 0 and "record" not in descriptor:
            return None

        display_end = match.end()

        remainder = text[display_end : display_end + 40]
        paren_match = RE_PARENTHESES_PUSH.match(remainder)
        if paren_match and third is None:
            third_candidate = next(
                int(group)
                for group in paren_match.groups()
                if group is not None
            )
            third = third_candidate
            display_end += paren_match.end()

        display = text[match.start() : display_end].strip()

        if third is not None:
            wins = first
            second_val = second
            third_val = third

            if any(tag in descriptor for tag in ("w-d-l", "w d l", "w-draw-l", "w draw l")):
                return RecordStats(wins=wins, losses=third_val, pushes=second_val, display=display)

            if any(tag in descriptor for tag in ("w-l-p", "w l p", "w-l-t", "w l t", "w-l-d", "w l d")):
                return RecordStats(wins=wins, losses=second_val, pushes=third_val, display=display)

            pushes = min(second_val, third_val)
            losses = max(second_val, third_val)
            return RecordStats(wins=wins, losses=losses, pushes=pushes, display=display)

        # Two-value records: wins-losses
        wins = first
        losses = second
        pushes = 0
        return RecordStats(wins=wins, losses=losses, pushes=pushes, display=display)

    record_line_match = RE_RECORD_LINE.search(text)
    if record_line_match:
        search_start = record_line_match.start()
        search_end = min(len(text), search_start + 120)
    else:
        search_start = 0
        search_end = min(len(text), 120)
    segment = text[search_start:search_end]
    token_matches = list(RE_LETTER_RECORD_TOKEN.finditer(segment))
    if not token_matches:
        return None

    counts: Dict[str, int] = {}
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    label_map = {
        "w": "wins",
        "win": "wins",
        "wins": "wins",
        "l": "losses",
        "loss": "losses",
        "losses": "losses",
        "p": "pushes",
        "push": "pushes",
        "pushes": "pushes",
        "t": "pushes",
        "tie": "pushes",
        "ties": "pushes",
        "d": "pushes",
        "draw": "pushes",
        "draws": "pushes",
    }

    for match in token_matches:
        value = int(match.group(1))
        raw_label = match.group(2).lower()
        normalized_label = label_map.get(raw_label)
        if not normalized_label:
            continue
        if normalized_label in counts:
            # Preserve the first observed value for each label to avoid noise.
            continue
        counts[normalized_label] = value
        if start_index is None:
            start_index = search_start + match.start()
        end_index = search_start + match.end()

    if "wins" not in counts or "losses" not in counts:
        return None

    pushes_value = counts.get("pushes", 0)
    wins_value = counts["wins"]
    losses_value = counts["losses"]
    total = wins_value + losses_value + pushes_value
    if total == 0:
        context_window = text[max(0, (start_index or 0) - 30) : (end_index or 0) + 30].lower()
        if "record" not in context_window:
            return None

    display_segment = text[start_index:end_index].strip() if start_index is not None and end_index is not None else ""
    display = display_segment or f"{wins_value}-{losses_value}{f'-{pushes_value}' if pushes_value else ''}"
    return RecordStats(wins=wins_value, losses=losses_value, pushes=pushes_value, display=display)


def compute_win_pct(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 0.0
    return wins / total


def compute_adjusted_pct(wins: int, losses: int, alpha: float = BETA_ALPHA, beta: float = BETA_BETA) -> float:
    total = wins + losses
    return (wins + alpha) / (total + alpha + beta) if total >= 0 else 0.0


def clean_pick_text(text: str) -> str:
    text = _normalize_ascii(text)
    text = MARKDOWN_LINK.sub(lambda m: m.group("label"), text)
    text = text.replace("**", "").replace("*", "")
    text = re.sub(r"[`_]+", "", text)
    text = re.sub(r"^[^A-Za-z0-9+\-]+", "", text)
    text = re.sub(r"\s+", " ", text)
    if text.count(")") > text.count("("):
        surplus = text.count(")") - text.count("(")
        for _ in range(surplus):
            idx = text.rfind(")")
            if idx == -1:
                break
            text = text[:idx] + text[idx + 1 :]
    return text.strip()


def peel_trailing_parenthetical(detail: str) -> Tuple[str, Optional[str]]:
    stripped = detail.rstrip()
    match = re.search(r"\(([^)]*\b\d+(?:\.\d+)?\s*(?:u|units?|unit)[^)]*)\)\s*$", stripped, re.I)
    if not match:
        return detail, None
    stake = _clean_stake_text(match.group(1))
    trimmed = stripped[: match.start()].rstrip(",; -")
    return trimmed, stake


def peel_trailing_stake(detail: str) -> Tuple[str, Optional[str]]:
    detail, paren_stake = peel_trailing_parenthetical(detail)
    if paren_stake:
        return detail, paren_stake
    stripped = detail.rstrip()
    stripped_for_match = stripped.rstrip(".! ")
    match = TRAILING_STAKE.search(stripped_for_match)
    if not match:
        alt_match = re.search(r"(\d+(?:\.\d+)?\s*(?:u|units?|unit))\s*\([^)]*\)\s*$", stripped, re.I)
        if not alt_match:
            return detail, None
        stake = alt_match.group(1)
        trimmed = stripped[: alt_match.start()].rstrip(" ,;-")
        return trimmed, _clean_stake_text(stake)
    stake = stripped_for_match[match.start() :].strip(" ,;-")
    trimmed = stripped_for_match[: match.start()].rstrip(" ,;-")
    return trimmed, _clean_stake_text(stake)


def _clean_stake_text(stake: Optional[str]) -> Optional[str]:
    if not stake:
        return None
    cleaned = stake.strip()
    cleaned = cleaned.lstrip("= ")
    # Remove trailing stake separators that sneak in from prefixes like "1u -"
    cleaned = cleaned.rstrip("-: ")
    paren_units = re.search(r"\(([^)]*?\b\d+(?:\.\d+)?\s*(?:u|units?|unit)\b[^)]*)\)\s*$", cleaned, re.I)
    if paren_units:
        inner = paren_units.group(1)
        inner_cleaned = _clean_stake_text(inner)
        if inner_cleaned:
            return inner_cleaned
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned).strip()
    return cleaned or None


def looks_like_bet_prefix(text: str) -> bool:
    lowered = text.strip().lower().lstrip("'\"")
    if not lowered:
        return False
    if lowered[0].isdigit() or lowered[0] in "+-":
        return True
    return any(lowered.startswith(prefix) for prefix in BET_PREFIXES)


def looks_like_plain_matchup(text: str) -> Optional[str]:
    original = text.strip()
    candidate = _normalize_ascii(original)
    if not candidate:
        return None
    if not re.search(r"[a-zA-Z]", candidate):
        return None
    if len(candidate) > 120:
        return None
    if candidate.count(".") >= 2:
        return None
    lowered = candidate.lower()
    if lowered.startswith("at ") and len(candidate) > 3 and not candidate[3].isalpha():
        return None
    if "at least" in lowered or "at most" in lowered:
        return None
    normalized_prefix = candidate.lstrip("_*•-> \t").lower()
    if normalized_prefix.startswith("week "):
        return None
    if re.search(r"\bbet\b", lowered):
        return None
    if any(
        word in lowered
        for word in ("record", "analysis", "units", "odds", "stake", "roi", "notes", "profit", "loss")
    ):
        return None
    if re.search(r"\b(gambling|responsibly|emotion|emotions|tail|tailed|fade|fading|thanks|thank|luck|thread|control|everyone)\b", lowered):
        return None
    if "best of luck" in lowered:
        return None
    if candidate.endswith(":"):
        return None
    match = LIKELY_GAME_TEXT.search(candidate)
    if match:
        # Require "at" separators to connect two textual entities (avoid phrases like "at +100")
        if " at " in lowered:
            at_match = re.search(r"\bat\b", candidate, re.I)
            if at_match:
                before = candidate[: at_match.start()].strip()
                after = candidate[at_match.end():].strip()
                if not before or not after or not before[-1].isalpha() or not after[0].isalpha():
                    return None
                before_words = [word for word in before.split() if word]
                after_words = [word for word in after.split() if word]
                if len(before_words) > 7 or len(after_words) > 7:
                    return None
                after_word = after.split(None, 1)[0].lower() if after.split() else ""
                if after_word in {"least", "most", "risk", "stake"}:
                    return None
                if after_word == "the":
                    if len(after_words) < 2 or not after_words[1][0].isalpha():
                        return None
        if match.group(0) == "@":
            after = candidate[match.end() :].lstrip()
            if not after or not after[0].isalpha():
                return None
        if match.group(0).strip().lower() == "at":
            after = candidate[match.end() :].strip()
            if after and after[0].isdigit():
                return None
        result = re.sub(r"\s+", " ", candidate)
        if result.lower().startswith("and "):
            result = result[4:].strip()
        return result
    if " - " in candidate:
        left, right = candidate.split(" - ", 1)
        if len(left.strip().split()) > 8 or len(right.strip().split()) > 8:
            return None
    if re.search(r"[A-Za-z].+\s-\s[A-Za-z]", candidate):
        # Avoid treating sport/league lines as matchups
        if _sport_token_from_text(candidate):
            return None
        result = re.sub(r"\s+", " ", candidate)
        if result.lower().startswith("and "):
            result = result[4:].strip()
        return result
    return None


def looks_like_record_heading(text: str) -> bool:
    candidate = text.strip()
    if not candidate:
        return False
    lowered = candidate.lower()
    if not RE_RECORD_LINE.search(lowered):
        return False
    if lowered.startswith(("record", "season record", "overall record")):
        return True
    if RE_RECORD.search(candidate):
        return True
    return False


def looks_like_sport_line(text: str) -> Optional[str]:
    candidate = _normalize_ascii(text).strip()
    if not candidate:
        return None
    lowered = candidate.lower()
    if any(term in lowered for term in ("profit", "loss", "summary", "result", "units")):
        return None
    if "record" in lowered or lowered.startswith(("net units", "units", "last pick", "previous pick")):
        return None
    if TIME_IN_TEXT.search(candidate):
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
    if not text:
        return None, None

    def finalize(game_candidate: Optional[str], detail_candidate: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        detail_clean = detail_candidate.strip() if detail_candidate else None
        if game_candidate:
            normalized = re.sub(r"\s+", " ", game_candidate).strip(" ,;-@/\t")
            if detail_clean:
                vs_match = re.search(r"\((?:vs\.?|v|versus)\s+[^)]+\)\s*$", detail_clean, re.I)
                if vs_match:
                    opponent_raw = vs_match.group(0)[1:-1].strip()
                    detail_clean = detail_clean[: vs_match.start()].rstrip(" ,;-@/\t") or None
                    if opponent_raw:
                        opponent_clean = re.sub(r"^(?:versus|vs\.?|v)\s+", "vs ", opponent_raw, flags=re.I)
                        opponent_clean = opponent_clean.replace("vs.", "vs").strip()
                        if not opponent_clean.lower().startswith("vs "):
                            opponent_clean = f"vs {opponent_clean}"
                        game_candidate = f"{game_candidate} {opponent_clean}".strip()
                        normalized = re.sub(r"\s+", " ", game_candidate).strip(" ,;-@/\t")
            matchup = looks_like_plain_matchup(normalized)
            if matchup:
                return matchup, detail_clean
            if _has_matchup_hint(normalized):
                return normalized, detail_clean
            if detail_clean and detail_clean.strip().startswith(("-", "+")) and any(ch.isalpha() for ch in normalized):
                return normalized, detail_clean
        fallback_detail = text.strip() or None
        if detail_clean and not fallback_detail:
            fallback_detail = detail_clean
        return None, fallback_detail

    dash_index = text.lower().find(" - ")
    if dash_index != -1 and " vs " in text.lower() and dash_index > text.lower().find(" vs "):
        game = text[:dash_index].strip(" ,;-@/\t")
        detail = text[dash_index + 3 :].strip()
        if game and looks_like_bet_prefix(game):
            return finalize(None, text)
        return finalize(game, detail)

    if "(" in text and _has_matchup_hint(text):
        idx = text.index("(")
        game = text[:idx].strip(" ,;-@/\t")
        if game and looks_like_bet_prefix(game):
            return finalize(None, text)
        if game and "|" in game:
            game_head, game_extra = game.split("|", 1)
            game = game_head.strip()
            extra = game_extra.strip(" ,;-@/\t")
            if extra:
                text = f"{extra} {text[idx:]}"
                return finalize(game, text)
        return finalize(game, text[idx:])

    ou_match = re.search(r"\s[ou][0-9]", text, re.I)
    if ou_match:
        idx = ou_match.start()
        game = text[:idx].strip(" ,;-@/\t")
        detail = text[idx + 1 :].strip()
        if game and "|" in game:
            game_head, game_extra = game.split("|", 1)
            game = game_head.strip()
            extra = game_extra.strip(" ,;-@/\t")
            if extra and detail:
                detail = f"{extra} {detail}".strip()
        if game and looks_like_bet_prefix(game):
            return finalize(None, text)
        return finalize(game, detail)

    for pattern in SPLIT_MARKERS:
        match = pattern.search(text)
        if match:
            idx = match.start()
            game = text[:idx].strip(" ,;-@/\t")
            detail = text[idx:].strip()
            if game and "|" in game:
                game_head, game_extra = game.split("|", 1)
                game = game_head.strip()
                extra = game_extra.strip(" ,;-@/\t")
                if extra and detail:
                    detail = f"{extra} {detail}".strip()
            if game and looks_like_bet_prefix(game):
                return finalize(None, text)
            if game and "+" in game and "+" not in detail:
                detail = f"{game} {detail}".strip()
            if detail.lower().startswith("to win"):
                parts = [segment.strip(" ,;-@/\t") for segment in re.split(r"\s*[-\u2013\u2014]\s*", game) if segment]
                if len(parts) >= 2:
                    candidate = parts[-1]
                    prefix = " - ".join(parts[:-1]).strip(" ,;-@/\t")
                    if candidate:
                        game = prefix or game
                        detail = f"{candidate} {detail}".strip()
            return finalize(game, detail)

    if "," in text:
        head, tail = text.split(",", 1)
        head = head.strip()
        tail = tail.strip()
        if head and tail:
            if looks_like_bet_prefix(head):
                return finalize(None, text)
            if "|" in head:
                head_main, head_extra = head.split("|", 1)
                head = head_main.strip()
                tail = f"{head_extra.strip()} {tail}".strip()
            return finalize(head, tail)

    return finalize(None, text.strip() or None)


def split_game_and_time(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    text = _normalize_ascii(text)
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
    stake: Optional[str] = None
    stake_match = STAKE_PREFIX.match(cleaned)
    if stake_match:
        stake = _clean_stake_text(stake_match.group(0))
        cleaned = cleaned[stake_match.end() :].strip()
        cleaned = cleaned.lstrip(") ")
    cleaned = cleaned.lstrip("-: ")
    game, detail = split_game_and_detail(cleaned)
    if game and detail:
        lowered_game = game.lower()
        vs_index = lowered_game.find(" vs ")
        if vs_index != -1:
            left_segment = game[:vs_index]
            spread_match = re.search(r"([+\-−\u2013\u2014]\s?\d+(?:\.\d+)?[A-Za-z%]*)\s*$", left_segment)
            if spread_match:
                spread_text = spread_match.group(1).strip()
                prefix = left_segment[: spread_match.start()].rstrip(" ,;-@/\t")
                suffix = game[vs_index:]
                game = f"{prefix}{suffix}".strip()
                if spread_text:
                    detail = f"{spread_text} {detail}".strip()
    if game and "+" in game and detail and " vs " in detail.lower() and "+" not in detail:
        prefix = detail
        remainder = ""
        at_index = detail.find("@")
        if at_index != -1:
            prefix = detail[:at_index]
            remainder = detail[at_index:]
        prefix_clean = prefix.strip(" ,;-/\t")
        if prefix_clean and " vs " in prefix_clean.lower():
            game = re.sub(r"\s+", " ", f"{game} {prefix_clean}").strip()
            detail = remainder.strip()
    if detail:
        detail, trailing_stake = peel_trailing_stake(detail)
        if trailing_stake and not stake:
            stake = _clean_stake_text(trailing_stake)
        leading_paren = re.match(r"\(\s*([^)]*?)\s*\)\s*(.*)", detail)
        if leading_paren and leading_paren.group(1):
            inner_detail = leading_paren.group(1).strip()
            tail_detail = leading_paren.group(2).strip()
            detail = f"{inner_detail} {tail_detail}".strip() if tail_detail else inner_detail
        paren_match = re.search(r"\(([^)]+)\)\s*$", detail)
        if paren_match:
            inner = paren_match.group(1).strip()
            matchup_text = _extract_matchup_from_text(inner) or (inner if _has_matchup_hint(inner) else None)
            if matchup_text:
                if not game or len(matchup_text) > len(game):
                    game = matchup_text
                detail = detail[: paren_match.start()].rstrip(" ,;-@/\t")
    combined_candidate = detail
    if game and detail:
        combined_candidate = f"{game} {detail}"
    if combined_candidate and not _has_matchup_hint(game):
        embedded_matchup = _extract_matchup_from_text(combined_candidate)
        if embedded_matchup:
            if not game or len(embedded_matchup) > len(game):
                game = embedded_matchup
            opponent = embedded_matchup.split(" vs ", 1)[-1]
            if detail and "+" not in opponent and " vs " not in opponent.lower():
                detail = re.sub(
                    r"\s+vs?\.?\s+" + re.escape(opponent),
                    " ",
                    detail,
                    flags=re.I,
                ).strip(" ,;-@/\t")
                detail = re.sub(r"\s+", " ", detail)
    return game, detail, _clean_stake_text(stake)


def _normalize_for_hint(value: str) -> str:
    simplified = _normalize_ascii(value).lower().replace("\u2019", "'")
    simplified = re.sub(r"^\s*[^a-z]+", "", simplified)
    simplified = simplified.replace("\t", " ")
    simplified = re.sub(r"\s+", " ", simplified)
    return simplified


def _prefix_matches_hint(prefix: str, hints: Tuple[str, ...]) -> bool:
    if not prefix:
        return False
    normalized = prefix.replace("'", "").replace("\u2019", "")
    window = normalized[:40]
    for hint in hints:
        hint_normalized = hint.replace("'", "")
        if window.startswith(hint_normalized):
            return True
        if hint_normalized in window:
            return True
    return False


def _is_field_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    prefix = _normalize_for_hint(stripped)
    normalized_line = _normalize_ascii(stripped).replace("\u2019", "'")
    sanitized_line = normalized_line.replace("__", "").replace("**", "")
    for key, patterns in FIELD_PATTERNS.items():
        hints = FIELD_KEYWORD_HINTS.get(key)
        if hints and not _prefix_matches_hint(prefix, hints):
            continue
        for pattern in patterns:
            if pattern.match(sanitized_line):
                return True
    for key, patterns in AUX_FIELD_PATTERNS.items():
        hints = AUX_FIELD_KEYWORD_HINTS.get(key)
        if hints and not _prefix_matches_hint(prefix, hints):
            continue
        for pattern in patterns:
            if pattern.match(sanitized_line):
                return True
    return False


def _strip_pick_value_prefix(value: str) -> str:
    trimmed = _normalize_ascii(value).strip()
    trimmed = trimmed.strip("* ")
    trimmed = re.sub(
        r"^(?:of\s+the\s+day|today'?s\s+picks?|today'?s\s+play|today'?s\s+potd|todays\s+picks?|todays\s+play|todays\s+potd|today'?s\s+bet|todays\s+bet|todays\s+fixtures|potd|picks?|play|bet|match|event)\b[\W_]*",
        "",
        trimmed,
        flags=re.I,
    )
    return trimmed.strip()


def _clean_odds_value(value: str) -> Optional[str]:
    cleaned = _normalize_ascii(value).strip()
    cleaned = re.sub(r"^(?:of|at|=)\s+", "", cleaned, flags=re.I)
    match = re.search(r"[-+@]?\s*\d+(?:\.\d+)?(?:\s*\([^)]+\))?", cleaned)
    if match:
        odds = cleaned[match.start() : match.end()].strip()
        return odds or None
    return cleaned if cleaned and len(cleaned) <= 20 else None


def _find_first_matchup(lines: Iterable[str]) -> Optional[str]:
    for raw_line in lines:
        normalized_line = _normalize_ascii(raw_line).replace("**", "").replace("__", "").strip()
        if not normalized_line:
            continue
        lowered = normalized_line.lower()
        if lowered.startswith(
            (
                "last pick",
                "previous pick",
                "record",
                "profit",
                "summary",
                "odds",
                "units",
                "stake",
                "write up",
                "write-up",
                "writeup",
            )
        ):
            continue
        candidate = _strip_pick_value_prefix(normalized_line)
        lowered_candidate = candidate.lower()
        if lowered_candidate.startswith(("he ", "she ", "i ", "we ", "they ", "last ")):
            continue
        if " scored" in lowered_candidate or " ago " in lowered_candidate:
            continue
        matchup = looks_like_plain_matchup(candidate) or looks_like_plain_matchup(normalized_line)
        if matchup:
            return matchup
    return None


BET_DETAIL_KEYWORDS = (
    " over ",
    " under ",
    " team total",
    "both teams",
    " to score",
    " moneyline",
    " ml",
    " spread",
    " draw no bet",
    " double chance",
    " alt",
    " asian",
    " handicap",
    " tt",
    " total",
    " parlay",
    " cards",
    " corners",
    " points",
    " pts",
    " yards",
    " goals",
    " runs",
    " assists",
    " rebounds",
    " kills",
    " map",
    " set",
)


def _looks_like_bet_detail(text: str) -> bool:
    candidate = _normalize_ascii(text).strip()
    if not candidate:
        return False
    if len(candidate) > 140:
        return False
    lowered_flat = candidate.lower()
    has_match_marker = _has_matchup_hint(candidate)
    if " roi" in lowered_flat or lowered_flat.startswith("roi"):
        return False
    if "return on investment" in lowered_flat:
        return False
    if "net unit" in lowered_flat or "net units" in lowered_flat:
        return False
    if "season record" in lowered_flat or "current record" in lowered_flat:
        return False
    if "record" in lowered_flat and not has_match_marker:
        return False
    if re.search(r"\b\d{1,3}-\d{1,3}\b", candidate) and not has_match_marker:
        return False
    lowered = f" {lowered_flat} "
    tokens_hit = [token for token in BET_DETAIL_KEYWORDS if token in lowered]
    if looks_like_plain_matchup(candidate):
        if not tokens_hit:
            return False
        if set(tokens_hit) == {" -"} and not re.search(r"\d", candidate):
            return False
    if looks_like_sport_line(candidate):
        return False
    if tokens_hit:
        return True
    if re.search(r"[+\-−\u2013\u2014]\s*\d", candidate):
        context_keywords = (
            "unit",
            "units",
            "point",
            "points",
            "pts",
            "yard",
            "yards",
            "yds",
            "line",
            "spread",
            "ah",
            "ml",
            "moneyline",
            "total",
            "over",
            "under",
            "run",
            "runs",
            "goal",
            "goals",
        )
        if (
            any(keyword in lowered_flat for keyword in context_keywords)
            or has_match_marker
            or re.search(r"@\s*-?\d", candidate)
            or re.search(r"\(\s*-?\d+(?:\.\d+)?\s*\)", candidate)
            or "bet" in lowered_flat
        ):
            return True
    if re.search(r"@\s*-?\d", candidate):
        return True
    if re.search(
        r"\b\d+(?:\.\d+)?\s*(?:u|units?|pts|points|yards|yds|goals|runs|aces|reb|rebounds|assists|asts|kills|maps?|sets?|cards?|corners?|outs|shots|ml)\b",
        lowered,
    ):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:to\s+win|parlay|leg|alt\s+line)\b", lowered):
        return True
    if " win" in lowered_flat and len(candidate.split()) <= 4:
        return True
    if re.search(r"\d", lowered_flat):
        return False
    return False


MATCHUP_SEPARATORS: Tuple[Tuple[str, str], ...] = (
    (" vs ", "vs"),
    (" vs.", "vs"),
    (" @ ", "@"),
    (" v ", "v"),
    (" versus ", "vs"),
)


SPORT_PREFIXES = (
    "nfl football",
    "college football",
    "college basketball",
    "ncaa football",
    "ncaa basketball",
    "mlb",
    "nba",
    "nfl",
    "nhl",
    "nrl",
    "afl",
    "ufc",
    "mma",
    "football",
    "basketball",
    "baseball",
    "hockey",
    "soccer",
    "rugby",
    "tennis",
)


def _strip_leading_sport_words(text: str) -> str:
    lowered = text.lower().strip()
    for prefix in SPORT_PREFIXES:
        if lowered.startswith(prefix):
            trimmed = text[len(prefix) :]
            return trimmed.strip(" -,:") or text.strip()
    return text.strip()


def _clean_team_fragment(fragment: str, take_last: bool) -> str:
    segment = _normalize_ascii(fragment)
    for splitter in ("|", "/"):
        parts = segment.split(splitter)
        segment = parts[-1] if take_last else parts[0]
    for splitter in (" - ", "-", ":", ","):
        if splitter in segment:
            parts = segment.split(splitter)
            candidate = parts[-1] if take_last else parts[0]
            if take_last:
                other = parts[0]
            else:
                other = parts[-1]
            if not any(ch.isalpha() for ch in candidate) and any(ch.isalpha() for ch in other):
                segment = other
            else:
                segment = candidate
    lowered_segment = segment.lower()
    if " in " in lowered_segment:
        parts = re.split(r"\s+in\s+", segment, maxsplit=1)
        segment = parts[-1] if take_last else parts[0]
        lowered_segment = segment.lower()
    if " at " in lowered_segment:
        parts = re.split(r"\s+at\s+", segment, maxsplit=1)
        segment = parts[-1] if take_last else parts[0]
    segment = _strip_leading_sport_words(segment)
    segment = re.sub(r"\s*[+\-−\u2013\u2014]\d+(?:\.\d+)?[A-Za-z]*$", "", segment)
    segment = re.sub(r"\s*\(\s*[-+@]?\d+(?:\.\d+)?\s*\)\s*$", "", segment)
    segment = re.sub(r"\s*\([^)]*\)\s*$", "", segment)
    segment = re.sub(r"\s*(?:ml|ah|moneyline)$", "", segment, flags=re.I)
    segment = re.sub(r"^[\s(]+", "", segment)
    segment = re.sub(r"\(+\s*$", "", segment)
    segment = re.sub(r"[\s)]+$", "", segment)
    segment = re.sub(r"\s+", " ", segment).strip()
    segment = segment.strip("] ")
    return segment


def _extract_matchup_from_text(text: str) -> Optional[str]:
    text = _normalize_ascii(text)
    lowered = text.lower()
    for raw_sep, normalized_sep in MATCHUP_SEPARATORS:
        pos = lowered.find(raw_sep)
        if pos == -1:
            continue
        left = text[:pos]
        right = text[pos + len(raw_sep) :]
        team_left = _clean_team_fragment(left, take_last=True)
        team_right = _clean_team_fragment(right, take_last=False)
        if _looks_like_bet_detail(team_left):
            alt_left = _clean_team_fragment(left, take_last=False)
            if alt_left and not _looks_like_bet_detail(alt_left):
                team_left = alt_left
        if _looks_like_bet_detail(team_right):
            alt_right = _clean_team_fragment(right, take_last=True)
            if alt_right and not _looks_like_bet_detail(alt_right):
                team_right = alt_right
        if (
            not team_left
            or not team_right
            or not any(ch.isalpha() for ch in team_left)
            or not any(ch.isalpha() for ch in team_right)
            or _looks_like_bet_detail(team_left)
            or _looks_like_bet_detail(team_right)
        ):
            continue
        return f"{team_left} {normalized_sep} {team_right}"
    return None


def _find_followup_bet(
    lines: List[str], start_index: int, max_lookahead: int = 20
) -> Optional[str]:
    end_index = min(len(lines), start_index + max_lookahead)
    blank_streak = 0
    for idx in range(start_index, end_index):
        candidate_line = lines[idx]
        candidate = candidate_line.strip()
        if not candidate:
            blank_streak += 1
            if blank_streak >= 2:
                break
            continue
        blank_streak = 0
        candidate_ascii = _normalize_ascii(candidate)
        if _is_field_line(candidate_ascii):
            continue
        normalized_lower = candidate_ascii.lower().lstrip("_*•-> \t")
        if normalized_lower.startswith(
            (
                "write up",
                "write-up",
                "analysis",
                "summary",
                "last pick",
                "previous pick",
                "prior pick",
            )
        ):
            continue
        if normalized_lower.startswith("record"):
            continue
        if normalized_lower.startswith(
            (
                "net profit",
                "net units",
                "net roi",
                "net record",
                "net balance",
                "net bankroll",
                "profit",
                "roi",
                "units won",
            )
        ):
            continue
        if " net profit" in normalized_lower or " net units" in normalized_lower:
            continue
        if looks_like_record_heading(candidate_ascii):
            continue
        if len(candidate_ascii) > 250:
            continue
        if _looks_like_bet_detail(candidate_ascii):
            return candidate_ascii
    return None


PICK_HEADER_RE = re.compile(
    r"^\s*[*_>\-•\u2022]*\s*(?:(?:today'?s|todays|next)\s+)?(?:potd|picks?|bet|play|selection)\b[:\-\u2013\u2014|]?\s*(?P<value>.*)$",
    re.I,
)


def _is_reasonable_pick_text(value: Optional[str]) -> bool:
    if not value:
        return False
    normalized = _normalize_ascii(value).lower()
    keywords = (
        " over ",
        " under ",
        " moneyline",
        " ml",
        " to win",
        " to score",
        " btts",
        " both teams",
        " draw",
        " double chance",
        " spread",
        " handicap",
        " total",
        " asian",
        " alt",
        " cards",
        " corners",
        " sog",
        " shots",
        " assists",
        " rebounds",
        " points",
        " goals",
        " win",
        " parlay",
        " team total",
    )
    if any(keyword in normalized for keyword in keywords):
        return True
    if re.search(r"\b\d{1,3}-\d{1,3}\b", normalized) and "@" not in normalized:
        return False
    if "@" in normalized or re.search(r"[+-]\s*\d", normalized):
        return True
    if re.search(r"\d", normalized):
        return False
    return False


def _extract_pick_from_headers(lines: List[str]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    last_header_game: Optional[str] = None
    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        ascii_line = _normalize_ascii(stripped)
        lowered = ascii_line.lower().lstrip("_*•-> \t")
        if lowered.startswith(
            (
                "last pick",
                "previous pick",
                "prior pick",
                "last potd",
                "previous potd",
                "last bet",
                "previous bet",
                "last play",
                "previous play",
            )
        ):
            continue
        match = PICK_HEADER_RE.match(ascii_line)
        if not match:
            continue
        value = match.group("value") or ""
        value = _strip_pick_value_prefix(value).strip()
        value = value.lstrip(":|- ")
        header_game_candidate = None
        if value:
            if " vs " in value.lower():
                header_game_candidate = value.strip()
            else:
                header_game_candidate = _extract_matchup_from_text(value) or looks_like_plain_matchup(value)
            if header_game_candidate:
                header_game_candidate = header_game_candidate.strip()
                last_header_game = header_game_candidate
        game_candidate: Optional[str] = None
        if value:
            matchup_candidate = looks_like_plain_matchup(value)
            if matchup_candidate and not re.search(r"\d", value) and not _looks_like_bet_detail(value):
                game_candidate = matchup_candidate
                followup = _find_followup_bet(lines, idx + 1)
                value = followup or value
            elif matchup_candidate and not _is_reasonable_pick_text(value):
                game_candidate = matchup_candidate
                followup = _find_followup_bet(lines, idx + 1)
                if followup:
                    value = followup
            elif " - " in value and not value.strip().startswith(("-", "+")):
                left, right = value.split(" - ", 1)
                left_matchup = looks_like_plain_matchup(left)
                if left_matchup:
                    game_candidate = game_candidate or left_matchup
                    value = right.strip()
        if not value:
            value = _find_followup_bet(lines, idx + 1)
        if not value:
            continue
        cleaned_value = value.strip()
        lowered_cleaned = cleaned_value.lower()
        if lowered_cleaned.startswith(("record", "last pick", "todays fixtures", "today's fixtures")):
            continue
        game, detail, stake = parse_pick_text(cleaned_value)
        pick_text = detail.strip() if detail else cleaned_value.strip()
        if not _looks_like_bet_detail(pick_text):
            followup = _find_followup_bet(lines, idx + 1)
            if followup:
                pick_text = followup.strip()
                if stake is None:
                    _, _, stake_from_followup = parse_pick_text(followup)
                    if stake_from_followup and not stake:
                        stake = stake_from_followup
        if game:
            normalized_game = looks_like_plain_matchup(game) or game
            if normalized_game:
                if not game_candidate or len(normalized_game) > len(game_candidate):
                    game_candidate = normalized_game
                if pick_text.startswith(("+", "-")) and normalized_game:
                    pick_text = f"{normalized_game} {pick_text}".strip()
                elif detail and "+" not in detail and pick_text.lower().startswith(normalized_game.lower()):
                    remainder = pick_text[len(normalized_game) :].lstrip(" -,:@")
                    if remainder:
                        pick_text = remainder
        if game_candidate and " vs " in game_candidate.lower() and pick_text:
            opponent = game_candidate.split(" vs ", 1)[-1]
            if "+" not in opponent and " vs " not in opponent.lower():
                pick_text = re.sub(
                    r"\s+vs?\.?\s+" + re.escape(opponent),
                    " ",
                    pick_text,
                    flags=re.I,
                ).strip(" ,;-@/\t")
                pick_text = re.sub(r"\s+", " ", pick_text)
        candidate: Dict[str, Any] = {"pick": pick_text.strip(), "index": idx}
        chosen_game = game_candidate or header_game_candidate or last_header_game
        if chosen_game:
            cleaned_game = _strip_pick_value_prefix(chosen_game.strip())
            cleaned_game = re.split(r"@\s*-?\d", cleaned_game, maxsplit=1)[0].strip()
            cleaned_game = re.sub(r"\s+\d+(?:\.\d+)?\s*(?:u|units?)$", "", cleaned_game, flags=re.I)
            chosen_game_value = cleaned_game
            paren_match = re.search(r"\(([^)]+)\)\s*$", cleaned_game)
            if paren_match and _has_matchup_hint(paren_match.group(1)):
                inner = paren_match.group(1).strip()
                inner_matchup = _extract_matchup_from_text(inner) or inner
                chosen_game_value = inner_matchup
            else:
                matchup_from_cleaned = _extract_matchup_from_text(cleaned_game)
                if matchup_from_cleaned:
                    chosen_game_value = matchup_from_cleaned
                else:
                    normalized_candidate_game = looks_like_plain_matchup(cleaned_game)
                    if normalized_candidate_game:
                        chosen_game_value = normalized_candidate_game
            candidate["game"] = chosen_game_value
        if header_game_candidate and " vs " in header_game_candidate.lower():
            existing_game = candidate.get("game")
            header_value = header_game_candidate.strip()
            if not existing_game or len(header_value) > len(existing_game):
                candidate["game"] = header_value
        if stake:
            candidate["stake"] = stake
        if best is None or idx >= best.get("index", -1):
            best = candidate
    return best


def _sport_token_from_text(text: str) -> Optional[str]:
    text = _normalize_ascii(text)
    if "/" in text:
        for part in text.split("/"):
            candidate = _sport_token_from_text(part)
            if candidate:
                return candidate
    if "|" in text:
        for part in text.split("|"):
            candidate = _sport_token_from_text(part)
            if candidate:
                return candidate
    cleaned = re.sub(r"[\W_]+", " ", text).strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if "dart" in lowered or "darts" in lowered:
        return "Darts"
    if "hockey" in lowered or "nhl" in lowered:
        return "Hockey"
    if "t20" in lowered:
        return "Cricket"
    if "bundesliga" in lowered or "premier league" in lowered or "champions league" in lowered or "uefa" in lowered:
        return "Soccer"
    if "serie a" in lowered or "la liga" in lowered or "laliga" in lowered or "mls" in lowered:
        return "Soccer"
    if "college" in lowered and "football" in lowered:
        return "College Football"
    if "football" in lowered and any(
        token in lowered for token in ("bundesliga", "premier", "champions", "uefa", "laliga", "serie", "mls")
    ):
        return "Soccer"
    if "football" in lowered and "nfl" not in lowered and "college" not in lowered and "aussie" not in lowered:
        return "Football"
    if "uefa" in lowered and "league" in lowered:
        return "Soccer"
    if re.search(r'\b(kills?|maps?|esl|cs2|counter|valorant|nuke|inferno)\b', lowered) or "gentle mates" in lowered or lowered == "dm":
        return "Esports"
    if lowered in COMMON_SPORT_TOKENS:
        if cleaned.isupper() or len(cleaned) <= 4:
            return cleaned.upper()
        if cleaned.istitle():
            return cleaned
        return cleaned.title()
    for token in lowered.split():
        if token in COMMON_SPORT_TOKENS:
            return _sport_token_from_text(token)
    if "nfl" in lowered and "football" in lowered:
        return "NFL"
    inferred = looks_like_sport_line(cleaned)
    if inferred:
        inferred_lower = inferred.lower()
        if inferred_lower in COMMON_SPORT_TOKENS:
            return inferred
        if " " in inferred_lower:
            return None
        return inferred
    return None


GAME_PREFIX = re.compile(
    r"^(?:today'?s\s+)?(?:new\s+)?(?:event|match(?:up)?|game|fixture|bet)\s*(?:[:\-\u2013\u2014|]\s*)",
    re.I,
)


def _normalize_game_text(game: str) -> Tuple[Optional[str], Optional[str]]:
    ascii_game = _normalize_ascii(game)
    cleaned = re.sub(r"\s+", " ", ascii_game).strip(" -*|\t\u2013\u2014")
    cleaned = cleaned.lstrip("_*#> ")
    cleaned = cleaned.lstrip("] ")
    cleaned = re.sub(r"^[^A-Za-z0-9]+", "", cleaned)
    cleaned = "".join(ch for ch in cleaned if not 0x1F1E6 <= ord(ch) <= 0x1F1FF)
    cleaned = cleaned.replace("\u2019", "'")
    cleaned = cleaned.rstrip("* ")
    # Remove markdown formatting (** and __) from the middle of text
    cleaned = re.sub(r"[*_]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    bracket_match = re.search(r"\[([^\]]+)\]", cleaned)
    sport_from_bracket: Optional[str] = None
    if bracket_match:
        bracket_content = bracket_match.group(1).strip()
        sport_from_bracket = _sport_token_from_text(bracket_content) or bracket_content
        cleaned = (cleaned[: bracket_match.start()] + cleaned[bracket_match.end() :]).strip()
    if "]" in cleaned and "[" not in ascii_game:
        parts = cleaned.split("]", 1)
        tail = parts[1].strip() if len(parts) > 1 else ""
        if tail:
            cleaned = tail
    cleaned = GAME_PREFIX.sub("", cleaned)
    sport_from_emoji: Optional[str] = None
    for emoji, sport_name in EMOJI_SPORT_MAP.items():
        if emoji in cleaned:
            sport_from_emoji = sport_from_emoji or sport_name
            cleaned = cleaned.replace(emoji, "").strip()
    if ":" in cleaned:
        leading, remainder = cleaned.split(":", 1)
        candidate = _sport_token_from_text(leading.strip())
        if candidate and remainder.strip():
            cleaned = remainder.strip()
            sport_from_emoji = sport_from_emoji or candidate
    parts = [part.strip(" -/") for part in re.split(r"\s*\|\s*", cleaned) if part.strip()]
    sport_candidate: Optional[str] = None
    if len(parts) >= 2:
        possible_sport = parts[0]
        sport_candidate = _sport_token_from_text(possible_sport)
        if sport_candidate:
            cleaned = " | ".join(parts[1:])
    elif "," in cleaned:
        leading, remainder = cleaned.split(",", 1)
        if not _has_matchup_hint(leading):
            candidate = _sport_token_from_text(leading)
            if candidate:
                sport_candidate = candidate
                cleaned = remainder.strip()

    # Handle slash-separated sport prefixes like "NCAAB / TeamA vs TeamB"
    if not sport_candidate and " / " in cleaned:
        leading, remainder = cleaned.split(" / ", 1)
        if not _has_matchup_hint(leading) and remainder:
            candidate = _sport_token_from_text(leading.strip())
            if candidate:
                sport_candidate = candidate
                cleaned = remainder.strip()

    # Handle dash-separated sport prefixes like "EPL – TeamA vs TeamB" or "NFL TeamA @ TeamB"
    if not sport_candidate:
        # Try splitting on various dash characters
        for dash_char in ["\u2013", "\u2014", "-"]:
            if dash_char in cleaned:
                parts = cleaned.split(dash_char, 1)
                if len(parts) == 2:
                    leading = parts[0].strip()
                    remainder = parts[1].strip()
                    # Check if leading part is a sport token and remainder has matchup hints
                    if not _has_matchup_hint(leading) and remainder:
                        candidate = _sport_token_from_text(leading)
                        if candidate:
                            sport_candidate = candidate
                            cleaned = remainder
                            break

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|\t")
    cleaned = re.sub(r"\b(pick|play)\b\s*$", "", cleaned, flags=re.I)
    if sport_from_bracket:
        if not sport_candidate or sport_candidate.lower() in {"soccer", "football"}:
            sport_candidate = sport_from_bracket
    return cleaned or None, sport_candidate or sport_from_emoji or sport_from_bracket


def _clean_time_value(value: str) -> str:
    cleaned = _normalize_ascii(value).strip()
    cleaned = cleaned.replace("&amp;", "&")
    cleaned = cleaned.strip("* ")
    cleaned = re.sub(r"^(?:&\s*)?tv[:\-\s]*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(?:kick(?:-?off)?)(?:\s*time)?\b[:\-\s]*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^start(?:\s*time)?\b[:\-\s]*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned)
    return cleaned.strip()


def _cleanup_pick_detail(detail: str) -> str:
    trimmed = detail.strip()
    trimmed = re.sub(r"[,;\-\s]*(?:for|risk)\s*$", "", trimmed, flags=re.I)
    trimmed = trimmed.rstrip("| ")
    trimmed = trimmed.rstrip("@ ")
    trimmed = re.sub(r"\s*\(price\s*=.*$", "", trimmed, flags=re.I)
    while trimmed.count(")") > trimmed.count("("):
        trimmed = trimmed[::-1].replace(")", "", 1)[::-1]
    trimmed = re.sub(r"\s+", " ", trimmed)
    return trimmed.strip(" ,;")


def _primary_team_from_game(game: Optional[str]) -> Optional[str]:
    if not game:
        return None
    matchup = _extract_matchup_from_text(game) or looks_like_plain_matchup(game) or game
    if not matchup:
        return None
    lowered_matchup = matchup.lower()
    for delimiter in (" vs ", " @ ", " v ", " versus ", " at "):
        idx = lowered_matchup.find(delimiter)
        if idx != -1:
            primary = matchup[:idx].strip(" ,;-@/\t")
            return _clean_team_fragment(primary, take_last=False) or primary
    return _clean_team_fragment(matchup, take_last=False)


def _strip_team_moneyline_prefix(detail: str, game: Optional[str]) -> str:
    if not detail or not game:
        return detail
    first_team = _primary_team_from_game(game)
    if not first_team:
        return detail
    variants = {first_team}
    normalized = _normalize_ascii(first_team)
    if normalized:
        variants.add(normalized)
    parts = [part for part in re.split(r"\s+", normalized) if part]
    if parts:
        variants.add(parts[0])
        variants.add(parts[-1])
    for variant in sorted({v for v in variants if v}, key=len, reverse=True):
        pattern = re.compile(rf"^{re.escape(variant)}\s+(?=(ml|moneyline)\b)", re.I)
        match = pattern.match(detail)
        if not match:
            continue
        remainder = detail[match.end() :].lstrip(" ,;-@/\t")
        if remainder and (" + " in remainder or " & " in remainder):
            return remainder
    return detail


PICK_PREFIX_CANDIDATE = re.compile(r"^\s*([A-Za-z0-9 .&'()/]+?)\s*(?:[-+\u2013\u2014])")


def _extract_pick_prefix_from_candidate(pick: Optional[str]) -> Optional[str]:
    if not pick:
        return None
    match = PICK_PREFIX_CANDIDATE.match(pick)
    if not match:
        return None
    prefix = match.group(1).strip(" ,;-@/\t")
    if not prefix or _looks_like_bet_detail(prefix):
        return None
    return prefix


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


def _find_explicit_match_line(lines: Iterable[str]) -> Optional[str]:
    for raw_line in lines:
        ascii_line = _normalize_ascii(raw_line).strip()
        if not ascii_line:
            continue
        lowered = ascii_line.lower()
        if lowered.startswith(("match:", "game:", "event:")):
            _, _, remainder = ascii_line.partition(":")
            candidate = remainder.strip()
            candidate = _strip_pick_value_prefix(candidate)
            normalized = looks_like_plain_matchup(candidate) or candidate
            if normalized:
                return normalized
    return None


def _find_explicit_sport_line(lines: Iterable[str]) -> Optional[str]:
    for raw_line in lines:
        ascii_line = _normalize_ascii(raw_line).strip()
        if not ascii_line:
            continue
        lowered = ascii_line.lower()
        if lowered.startswith(("sport:", "competition:", "league:")):
            _, _, remainder = ascii_line.partition(":")
            candidate = remainder.strip()
            cleaned = _sport_token_from_text(candidate) or candidate
            if cleaned and not cleaned.lower().startswith("bonus"):
                return cleaned
    return None


def extract_pick_fields(lines: Iterable[str]) -> dict:
    material = list(lines)
    expanded: List[str] = []
    for raw_line in material:
        if "|" in raw_line and not raw_line.strip().startswith("|"):
            segments = [segment.strip() for segment in raw_line.split("|") if segment.strip()]
            expanded.extend(segments)
        else:
            expanded.append(raw_line)
    material = expanded
    result = {
        "pick": None,
        "game": None,
        "sport": None,
        "time": None,
        "recommended_wager": None,
    }
    aux: Dict[str, Optional[str]] = {"odds": None, "book": None}
    game_line_index: Optional[int] = None
    sport_line_index: Optional[int] = None
    pick_line_index: Optional[int] = None
    in_previous_pick_block = False

    for idx, line in enumerate(material):
        stripped = line.strip()
        if not stripped:
            continue
        ascii_line = _normalize_ascii(stripped)
        lowered = stripped.lower()
        ascii_lowered = ascii_line.lower()
        normalized_lowered = ascii_lowered.lstrip("_*•-> \t")
        if normalized_lowered.startswith(("write up", "write-up", "writeup")):
            continue
        if normalized_lowered.startswith(
            (
                "last pick",
                "previous pick",
                "prior pick",
                "prior potd",
                "last potd",
                "previous potd",
                "last bet",
                "previous bet",
                "last play",
                "previous play",
                "previous :",
                "previous:",
                "previous -",
            )
        ):
            in_previous_pick_block = True
            continue
        if normalized_lowered.startswith(
            (
                "today's pick",
                "todays pick",
                "today's bet",
                "todays bet",
                "today's play",
                "todays play",
                "potd",
                "event",
                "new event",
                "todays event",
                "today's event",
                "game",
                "pick",
                "next pick",
                "next play",
            )
        ):
            in_previous_pick_block = False
        if in_previous_pick_block and "|" in stripped:
            in_previous_pick_block = False
        # Also exit previous block if line looks like a game/matchup
        if in_previous_pick_block and _has_matchup_hint(stripped):
            in_previous_pick_block = False
        if normalized_lowered.startswith(("units won", "profit/loss", "profit loss", "profit", "loss")):
            continue
        if normalized_lowered.startswith(
            (
                "roi",
                "return on investment",
                "net units",
                "net unit",
                "net profit",
                "net roi",
                "expected goals",
                "model fair",
                "poisson",
                "lambda",
            )
        ):
            continue
        if looks_like_record_heading(stripped):
            continue
        if in_previous_pick_block:
            continue
        captured = False
        for key, patterns in FIELD_PATTERNS.items():
            if key == "pick" and result["pick"] and pick_line_index is not None and idx > pick_line_index:
                if not normalized_lowered.startswith(
                    (
                        "pick",
                        "today's pick",
                        "todays pick",
                        "today's bet",
                        "todays bet",
                        "today's play",
                        "todays play",
                        "potd",
                        "bet",
                        "play",
                    )
                ):
                    continue
            if key == "recommended_wager" and any(
                token in normalized_lowered for token in ("net", "record", "units won", "balance")
            ):
                continue
            for pattern in patterns:
                if key == "pick":
                    normalized_preview = ascii_line.lower().replace("\u2019", "'")
                    if not any(token in normalized_preview for token in PICK_KEY_TOKENS):
                        continue
                normalized_line = ascii_line.replace("\u2019", "'")
                normalized_line = "".join(
                    ch
                    for ch in normalized_line
                    if not (0x2600 <= ord(ch) <= 0x27FF or 0x1F300 <= ord(ch) <= 0x1FAFF)
                )
                sanitized_line = normalized_line.replace("__", "").replace("**", "")
                match = pattern.match(sanitized_line)
                if match:
                    value = match.group(1).strip()
                    value = _normalize_ascii(value)
                    if not value:
                        value = _next_non_empty(material, idx + 1) or ""
                        if value and _is_field_line(value):
                            value = ""
                        value = value.strip()
                        value = _normalize_ascii(value)
                    if value and key == "pick":
                        if "👉" in line:
                            continue
                        value = _strip_pick_value_prefix(value)
                        value = value.strip()
                        matchup_candidate = looks_like_plain_matchup(value) if value else None
                        if matchup_candidate:
                            value_lower = value.lower()
                            has_digits = bool(re.search(r"\d", value))
                            has_odds_marker = "@" in value_lower or re.search(r"[+-]\s*\d", value)
                            keyword_hits = (
                                " over " in value_lower
                                or " under " in value_lower
                                or " moneyline" in value_lower
                                or " ml" in value_lower
                                or " spread" in value_lower
                                or " handicap" in value_lower
                                or " to win" in value_lower
                                or " to score" in value_lower
                                or " total" in value_lower
                                or " cards" in value_lower
                                or " corners" in value_lower
                                or " btts" in value_lower
                                or " both teams" in value_lower
                                or " goals" in value_lower
                                or " shots" in value_lower
                                or " assists" in value_lower
                                or " rebounds" in value_lower
                                or " points" in value_lower
                                or " draw no bet" in value_lower
                            )
                            if not has_digits and not has_odds_marker and not keyword_hits:
                                existing_game = result.get("game")
                                if not existing_game or _should_replace_game(existing_game, matchup_candidate):
                                    result["game"] = matchup_candidate
                                    game_line_index = idx
                                followup = _find_followup_bet(material, idx + 1)
                                if followup:
                                    value = followup.strip()
                        if not value:
                            value = _find_followup_bet(material, idx + 1) or ""
                        elif not _looks_like_bet_detail(value):
                            followup = _find_followup_bet(material, idx + 1)
                            if followup:
                                value = followup
                        value = value.strip()
                        in_previous_pick_block = False
                    if value and key == "time":
                        value = _clean_time_value(value)
                    if value and key == "sport":
                        time_match = TIME_IN_TEXT.search(value)
                        if time_match and not result["time"]:
                            extracted_time = _clean_time_value(time_match.group(1))
                            if extracted_time:
                                result["time"] = extracted_time
                            value = (value[: time_match.start()] + value[time_match.end() :]).strip(" ,;-/\t")
                        normalized_sport = _sport_token_from_text(value)
                        if normalized_sport:
                            value = normalized_sport
                    if value and key == "recommended_wager":
                        if _is_summary_units_value(value, normalized_lowered):
                            value = None
                        else:
                            value = _clean_stake_text(value)
                    if value:
                        if key == "pick" and looks_like_record_heading(value):
                            continue
                        if key == "pick" and not any(ch.isalpha() for ch in value):
                            logger.debug(
                                "Ignoring pick candidate without letters: '%s' (line %s)",
                                value,
                                line.strip(),
                            )
                            continue
                        existing = result.get(key)
                        if key == "pick" and existing:
                            has_letters_existing = any(ch.isalpha() for ch in existing)
                            has_letters_new = any(ch.isalpha() for ch in value)
                            should_replace = False
                            if has_letters_new and not has_letters_existing:
                                should_replace = True
                            elif len(value) > len(existing):
                                should_replace = True
                            if should_replace:
                                logger.debug(
                                    "Replacing pick '%s' with '%s'", existing, value
                                )
                                result[key] = value
                        elif key == "game":
                            if not existing or _should_replace_game(existing, value):
                                result[key] = value
                                game_line_index = idx
                                if not result.get("sport"):
                                    sport_from_game = _sport_token_from_text(value)
                                    if sport_from_game:
                                        result["sport"] = sport_from_game
                                        sport_line_index = idx
                            else:
                                continue
                        elif key == "sport":
                            if not existing or _should_replace_sport(existing, value):
                                result[key] = value
                                sport_line_index = idx
                            else:
                                continue
                        elif key == "recommended_wager":
                            if not existing or _clean_stake_text(existing) is None:
                                result[key] = value
                            else:
                                continue
                        else:
                            result[key] = value
                        if key == "pick" and value and not result.get("sport"):
                            for emoji, sport_name in EMOJI_SPORT_MAP.items():
                                if emoji in line:
                                    result["sport"] = sport_name
                                    break
                        if key == "pick" and value:
                            pick_line_index = idx
                        if key == "game" and not result.get("sport"):
                            for emoji, sport_name in EMOJI_SPORT_MAP.items():
                                if emoji in line:
                                    result["sport"] = sport_name
                                    break
                    captured = True
                    break
            if captured:
                break
        if captured:
            continue
        if not result["pick"]:
            ascii_candidate = ascii_line
            stake_prefix_match = STAKE_PREFIX.match(ascii_candidate)
            stake_value = None
            if stake_prefix_match:
                stake_value = _clean_stake_text(stake_prefix_match.group(0))
                ascii_candidate = ascii_candidate[stake_prefix_match.end() :].strip(" -:|\t")
            candidate_pick_ascii = ascii_candidate.strip()
            candidate_pick_ascii = _strip_pick_value_prefix(candidate_pick_ascii)
            lowered_candidate_pick = candidate_pick_ascii.lower()
            if (
                lowered_candidate_pick.startswith(("unit", "record", "profit", "net", "form", "roi"))
                or "roi" in lowered_candidate_pick
                or "net unit" in lowered_candidate_pick
                or "net profit" in lowered_candidate_pick
            ):
                candidate_pick_ascii = ""
            if candidate_pick_ascii and _looks_like_bet_detail(candidate_pick_ascii):
                result["pick"] = candidate_pick_ascii
                if stake_value and not result.get("recommended_wager"):
                    result["recommended_wager"] = stake_value
                pick_line_index = idx
                in_previous_pick_block = False
                continue
        if not result["time"] and stripped.lower().startswith("date"):
            remainder = stripped.split(":", 1)[1] if ":" in stripped else stripped[4:]
            remainder = remainder.strip(" -:\t")
            if remainder:
                result["time"] = _clean_time_value(remainder)
                continue
        current_game = result.get("game")
        if current_game is None:
            needs_game = True
        else:
            needs_game = False
            has_matchup_marker = _has_matchup_hint(current_game) or re.search(
                r"[A-Za-z][^\n]{0,40}\s[-/]\s[^\n]{0,40}[A-Za-z]",
                current_game,
            )
            if not has_matchup_marker:
                needs_game = True
            elif game_line_index is not None and idx > game_line_index and (idx - game_line_index) <= 3:
                needs_game = True
        if needs_game:
            lowered_stripped = stripped.lower()
            if lowered_stripped.startswith("at ") or lowered_stripped.startswith("last pick"):
                continue
            candidate_game_line = _strip_pick_value_prefix(stripped)
            matchup = looks_like_plain_matchup(candidate_game_line) or looks_like_plain_matchup(stripped)
            if matchup:
                result["game"] = matchup
                game_line_index = idx
                continue
        if not result["sport"]:
            sport_line = looks_like_sport_line(stripped)
            if sport_line:
                lowered_sport_line = sport_line.lower()
                if "potd" not in lowered_sport_line and not lowered_sport_line.startswith("bonus tip"):
                    normalized_sport = _sport_token_from_text(sport_line) or sport_line
                    result["sport"] = normalized_sport
                    sport_line_index = idx
                    continue
            if not looks_like_record_heading(stripped):
                has_multiple_sport_tokens = sum(
                    1 for token in COMMON_SPORT_TOKENS if token in stripped.lower()
                ) >= 2
                is_boxing_day = "boxing day" in stripped.lower()
                looks_like_record_snippet = bool(re.search(r'\b\d+\s*-\s*\d+\b', stripped))
                if not has_multiple_sport_tokens and not is_boxing_day and not looks_like_record_snippet:
                    auto_sport = _sport_token_from_text(stripped)
                    if auto_sport and "potd" not in normalized_lowered and not normalized_lowered.startswith("bonus tip"):
                        result["sport"] = auto_sport
                        sport_line_index = idx
                        continue
            for emoji, sport_name in EMOJI_SPORT_MAP.items():
                if emoji in stripped:
                    result["sport"] = sport_name
                    sport_line_index = idx
                    break
            if result["sport"]:
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
                        if key == "recommended_wager":
                            aux[key] = _clean_stake_text(value) or value
                        elif key == "odds":
                            cleaned_odds = _clean_odds_value(value)
                            if cleaned_odds:
                                aux[key] = cleaned_odds
                        else:
                            aux[key] = value
                    break

    best_pick_candidate = _extract_pick_from_headers(material)
    if best_pick_candidate:
        candidate_pick = best_pick_candidate.get("pick")
        if candidate_pick:
            existing_pick = result.get("pick")
            candidate_is_valid = _is_reasonable_pick_text(candidate_pick)
            replace = False
            if not _is_reasonable_pick_text(existing_pick):
                replace = True
            elif candidate_is_valid and best_pick_candidate.get("index", -1) >= (pick_line_index or -1):
                replace = True
            if replace:
                result["pick"] = candidate_pick
                pick_line_index = best_pick_candidate.get("index", pick_line_index)
                candidate_game = best_pick_candidate.get("game")
                if candidate_game:
                    existing_game = result.get("game")
                    if not existing_game or _should_replace_game(existing_game, candidate_game):
                        result["game"] = candidate_game
                candidate_stake = best_pick_candidate.get("stake")
                if candidate_stake and not result.get("recommended_wager"):
                    result["recommended_wager"] = candidate_stake

    if not result.get("game"):
        for raw_line in material:
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            ascii_line = _normalize_ascii(stripped_line)
            lowered_line = ascii_line.lower()
            if lowered_line.startswith("match:") or lowered_line.startswith("event:") or lowered_line.startswith("game:"):
                _, _, remainder = ascii_line.partition(":")
                candidate_game = remainder.strip()
                if candidate_game:
                    normalized_candidate = looks_like_plain_matchup(candidate_game) or candidate_game
                    if normalized_candidate:
                        result["game"] = normalized_candidate.strip()
                        break

    if result["pick"]:
        original_pick = result["pick"].strip()
        game, detail, stake = parse_pick_text(original_pick)
        normalized_game: Optional[str] = None
        if game:
            raw_game = re.sub(r"\s+", " ", game).strip()
            if "+" in raw_game and not _has_matchup_hint(raw_game):
                parts = [part.strip() for part in raw_game.split("+") if part.strip()]
                if parts:
                    raw_game = " / ".join(parts)
            if _has_matchup_hint(raw_game):
                normalized_game = looks_like_plain_matchup(raw_game) or raw_game
        cleaned_detail = detail.strip() if detail else ""
        if cleaned_detail:
            cleaned_detail = _cleanup_pick_detail(cleaned_detail)
        if normalized_game:
            existing_game = result.get("game")
            normalized_lower = normalized_game.lower()
            should_replace_game = not existing_game
            if isinstance(existing_game, str) and existing_game:
                existing_lower = existing_game.lower()
                if not _has_matchup_hint(existing_game) and _has_matchup_hint(normalized_game):
                    should_replace_game = True
                elif normalized_lower in existing_lower and normalized_lower != existing_lower:
                    should_replace_game = True
                elif len(normalized_game) > len(existing_game):
                    should_replace_game = True
                elif looks_like_plain_matchup(normalized_game) and not looks_like_plain_matchup(existing_game):
                    should_replace_game = True
            if should_replace_game:
                result["game"] = normalized_game
            if cleaned_detail:
                cleaned_detail = _strip_team_moneyline_prefix(cleaned_detail, normalized_game)
                lowered_detail = cleaned_detail.lower()
                lowered_game = normalized_lower
                if lowered_detail.startswith(lowered_game):
                    remainder = cleaned_detail[len(normalized_game) :].lstrip()
                    if remainder and remainder[0] not in "-+":
                        trimmed = remainder.lstrip(" ,;-@")
                        if trimmed:
                            cleaned_detail = trimmed
        if cleaned_detail:
            prefixed = False
            if normalized_game and cleaned_detail.strip().startswith(("-", "+")):
                lower_game = normalized_game.lower()
                primary = normalized_game
                for delimiter in (" vs ", " @ ", " v ", " versus ", " at "):
                    idx = lower_game.find(delimiter)
                    if idx != -1:
                        primary = normalized_game[:idx].strip(" ,;-@/\t")
                        break
                primary = _clean_team_fragment(primary, take_last=False) if primary else primary
                cleaned_detail = f"{primary} {cleaned_detail}".strip()
                prefixed = True
            if normalized_game and not prefixed and normalized_game.lower() not in cleaned_detail.lower():
                simple_game = normalized_game.lower()
                multi_matchup = "+" in normalized_game or "/" in normalized_game
                markers = (" vs", " @", " v ", " versus ", " at ")
                if multi_matchup or not any(marker in simple_game for marker in markers):
                    cleaned_detail = f"{normalized_game} {cleaned_detail}".strip()
            result["pick"] = cleaned_detail.strip()
        else:
            result["pick"] = original_pick
        if stake and not result["recommended_wager"]:
            result["recommended_wager"] = stake

    if result["pick"]:
        for emoji, sport_name in EMOJI_SPORT_MAP.items():
            if emoji in result["pick"]:
                result["pick"] = result["pick"].replace(emoji, "").strip()
                if not result["sport"]:
                    result["sport"] = sport_name
    if result["pick"] and not result.get("sport"):
        parenthetical_sport_match = re.search(r'\(([A-Za-z0-9 /]+)\)\s*$', result["pick"])
        if parenthetical_sport_match:
            parenthetical_text = parenthetical_sport_match.group(1).strip()
            inferred_sport = _sport_token_from_text(parenthetical_text)
            if inferred_sport:
                result["sport"] = inferred_sport
                result["pick"] = result["pick"][: parenthetical_sport_match.start()].strip()
        if not result.get("sport"):
            trailing_segment_match = re.search(r"\s-\s([A-Za-z0-9 .&'()/]+)$", result["pick"])
            if trailing_segment_match:
                trailing = trailing_segment_match.group(1).strip()
                if trailing and not re.search(r"\d", trailing):
                    inferred_sport = _sport_token_from_text(trailing)
                    if inferred_sport:
                        result["sport"] = inferred_sport
                        result["pick"] = re.sub(r"\s+", " ", result["pick"][: trailing_segment_match.start()]).strip(" ,:/\t")
        if " + " in result["pick"] and (not result.get("sport") or result["sport"] == "Parlay"):
            result["sport"] = result.get("sport") or "Parlay"
        lowered_pick = result["pick"].lower()
        if re.search(r'\b(kills?|maps?|valorant|cs2|cs:?go|headshots?)\b', lowered_pick) or "gentle mates" in lowered_pick:
            result["sport"] = result.get("sport") or "Esports"

    if not result["game"] and result["pick"]:
        raw_pick = result["pick"]
        embedded_matchup = _extract_matchup_from_text(raw_pick) or looks_like_plain_matchup(raw_pick)
        if embedded_matchup:
            result["game"] = embedded_matchup
            pattern = re.compile(re.escape(embedded_matchup), re.I)
            stripped_pick = pattern.sub("", raw_pick, count=1)
            stripped_pick = re.sub(r"\s*-\s*-\s*", " - ", stripped_pick)
            stripped_pick = re.sub(r"\s+", " ", stripped_pick).strip(" ,:/\t")
            if stripped_pick and any(ch.isalpha() for ch in stripped_pick):
                result["pick"] = stripped_pick

    if result["game"]:
        raw_game = result["game"]
        game_text, inferred_time = split_game_and_time(raw_game)
        if inferred_time and not result["time"]:
            cleaned_time = _clean_time_value(inferred_time)
            if cleaned_time:
                result["time"] = cleaned_time
        sport_from_game = None
        if game_text:
            normalized_game_text, sport_candidate = _normalize_game_text(game_text)
            if normalized_game_text is not None:
                game_text = normalized_game_text
            sport_from_game = sport_candidate
            if game_text and "(" in game_text and ")" in game_text:
                match = re.search(r"\(([^)]+)\)\s*$", game_text)
                if match:
                    inner_text = match.group(1).strip()
                    inner_matchup: Optional[str] = None
                    if inner_text and _has_matchup_hint(inner_text):
                        inner_matchup = looks_like_plain_matchup(inner_text) or _extract_matchup_from_text(inner_text)
                    if inner_matchup:
                        game_text = inner_matchup
                    else:
                        maybe_sport = inner_text
                        cleaned_sport = _sport_token_from_text(maybe_sport) or looks_like_sport_line(maybe_sport)
                        sport_from_game = sport_from_game or cleaned_sport or (
                            maybe_sport.title()
                            if maybe_sport and maybe_sport.isupper() and 2 <= len(maybe_sport) <= 40
                            else None
                        )
                        game_text = game_text[: match.start()].rstrip(" ,;-@/\t")
            elif game_text and "(" in game_text and ")" not in game_text:
                game_text = game_text.split("(", 1)[0].rstrip(" ,;-@/\t")
            if game_text and _has_matchup_hint(game_text):
                extracted_game = _extract_matchup_from_text(game_text)
                if extracted_game:
                    game_text = extracted_game
                    if not sport_from_game:
                        prefix_segment = raw_game.split(game_text, 1)[0].strip(" -*,:/\t")
                        inferred_prefix_sport = (
                            _sport_token_from_text(prefix_segment) if prefix_segment else None
                        )
                        if inferred_prefix_sport:
                            sport_from_game = inferred_prefix_sport
        game_text = re.sub(r"\s+", " ", game_text or "").strip(" -|\t") if game_text else None
        if game_text:
            for emoji, sport_name in EMOJI_SPORT_MAP.items():
                if emoji in game_text:
                    game_text = game_text.replace(emoji, "").strip()
                    sport_from_game = sport_from_game or sport_name
        result["game"] = game_text
        if sport_from_game and _should_replace_sport(result.get("sport"), sport_from_game):
            result["sport"] = sport_from_game
        # If still no sport, try detecting from team names in the game
        if not result.get("sport") and result.get("game"):
            team_sport = _detect_sport_from_team_names(result["game"])
            if team_sport:
                result["sport"] = team_sport
        if result["game"]:
            stripped_game = _strip_pick_value_prefix(str(result["game"]))
            if stripped_game and stripped_game != result["game"]:
                result["game"] = stripped_game
            sanitized_game = re.split(r"@\s*-?\d+(?:\.\d+)?", result["game"], maxsplit=1)[0]
            sanitized_game = re.sub(r"\s+\d+(?:\.\d+)?\s*(?:u|units?)$", "", sanitized_game, flags=re.I)
            sanitized_game = sanitized_game.strip(" ,;/@\t")
            sanitized_game = re.sub(r"\s*\((?![^)]*vs)[^)]*\)\s*$", "", sanitized_game, flags=re.I)
            result["game"] = sanitized_game

    if result["time"] and "|" in result["time"]:
        time_part, sport_hint = [part.strip() for part in result["time"].split("|", 1)]
        if time_part:
            result["time"] = time_part
        if sport_hint and not result.get("sport"):
            inferred = _sport_token_from_text(sport_hint)
            if inferred:
                result["sport"] = inferred
            elif len(sport_hint) <= 30:
                result["sport"] = sport_hint

    if result["sport"] and "potd" in result["sport"].lower():
        result["sport"] = None
    if result["pick"] and result["sport"] and result["pick"].strip().lower() == result["sport"].strip().lower():
        result["pick"] = None

    if result.get("pick") and not _is_reasonable_pick_text(result["pick"]):
        result["pick"] = None
    if result.get("game"):
        game_text = str(result["game"])
        lowered_game = game_text.lower()
        has_matchup_marker = _has_matchup_hint(game_text) or re.search(r"\bvs\b|\b@\b|\bv\b", lowered_game)
        if lowered_game.startswith("at ") and not re.search(r"\bat\s+[A-Za-z]", lowered_game):
            result["game"] = None
        elif not has_matchup_marker:
            result["game"] = None

    if not result.get("game"):
        alt_game = _find_first_matchup(material)
        if alt_game:
            result["game"] = alt_game

    canonical_pick = best_pick_candidate.get("pick") if best_pick_candidate else None
    final_pick = result.get("pick")
    if canonical_pick:
        if not final_pick or not _is_reasonable_pick_text(final_pick):
            final_pick = canonical_pick
        else:
            stripped_final = final_pick.lstrip()
            if stripped_final.startswith(("-", "+")):
                prefix = _extract_pick_prefix_from_candidate(canonical_pick)
                if prefix and prefix.lower() not in stripped_final.lower():
                    final_pick = f"{prefix} {stripped_final}"
    if final_pick:
        final_pick = _cleanup_pick_detail(final_pick)
        odds_suffix = aux.get("odds")
        if odds_suffix and len(odds_suffix) <= 30 and odds_suffix.lower() not in final_pick.lower():
            final_pick = f"{final_pick} @ {odds_suffix}"
        book_suffix = aux.get("book")
        if book_suffix and len(book_suffix) <= 30 and book_suffix.lower() not in final_pick.lower():
            final_pick = f"{final_pick} ({book_suffix})"
        result["pick"] = _cleanup_pick_detail(final_pick)

    if result.get("game"):
        normalized_game = looks_like_plain_matchup(result["game"])
        if normalized_game:
            result["game"] = normalized_game
        elif not _has_matchup_hint(result["game"]):
            result["game"] = None
    if not result.get("game"):
        alt_game = _find_first_matchup(material)
        if alt_game:
            result["game"] = alt_game
    if result.get("game"):
        lowered_game = str(result["game"]).lower()
        if lowered_game.startswith(("he ", "she ", "i ", "we ", "they ", "bonus", "record", "profit")):
            explicit = _find_explicit_match_line(material)
            if explicit:
                result["game"] = explicit
    if result.get("sport"):
        normalized_sport_value = _normalize_ascii(str(result["sport"])).lstrip("*_ ")
        lowered_sport = normalized_sport_value.lower()
        if lowered_sport.startswith(("bonus", "record", "he ", "she ", "i ", "we ", "they ")):
            explicit_sport = _find_explicit_sport_line(material)
            if explicit_sport:
                result["sport"] = explicit_sport
    else:
        explicit_sport = _find_explicit_sport_line(material)
        if explicit_sport:
            result["sport"] = explicit_sport

    if result.get("pick") and not any(ch.isalpha() for ch in str(result["pick"])):
        restored_pick: Optional[str] = None
        current_pick = str(result["pick"])
        for raw_line in material:
            ascii_line = _normalize_ascii(raw_line).replace("**", "").replace("__", "").strip()
            if not ascii_line:
                continue
            candidate = _strip_pick_value_prefix(ascii_line)
            if current_pick.strip() and current_pick.strip() not in candidate:
                if not candidate.endswith(current_pick.strip()):
                    continue
            if _looks_like_bet_detail(candidate) and any(ch.isalpha() for ch in candidate):
                restored_pick = candidate.strip()
                break
        if restored_pick:
            result["pick"] = restored_pick

    if result["pick"] and not result.get("recommended_wager"):
        trimmed_pick, stake_from_pick = peel_trailing_stake(result["pick"])
        if stake_from_pick:
            result["pick"] = trimmed_pick
            result["recommended_wager"] = stake_from_pick
    if result.get("pick"):
        result["pick"] = _cleanup_pick_detail(result["pick"])
        if result.get("game"):
            result["pick"] = _strip_team_moneyline_prefix(result["pick"], result["game"])
            result["pick"] = _cleanup_pick_detail(result["pick"])
            primary_team = _primary_team_from_game(result["game"])
            if primary_team:
                stripped_pick = result["pick"].lstrip()
                if stripped_pick.startswith(("-", "+")) and primary_team.lower() not in stripped_pick.lower():
                    result["pick"] = _cleanup_pick_detail(f"{primary_team} {result['pick']}")

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
    debug_entries: Optional[List[Dict[str, Any]]] = None,
    debug_output_path: Optional[Path] = None,
    debug_metadata: Optional[Dict[str, Any]] = None,
) -> List[PickEntry]:
    picks: List[PickEntry] = []
    for idx, comment in enumerate(comments):
        logger.debug("Processing comment index %d", idx)
        body = comment.get("body", "")
        logger.debug("Comment %d snippet: %s", idx, body[:120].replace("\n", " "))
        comment_debug: Optional[Dict[str, Any]] = None
        if debug_entries is not None:
            comment_debug = {
                "comment_index": idx,
                "comment_id": comment.get("id"),
                "author": comment.get("author", "unknown"),
                "source": source,
                "thread_title": thread_title,
                "permalink": f"{REDDIT_BASE}{comment.get('permalink', base_permalink)}",
                "body": body,
                "record": None,
                "record_error": None,
                "fields": None,
                "included": False,
                "skip_reason": None,
            }
            debug_entries.append(comment_debug)
            _flush_debug_snapshot(
                picks,
                debug_entries,
                debug_output_path,
                debug_metadata,
                reason="comment_initialized",
                comment_index=idx,
            )
        logger.debug("Starting record parse for comment %d", idx)
        try:
            record = parse_record(body)
        except Exception as exc:
            logger.error("Failed to parse record on comment %d: %s", idx, exc, exc_info=True)
            if comment_debug is not None:
                comment_debug["record_error"] = str(exc)
                comment_debug["skip_reason"] = "record_parse_error"
                _flush_debug_snapshot(
                    picks,
                    debug_entries,
                    debug_output_path,
                    debug_metadata,
                    reason="record_parse_error",
                    comment_index=idx,
                )
            continue
        if not record:
            logger.debug("No record found for comment %d", idx)
            if comment_debug is not None:
                comment_debug["skip_reason"] = "record_not_found"
                _flush_debug_snapshot(
                    picks,
                    debug_entries,
                    debug_output_path,
                    debug_metadata,
                    reason="record_missing",
                    comment_index=idx,
                )
            continue
        logger.debug("Record parsed for comment %d: %s", idx, record)
        wins = record.wins
        losses = record.losses
        pushes = record.pushes
        if comment_debug is not None:
            comment_debug["record"] = asdict(record)
            _flush_debug_snapshot(
                picks,
                debug_entries,
                debug_output_path,
                debug_metadata,
                reason="record_parsed",
                comment_index=idx,
            )
        lines = body.splitlines()
        logger.debug("Starting field extraction for comment %d (%d lines)", idx, len(lines))
        fields = extract_pick_fields(lines)
        logger.debug("Fields extracted for comment %d: %s", idx, fields)
        if fields.get("pick") is None and fields.get("game"):
            logger.debug("No explicit pick found for comment %d; using game as bet", idx)
            fields["pick"] = fields["game"]
        if fields.get("game") is None and fields.get("pick"):
            pick_text = str(fields["pick"])
            embedded_matchup = _extract_matchup_from_text(pick_text)
            if not embedded_matchup and _has_matchup_hint(pick_text):
                embedded_matchup = pick_text
            if embedded_matchup:
                logger.debug("No explicit game found for comment %d; using embedded matchup", idx)
                fields["game"] = embedded_matchup
        if comment_debug is not None:
            comment_debug["record"] = asdict(record)
            comment_debug["fields"] = dict(fields)
            _flush_debug_snapshot(
                picks,
                debug_entries,
                debug_output_path,
                debug_metadata,
                reason="fields_extracted",
                comment_index=idx,
            )
        required_keys = ("pick", "game")
        missing_required = [key for key in required_keys if not fields.get(key)]
        if missing_required:
            logger.debug("Skipping comment %d due to missing required fields: %s", idx, missing_required)
            if comment_debug is not None:
                comment_debug["skip_reason"] = f"missing_fields:{','.join(missing_required)}"
                comment_debug["fields"] = dict(fields)
                _flush_debug_snapshot(
                    picks,
                    debug_entries,
                    debug_output_path,
                    debug_metadata,
                    reason="missing_required",
                    comment_index=idx,
                )
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
        logger.debug(
            "Appended pick for comment %d: author=%s pick=%s", idx, comment.get("author", "unknown"), fields["pick"]
        )
        if comment_debug is not None:
            comment_debug["included"] = True
            comment_debug["pick_entry"] = asdict(picks[-1])
            _flush_debug_snapshot(
                picks,
                debug_entries,
                debug_output_path,
                debug_metadata,
                reason="comment_included",
                comment_index=idx,
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


def _flush_debug_snapshot(
    picks: List[PickEntry],
    entries: Optional[List[Dict[str, Any]]],
    output_path: Optional[Path],
    metadata: Optional[Dict[str, Any]],
    *,
    reason: str,
    comment_index: int,
) -> None:
    if entries is None or output_path is None:
        return
    snapshot_metadata: Dict[str, Any] = dict(metadata or {})
    snapshot_metadata.setdefault("partial", True)
    snapshot_metadata["last_comment_index"] = comment_index
    snapshot_metadata["snapshot_reason"] = reason
    try:
        write_debug_output(picks, entries, output_path, metadata=snapshot_metadata)
        logger.debug(
            "Wrote debug snapshot (%s) for comment %d to %s",
            reason,
            comment_index,
            output_path,
        )
    except Exception as exc:
        logger.error(
            "Failed to write debug snapshot (%s) for comment %d: %s",
            reason,
            comment_index,
            exc,
            exc_info=True,
        )


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


def write_debug_output(
    picks: List[PickEntry],
    entries: List[Dict[str, Any]],
    output: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    included_comments = sum(1 for entry in entries if entry.get("included"))
    payload: Dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metadata": metadata or {},
        "summary": {
            "total_comments": len(entries),
            "comments_with_picks": included_comments,
        },
        "picks": [asdict(pick) for pick in picks],
        "comments": entries,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    parser.add_argument(
        "--debug-output",
        default=None,
        help="Optional JSON file capturing comment parsing diagnostics",
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
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.debug("Parsed arguments: %s", args)
    client = RedditClient(user_agent=args.user_agent)

    all_picks: List[PickEntry] = []
    thread_titles: List[str] = []
    debug_entries: Optional[List[Dict[str, Any]]] = [] if args.debug_output else None
    debug_output_path: Optional[Path] = Path(args.debug_output) if args.debug_output else None

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
            debug_entries=debug_entries,
            debug_output_path=debug_output_path,
            debug_metadata=(
                {
                    "source": source_label,
                    "thread_title": thread_title,
                    "thread_permalink": permalink,
                }
                if debug_entries is not None
                else None
            ),
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
                debug_entries=debug_entries,
                debug_output_path=debug_output_path,
                debug_metadata=(
                    {
                        "source": config.subreddit,
                        "thread_title": thread_title,
                        "thread_permalink": thread_data.get("permalink", "/"),
                    }
                    if debug_entries is not None
                    else None
                ),
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

    total_picks_before_limit = len(all_picks)
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

    if debug_entries is not None and debug_output_path is not None:
        metadata = {
            "report_title": report_title,
            "limit": args.limit,
            "subreddit_args": args.subreddits,
            "thread_url": args.thread_url,
            "base_url": args.base_url,
            "total_picks_before_limit": total_picks_before_limit,
            "total_picks_after_limit": len(all_picks),
            "thread_titles": thread_titles,
        }
        write_debug_output(all_picks, debug_entries, debug_output_path, metadata=metadata)
        print(f"Wrote debug details for {len(debug_entries)} comments to {debug_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
