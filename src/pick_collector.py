#!/usr/bin/env python3
"""Collect top picks from Reddit betting threads (Pick of the Day, Best Bets, etc.)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
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

RE_RECORD = re.compile(r"\b(\d{1,3})-(\d{1,3})(?:-(\d{1,3}))?\b")
RE_RECORD_LINE = re.compile(r"\brecord\b", re.I)
RE_PARENTHESES_PUSH = re.compile(
    r"^\s*\(\s*(\d{1,3})\s*(?:push(?:es)?|ties?|draws?)\s*\)"
    r"|^\s*\(\s*(\d{1,3})\s*\)"
    r"|^\s*[\-–—]\s*(\d{1,3})\s*(?:push(?:es)?|ties?|draws?)\b",
    re.I,
)
BETA_ALPHA = 5.0
BETA_BETA = 5.0
DEFAULT_TITLE_QUERIES = {
    "sportsbook": 'title:"Pick of the Day"',
    "sportsbetting": 'title:"Best Bets"',
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
            r"^\s*[-•*>\u2022\u2013\u2014]*\s*(?:units?(?:\s+played)?|unit\s*size|stake|risk|(?:recommended\s*)?wager(?:\s+amount)?|bet\s*size|investment|units?\s*risked)\s*(?:[:\-\u2013\u2014|]\s*)?(.*)$",
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
    re.compile(r"[-+]\s?\d+(?:\.\d+)?"),
]
TIME_IN_TEXT = re.compile(
    r"(\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)\s*(?:[A-Z]{2,5})?)",
    re.I,
)
LIKELY_GAME_TEXT = re.compile(r"(?:\bvs?\.?\b|\bversus\b|\bat\b|@|\bv\b)", re.I)
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
    match = TRAILING_STAKE.search(stripped)
    if not match:
        return detail, None
    stake = stripped[match.start() :].strip(" ,;-")
    trimmed = stripped[: match.start()].rstrip(" ,;-")
    return trimmed, _clean_stake_text(stake)


def _clean_stake_text(stake: Optional[str]) -> Optional[str]:
    if not stake:
        return None
    cleaned = stake.strip()
    # Remove trailing stake separators that sneak in from prefixes like "1u -"
    cleaned = cleaned.rstrip("-: ")
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
    candidate = text.strip()
    if not candidate:
        return None
    if not re.search(r"[a-zA-Z]", candidate):
        return None
    if len(candidate) > 120:
        return None
    if candidate.count(".") >= 2:
        return None
    lowered = candidate.lower()
    normalized_prefix = candidate.lstrip("_*•-> \t").lower()
    if normalized_prefix.startswith("week "):
        return None
    if "bet" in lowered and not LIKELY_GAME_TEXT.search(candidate):
        return None
    if any(word in lowered for word in ("record", "analysis", "units", "odds", "stake", "roi", "notes")):
        return None
    if LIKELY_GAME_TEXT.search(candidate):
        return re.sub(r"\s+", " ", candidate)
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
    candidate = text.strip()
    if not candidate:
        return None
    lowered = candidate.lower()
    if "record" in lowered or lowered.startswith(("net units", "units", "last pick", "previous pick")):
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
            matchup = looks_like_plain_matchup(normalized)
            if matchup:
                return matchup, detail_clean
            if LIKELY_GAME_TEXT.search(normalized):
                return normalized, detail_clean
        fallback_detail = text.strip() or None
        if detail_clean and not fallback_detail:
            fallback_detail = detail_clean
        return None, fallback_detail

    for pattern in SPLIT_MARKERS:
        match = pattern.search(text)
        if match:
            idx = match.start()
            game = text[:idx].strip(" ,;-@/\t")
            detail = text[idx:].strip()
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

    ou_match = re.search(r"\s[ou][0-9]", text, re.I)
    if ou_match:
        idx = ou_match.start()
        game = text[:idx].strip(" ,;-@/\t")
        detail = text[idx + 1 :].strip()
        if game and looks_like_bet_prefix(game):
            return finalize(None, text)
        return finalize(game, detail)

    if "(" in text and LIKELY_GAME_TEXT.search(text):
        idx = text.index("(")
        game = text[:idx].strip(" ,;-@/\t")
        if game and looks_like_bet_prefix(game):
            return finalize(None, text)
        return finalize(game, text[idx:])

    if "," in text:
        head, tail = text.split(",", 1)
        head = head.strip()
        tail = tail.strip()
        if head and tail:
            if looks_like_bet_prefix(head):
                return finalize(None, text)
            return finalize(head, tail)

    return finalize(None, text.strip() or None)


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
    stake: Optional[str] = None
    stake_match = STAKE_PREFIX.match(cleaned)
    if stake_match:
        stake = _clean_stake_text(stake_match.group(0))
        cleaned = cleaned[stake_match.end() :].strip()
    cleaned = cleaned.lstrip("-: ")
    game, detail = split_game_and_detail(cleaned)
    if detail:
        detail, trailing_stake = peel_trailing_stake(detail)
        if trailing_stake and not stake:
            stake = _clean_stake_text(trailing_stake)
    return game, detail, _clean_stake_text(stake)


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


def _strip_pick_value_prefix(value: str) -> str:
    trimmed = value.strip()
    trimmed = re.sub(
        r"^(?:of\s+the\s+day|today'?s\s+pick|today'?s\s+play|todays\s+pick|todays\s+play|potd|pick|play|bet)\b[:\-\u2013\u2014|\s]*",
        "",
        trimmed,
        flags=re.I,
    )
    return trimmed.strip()


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
    " +",
    " -",
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
    candidate = text.strip()
    if not candidate:
        return False
    if len(candidate) > 140:
        return False
    lowered = f" {candidate.lower()} "
    if looks_like_plain_matchup(candidate):
        if not any(token in lowered for token in BET_DETAIL_KEYWORDS):
            return False
    if looks_like_sport_line(candidate):
        return False
    if any(token in lowered for token in BET_DETAIL_KEYWORDS):
        return True
    if re.search(r"[+-]\s*\d", candidate):
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
    segment = fragment
    for splitter in ("|", "/"):
        parts = segment.split(splitter)
        segment = parts[-1] if take_last else parts[0]
    for splitter in (" - ", "-", ":", ","):
        if splitter in segment:
            parts = segment.split(splitter)
            segment = parts[-1] if take_last else parts[0]
    lowered_segment = segment.lower()
    if " in " in lowered_segment:
        parts = re.split(r"\s+in\s+", segment, 1)
        segment = parts[-1] if take_last else parts[0]
        lowered_segment = segment.lower()
    if " at " in lowered_segment:
        parts = re.split(r"\s+at\s+", segment, 1)
        segment = parts[-1] if take_last else parts[0]
    segment = _strip_leading_sport_words(segment)
    segment = re.sub(r"\s+", " ", segment).strip()
    return segment


def _extract_matchup_from_text(text: str) -> Optional[str]:
    lowered = text.lower()
    for raw_sep, normalized_sep in MATCHUP_SEPARATORS:
        pos = lowered.find(raw_sep)
        if pos == -1:
            continue
        left = text[:pos]
        right = text[pos + len(raw_sep) :]
        team_left = _clean_team_fragment(left, take_last=True)
        team_right = _clean_team_fragment(right, take_last=False)
        if team_left and team_right and any(ch.isalpha() for ch in team_left) and any(ch.isalpha() for ch in team_right):
            return f"{team_left} {normalized_sep} {team_right}"
    return None


def _find_followup_bet(lines: List[str], start_index: int) -> Optional[str]:
    for idx in range(start_index, len(lines)):
        candidate_line = lines[idx]
        candidate = candidate_line.strip()
        if not candidate:
            continue
        if _is_field_line(candidate_line):
            continue
        normalized_lower = candidate.lower().lstrip("_*•-> \t")
        if normalized_lower.startswith(
            ("write up", "write-up", "analysis", "last pick", "previous pick", "prior pick")
        ):
            continue
        if normalized_lower.startswith("record"):
            continue
        if looks_like_record_heading(candidate):
            continue
        if _looks_like_bet_detail(candidate):
            return candidate
    return None


def _sport_token_from_text(text: str) -> Optional[str]:
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
    r"^(?:today'?s\s+)?(?:event|match(?:up)?|game|fixture|bet)\s*(?:[:\-\u2013\u2014|]\s*)",
    re.I,
)


def _normalize_game_text(game: str) -> Tuple[Optional[str], Optional[str]]:
    cleaned = re.sub(r"\s+", " ", game).strip(" -*|\t")
    cleaned = cleaned.lstrip("_*#> ")
    cleaned = re.sub(r"^[^A-Za-z0-9]+", "", cleaned)
    cleaned = "".join(ch for ch in cleaned if not 0x1F1E6 <= ord(ch) <= 0x1F1FF)
    cleaned = cleaned.replace("\u2019", "'")
    cleaned = cleaned.rstrip("* ")
    bracket_match = re.search(r"\[([^\]]+)\]", cleaned)
    sport_from_bracket: Optional[str] = None
    if bracket_match:
        bracket_content = bracket_match.group(1).strip()
        sport_from_bracket = _sport_token_from_text(bracket_content) or bracket_content
        cleaned = (cleaned[: bracket_match.start()] + cleaned[bracket_match.end() :]).strip()
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
        if not LIKELY_GAME_TEXT.search(leading):
            candidate = _sport_token_from_text(leading)
            if candidate:
                sport_candidate = candidate
                cleaned = remainder.strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|\t")
    if sport_from_bracket:
        if not sport_candidate or sport_candidate.lower() in {"soccer", "football"}:
            sport_candidate = sport_from_bracket
    return cleaned or None, sport_candidate or sport_from_emoji or sport_from_bracket


def _clean_time_value(value: str) -> str:
    cleaned = value.strip()
    cleaned = cleaned.replace("&amp;", "&")
    cleaned = cleaned.strip("* ")
    cleaned = re.sub(r"^(?:&\s*)?tv[:\-\s]*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(?:kick(?:-?off)?\s*)", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^start\s*time[:\-\s]*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned)
    return cleaned.strip()


def _cleanup_pick_detail(detail: str) -> str:
    trimmed = detail.strip()
    trimmed = re.sub(r"[,;\-\s]*(?:for|risk)\s*$", "", trimmed, flags=re.I)
    return trimmed.strip(" ,;-")


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
    game_line_index: Optional[int] = None
    sport_line_index: Optional[int] = None
    pick_line_index: Optional[int] = None
    in_previous_pick_block = False

    for idx, line in enumerate(material):
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        normalized_lowered = lowered.lstrip("_*•-> \t")
        if normalized_lowered.startswith(("last pick", "previous pick", "prior pick", "prior potd")):
            in_previous_pick_block = True
            continue
        if normalized_lowered.startswith(("today's pick", "todays pick", "potd", "event", "game", "pick")):
            in_previous_pick_block = False
        if in_previous_pick_block and "|" in stripped:
            in_previous_pick_block = False
        if normalized_lowered.startswith("units won"):
            continue
        if looks_like_record_heading(stripped):
            continue
        if in_previous_pick_block:
            continue
        captured = False
        for key, patterns in FIELD_PATTERNS.items():
            if key != "pick" and result[key]:
                continue
            for pattern in patterns:
                if key == "pick":
                    normalized_preview = line.lower().replace("\u2019", "'")
                    if not any(token in normalized_preview for token in PICK_KEY_TOKENS):
                        continue
                normalized_line = line.replace("\u2019", "'")
                normalized_line = "".join(
                    ch
                    for ch in normalized_line
                    if not (0x2600 <= ord(ch) <= 0x27FF or 0x1F300 <= ord(ch) <= 0x1FAFF)
                )
                match = pattern.match(normalized_line)
                if match:
                    value = match.group(1).strip()
                    if not value:
                        value = _next_non_empty(material, idx + 1) or ""
                        if value and _is_field_line(value):
                            value = ""
                        value = value.strip()
                    if value and key == "pick":
                        if "👉" in line:
                            continue
                        value = _strip_pick_value_prefix(value)
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
                        normalized_sport = _sport_token_from_text(value)
                        if normalized_sport:
                            value = normalized_sport
                    if value and key == "recommended_wager":
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
                        else:
                            result[key] = value
                        if key == "pick" and value and not result.get("sport"):
                            for emoji, sport_name in EMOJI_SPORT_MAP.items():
                                if emoji in line:
                                    result["sport"] = sport_name
                                    break
                        if key == "pick" and value:
                            pick_line_index = idx
                        if key == "game" and value:
                            game_line_index = idx
                        if key == "sport" and value:
                            sport_line_index = idx
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
            if not LIKELY_GAME_TEXT.search(current_game):
                needs_game = True
            elif game_line_index is not None and idx > game_line_index and (idx - game_line_index) <= 8:
                needs_game = True
        if needs_game:
            matchup = looks_like_plain_matchup(stripped)
            if matchup:
                result["game"] = matchup
                game_line_index = idx
                continue
        if not result["sport"]:
            sport_line = looks_like_sport_line(stripped)
            if sport_line:
                normalized_sport = _sport_token_from_text(sport_line) or sport_line
                result["sport"] = normalized_sport
                sport_line_index = idx
                continue
            auto_sport = _sport_token_from_text(stripped)
            if auto_sport:
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
                        else:
                            aux[key] = value
                    break

    if result["pick"]:
        original_pick = result["pick"].strip()
        game, detail, stake = parse_pick_text(original_pick)
        normalized_game: Optional[str] = None
        if game:
            raw_game = re.sub(r"\s+", " ", game).strip()
            if "+" in raw_game and not LIKELY_GAME_TEXT.search(raw_game):
                parts = [part.strip() for part in raw_game.split("+") if part.strip()]
                if parts:
                    raw_game = " / ".join(parts)
            if LIKELY_GAME_TEXT.search(raw_game):
                normalized_game = looks_like_plain_matchup(raw_game) or raw_game
        cleaned_detail = detail.strip() if detail else ""
        if cleaned_detail:
            cleaned_detail = _cleanup_pick_detail(cleaned_detail)
        if normalized_game:
            existing_game = result.get("game") or ""
            if not existing_game or len(normalized_game) > len(existing_game):
                result["game"] = normalized_game
            if cleaned_detail:
                lowered_detail = cleaned_detail.lower()
                lowered_game = normalized_game.lower()
                if lowered_detail.startswith(lowered_game):
                    trimmed = cleaned_detail[len(normalized_game) :].lstrip(" ,;-@")
                    if trimmed:
                        cleaned_detail = trimmed
        if cleaned_detail:
            if normalized_game and normalized_game.lower() not in cleaned_detail.lower():
                simple_game = normalized_game.lower()
                if not any(marker in simple_game for marker in (" vs", " @", " v ", " versus ", " at ")):
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
        trailing_segment_match = re.search(r"\s-\s([A-Za-z0-9 .&'()/]+)$", result["pick"])
        if trailing_segment_match:
            trailing = trailing_segment_match.group(1).strip()
            if trailing and not re.search(r"\d", trailing):
                inferred_sport = _sport_token_from_text(trailing)
                if inferred_sport:
                    result["sport"] = inferred_sport
                    result["pick"] = re.sub(r"\s+", " ", result["pick"][: trailing_segment_match.start()]).strip(" -,:/\t")
        if " + " in result["pick"]:
            result["sport"] = "Parlay"

    if not result["game"] and result["pick"]:
        raw_pick = result["pick"]
        embedded_matchup = _extract_matchup_from_text(raw_pick) or looks_like_plain_matchup(raw_pick)
        if embedded_matchup:
            result["game"] = embedded_matchup
            pattern = re.compile(re.escape(embedded_matchup), re.I)
            stripped_pick = pattern.sub("", raw_pick, count=1)
            stripped_pick = re.sub(r"\s*-\s*-\s*", " - ", stripped_pick)
            stripped_pick = re.sub(r"\s+", " ", stripped_pick).strip(" -,:/\t")
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
                    maybe_sport = match.group(1).strip()
                    cleaned_sport = _sport_token_from_text(maybe_sport) or looks_like_sport_line(maybe_sport)
                    sport_from_game = sport_from_game or cleaned_sport or (
                        maybe_sport.title()
                        if maybe_sport.isupper() and 2 <= len(maybe_sport) <= 40
                        else None
                    )
                    game_text = game_text[: match.start()].rstrip(" ,;-@/\t")
            if game_text and LIKELY_GAME_TEXT.search(game_text):
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
    debug_entries: Optional[List[Dict[str, Any]]] = None,
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
        try:
            record = parse_record(body)
        except Exception as exc:
            logger.error("Failed to parse record on comment %d: %s", idx, exc, exc_info=True)
            if comment_debug is not None:
                comment_debug["record_error"] = str(exc)
                comment_debug["skip_reason"] = "record_parse_error"
                debug_entries.append(comment_debug)
            continue
        if not record:
            logger.debug("No record found for comment %d", idx)
            if comment_debug is not None:
                comment_debug["skip_reason"] = "record_not_found"
                debug_entries.append(comment_debug)
            continue
        logger.debug("Record parsed for comment %d: %s", idx, record)
        wins = record.wins
        losses = record.losses
        pushes = record.pushes
        fields = extract_pick_fields(body.splitlines())
        logger.debug("Fields extracted for comment %d: %s", idx, fields)
        if fields.get("pick") is None and fields.get("game"):
            logger.debug("No explicit pick found for comment %d; using game as bet", idx)
            fields["pick"] = fields["game"]
        if comment_debug is not None:
            comment_debug["record"] = asdict(record)
            comment_debug["fields"] = dict(fields)
        required_keys = ("pick", "game")
        missing_required = [key for key in required_keys if not fields.get(key)]
        if missing_required:
            logger.debug("Skipping comment %d due to missing required fields: %s", idx, missing_required)
            if comment_debug is not None:
                comment_debug["skip_reason"] = f"missing_fields:{','.join(missing_required)}"
                comment_debug["fields"] = dict(fields)
                debug_entries.append(comment_debug)
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
            debug_entries.append(comment_debug)
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

    if debug_entries is not None and args.debug_output:
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
        write_debug_output(all_picks, debug_entries, Path(args.debug_output), metadata=metadata)
        print(f"Wrote debug details for {len(debug_entries)} comments to {args.debug_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
