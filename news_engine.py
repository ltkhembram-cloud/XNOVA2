#!/usr/bin/env python3
"""Minimal World News Engine.

Query -> Provider Search -> Normalize -> Deduplicate -> Evidence Assessment -> SQLite -> Results

Providers: Tavily Search API, News API (/v2/everything), Google News RSS.
No LLM is used. Evidence assessment is deliberately conservative: it works from
headlines/snippets only, so it never assigns CORROBORATED or CONFLICTING (those
need claim-level comparison of full article text, which this version does not do).
"""
from __future__ import annotations

import argparse
import hashlib
import html
import logging
import os
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import requests

log = logging.getLogger("news_engine")

# --------------------------------------------------------------------------- #
# Configuration and constants
# --------------------------------------------------------------------------- #
TAVILY_URL = "https://api.tavily.com/search"
NEWSAPI_URL = "https://newsapi.org/v2/everything"
GOOGLE_RSS_URL = "https://news.google.com/rss/search"

# Languages accepted by News API's `language` filter (check current docs; Bengali is not among them).
NEWSAPI_LANGUAGES = {"ar", "de", "en", "es", "fr", "he", "it", "nl", "no", "pt", "ru", "sv", "ud", "zh"}
TRACKING_PARAMS = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "igshid", "yclid", "_hsenc", "_hsmi"}
RETRY_STATUSES = {429, 500, 502, 503, 504}
EVIDENCE_STATUSES = ("CORROBORATED", "PARTIALLY_CORROBORATED", "UNVERIFIED", "CONFLICTING", "NOT_ASSESSED")
DUP_WINDOW = timedelta(hours=48)  # same-story window for headline matches
MIN_TITLE_WORDS = 4               # shorter headlines are too generic to match on
RANK_WEIGHTS = {"relevance": 0.45, "recency": 0.25, "source": 0.10, "evidence": 0.20}  # ranking only, not truth
DUPLICATE_PENALTY = 0.15
EVIDENCE_STRENGTH = {"CORROBORATED": 1.0, "PARTIALLY_CORROBORATED": 0.5, "UNVERIFIED": 0.1,
                     "CONFLICTING": 0.0, "NOT_ASSESSED": 0.0}
RECENCY_HALF_LIFE_HOURS = 48.0
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class Config:
    db_path: str = "news_engine.db"
    timeout: float = 10.0            # per HTTP request, seconds
    deadline: float = 30.0           # overall provider deadline, seconds
    retries: int = 2                 # extra attempts on 429/5xx/network errors (bounded backoff)
    max_results: int = 20            # per provider
    show: int = 20                   # results displayed
    days: Optional[int] = None       # restrict to the last N days where the provider supports it
    language: Optional[str] = None   # e.g. "en", "bn"; auto-detected from the query if None
    region: Optional[str] = None     # e.g. "US", "BD"
    trusted_domains: frozenset = frozenset()  # optional, user-supplied ranking boost


class ProviderError(Exception):
    """A provider failed in a way we can describe safely (never contains secrets)."""


@dataclass
class Article:
    title: str
    canonical_url: str
    url_hash: str
    title_norm: str
    description: str
    source_name: Optional[str]
    source_domain: Optional[str]
    published_at: Optional[str]          # UTC "YYYY-MM-DDTHH:MM:SSZ" or None
    providers: set = field(default_factory=set)


@dataclass
class Ctx:
    cfg: Config
    session: requests.Session
    deadline_at: float                   # time.monotonic() value

    @classmethod
    def default(cls) -> "Ctx":
        cfg = Config()
        return cls(cfg, make_session(), time.monotonic() + cfg.deadline)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "minimal-world-news-engine/1.0"
    return s


# --------------------------------------------------------------------------- #
# Normalization helpers (pure functions)
# --------------------------------------------------------------------------- #
def _is_tracking(key: str) -> bool:
    k = key.lower()
    return k.startswith("utm_") or k in TRACKING_PARAMS


def normalize_url(url: object) -> Optional[str]:
    """Canonicalize an http(s) URL. Keeps path case and meaningful query parameters."""
    if not isinstance(url, str):
        return None
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    query = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k))
    return urlunsplit((scheme, netloc, path, urlencode(query), ""))  # userinfo and fragment dropped


def url_fingerprint(canonical_url: str) -> str:
    return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()


def domain_of(url: object) -> Optional[str]:
    canon = normalize_url(url)
    if not canon:
        return None
    return urlsplit(canon).hostname or None


def normalize_text(text: Optional[str]) -> str:
    """Case-fold and keep letters, combining marks and digits (so Bengali is preserved)."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    chars = [c if unicodedata.category(c)[0] in "LMN" else " " for c in text]
    return " ".join("".join(chars).split())


def clean_text(raw: object, limit: int = 1000) -> str:
    if not isinstance(raw, str):
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return " ".join(text.split())[:limit]


def parse_timestamp(value: object, now: Optional[datetime] = None) -> Optional[str]:
    """ISO 8601 or RFC 822 -> UTC string. Returns None for anything invalid; never invents a date.
    Naive timestamps are assumed to be UTC."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    now = now or datetime.now(timezone.utc)
    if dt.year < 1990 or dt > now + timedelta(days=1):
        return None
    return dt.strftime(TS_FORMAT)


def _load_ts(value: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.strptime(value, TS_FORMAT).replace(tzinfo=timezone.utc) if value else None
    except ValueError:
        return None


def resolve_locale(query: str, cfg: Config) -> tuple[str, str]:
    if cfg.language:
        return cfg.language, (cfg.region or "US").upper()
    if re.search(r"[\u0980-\u09FF]", query):  # Bengali script
        return "bn", (cfg.region or "BD").upper()
    return "en", (cfg.region or "US").upper()


def build_article(*, provider: str, title: object, url: object, description: object = None,
                  source_name: object = None, source_domain: Optional[str] = None,
                  published: object = None) -> Optional[Article]:
    """Normalize one provider record. Returns None if title or URL is unusable (nothing is invented)."""
    canonical = normalize_url(url)
    title_clean = clean_text(title, 500)
    if not canonical or not title_clean:
        return None
    domain = source_domain or domain_of(canonical)
    name = clean_text(source_name, 200) or domain
    return Article(
        title=title_clean, canonical_url=canonical, url_hash=url_fingerprint(canonical),
        title_norm=normalize_text(title_clean), description=clean_text(description),
        source_name=name, source_domain=domain, published_at=parse_timestamp(published),
        providers={provider},
    )


# --------------------------------------------------------------------------- #
# HTTP with bounded retries
# --------------------------------------------------------------------------- #
def request_with_retry(session, method: str, url: str, cfg: Config, deadline_at: float, **kwargs):
    """Bounded exponential backoff; error text never includes URLs, headers or keys."""
    attempt = 0
    while True:
        remaining = deadline_at - time.monotonic()
        if remaining <= 0:
            raise ProviderError("search deadline exceeded")
        try:
            resp = session.request(method, url, timeout=min(cfg.timeout, remaining), **kwargs)
        except (requests.Timeout, requests.ConnectionError) as exc:
            reason = type(exc).__name__
        except requests.RequestException as exc:
            raise ProviderError(type(exc).__name__) from None
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code not in RETRY_STATUSES:
                raise ProviderError(f"HTTP {resp.status_code}")
            reason = f"HTTP {resp.status_code}"
        if attempt >= cfg.retries:
            raise ProviderError(f"{reason} after {attempt + 1} attempts")
        time.sleep(max(0.0, min(0.5 * 2 ** attempt, 4.0, deadline_at - time.monotonic())))
        attempt += 1


def _json_object(resp) -> dict:
    try:
        data = resp.json()
    except ValueError:
        raise ProviderError("invalid JSON response") from None
    if not isinstance(data, dict):
        raise ProviderError("unexpected JSON structure")
    return data


# --------------------------------------------------------------------------- #
# Providers: each returns list[Article] or raises ProviderError
# --------------------------------------------------------------------------- #
def search_tavily(query: str, ctx: Optional[Ctx] = None) -> list[Article]:
    ctx = ctx or Ctx.default()
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise ProviderError("TAVILY_API_KEY not set")
    body: dict = {"query": query, "topic": "news", "search_depth": "basic",
                  "max_results": max(1, min(ctx.cfg.max_results, 20))}
    if ctx.cfg.days:
        body["days"] = ctx.cfg.days  # otherwise Tavily's own default window applies
    resp = request_with_retry(ctx.session, "POST", TAVILY_URL, ctx.cfg, ctx.deadline_at,
                              json=body, headers={"Authorization": f"Bearer {key}"})
    items = _json_object(resp).get("results")
    if not isinstance(items, list):
        raise ProviderError("response has no 'results' list")
    out = []
    for it in items:
        if isinstance(it, dict):
            art = build_article(provider="tavily", title=it.get("title"), url=it.get("url"),
                                description=it.get("content"), published=it.get("published_date"))
            if art:
                out.append(art)
    return out


def search_newsapi(query: str, ctx: Optional[Ctx] = None) -> list[Article]:
    ctx = ctx or Ctx.default()
    key = os.environ.get("NEWSAPI_API_KEY")
    if not key:
        raise ProviderError("NEWSAPI_API_KEY not set")
    lang, _ = resolve_locale(query, ctx.cfg)
    params: dict = {"q": query, "sortBy": "relevancy", "pageSize": max(1, min(ctx.cfg.max_results, 100))}
    if lang in NEWSAPI_LANGUAGES:  # unsupported languages (e.g. Bengali) are searched without a filter
        params["language"] = lang
    if ctx.cfg.days:
        since = datetime.now(timezone.utc) - timedelta(days=ctx.cfg.days)
        params["from"] = since.strftime("%Y-%m-%dT%H:%M:%S")
    resp = request_with_retry(ctx.session, "GET", NEWSAPI_URL, ctx.cfg, ctx.deadline_at,
                              params=params, headers={"X-Api-Key": key})  # key in header, never in the URL
    data = _json_object(resp)
    if data.get("status") == "error":
        raise ProviderError(f"API error {data.get('code', 'unknown')}")
    items = data.get("articles")
    if not isinstance(items, list):
        raise ProviderError("response has no 'articles' list")
    out = []
    for it in items:
        if not isinstance(it, dict) or it.get("title") == "[Removed]":
            continue
        src = it.get("source") if isinstance(it.get("source"), dict) else {}
        art = build_article(provider="newsapi", title=it.get("title"), url=it.get("url"),
                            description=it.get("description"), source_name=src.get("name"),
                            published=it.get("publishedAt"))
        if art:
            out.append(art)
    return out


def search_google_news_rss(query: str, ctx: Optional[Ctx] = None) -> list[Article]:
    """Google News RSS is an unofficial feed, not a search API. Links are Google redirect URLs, so
    the publisher domain comes from the feed's <source> element, not from the link."""
    ctx = ctx or Ctx.default()
    lang, region = resolve_locale(query, ctx.cfg)
    q = f"{query} when:{ctx.cfg.days}d" if ctx.cfg.days else query
    params = {"q": q, "hl": f"{lang}-{region}" if lang == "en" else lang, "gl": region, "ceid": f"{region}:{lang}"}
    resp = request_with_retry(ctx.session, "GET", GOOGLE_RSS_URL, ctx.cfg, ctx.deadline_at, params=params)
    feed = feedparser.parse(resp.content)
    if getattr(feed, "bozo", False) and not feed.entries:
        raise ProviderError("malformed RSS feed")
    out = []
    for e in feed.entries[: ctx.cfg.max_results]:
        src = e.get("source") or {}
        name = clean_text(src.get("title"), 200)
        title = clean_text(e.get("title"), 500)
        if name and title.endswith(f" - {name}"):
            title = title[: -len(name) - 3]  # Google appends " - Publisher" to headlines
        art = build_article(provider="google_news_rss", title=title, url=e.get("link"), description="",
                            source_name=name or None, source_domain=domain_of(src.get("href")),
                            published=e.get("published"))
        if art:
            out.append(art)
    return out


PROVIDERS: dict[str, Callable[[str, Ctx], list[Article]]] = {
    "tavily": search_tavily,
    "newsapi": search_newsapi,
    "google_news_rss": search_google_news_rss,
}


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #
def merge_batch(batches: Iterable[list[Article]]) -> list[Article]:
    """Stage 1-3: identical canonical URLs collapse into one article with several discovery routes."""
    merged: dict[str, Article] = {}
    for batch in batches:
        for art in batch:
            cur = merged.get(art.url_hash)
            if cur is None:
                merged[art.url_hash] = art
                continue
            cur.providers |= art.providers
            cur.description = cur.description or art.description
            cur.published_at = cur.published_at or art.published_at
            cur.source_name = cur.source_name or art.source_name
    return list(merged.values())


def same_story_context(domain_a: Optional[str], pub_a: Optional[str],
                       domain_b: Optional[str], pub_b: Optional[str]) -> bool:
    """Context check applied AFTER an exact normalized-title match (title alone is never enough).
    Both dates known -> must be within the window. A missing date -> only the same publisher counts."""
    ta, tb = _load_ts(pub_a), _load_ts(pub_b)
    if ta and tb:
        return abs(ta - tb) <= DUP_WINDOW
    return bool(domain_a) and domain_a == domain_b


def find_possible_duplicate(conn: sqlite3.Connection, art: Article) -> Optional[int]:
    if len(art.title_norm.split()) < MIN_TITLE_WORDS:
        return None
    rows = conn.execute(
        "SELECT COALESCE(possible_duplicate_of, id) AS root, source_domain, published_at "
        "FROM articles WHERE title_norm = ? AND url_hash <> ? ORDER BY id LIMIT 50",
        (art.title_norm, art.url_hash),
    ).fetchall()
    for r in rows:
        if same_story_context(art.source_domain, art.published_at, r["source_domain"], r["published_at"]):
            return r["root"]
    return None


# --------------------------------------------------------------------------- #
# SQLite
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id                    INTEGER PRIMARY KEY,
    canonical_url         TEXT NOT NULL,
    url_hash              TEXT NOT NULL UNIQUE,
    title                 TEXT NOT NULL,
    title_norm            TEXT NOT NULL,
    description           TEXT,
    source_name           TEXT,
    source_domain         TEXT,
    published_at          TEXT,
    first_seen_at         TEXT NOT NULL,
    last_seen_at          TEXT NOT NULL,
    verification_status   TEXT NOT NULL DEFAULT 'NOT_ASSESSED'
        CHECK (verification_status IN ('CORROBORATED','PARTIALLY_CORROBORATED','UNVERIFIED','CONFLICTING','NOT_ASSESSED')),
    verification_notes    TEXT NOT NULL DEFAULT '',
    possible_duplicate_of INTEGER REFERENCES articles(id)
);
CREATE INDEX IF NOT EXISTS ix_articles_title_norm ON articles(title_norm);
CREATE TABLE IF NOT EXISTS article_sources (
    article_id    INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    provider      TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL,
    PRIMARY KEY (article_id, provider)
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def upsert_article(conn: sqlite3.Connection, art: Article, now_iso: str) -> tuple[int, str]:
    """Insert or refresh one article. Returns (article_id, 'new' | 'seen' | 'possible_duplicate').
    Call inside a transaction (`with conn:`)."""
    row = conn.execute("SELECT id, possible_duplicate_of FROM articles WHERE url_hash = ?",
                       (art.url_hash,)).fetchone()
    if row:
        article_id = row["id"]
        conn.execute(
            "UPDATE articles SET last_seen_at = ?, "
            "published_at = COALESCE(published_at, ?), "
            "description = CASE WHEN description IS NULL OR description = '' THEN ? ELSE description END "
            "WHERE id = ?",
            (now_iso, art.published_at, art.description, article_id),
        )
        novelty = "possible_duplicate" if row["possible_duplicate_of"] else "seen"
    else:
        dup_of = find_possible_duplicate(conn, art)
        cur = conn.execute(
            "INSERT INTO articles (canonical_url, url_hash, title, title_norm, description, source_name, "
            "source_domain, published_at, first_seen_at, last_seen_at, possible_duplicate_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (art.canonical_url, art.url_hash, art.title, art.title_norm, art.description, art.source_name,
             art.source_domain, art.published_at, now_iso, now_iso, dup_of),
        )
        article_id = cur.lastrowid
        novelty = "possible_duplicate" if dup_of else "new"
    for provider in sorted(art.providers):
        conn.execute(
            "INSERT INTO article_sources (article_id, provider, first_seen_at, last_seen_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(article_id, provider) DO UPDATE SET last_seen_at = excluded.last_seen_at",
            (article_id, provider, now_iso, now_iso),
        )
    return article_id, novelty


# --------------------------------------------------------------------------- #
# Evidence assessment (snippet-level, conservative)
# --------------------------------------------------------------------------- #
def assess_article(conn: sqlite3.Connection, article_id: int) -> tuple[str, str]:
    me = conn.execute("SELECT id, title_norm, source_domain, published_at, description FROM articles WHERE id = ?",
                      (article_id,)).fetchone()
    group = [me]
    if len(me["title_norm"].split()) >= MIN_TITLE_WORDS:
        for r in conn.execute(
            "SELECT id, source_domain, published_at, description FROM articles "
            "WHERE title_norm = ? AND id <> ? LIMIT 50", (me["title_norm"], article_id)):
            if same_story_context(me["source_domain"], me["published_at"], r["source_domain"], r["published_at"]):
                group.append(r)
    per_domain: dict[str, str] = {}
    for r in group:
        per_domain.setdefault(r["source_domain"] or "unknown", normalize_text(r["description"]))
    # Publishers whose snippets are identical are treated as ONE report (likely copied/syndicated text).
    independent_keys = {desc or f"domain:{dom}" for dom, desc in per_domain.items()}
    notes = []
    if len(per_domain) >= 2 and len(independent_keys) >= 2:
        status = "PARTIALLY_CORROBORATED"
        notes.append(f"Same headline from {len(independent_keys)} publishers within 48h.")
        notes.append("Independence from a shared wire/press release is not established and claims were not compared.")
    elif len(per_domain) >= 2:
        status = "UNVERIFIED"
        notes.append(f"{len(per_domain)} publishers carry identical snippet text; treated as one report (likely copied).")
    else:
        status = "UNVERIFIED"
        notes.append("Single publisher found; no matching report from another publisher.")
    notes.append("Based on headline/snippet metadata only; full-content verification was not performed."
                 if me["description"] else
                 "No snippet supplied by provider; headline/metadata only; full-content verification was not performed.")
    if not me["published_at"]:
        notes.append("Publication date unavailable.")
    return status, " ".join(notes)


def store_assessment(conn: sqlite3.Connection, article_id: int, status: str, notes: str) -> None:
    conn.execute(  # only writes when something actually changed
        "UPDATE articles SET verification_status = ?, verification_notes = ? "
        "WHERE id = ? AND (verification_status <> ? OR verification_notes <> ?)",
        (status, notes, article_id, status, notes),
    )


# --------------------------------------------------------------------------- #
# Ranking (transparent; a high score is NOT evidence of truth)
# --------------------------------------------------------------------------- #
def relevance(query: str, art: Article) -> float:
    tokens = normalize_text(query).split()
    if not tokens:
        return 0.0
    body = f"{art.title_norm} {normalize_text(art.description)}"
    title_hits = sum(t in art.title_norm for t in tokens)  # substring match tolerates Bengali inflection
    body_hits = sum(t in body for t in tokens)
    return (0.6 * title_hits + 0.4 * body_hits) / len(tokens)


def recency(published_at: Optional[str], now: datetime) -> float:
    ts = _load_ts(published_at)
    if not ts:
        return 0.0
    age_hours = max(0.0, (now - ts).total_seconds() / 3600)
    return 0.5 ** (age_hours / RECENCY_HALF_LIFE_HOURS)


def rank_score(query: str, art: Article, status: str, novelty: str, cfg: Config, now: datetime) -> float:
    quality = 1.0 if art.source_domain in cfg.trusted_domains else 0.0
    score = (RANK_WEIGHTS["relevance"] * relevance(query, art)
             + RANK_WEIGHTS["recency"] * recency(art.published_at, now)
             + RANK_WEIGHTS["source"] * quality
             + RANK_WEIGHTS["evidence"] * EVIDENCE_STRENGTH[status])
    return score - (DUPLICATE_PENALTY if novelty == "possible_duplicate" else 0.0)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    article: Article
    article_id: int
    novelty: str
    status: str
    notes: str
    score: float


@dataclass
class SearchReport:
    query: str
    results: list
    provider_counts: dict
    failures: dict
    novelty_counts: Counter
    status_counts: Counter


def run_search(query: str, cfg: Config, conn: sqlite3.Connection, session=None,
               providers: Optional[dict] = None, now: Optional[datetime] = None) -> SearchReport:
    providers = providers if providers is not None else PROVIDERS
    session = session or make_session()
    now = now or datetime.now(timezone.utc)
    ctx = Ctx(cfg, session, time.monotonic() + cfg.deadline)

    batches: list[list[Article]] = []
    counts: dict[str, int] = {}
    failures: dict[str, str] = {}
    pool = ThreadPoolExecutor(max_workers=max(1, len(providers)))  # network-bound: 3 calls in parallel
    futures = {pool.submit(fn, query, ctx): name for name, fn in providers.items()}
    done, not_done = wait(futures, timeout=cfg.deadline)
    pool.shutdown(wait=False, cancel_futures=True)
    for name in providers:  # stable order
        fut = next(f for f, n in futures.items() if n == name)
        if fut in not_done:
            failures[name] = "deadline exceeded"
            continue
        try:
            arts = fut.result()
        except ProviderError as exc:
            failures[name] = str(exc)
            log.warning("provider %s failed: %s", name, exc)
        except Exception as exc:  # a provider bug must not take the others down
            failures[name] = f"unexpected {type(exc).__name__}"
            log.warning("provider %s crashed: %s", name, type(exc).__name__)
        else:
            batches.append(arts)
            counts[name] = len(arts)

    merged = merge_batch(batches)
    now_iso = now.strftime(TS_FORMAT)
    results: list[Result] = []
    with conn:  # one transaction: commit on success, rollback on any error
        for art in merged:
            article_id, novelty = upsert_article(conn, art, now_iso)
            status, notes = assess_article(conn, article_id)
            store_assessment(conn, article_id, status, notes)
            score = rank_score(query, art, status, novelty, cfg, now)
            results.append(Result(art, article_id, novelty, status, notes, score))
    results.sort(key=lambda r: r.score, reverse=True)
    return SearchReport(query, results, counts, failures,
                        Counter(r.novelty for r in results), Counter(r.status for r in results))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def format_report(rep: SearchReport, cfg: Config) -> str:
    lines: list[str] = []
    shown = rep.results[: cfg.show]
    for i, r in enumerate(shown, 1):
        a = r.article
        label = {"new": "NEW", "seen": "PREVIOUSLY SEEN", "possible_duplicate": "POSSIBLE DUPLICATE"}[r.novelty]
        lines += [
            f"{i}. {a.title}",
            f"   {a.source_name or 'unknown publisher'} | {a.published_at or 'date unavailable'} | via {', '.join(sorted(a.providers))}",
            f"   {a.canonical_url}",
            f"   Evidence: {r.status} - {r.notes}",
            f"   Status: {label} (score {r.score:.2f})",
            "",
        ]
    if not rep.results:
        lines.append("No results returned by the providers. This does not mean no news exists.\n")
    lines.append("--- Summary ---")
    lines.append(f"Results collected (after URL dedup): {len(rep.results)} (showing {len(shown)})")
    lines.append(f"New: {rep.novelty_counts['new']} | Previously seen: {rep.novelty_counts['seen']} | "
                 f"Possible duplicates: {rep.novelty_counts['possible_duplicate']}")
    lines.append("Verification: " + (", ".join(f"{s}={rep.status_counts[s]}" for s in EVIDENCE_STATUSES
                                              if rep.status_counts[s]) or "n/a"))
    ok = ", ".join(f"{n} ({c})" for n, c in rep.provider_counts.items()) or "none"
    lines.append(f"Providers OK: {ok}")
    if rep.failures:
        lines.append("Provider failures: " + "; ".join(f"{n}: {m}" for n, m in rep.failures.items()))
        lines.append("WARNING: degraded coverage - results are incomplete.")
    lines.append("Note: results come from at most three providers and are not a complete search of world news.")
    return "\n".join(lines)


def config_from_args(args: argparse.Namespace) -> Config:
    trusted = frozenset(d.strip().lower() for d in os.environ.get("TRUSTED_DOMAINS", "").split(",") if d.strip())
    return Config(db_path=args.db, timeout=args.timeout, deadline=args.deadline, max_results=args.max_results,
                  show=args.show, days=args.days, language=args.lang, region=args.region, trusted_domains=trusted)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Minimal World News Engine")
    p.add_argument("query", nargs="*", help="search query (English or Bengali); prompts if omitted")
    p.add_argument("--db", default="news_engine.db")
    p.add_argument("--max-results", type=int, default=20, help="per provider")
    p.add_argument("--show", type=int, default=20, help="results to display")
    p.add_argument("--days", type=int, help="only the last N days (where supported)")
    p.add_argument("--lang", help="language code, e.g. en, bn")
    p.add_argument("--region", help="region code, e.g. US, BD")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--deadline", type=float, default=30.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    query = " ".join(args.query).strip() or input("Enter query: ").strip()
    if not query:
        print("Empty query.", file=sys.stderr)
        return 2
    cfg = config_from_args(args)
    conn = connect(cfg.db_path)
    try:
        report = run_search(query, cfg, conn)
    finally:
        conn.close()
    print(format_report(report, cfg))
    return 1 if report.failures and not report.provider_counts else 0


if __name__ == "__main__":
    sys.exit(main())
