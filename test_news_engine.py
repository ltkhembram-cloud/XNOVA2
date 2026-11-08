"""Mocked unit tests - no network, no API credits. Run: python -m unittest -v"""
import logging
import os
import sqlite3
import unittest
from datetime import datetime, timezone
from unittest import mock

import requests

import news_engine as ne

NOW = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
SECRET_T, SECRET_N = "tvly-SECRET-123", "news-SECRET-456"


class FakeResponse:
    def __init__(self, status=200, data=None, content=b"", bad_json=False):
        self.status_code, self._data, self.content, self._bad = status, data, content, bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._data


class FakeSession:
    """Routes by host. Handler is a FakeResponse, an Exception to raise, or a callable."""
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        for host, h in self.routes.items():
            if host in url:
                h = h() if callable(h) else h
                if isinstance(h, Exception):
                    raise h
                return h
        raise AssertionError(f"unrouted {url}")


ARTICLE_URL = "https://www.Example.com/world/story-1?utm_source=x&id=42"
TITLE = "Scientists announce new battery chemistry breakthrough"
TAVILY_OK = FakeResponse(data={"results": [{"title": TITLE, "url": ARTICLE_URL, "content": "Snippet A",
                                            "published_date": "2026-11-07T10:00:00Z"}]})
NEWSAPI_OK = FakeResponse(data={"status": "ok", "articles": [
    {"source": {"name": "Example"}, "title": TITLE, "url": "https://example.com/world/story-1?id=42&fbclid=z",
     "description": "Snippet A", "publishedAt": "2026-11-07T10:00:00+00:00"},
    {"title": "[Removed]", "url": "https://removed.com", "source": {"name": "x"}}]})
RSS_XML = ("<rss><channel><item><title>%s - Example</title><link>https://example.com/world/story-1?id=42</link>"
           "<pubDate>Sat, 07 Nov 2026 10:00:00 GMT</pubDate><source url='https://www.example.com'>Example</source>"
           "</item></channel></rss>" % TITLE).encode()
RSS_OK = FakeResponse(content=RSS_XML)


def mem_conn():
    return ne.connect(":memory:")


def art(title, url, domain=None, pub=None, desc="", provider="tavily"):
    a = ne.build_article(provider=provider, title=title, url=url, description=desc,
                         source_domain=domain, published=pub)
    assert a is not None
    return a


class UrlTests(unittest.TestCase):
    def test_strips_tracking_keeps_meaningful_params(self):
        u = ne.normalize_url("HTTPS://WWW.Example.COM:443/News/Story/?utm_medium=a&id=7&fbclid=1#frag")
        self.assertEqual(u, "https://example.com/News/Story?id=7")

    def test_same_article_same_hash_different_ids_differ(self):
        a = ne.normalize_url("https://example.com/a?id=1&utm_campaign=z")
        b = ne.normalize_url("http://x.com")
        self.assertEqual(ne.url_fingerprint(a), ne.url_fingerprint(ne.normalize_url("https://example.com/a?id=1")))
        self.assertNotEqual(ne.url_fingerprint(a), ne.url_fingerprint(ne.normalize_url("https://example.com/a?id=2")))
        self.assertNotEqual(a, b)

    def test_rejects_bad_urls(self):
        for bad in (None, "", "ftp://x.com/a", "javascript:alert(1)", "https://", 12, "http://[bad"):
            self.assertIsNone(ne.normalize_url(bad), bad)

    def test_bengali_title_preserved(self):
        self.assertEqual(ne.normalize_text("বাংলাদেশের অর্থনীতি: নতুন প্রতিবেদন!"), "বাংলাদেশের অর্থনীতি নতুন প্রতিবেদন")


class TimestampTests(unittest.TestCase):
    def test_valid_formats_to_utc(self):
        self.assertEqual(ne.parse_timestamp("2026-11-07T10:00:00Z", NOW), "2026-11-07T10:00:00Z")
        self.assertEqual(ne.parse_timestamp("2026-11-07T10:00:00+05:30", NOW), "2026-11-07T04:30:00Z")
        self.assertEqual(ne.parse_timestamp("Sat, 07 Nov 2026 10:00:00 GMT", NOW), "2026-11-07T10:00:00Z")

    def test_invalid_returns_none(self):
        for bad in (None, "", "   ", "yesterday-ish", 1700000000, "2026-13-45", "2099-01-01T00:00:00Z", "1970-01-01T00:00:00Z"):
            self.assertIsNone(ne.parse_timestamp(bad, NOW), bad)

    def test_bad_timestamp_does_not_crash_article_build(self):
        a = ne.build_article(provider="tavily", title="A valid long headline here", url="https://a.com/x", published="garbage")
        self.assertIsNone(a.published_at)


class DuplicateTests(unittest.TestCase):
    def test_context_rules(self):
        f = ne.same_story_context
        self.assertTrue(f("a.com", "2026-11-07T10:00:00Z", "b.com", "2026-11-08T09:00:00Z"))
        self.assertFalse(f("a.com", "2026-11-01T10:00:00Z", "b.com", "2026-11-08T09:00:00Z"))
        self.assertTrue(f("a.com", None, "a.com", "2026-11-08T09:00:00Z"))   # same publisher, no date
        self.assertFalse(f("a.com", None, "b.com", "2026-11-08T09:00:00Z"))  # different publisher, no date

    def test_similar_headlines_from_distinct_events_not_merged(self):
        conn = mem_conn()
        with conn:
            ne.upsert_article(conn, art("Earthquake strikes northern Japan, dozens injured", "https://a.com/1", pub="2026-11-07T10:00:00Z"), "t1")
            _, nov = ne.upsert_article(conn, art("Earthquake strikes northern Chile, dozens injured", "https://b.com/2", pub="2026-11-07T11:00:00Z"), "t1")
        self.assertEqual(nov, "new")
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM articles WHERE possible_duplicate_of IS NOT NULL").fetchone()[0], 0)

    def test_same_headline_other_publisher_flagged_not_deleted(self):
        conn = mem_conn()
        with conn:
            ne.upsert_article(conn, art(TITLE, "https://a.com/1", pub="2026-11-07T10:00:00Z"), "t1")
            _, nov = ne.upsert_article(conn, art(TITLE, "https://b.com/9", pub="2026-11-07T12:00:00Z"), "t1")
        self.assertEqual(nov, "possible_duplicate")
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0], 2)

    def test_short_generic_titles_never_flagged(self):
        conn = mem_conn()
        with conn:
            ne.upsert_article(conn, art("Live updates", "https://a.com/1", pub="2026-11-07T10:00:00Z"), "t1")
            _, nov = ne.upsert_article(conn, art("Live updates", "https://b.com/1", pub="2026-11-07T10:30:00Z"), "t1")
        self.assertEqual(nov, "new")


class DatabaseTests(unittest.TestCase):
    def test_unique_constraint_enforced(self):
        conn = mem_conn()
        a = art(TITLE, "https://a.com/1")
        with conn:
            ne.upsert_article(conn, a, "t1")
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO articles (canonical_url,url_hash,title,title_norm,first_seen_at,last_seen_at) VALUES (?,?,?,?,?,?)",
                         (a.canonical_url, a.url_hash, "x", "x", "t", "t"))

    def test_foreign_keys_enforced(self):
        with self.assertRaises(sqlite3.IntegrityError):
            mem_conn().execute("INSERT INTO article_sources VALUES (999,'tavily','t','t')")

    def test_rediscovery_updates_last_seen_keeps_first_seen_and_adds_provider(self):
        conn = mem_conn()
        with conn:
            ne.upsert_article(conn, art(TITLE, ARTICLE_URL, provider="tavily"), "2026-11-01T00:00:00Z")
        with conn:
            _, nov = ne.upsert_article(conn, art(TITLE, ARTICLE_URL, provider="newsapi"), "2026-11-08T00:00:00Z")
        self.assertEqual(nov, "seen")
        row = conn.execute("SELECT first_seen_at,last_seen_at FROM articles").fetchone()
        self.assertEqual((row[0], row[1]), ("2026-11-01T00:00:00Z", "2026-11-08T00:00:00Z"))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0], 1)
        provs = [r[0] for r in conn.execute("SELECT provider FROM article_sources ORDER BY provider")]
        self.assertEqual(provs, ["newsapi", "tavily"])

    def test_sql_injection_text_stored_literally(self):
        conn = mem_conn()
        evil = "x'); DROP TABLE articles; -- injected headline words"
        with conn:
            ne.upsert_article(conn, art(evil, "https://a.com/evil"), "t1")
        self.assertEqual(conn.execute("SELECT title FROM articles").fetchone()[0], evil)

    def test_failed_transaction_rolls_back(self):
        conn = mem_conn()
        with self.assertRaises(RuntimeError):
            with conn:
                ne.upsert_article(conn, art(TITLE, "https://a.com/1"), "t1")
                raise RuntimeError("boom")
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0], 0)


class EvidenceTests(unittest.TestCase):
    def _assess(self, *arts):
        conn = mem_conn()
        ids = []
        with conn:
            for a in arts:
                ids.append(ne.upsert_article(conn, a, "t1")[0])
        return [ne.assess_article(conn, i) for i in ids]

    def test_single_publisher_unverified_and_states_limitation(self):
        (status, notes), = self._assess(art(TITLE, "https://a.com/1", desc="snippet", pub="2026-11-07T10:00:00Z"))
        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("full-content verification was not performed", notes)

    def test_two_publishers_distinct_snippets_only_partial(self):
        res = self._assess(art(TITLE, "https://a.com/1", desc="One account", pub="2026-11-07T10:00:00Z"),
                           art(TITLE, "https://b.com/2", desc="Different account", pub="2026-11-07T14:00:00Z"))
        for status, notes in res:
            self.assertEqual(status, "PARTIALLY_CORROBORATED")
            self.assertIn("not established", notes)

    def test_identical_snippets_not_counted_as_independent(self):
        res = self._assess(art(TITLE, "https://a.com/1", desc="Same press release text", pub="2026-11-07T10:00:00Z"),
                           art(TITLE, "https://b.com/2", desc="Same press release text", pub="2026-11-07T11:00:00Z"))
        self.assertTrue(all(s == "UNVERIFIED" for s, _ in res))

    def test_multiple_providers_same_url_is_not_corroboration(self):
        conn = mem_conn()
        a = art(TITLE, "https://a.com/1", desc="x", pub="2026-11-07T10:00:00Z")
        a.providers = {"tavily", "newsapi", "google_news_rss"}
        with conn:
            i, _ = ne.upsert_article(conn, a, "t1")
        self.assertEqual(ne.assess_article(conn, i)[0], "UNVERIFIED")


class ProviderTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"TAVILY_API_KEY": SECRET_T, "NEWSAPI_API_KEY": SECRET_N})
        patcher.start(); self.addCleanup(patcher.stop)
        sleep = mock.patch.object(ne.time, "sleep"); sleep.start(); self.addCleanup(sleep.stop)
        self.cfg = ne.Config(db_path=":memory:", retries=2)

    def routes(self, **over):
        r = {"tavily.com": TAVILY_OK, "newsapi.org": NEWSAPI_OK, "news.google.com": RSS_OK}
        r.update(over)
        return r

    def test_same_article_from_all_three_providers_stored_once(self):
        conn = mem_conn()
        rep = ne.run_search("battery breakthrough", self.cfg, conn, session=FakeSession(self.routes()), now=NOW)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0], 1)
        self.assertEqual(len(rep.results), 1)
        self.assertEqual(rep.results[0].article.providers, {"tavily", "newsapi", "google_news_rss"})
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM article_sources").fetchone()[0], 3)
        self.assertFalse(rep.failures)
        # rerun: still one row, now 'seen', last_seen updated
        rep2 = ne.run_search("battery breakthrough", self.cfg, conn, session=FakeSession(self.routes()),
                             now=datetime(2026, 11, 9, tzinfo=timezone.utc))
        self.assertEqual(rep2.novelty_counts["seen"], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT last_seen_at FROM articles").fetchone()[0], "2026-11-09T00:00:00Z")

    def test_failed_providers_do_not_crash_and_are_reported(self):
        conn = mem_conn()
        sess = FakeSession(self.routes(**{"tavily.com": requests.ConnectionError(f"boom {SECRET_T}"),
                                          "newsapi.org": FakeResponse(status=429)}))
        rep = ne.run_search("battery", self.cfg, conn, session=sess, now=NOW)
        self.assertEqual(set(rep.failures), {"tavily", "newsapi"})
        self.assertEqual(len(rep.results), 1)  # RSS still delivered
        out = ne.format_report(rep, self.cfg)
        self.assertIn("degraded coverage", out)
        self.assertNotIn(SECRET_T, out)

    def test_retries_are_bounded(self):
        sess = FakeSession(self.routes(**{"newsapi.org": FakeResponse(status=503)}))
        with self.assertRaises(ne.ProviderError):
            ne.search_newsapi("q", ne.Ctx(self.cfg, sess, ne.time.monotonic() + 30))
        self.assertEqual(sum("newsapi.org" in c[1] for c in sess.calls), 3)  # 1 try + 2 retries

    def test_invalid_json_and_bad_shape_are_provider_errors(self):
        ctx = lambda s: ne.Ctx(self.cfg, s, ne.time.monotonic() + 30)
        with self.assertRaises(ne.ProviderError):
            ne.search_tavily("q", ctx(FakeSession({"tavily.com": FakeResponse(bad_json=True)})))
        with self.assertRaises(ne.ProviderError):
            ne.search_newsapi("q", ctx(FakeSession({"newsapi.org": FakeResponse(data={"unexpected": 1})})))
        with self.assertRaises(ne.ProviderError):
            ne.search_google_news_rss("q", ctx(FakeSession({"news.google.com": FakeResponse(content=b"<<not xml")})))

    def test_missing_key_is_a_reported_failure(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            rep = ne.run_search("q", self.cfg, mem_conn(), session=FakeSession(self.routes()), now=NOW)
        self.assertIn("TAVILY_API_KEY not set", rep.failures["tavily"])

    def test_unexpected_provider_exception_isolated(self):
        providers = dict(ne.PROVIDERS, tavily=lambda q, c: 1 / 0)
        rep = ne.run_search("q", self.cfg, mem_conn(), session=FakeSession(self.routes()), providers=providers, now=NOW)
        self.assertIn("ZeroDivisionError", rep.failures["tavily"])
        self.assertEqual(len(rep.results), 1)

    def test_api_keys_not_logged_or_in_urls(self):
        sess = FakeSession(self.routes(**{"tavily.com": FakeResponse(status=401)}))
        with self.assertLogs("news_engine", level="WARNING") as cm:
            ne.run_search("q", self.cfg, mem_conn(), session=sess, now=NOW)
        self.assertNotIn(SECRET_T, "\n".join(cm.output))
        for method, url, kw in sess.calls:
            self.assertNotIn(SECRET_N, url)
            self.assertNotIn(SECRET_N, str(kw.get("params", "")))

    def test_bengali_query_skips_unsupported_newsapi_language_and_sets_rss_locale(self):
        sess = FakeSession(self.routes())
        ne.run_search("বাংলাদেশের অর্থনীতি", self.cfg, mem_conn(), session=sess, now=NOW)
        news = next(c for c in sess.calls if "newsapi.org" in c[1])
        self.assertNotIn("language", news[2]["params"])
        rss = next(c for c in sess.calls if "news.google.com" in c[1])
        self.assertEqual((rss[2]["params"]["gl"], rss[2]["params"]["ceid"]), ("BD", "BD:bn"))

    def test_empty_results_not_described_as_no_news(self):
        empty = {"tavily.com": FakeResponse(data={"results": []}),
                 "newsapi.org": FakeResponse(data={"status": "ok", "articles": []}),
                 "news.google.com": FakeResponse(content=b"<rss><channel></channel></rss>")}
        rep = ne.run_search("zzz", self.cfg, mem_conn(), session=FakeSession(empty), now=NOW)
        self.assertIn("does not mean no news exists", ne.format_report(rep, self.cfg))


if __name__ == "__main__":
    unittest.main()
