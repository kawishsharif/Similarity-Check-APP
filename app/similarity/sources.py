"""Source connectors. Each source is a separate module-level class so a source that changes
its terms or limits can be switched off (SOURCES=...) without touching the rest.

A connector's `search()` returns Candidates. A candidate carries either `text` (an abstract,
already in hand) or `fetch_url` (a page or PDF that is downloaded, cleaned and cached later).
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..config import Settings

log = logging.getLogger(__name__)
UA = "SimilarityCheckBot/1.0 (+{contact})"
MAX_DOWNLOAD = 15 * 1024 * 1024


@dataclass
class Candidate:
    key: str                   # de-duplication key (DOI, URL, archive id)
    title: str
    address: str               # shown in the report
    type: str                  # "internet" | "publication" | "submitted"
    text: str | None = None
    fetch_url: str | None = None
    hits: int = 0
    meta: dict = field(default_factory=dict)


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval, self._last, self._lock = min_interval, 0.0, threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._last + self.min_interval - now
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class Source:
    name = "base"
    interval = 0.2
    query_cap: int | None = None  # some sources are slow or strict, so they only see the first N queries
    calls = 0
    errors = 0

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.s = settings
        self.client = client or httpx.Client(timeout=15, follow_redirects=True, headers={"User-Agent": UA.format(contact=settings.contact_email)})
        self.limiter = RateLimiter(self.interval)

    def get_json(self, url: str, **kw):
        self.limiter.wait()
        try:
            r = self.client.get(url, **kw)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            self.errors += 1
            log.info("%s: request failed: %s", self.name, exc)
            return None

    def search(self, query: str, lang: str) -> list[Candidate]:  # pragma: no cover - interface
        raise NotImplementedError


def _clean_abstract(a: str | None) -> str | None:
    if not a:
        return None
    return html.unescape(re.sub(r"<[^>]+>", " ", a)).strip() or None


class OpenAlex(Source):
    name = "openalex"
    interval = 0.15

    def search(self, query, lang):
        params = {"search": query, "per-page": 5, "mailto": self.s.contact_email}
        if self.s.openalex_key:
            params["api_key"] = self.s.openalex_key
        data = self.get_json("https://api.openalex.org/works", params=params)
        out = []
        for w in (data or {}).get("results", []):
            inv = w.get("abstract_inverted_index") or {}
            words = sorted(((p, word) for word, ps in inv.items() for p in ps))
            abstract = " ".join(word for _, word in words) or None
            doi = w.get("doi") or w.get("id")
            oa = (w.get("open_access") or {}).get("oa_url")
            out.append(Candidate(key=(doi or w.get("id") or "").lower(), title=w.get("title") or doi or "Untitled", address=doi or w.get("id", ""),
                                 type="publication", text=abstract, fetch_url=oa if oa and not oa.endswith(".zip") else None))
        return out


class Crossref(Source):
    name = "crossref"
    interval = 0.2

    def search(self, query, lang):
        data = self.get_json("https://api.crossref.org/works", params={"query.bibliographic": query, "rows": 5, "select": "DOI,title,abstract,URL", "mailto": self.s.contact_email})
        out = []
        for w in ((data or {}).get("message") or {}).get("items", []):
            doi = (w.get("DOI") or "").lower()
            out.append(Candidate(key=f"https://doi.org/{doi}", title=(w.get("title") or [doi])[0], address=w.get("URL") or f"https://doi.org/{doi}",
                                 type="publication", text=_clean_abstract(w.get("abstract"))))
        return out


class SemanticScholar(Source):
    name = "semanticscholar"
    interval = 1.1
    query_cap = 40

    def search(self, query, lang):
        headers = {"x-api-key": self.s.s2_key} if self.s.s2_key else {}
        data = self.get_json("https://api.semanticscholar.org/graph/v1/paper/search", params={"query": query[:300], "limit": 5, "fields": "title,abstract,url,openAccessPdf"}, headers=headers)
        out = []
        for w in (data or {}).get("data", []):
            pdf = (w.get("openAccessPdf") or {}).get("url")
            out.append(Candidate(key=w.get("url") or w.get("paperId", ""), title=w.get("title") or "Untitled", address=w.get("url") or "", type="publication",
                                 text=w.get("abstract"), fetch_url=pdf))
        return out


class Core(Source):
    name = "core"
    interval = 1.0

    def search(self, query, lang):
        if not self.s.core_key:
            return []
        data = self.get_json("https://api.core.ac.uk/v3/search/works", params={"q": f'"{query}"', "limit": 5}, headers={"Authorization": f"Bearer {self.s.core_key}"})
        out = []
        for w in (data or {}).get("results", []):
            url = w.get("downloadUrl") or (f"https://core.ac.uk/works/{w.get('id')}")
            out.append(Candidate(key=url, title=w.get("title") or "Untitled", address=url, type="publication", text=w.get("fullText") or w.get("abstract")))
        return out


class Arxiv(Source):
    name = "arxiv"
    interval = 3.0  # arXiv asks for one request every three seconds
    query_cap = 25

    def search(self, query, lang):
        self.limiter.wait()
        try:
            r = self.client.get("http://export.arxiv.org/api/query", params={"search_query": f'all:"{query[:200]}"', "max_results": 4})
            r.raise_for_status()
            root = ET.fromstring(r.text)
        except Exception as exc:
            self.errors += 1
            log.info("arxiv: %s", exc)
            return []
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for e in root.findall("a:entry", ns):
            url = (e.findtext("a:id", "", ns) or "").strip()
            out.append(Candidate(key=url, title=" ".join((e.findtext("a:title", "", ns) or "").split()), address=url, type="publication",
                                 text=(e.findtext("a:summary", "", ns) or "").strip(), fetch_url=url.replace("/abs/", "/pdf/") if url else None))
        return out


class EuropePMC(Source):
    name = "europepmc"
    interval = 0.3

    def search(self, query, lang):
        data = self.get_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={"query": f'"{query[:200]}"', "format": "json", "pageSize": 5, "resultType": "core"})
        out = []
        for w in ((data or {}).get("resultList") or {}).get("result", []):
            url = f"https://europepmc.org/article/{w.get('source')}/{w.get('id')}"
            out.append(Candidate(key=url, title=w.get("title") or "Untitled", address=url, type="publication", text=_clean_abstract(w.get("abstractText"))))
        return out


class Wikipedia(Source):
    name = "wikipedia"
    interval = 0.2

    def search(self, query, lang):
        out = []
        for code in ("zh", "en") if lang == "zh" else ("en",):
            base = f"https://{code}.wikipedia.org/w/api.php"
            data = self.get_json(base, params={"action": "query", "list": "search", "srsearch": f'"{query[:250]}"', "srlimit": 3, "format": "json"})
            for hit in ((data or {}).get("query") or {}).get("search", []):
                title = hit["title"]
                url = f"https://{code}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                api = f"{base}?action=query&prop=extracts&explaintext=1&format=json&redirects=1&titles={quote(title)}"
                out.append(Candidate(key=url, title=f"{title} - Wikipedia", address=url, type="internet", fetch_url=api, meta={"kind": "wikipedia"}))
        return out


class SearXNG(Source):
    name = "searxng"
    interval = 1.0

    def search(self, query, lang):
        if not self.s.searxng_url:
            return []
        data = self.get_json(f"{self.s.searxng_url.rstrip('/')}/search", params={"q": f'"{query[:250]}"', "format": "json", "language": "zh" if lang == "zh" else "en"})
        out = []
        for w in (data or {}).get("results", [])[:6]:
            url = w.get("url")
            if url:
                out.append(Candidate(key=url, title=w.get("title") or url, address=url, type="internet", fetch_url=url, text=None))
        return out


class DirectorySource(Source):
    """Compare against a folder of .txt files (offline use, the command-line tool, tests)."""
    name = "directory"

    def __init__(self, settings: Settings, folder: Path):
        self.s, self.folder, self.limiter = settings, Path(folder), RateLimiter(0)
        self.client = None

    def search(self, query, lang):
        out = []
        for f in sorted(self.folder.glob("*.txt")):
            out.append(Candidate(key=f"file:{f.name}", title=f.stem, address=f.name, type="internet", text=f.read_text(encoding="utf-8", errors="replace")))
        return out


REGISTRY = {c.name: c for c in (OpenAlex, Crossref, SemanticScholar, Core, Arxiv, EuropePMC, Wikipedia, SearXNG)}


def build_sources(settings: Settings) -> list[Source]:
    shared = httpx.Client(timeout=15, follow_redirects=True, headers={"User-Agent": UA.format(contact=settings.contact_email)})
    out = []
    for name in settings.sources:
        cls = REGISTRY.get(name)
        if cls:
            out.append(cls(settings, shared))
    return out


# --------------------------------------------------------------------------------------------
# Downloading and caching candidate full text
# --------------------------------------------------------------------------------------------
class TextFetcher:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.s = settings
        self.client = client or httpx.Client(timeout=20, follow_redirects=False, headers={"User-Agent": UA.format(contact=settings.contact_email)})
        self.dir = settings.path("cache", "sources")

    def _cache_file(self, url: str) -> Path:
        return self.dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")

    def fetch(self, cand: Candidate) -> str | None:
        """Full text for a candidate (cached 30 days by default); falls back to the abstract."""
        if not cand.fetch_url:
            return cand.text
        f = self._cache_file(cand.fetch_url)
        if f.exists() and time.time() - f.stat().st_mtime < self.s.cache_days * 86400:
            try:
                return json.loads(f.read_text(encoding="utf-8"))["text"] or cand.text
            except Exception:
                pass
        text = None
        try:
            text = self._download(cand)
        except Exception as exc:
            log.info("fetch failed for %s: %s", cand.fetch_url, exc)
        f.write_text(json.dumps({"url": cand.fetch_url, "text": text}), encoding="utf-8")
        return text or cand.text

    def _download(self, cand: Candidate) -> str | None:
        url = cand.fetch_url
        for _ in range(4):  # follow redirects by hand so every hop is checked
            if not self._public_url(url):
                return None
            with self.client.stream("GET", url) as r:
                if r.is_redirect:
                    url = str(r.url.join(r.headers.get("location", "")))
                    continue
                r.raise_for_status()
                buf = io.BytesIO()
                for chunk in r.iter_bytes():
                    buf.write(chunk)
                    if buf.tell() > MAX_DOWNLOAD:
                        return None
                ctype = r.headers.get("content-type", "")
                encoding = r.encoding
            break
        else:
            return None
        data = buf.getvalue()
        if cand.meta.get("kind") == "wikipedia":
            pages = (json.loads(data).get("query") or {}).get("pages", {})
            return "\n".join(p.get("extract", "") for p in pages.values()) or None
        if data[:5] == b"%PDF-" or "pdf" in ctype:
            return self._pdf_text(data)
        import trafilatura

        return trafilatura.extract(data.decode(encoding or "utf-8", "replace"), include_comments=False, include_tables=False)

    @staticmethod
    def _public_url(url: str | None) -> bool:
        """Refuse to fetch internal addresses that search results might point at (SSRF guard)."""
        import ipaddress
        import socket
        from urllib.parse import urlparse

        p = urlparse(url or "")
        if p.scheme not in ("http", "https") or not p.hostname:
            return False
        try:
            for fam, _, _, _, addr in socket.getaddrinfo(p.hostname, None):
                ip = ipaddress.ip_address(addr[0])
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                    return False
        except OSError:
            return False
        return True

    @staticmethod
    def _pdf_text(data: bytes) -> str | None:
        import pdfplumber

        parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages[:60]:
                parts.append(page.extract_text() or "")
        return "\n".join(parts) or None
