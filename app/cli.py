"""Command-line check (no web app, no database): `python -m app.cli check paper.pdf --corpus ./texts`.

Prints the similarity sources and percentages and the AI-writing estimate, and can write both PDF reports.
Without --corpus it uses the sources in SOURCES (needs network access).
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from .aidetect.engine import run_ai_detection
from .config import get_settings, load_brand
from .ingest import IngestError, ingest_file
from .reports.build import _pct, build_report
from .similarity.engine import SimOptions, run_similarity
from .similarity.sources import DirectorySource, TextFetcher, build_sources


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="app.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check one file")
    c.add_argument("file", type=Path)
    c.add_argument("--corpus", type=Path, help="folder of .txt files to compare against instead of online sources")
    c.add_argument("--reports", type=Path, help="write the two PDF reports into this folder")
    c.add_argument("--include-biblio", action="store_true")
    c.add_argument("--exclude-quotes", action="store_true")
    c.add_argument("--min-words", type=int, default=None)
    a = ap.parse_args(argv)

    s = get_settings()
    work = Path(tempfile.mkdtemp(prefix="cli_"))
    try:
        doc, pdf, _ = ingest_file(a.file, work, a.file.name, s)
    except IngestError as exc:
        print(f"error: {exc}")
        return 2
    print(f"{a.file.name}: {len(doc.pages)} pages, {doc.n:,} words, language {doc.language}")
    sources = [DirectorySource(s, a.corpus)] if a.corpus else [x for x in build_sources(s)]
    sim = run_similarity(doc, SimOptions(not a.include_biblio, a.exclude_quotes, a.min_words or s.min_match_words), sources, s, TextFetcher(s),
                         progress=lambda f, m: print(f"  [{f:4.0%}] {m}"))
    print(f"\nSimilarity: {_pct(sim['percent'])}%  ({sim['matched_words']:,} of {sim['total_words']:,} counted words)")
    for src in sim["sources"]:
        print(f"  {src['rank']:>2}. {src['percent']:5.1f}%  [{src['type']}] {src['title']}  {src['address']}")
    for w in sim["warnings"]:
        print("  warning:", w)
    ai = run_ai_detection(doc, s)
    print(f"\nAI writing: {ai['display']}  (detector: {ai['detector']}; estimate only)" + (f"  {ai['message']}" if ai["message"] else ""))
    if a.reports:
        brand = load_brand(s)
        sub = {"id": f"{brand['id_prefix']}-CLI", "organisation": brand["organisation"], "title": doc.title, "author": "", "submitted": "-", "downloaded": "-",
               "file_name": a.file.name, "file_size": f"{a.file.stat().st_size / 1024:.1f} KB", "pages": len(doc.pages), "words": doc.n, "characters": doc.char_count, "lang": doc.language}
        a.reports.mkdir(parents=True, exist_ok=True)
        for kind, res in (("similarity", sim), ("ai", ai)):
            build_report(kind, doc, res, pdf, a.reports / f"{kind}-report.pdf", sub, f"{brand['id_prefix']}-R-CLI-{kind}", brand)
        print(f"\nReports written to {a.reports}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
