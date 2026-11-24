from pathlib import Path

import pytest

from app.config import get_settings
from app.ingest import extract_pdf
from app.similarity import align as al
from app.similarity.engine import SimOptions, run_similarity, select_queries
from app.similarity.sources import DirectorySource
from conftest import ORIGINAL, OWN_WORDS, make_pdf

S = get_settings()


def test_align_finds_shared_passage_with_small_edit():
    a = "the quick brown fox jumps over the lazy dog while the cat sleeps by the warm fire tonight".split()
    b = "intro words here the quick brown fox leaps over the lazy dog while the cat sleeps by the warm fire tonight outro".split()
    ms = al.align(a, al.build_index(a, 5), b, 5, 8)
    assert len(ms) == 1 and ms[0].s0 == 0 and ms[0].length >= 15


def test_align_ignores_short_overlap():
    a = "alpha beta gamma delta epsilon zeta eta theta".split()
    b = "one two alpha beta gamma delta epsilon three four".split()
    assert al.align(a, al.build_index(a, 5), b, 5, 8) == []


def test_similarity_end_to_end_with_directory_source(tmp):
    corpus = tmp / "corpus"; corpus.mkdir()
    (corpus / "history-book.txt").write_text("Chapter one. " + ORIGINAL + " Chapter two continues with unrelated material about canals and rail.")
    (corpus / "other.txt").write_text("Completely unrelated content about gardening, tomatoes, compost heaps and the best way to prune apple trees in winter.")
    pdf = make_pdf(tmp / "s.pdf", [ORIGINAL, OWN_WORDS])
    doc = extract_pdf(pdf, S)
    res = run_similarity(doc, SimOptions(), [DirectorySource(S, corpus)], S)
    assert [s["title"] for s in res["sources"]] == ["history-book"]
    src = res["sources"][0]
    # the copied paragraph is about half of the document
    assert 40 < res["percent"] < 75
    assert src["percent"] == res["percent"]                       # per-source figures add up to the overall one
    assert res["groups"][0]["key"] == "not_cited_or_quoted" and res["groups"][0]["count"] == 1
    m = res["matches"][0]
    assert doc.text(m["s0"], m["s0"] + 6).startswith("The Industrial Revolution")


def test_bibliography_and_quotes_can_be_excluded(tmp):
    corpus = tmp / "c"; corpus.mkdir()
    (corpus / "src.txt").write_text(ORIGINAL)
    quoted = "“" + ORIGINAL + "” (Smith, 2019)."
    pdf = make_pdf(tmp / "q.pdf", [OWN_WORDS, quoted])
    doc = extract_pdf(pdf, S)
    src = [DirectorySource(S, corpus)]
    r1 = run_similarity(doc, SimOptions(), src, S)
    assert r1["groups"][3]["key"] == "cited_quoted" and r1["groups"][3]["count"] >= 1
    r2 = run_similarity(doc, SimOptions(exclude_quotes=True), src, S)
    assert r2["percent"] == 0 and r2["sources"] == []


def test_overlap_goes_to_largest_source(tmp):
    corpus = tmp / "c"; corpus.mkdir()
    (corpus / "big.txt").write_text(ORIGINAL + " " + OWN_WORDS)
    (corpus / "small.txt").write_text(ORIGINAL)
    pdf = make_pdf(tmp / "s.pdf", [ORIGINAL, OWN_WORDS])
    res = run_similarity(extract_pdf(pdf, S), SimOptions(), [DirectorySource(S, corpus)], S)
    assert res["sources"][0]["title"] == "big"
    assert sum(s["words"] for s in res["sources"]) == res["matched_words"]   # every word counted once
    assert res["percent"] > 90


def test_query_selection_skips_bibliography_and_caps(tmp):
    pdf = make_pdf(tmp / "r.pdf", [ORIGINAL, OWN_WORDS, "References", "Smith, J. (2019). Some very distinctive reference title about quantum fisheries economics."], heading_first=False)
    doc = extract_pdf(pdf, S)
    qs = select_queries(doc, 80)
    assert qs and all("fisheries" not in q for q in qs)
    assert len(select_queries(doc, 1)) == 1


def test_paraphrase_near_match(tmp):
    corpus = tmp / "c"; corpus.mkdir()
    (corpus / "src.txt").write_text("Mechanised textile production moved workers from farms into rapidly growing factory towns and the demand for coal and iron reshaped the landscape of northern England entirely.")
    changed = ("Mechanised textile production moved workers off farms into rapidly growing factory towns, and the demand for coal and iron reshaped the landscape of northern England entirely.")
    pdf = make_pdf(tmp / "p.pdf", [OWN_WORDS, changed])
    res = run_similarity(extract_pdf(pdf, S), SimOptions(), [DirectorySource(S, corpus)], S)
    assert res["percent"] > 30
