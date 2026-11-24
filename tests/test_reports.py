from pathlib import Path

from pypdf import PdfReader

from app.aidetect.engine import run_ai_detection
from app.config import get_settings, load_brand
from app.ingest import extract_pdf
from app.reports.build import build_report
from app.similarity.engine import SimOptions, run_similarity
from app.similarity.sources import DirectorySource
from conftest import ORIGINAL, OWN_WORDS, make_pdf
from samples import AI_TEXT, HUMAN_TEXT

S = get_settings()
SUB = {"id": "YB-2026-00012345", "organisation": "Example University", "title": "Sample Paper", "author": "A. Author", "submitted": "Nov 24, 2026, 10:13 PM",
       "downloaded": "Nov 24, 2026, 10:16 PM", "file_name": "sample.pdf", "file_size": "40 KB", "pages": 1, "words": 100, "characters": 500, "lang": "en"}


def test_similarity_report_pdf(tmp):
    corpus = tmp / "c"; corpus.mkdir()
    (corpus / "history-book.txt").write_text(ORIGINAL)
    pdf = make_pdf(tmp / "s.pdf", [ORIGINAL, OWN_WORDS], white_text="hidden words here for the test", )
    doc = extract_pdf(pdf, S)
    assert doc.hidden_chars > 0
    res = run_similarity(doc, SimOptions(), [DirectorySource(S, corpus)], S)
    out = tmp / "sim.pdf"
    sha = build_report("similarity", doc, res, pdf, out, {**SUB, "pages": 1}, "YB-R-2026-00000001", load_brand(S))
    r = PdfReader(str(out))
    assert len(r.pages) >= 3                                     # cover, overview, sources, + 1 document page
    text = " ".join("\n".join(p.extract_text() for p in r.pages).split())
    assert "overall similarity" in text and "history-book" in text and "Hidden text" in text
    assert "YB-R-2026-00000001" in text and "Submission ID YB-2026-00012345" in text
    assert f"Page {len(r.pages)} of {len(r.pages)}" in text     # page numbering is consistent
    assert len(sha) == 64 and r.metadata.producer == "Your Brand"


def test_ai_report_pdf_and_unscored(tmp):
    pdf = make_pdf(tmp / "a.pdf", [AI_TEXT + HUMAN_TEXT])
    doc = extract_pdf(pdf, S)
    ai = run_ai_detection(doc, S)
    out = tmp / "ai.pdf"
    build_report("ai", doc, ai, pdf, out, SUB, "YB-R-2026-00000002", load_brand(S))
    text = "\n".join(p.extract_text() for p in PdfReader(str(out)).pages)
    flat = " ".join(text.split())
    assert "estimated as AI" in flat and "Review required" in flat and "Disclaimer" in flat

    short = make_pdf(tmp / "short.pdf", [OWN_WORDS])
    d2 = extract_pdf(short, S)
    r2 = run_ai_detection(d2, S)
    assert r2["scored"] is False and r2["display"] == "--"
    build_report("ai", d2, r2, short, tmp / "ai2.pdf", SUB, "YB-R-2026-00000003", load_brand(S))
    assert "score not available" in " ".join("\n".join(p.extract_text() for p in PdfReader(str(tmp / "ai2.pdf")).pages).split())
