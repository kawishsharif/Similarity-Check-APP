"""Orchestrates one submission: ingest -> (similarity || AI-writing) -> reports -> notify."""
from __future__ import annotations

import gzip
import json
import logging
import secrets
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy import delete

from . import archive as archive_mod
from . import mail
from .aidetect.engine import run_ai_detection
from .config import get_settings, load_brand
from .db import (AIResult, AISpanRow, MatchRow, Report, SimilarityResult, SourceRow, Submission, User, session_scope, utcnow)
from .ingest import Doc, IngestError, ingest_file
from .reports.build import _pct, build_report, integrity_flags
from .similarity.engine import SimOptions, run_similarity
from .similarity.sources import TextFetcher, build_sources

log = logging.getLogger(__name__)


def new_id(db, prefix: str, mid: str = "") -> str:
    year = utcnow().year
    while True:
        cand = f"{prefix}-{mid + '-' if mid else ''}{year}-{secrets.randbelow(10**8):08d}"
        if not db.query(Submission).filter_by(public_id=cand).first() and not db.query(Report).filter_by(report_id=cand).first():
            return cand


def sub_dir(sid: int) -> Path:
    return get_settings().path("subs", str(sid))


def doc_path(sid: int) -> Path:
    return sub_dir(sid) / "doc.json.gz"


def save_doc(sid: int, doc: Doc) -> None:
    with gzip.open(doc_path(sid), "wt", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f)


def load_doc(sid: int) -> Doc:
    with gzip.open(doc_path(sid), "rt", encoding="utf-8") as f:
        return Doc.from_dict(json.load(f))


def _update(sid: int, **fields) -> None:
    with session_scope() as db:
        db.query(Submission).filter_by(id=sid).update(fields)


def _fmt_size(n: int) -> str:
    return f"{n / 1024:.1f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


def _fmt_dt(d) -> str:
    return d.strftime("%b %d, %Y, %I:%M %p UTC").replace(" 0", " ")


def process_submission(sid: int) -> None:
    """Run the whole check for submission `sid`. Safe to call from a worker thread or RQ."""
    settings = get_settings()
    try:
        _process(sid, settings)
    except IngestError as exc:
        _fail(sid, str(exc))
    except Exception:
        log.error("submission %s failed:\n%s", sid, traceback.format_exc())
        _fail(sid, "Processing failed unexpectedly. Please try again; if it keeps failing, contact support.")


def _fail(sid: int, message: str) -> None:
    with session_scope() as db:
        sub = db.get(Submission, sid)
        if not sub:
            return
        if sub.status != "failed":  # refund the credit once
            user = db.get(User, sub.user_id)
            if user and user.role != "admin":
                user.quota += 1
        sub.status, sub.error, sub.stage, sub.finished_at = "failed", message, "Failed", utcnow()


def _process(sid: int, settings) -> None:
    brand = load_brand(settings)
    with session_scope() as db:
        sub = db.get(Submission, sid)
        if sub is None:
            return
        sub.status, sub.stage, sub.progress, sub.error = "ingesting", "Reading the document", 2.0, ""
        user = db.get(User, sub.user_id)
        user_id, file_name, options, consent = sub.user_id, sub.file_name, dict(sub.options or {}), sub.consent_archive
        title_in, author_in, public_id, created = sub.title, sub.author, sub.public_id, sub.created_at
        author_default = (user.name or user.email.split("@")[0]) if user else ""
        email = user.email if user else None
        file_size = sub.file_size

    d = sub_dir(sid)
    upload = d / "upload.bin"
    doc, pdf, kind = ingest_file(upload, d, file_name, settings)
    upload.unlink(missing_ok=True)
    save_doc(sid, doc)
    title = title_in or doc.title
    _update(sid, kind=kind, pages=len(doc.pages), words=doc.n, characters=doc.char_count, language=doc.language, title=title[:300],
            status="checking", stage="Checking sources and writing style", progress=10.0)

    # ---- similarity and AI-writing run in parallel ------------------------------------------------
    lock = threading.Lock()
    prog = {"sim": 0.0, "ai": 0.0, "msg": ""}

    def report(which):
        def cb(frac, msg):
            with lock:
                prog[which] = max(prog[which], frac)
                prog["msg"] = msg
                _update(sid, progress=10 + 75 * (0.75 * prog["sim"] + 0.25 * prog["ai"]), stage=msg)
        return cb

    opts = SimOptions(exclude_biblio=bool(options.get("exclude_biblio", True)), exclude_quotes=bool(options.get("exclude_quotes", False)),
                      min_words=int(options.get("min_words", settings.min_match_words)))
    sources = build_sources(settings)

    def archive_cb(hashes):
        from .db import session

        with_db = session()
        try:
            return archive_mod.lookup(with_db, settings, hashes, user_id)
        finally:
            with_db.close()

    use_archive = "archive" in settings.sources
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_sim = pool.submit(run_similarity, doc, opts, sources, settings, TextFetcher(settings), archive_cb if use_archive else None, report("sim"))
        f_ai = pool.submit(run_ai_detection, doc, settings, report("ai"))
        sim, ai = f_sim.result(), f_ai.result()

    # ---- persist results ------------------------------------------------------------------------
    flags = {"filters": sim["filters"], "warnings": sim["warnings"], "integrity": integrity_flags(doc), "stats": sim["stats"], "language": sim["language"]}
    with session_scope() as db:
        for model in (SimilarityResult, SourceRow, MatchRow, AIResult, AISpanRow, Report):
            col = model.submission_id
            db.execute(delete(model).where(col == sid))
        db.add(SimilarityResult(submission_id=sid, percent=sim["percent"], groups=sim["groups"], source_types=sim["source_types"], flags=flags,
                                matched_words=sim["matched_words"], total_words=sim["total_words"]))
        for s in sim["sources"]:
            db.add(SourceRow(submission_id=sid, rank=s["rank"], type=s["type"], title=s["title"], address=s["address"], matched_words=s["words"], percent=s["percent"], color=s["color"]))
        for m in sim["matches"]:
            db.add(MatchRow(submission_id=sid, source_rank=m["rank"], page=doc.page[m["s0"]], start_word=m["s0"], end_word=m["s1"], category=m["category"], kind=m["kind"]))
        db.add(AIResult(submission_id=sid, percent=ai["percent"], display=ai["display"], scored=ai["scored"], groups=ai["groups"], qualifying_words=ai["qualifying_words"],
                        flagged_words=ai["flagged_words"], detector=ai["detector"], message=ai["message"]))
        for sp in ai["spans"]:
            db.add(AISpanRow(submission_id=sid, page=doc.page[sp["s0"]], start_word=sp["s0"], end_word=sp["s1"], group=sp["group"], score=sp["score"]))

    # ---- reports ------------------------------------------------------------------------------
    _update(sid, status="reporting", stage="Building reports", progress=88.0)
    now = utcnow()
    sub_info = {"id": public_id, "organisation": brand["organisation"], "title": title, "author": author_in or author_default,
                "submitted": _fmt_dt(created), "downloaded": _fmt_dt(now), "file_name": file_name, "file_size": _fmt_size(file_size),
                "pages": len(doc.pages), "words": doc.n, "characters": doc.char_count, "lang": doc.language}
    made = []
    with session_scope() as db:
        for rtype, result, figure in (("similarity", sim, f"{_pct(sim['percent'])}%"), ("ai", ai, ai["display"])):
            rid = new_id(db, brand["id_prefix"], "R")
            out = d / f"{rtype}.pdf"
            checksum = build_report(rtype, doc, result, pdf, out, sub_info, rid, brand)
            db.add(Report(report_id=rid, submission_id=sid, type=rtype, file_path=str(out), checksum=checksum, figure=figure))
            made.append(rid)

    if consent:
        from .db import session

        db2 = session()
        try:
            archive_mod.add_document(db2, settings, doc, user_id, sid)
        finally:
            db2.close()

    _update(sid, status="done", stage="Complete", progress=100.0, finished_at=utcnow())
    if email:
        mail.send_email(email, f"Your {brand['brand_name']} reports are ready",
                        f"Your submission {public_id} has been checked.\n\nSimilarity: {_pct(sim['percent'])}%\nAI writing: {ai['display']}\n\n"
                        f"Open {settings.base_url}/submissions/{sid} to download the reports.")
