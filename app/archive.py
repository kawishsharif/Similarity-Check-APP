"""Platform archive: papers uploaded with the uploader's consent, searchable by fingerprint."""
from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from . import textproc as tp
from .config import Settings
from .db import ArchiveDocument, ArchiveFingerprint
from .ingest import Doc
from .similarity.align import h64, sampled, shingle_size
from .similarity.sources import Candidate

MIN_SHARED = 3


def _signed(h: int) -> int:
    return h - (1 << 63)  # unsigned 64-bit -> signed so it fits BIGINT in SQLite and PostgreSQL


def add_document(db: Session, settings: Settings, doc: Doc, user_id: int, submission_id: int) -> int:
    """Store the text (without the reference list) and its sampled fingerprints."""
    end = doc.biblio_start if doc.biblio_start is not None else doc.n
    text = tp.surface_text(doc.raw[:end])
    row = ArchiveDocument(user_id=user_id, submission_id=submission_id, text_path="")
    db.add(row)
    db.flush()
    path = settings.path("archive") / f"{row.id}.txt"
    path.write_text(text, encoding="utf-8")
    row.text_path = str(path)
    k = shingle_size(doc.cjk)
    norms = doc.norm[:end]
    fps = []
    for i in range(len(norms) - k + 1):
        h = h64(norms[i:i + k])
        if sampled(h):
            fps.append({"hash": _signed(h), "document_id": row.id, "position": i})
    if fps:
        db.execute(ArchiveFingerprint.__table__.insert(), fps)
    db.commit()
    return row.id


def lookup(db: Session, settings: Settings, hashes: list[int], user_id: int) -> list[Candidate]:
    """Archive documents sharing at least MIN_SHARED sampled shingles with the submission."""
    if not hashes:
        return []
    counts: dict[int, int] = {}
    uniq = list({_signed(h) for h in hashes})
    for i in range(0, len(uniq), 500):
        q = (select(ArchiveFingerprint.document_id, func.count(func.distinct(ArchiveFingerprint.hash)))
             .where(ArchiveFingerprint.hash.in_(uniq[i:i + 500])).group_by(ArchiveFingerprint.document_id))
        for doc_id, c in db.execute(q):
            counts[doc_id] = counts.get(doc_id, 0) + c
    out = []
    for doc_id, c in sorted(counts.items(), key=lambda x: -x[1])[:30]:
        if c < MIN_SHARED:
            continue
        d = db.get(ArchiveDocument, doc_id)
        if not d or (settings.exclude_own_archive and d.user_id == user_id):
            continue
        try:
            text = open(d.text_path, encoding="utf-8").read()
        except OSError:
            continue
        # Other people's titles and names are never revealed: only the submission month.
        when = d.created_at.strftime("%b %Y")
        out.append(Candidate(key=f"archive:{doc_id}", title=f"Paper in the platform archive ({when})", address=f"Platform archive, submitted {when}", type="submitted", text=text))
    return out


def remove_for_submission(db: Session, submission_id: int) -> None:
    ids = [r for (r,) in db.execute(select(ArchiveDocument.id).where(ArchiveDocument.submission_id == submission_id))]
    for doc_id in ids:
        d = db.get(ArchiveDocument, doc_id)
        try:
            import os

            os.unlink(d.text_path)
        except OSError:
            pass
    if ids:
        db.execute(delete(ArchiveFingerprint).where(ArchiveFingerprint.document_id.in_(ids)))
        db.execute(delete(ArchiveDocument).where(ArchiveDocument.id.in_(ids)))
    db.commit()
