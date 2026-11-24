"""Account, submission and retention operations shared by the web app, API and CLI."""
from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from . import archive as archive_mod
from . import jobs
from .config import get_settings, load_brand
from .db import AIResult, AISpanRow, MatchRow, Report, SimilarityResult, SourceRow, Submission, User, utcnow
from .ingest import IngestError, sniff_type
from .pipeline import new_id, sub_dir
from .security import hash_password, verify_password


class ServiceError(Exception):
    """A problem with a request that is safe to show to the user."""


def create_user(db: Session, email: str, password: str, name: str = "", role: str = "user", quota: int | None = None) -> User:
    email = email.strip().lower()
    if "@" not in email or len(email) > 254:
        raise ServiceError("Enter a valid email address.")
    if len(password) < 8:
        raise ServiceError("Password must be at least 8 characters.")
    if db.query(User).filter_by(email=email).first():
        raise ServiceError("An account with this email already exists.")
    u = User(email=email, name=name.strip()[:120], password_hash=hash_password(password), role=role,
             quota=get_settings().default_quota if quota is None else quota)
    db.add(u)
    db.commit()
    return u


def authenticate(db: Session, email: str, password: str) -> User | None:
    u = db.query(User).filter_by(email=email.strip().lower()).first()
    if u and u.active and verify_password(password, u.password_hash):
        return u
    return None


def bootstrap_admin(db: Session) -> None:
    s = get_settings()
    if s.admin_email and s.admin_password and not db.query(User).filter_by(email=s.admin_email.lower()).first():
        create_user(db, s.admin_email, s.admin_password, "Administrator", "admin", quota=0)


def create_submission(db: Session, user: User, upload: Path, filename: str, title: str = "", author: str = "", options: dict | None = None,
                      consent_archive: bool = False, keep: bool = False, start: bool = True) -> Submission:
    """Validate an uploaded file, charge one credit, store it and queue the check."""
    s = get_settings()
    if user.role != "admin" and user.quota < 1:
        raise ServiceError("You have no checks left. Ask an administrator to add credits.")
    if db.query(Submission).filter(Submission.user_id == user.id, Submission.status.in_(["queued", "ingesting", "checking", "reporting"])).count() >= 3:
        raise ServiceError("You already have three checks in progress. Wait for one to finish.")
    kind = sniff_type(upload)
    if kind is None:
        raise ServiceError("Unsupported file. Please upload a PDF, DOCX or TXT file.")
    brand = load_brand(s)
    opts = {"exclude_biblio": True, "exclude_quotes": False, "min_words": s.min_match_words, **(options or {})}
    opts["min_words"] = max(5, min(int(opts["min_words"]), 100))
    sub = Submission(public_id=new_id(db, brand["id_prefix"]), user_id=user.id, file_name=Path(filename).name[:300] or "document",
                     file_size=upload.stat().st_size, kind=kind, title=title.strip()[:300], author=author.strip()[:200], options=opts,
                     consent_archive=consent_archive, keep=keep)
    db.add(sub)
    db.flush()
    shutil.move(str(upload), str(sub_dir(sub.id) / "upload.bin"))
    if user.role != "admin":
        user.quota -= 1
    db.commit()
    if start:
        jobs.enqueue(sub.id)
    return sub


def delete_submission(db: Session, sub: Submission, remove_archive: bool = True) -> None:
    sid = sub.id
    if remove_archive:
        archive_mod.remove_for_submission(db, sid)
    for model in (MatchRow, SourceRow, SimilarityResult, AISpanRow, AIResult, Report):
        db.execute(delete(model).where(model.submission_id == sid))
    db.delete(sub)
    db.commit()
    shutil.rmtree(get_settings().data_dir / "subs" / str(sid), ignore_errors=True)


def purge_expired(db: Session) -> int:
    """Delete uploads and reports older than the retention period (archive entries stay: the uploader consented)."""
    s = get_settings()
    cutoff = utcnow() - dt.timedelta(days=s.retention_days)
    old = db.query(Submission).filter(Submission.created_at < cutoff, Submission.keep.is_(False),
                                      Submission.status.in_(["done", "failed"])).all()
    for sub in old:
        delete_submission(db, sub, remove_archive=False)
    return len(old)
