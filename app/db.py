"""Database models (SQLAlchemy 2.0). SQLite by default, PostgreSQL in production via DATABASE_URL."""
from __future__ import annotations

import datetime as dt
from contextlib import contextmanager

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import get_settings


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="user")  # user | admin
    quota: Mapped[int] = mapped_column(Integer, default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class Submission(Base):
    __tablename__ = "submissions"
    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    file_name: Mapped[str] = mapped_column(String(300))
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    kind: Mapped[str] = mapped_column(String(10), default="")
    title: Mapped[str] = mapped_column(String(300), default="")
    author: Mapped[str] = mapped_column(String(200), default="")
    pages: Mapped[int] = mapped_column(Integer, default=0)
    words: Mapped[int] = mapped_column(Integer, default=0)
    characters: Mapped[int] = mapped_column(Integer, default=0)
    language: Mapped[str] = mapped_column(String(10), default="")
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)  # queued|ingesting|checking|reporting|done|failed
    stage: Mapped[str] = mapped_column(String(200), default="Waiting in queue")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str] = mapped_column(Text, default="")
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    consent_archive: Mapped[bool] = mapped_column(Boolean, default=False)
    keep: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class SimilarityResult(Base):
    __tablename__ = "similarity_results"
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"), primary_key=True)
    percent: Mapped[float] = mapped_column(Float)
    groups: Mapped[list] = mapped_column(JSON)
    source_types: Mapped[list] = mapped_column(JSON)
    flags: Mapped[dict] = mapped_column(JSON)  # filters, warnings, integrity flags, stats
    matched_words: Mapped[int] = mapped_column(Integer, default=0)
    total_words: Mapped[int] = mapped_column(Integer, default=0)


class SourceRow(Base):
    __tablename__ = "sources"
    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(20))
    title: Mapped[str] = mapped_column(Text)
    address: Mapped[str] = mapped_column(Text)
    matched_words: Mapped[int] = mapped_column(Integer)
    percent: Mapped[float] = mapped_column(Float)
    color: Mapped[str] = mapped_column(String(10), default="#888888")


class MatchRow(Base):
    __tablename__ = "matches"
    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"), index=True)
    source_rank: Mapped[int] = mapped_column(Integer)
    page: Mapped[int] = mapped_column(Integer, default=0)
    start_word: Mapped[int] = mapped_column(Integer)
    end_word: Mapped[int] = mapped_column(Integer)
    category: Mapped[str] = mapped_column(String(30))
    kind: Mapped[str] = mapped_column(String(12), default="exact")


class AIResult(Base):
    __tablename__ = "ai_results"
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"), primary_key=True)
    percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    display: Mapped[str] = mapped_column(String(10), default="--")
    scored: Mapped[bool] = mapped_column(Boolean, default=False)
    groups: Mapped[list] = mapped_column(JSON)
    qualifying_words: Mapped[int] = mapped_column(Integer, default=0)
    flagged_words: Mapped[int] = mapped_column(Integer, default=0)
    detector: Mapped[str] = mapped_column(String(30), default="")
    message: Mapped[str] = mapped_column(Text, default="")


class AISpanRow(Base):
    __tablename__ = "ai_spans"
    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"), index=True)
    page: Mapped[int] = mapped_column(Integer, default=0)
    start_word: Mapped[int] = mapped_column(Integer)
    end_word: Mapped[int] = mapped_column(Integer)
    group: Mapped[str] = mapped_column(String(10))
    score: Mapped[float] = mapped_column(Float, default=0.0)


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    report_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"), index=True)
    type: Mapped[str] = mapped_column(String(12))  # similarity | ai
    file_path: Mapped[str] = mapped_column(Text)
    checksum: Mapped[str] = mapped_column(String(64))
    figure: Mapped[str] = mapped_column(String(20), default="")  # the headline number as printed on the report
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class ArchiveDocument(Base):
    __tablename__ = "archive_documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    text_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class ArchiveFingerprint(Base):
    __tablename__ = "archive_fingerprints"
    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[int] = mapped_column(BigInteger, index=True)  # signed 64-bit
    document_id: Mapped[int] = mapped_column(ForeignKey("archive_documents.id"), index=True)
    position: Mapped[int] = mapped_column(Integer)


_engine = None
_Session: sessionmaker | None = None


def get_engine():
    global _engine, _Session
    if _engine is None:
        url = get_settings().database_url
        kw = {"connect_args": {"check_same_thread": False, "timeout": 30}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        _engine = create_engine(url, **kw)
        if url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def _pragmas(conn, _):  # WAL lets the web app read while workers write
                cur = conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        Base.metadata.create_all(_engine)
        _Session = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def reset_engine() -> None:
    global _engine, _Session
    if _engine is not None:
        _engine.dispose()
    _engine, _Session = None, None


def session() -> Session:
    get_engine()
    assert _Session is not None
    return _Session()


@contextmanager
def session_scope():
    s = session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
