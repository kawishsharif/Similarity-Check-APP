"""Job queue. With REDIS_URL set, jobs go to RQ (run `python -m app.worker`); without it, a thread
pool inside the web process runs them, which is enough for a single small server."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from .config import get_settings
from .db import Submission, session_scope

log = logging.getLogger(__name__)
_pool: ThreadPoolExecutor | None = None


def _run(sid: int) -> None:
    from .pipeline import process_submission

    process_submission(sid)


def enqueue(sid: int) -> None:
    s = get_settings()
    if s.redis_url:
        from redis import Redis
        from rq import Queue

        Queue("checks", connection=Redis.from_url(s.redis_url)).enqueue("app.pipeline.process_submission", sid, job_timeout=3600, failure_ttl=86400)
        return
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=s.workers, thread_name_prefix="check")
    _pool.submit(_run, sid)


def recover_stuck() -> int:
    """After a restart, push unfinished submissions back on the queue (in-process mode only)."""
    if get_settings().redis_url:
        return 0
    with session_scope() as db:
        ids = [i for (i,) in db.query(Submission.id).filter(Submission.status.in_(["queued", "ingesting", "checking", "reporting"]))]
    for i in ids:
        enqueue(i)
    return len(ids)


def shutdown() -> None:
    global _pool
    if _pool:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
