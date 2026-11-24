"""RQ worker entry point: `python -m app.worker` (needs REDIS_URL)."""
from redis import Redis
from rq import Queue, Worker

from .config import get_settings

if __name__ == "__main__":
    s = get_settings()
    if not s.redis_url:
        raise SystemExit("REDIS_URL is not set")
    conn = Redis.from_url(s.redis_url)
    Worker([Queue("checks", connection=conn)], connection=conn).work()
