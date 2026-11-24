"""FastAPI application factory."""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware

from . import jobs, services
from .config import get_settings
from .db import get_engine, session_scope

STATIC = Path(__file__).parent / "web" / "static"
log = logging.getLogger("app")

CSP = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"


def _purge_loop(stop: threading.Event) -> None:
    while not stop.wait(24 * 3600):
        try:
            with session_scope() as db:
                n = services.purge_expired(db)
            log.info("retention purge removed %d submissions", n)
        except Exception:
            log.exception("retention purge failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_engine()
    with session_scope() as db:
        services.bootstrap_admin(db)
        services.purge_expired(db)
    n = jobs.recover_stuck()
    if n:
        log.info("re-queued %d unfinished submissions", n)
    stop = threading.Event()
    threading.Thread(target=_purge_loop, args=(stop,), daemon=True).start()
    yield
    stop.set()
    jobs.shutdown()


def create_app() -> FastAPI:
    from fastapi.staticfiles import StaticFiles

    from .web.routes import router

    s = get_settings()
    app = FastAPI(title="Similarity and AI-Writing Checker", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.add_middleware(SessionMiddleware, secret_key=s.secret_key, session_cookie="sc_session", max_age=14 * 24 * 3600, same_site="lax", https_only=s.https_only)

    @app.middleware("http")
    async def security_headers(request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        if not request.url.path.startswith("/api/docs"):
            resp.headers.setdefault("Content-Security-Policy", CSP)
        return resp

    from fastapi import HTTPException, Request
    from fastapi.responses import JSONResponse, RedirectResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from .db import session as db_session
    from .web.routes import Redirect, render

    @app.exception_handler(Redirect)
    async def _redirect(request: Request, exc: Redirect):
        return RedirectResponse(exc.url, 303)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        db = db_session()
        try:
            return render(request, db, "error.html", exc.status_code, code=exc.status_code, message=exc.detail if isinstance(exc.detail, str) else "Something went wrong.")
        finally:
            db.close()

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    app.include_router(router)
    return app


app = create_app()
