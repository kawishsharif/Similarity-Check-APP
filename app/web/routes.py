"""Web pages and JSON API."""
from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import jobs, services
from ..config import get_settings, load_brand, save_brand
from ..db import AIResult, Report, SimilarityResult, SourceRow, Submission, User, session, utcnow
from ..reports.build import _pct
from ..security import csrf_ok, limiter, new_csrf

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.filters["pct"] = _pct
templates.env.filters["httplink"] = lambda s: bool(re.match(r"^https?://", s or ""))
COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
ACTIVE = ("queued", "ingesting", "checking", "reporting")


# ------------------------------------------------------------------------------------------ helpers
def get_db():
    db = session()
    try:
        yield db
    finally:
        db.close()


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "?"


def current_user(request: Request, db: Session) -> User | None:
    uid = request.session.get("uid")
    u = db.get(User, uid) if uid else None
    return u if u and u.active else None


def csrf_token(request: Request) -> str:
    if "csrf" not in request.session:
        request.session["csrf"] = new_csrf()
    return request.session["csrf"]


def render(request: Request, db: Session, name: str, status: int = 200, **ctx):
    user = current_user(request, db)
    ctx.update(request=request, user=user, brand=load_brand(), csrf=csrf_token(request), flash=request.session.pop("flash", None), settings=get_settings())
    return templates.TemplateResponse(request, name, ctx, status_code=status)


def flash(request: Request, msg: str, kind: str = "info") -> None:
    request.session["flash"] = {"msg": msg, "kind": kind}


class Redirect(Exception):
    def __init__(self, url: str):
        self.url = url


def need_user(request: Request, db: Session) -> User:
    u = current_user(request, db)
    if u is None:
        if request.url.path.startswith("/api/"):
            raise HTTPException(401, "Sign in required")
        raise Redirect("/login")
    return u


def need_admin(request: Request, db: Session) -> User:
    u = need_user(request, db)
    if u.role != "admin":
        raise HTTPException(403, "Administrators only")
    return u


def check_csrf(request: Request, form_token: str | None = None) -> None:
    given = form_token or request.headers.get("x-csrf-token")
    if not csrf_ok(request.session.get("csrf"), given):
        raise HTTPException(403, "Invalid or missing CSRF token. Reload the page and try again.")


def owned(db: Session, user: User, public_id: str) -> Submission:
    sub = db.query(Submission).filter_by(public_id=public_id).first()
    if sub is None or (sub.user_id != user.id and user.role != "admin"):
        raise HTTPException(404, "Submission not found")
    return sub


def result_payload(db: Session, sub: Submission) -> dict:
    out = {"id": sub.public_id, "status": sub.status, "stage": sub.stage, "progress": round(sub.progress, 1), "error": sub.error, "title": sub.title,
           "file_name": sub.file_name, "pages": sub.pages, "words": sub.words, "language": sub.language, "created_at": sub.created_at.isoformat() + "Z",
           "similarity": None, "ai": None, "reports": {}}
    if sub.status == "done":
        sim, ai = db.get(SimilarityResult, sub.id), db.get(AIResult, sub.id)
        srcs = db.query(SourceRow).filter_by(submission_id=sub.id).order_by(SourceRow.rank).all()
        if sim:
            out["similarity"] = {"percent": sim.percent, "groups": sim.groups, "source_types": sim.source_types, "warnings": sim.flags.get("warnings", []),
                                 "filters": sim.flags.get("filters"), "integrity": sim.flags.get("integrity", []),
                                 "sources": [{"rank": s.rank, "type": s.type, "title": s.title, "address": s.address, "percent": s.percent, "color": s.color} for s in srcs]}
        if ai:
            out["ai"] = {"percent": ai.percent, "display": ai.display, "scored": ai.scored, "groups": ai.groups, "qualifying_words": ai.qualifying_words, "message": ai.message}
        for r in db.query(Report).filter_by(submission_id=sub.id):
            out["reports"][r.type] = {"report_id": r.report_id, "url": f"/api/submissions/{sub.public_id}/report/{r.type}", "figure": r.figure}
    return out


async def save_upload(file: UploadFile) -> Path:
    s = get_settings()
    limit = s.max_upload_mb * 1024 * 1024
    tmp = s.path("tmp")
    fd, name = tempfile.mkstemp(dir=tmp, prefix="up_")
    size = 0
    with open(fd, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                Path(name).unlink(missing_ok=True)
                raise services.ServiceError(f"The file is larger than {s.max_upload_mb} MB.")
            out.write(chunk)
    if size == 0:
        Path(name).unlink(missing_ok=True)
        raise services.ServiceError("The file is empty.")
    return Path(name)


def options_from(exclude_biblio, exclude_quotes, min_words) -> dict:
    return {"exclude_biblio": bool(exclude_biblio), "exclude_quotes": bool(exclude_quotes), "min_words": min_words}


# ------------------------------------------------------------------------------------------ pages
@router.get("/healthz")
def healthz():
    return {"ok": True}


@router.get("/brand/logo")
def brand_logo():
    path = Path(load_brand()["logo_path"])
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png", headers={"Cache-Control": "no-cache"})


@router.get("/")
def home(request: Request, db: Session = Depends(get_db)):
    return render(request, db, "home.html")


@router.get("/help")
def help_page(request: Request, db: Session = Depends(get_db)):
    return render(request, db, "help.html")


@router.get("/signup")
def signup_form(request: Request, db: Session = Depends(get_db)):
    return render(request, db, "signup.html")


@router.post("/signup")
def signup(request: Request, email: str = Form(...), password: str = Form(...), name: str = Form(""), csrf: str = Form(""), db: Session = Depends(get_db)):
    check_csrf(request, csrf)
    if not get_settings().signup_open:
        return render(request, db, "signup.html", 403, error="Sign-up is closed. Accounts are created by an administrator.")
    if not limiter.allow(f"signup:{client_ip(request)}", 10, 3600):
        return render(request, db, "signup.html", 429, error="Too many sign-ups from this address. Try again later.")
    try:
        u = services.create_user(db, email, password, name)
    except services.ServiceError as exc:
        return render(request, db, "signup.html", 400, error=str(exc))
    request.session.clear()
    request.session["uid"] = u.id
    return RedirectResponse("/upload", 303)


@router.get("/login")
def login_form(request: Request, db: Session = Depends(get_db)):
    return render(request, db, "login.html")


@router.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...), csrf: str = Form(""), db: Session = Depends(get_db)):
    check_csrf(request, csrf)
    if not limiter.allow(f"login:{client_ip(request)}", 10, 60):
        return render(request, db, "login.html", 429, error="Too many attempts. Wait a minute and try again.")
    u = services.authenticate(db, email, password)
    if not u:
        return render(request, db, "login.html", 401, error="Incorrect email or password.")
    request.session.clear()
    request.session["uid"] = u.id
    return RedirectResponse("/upload", 303)


@router.post("/logout")
def logout(request: Request, csrf: str = Form("")):
    check_csrf(request, csrf)
    request.session.clear()
    return RedirectResponse("/", 303)


@router.get("/upload")
def upload_form(request: Request, db: Session = Depends(get_db)):
    need_user(request, db)
    return render(request, db, "upload.html")


@router.post("/upload")
async def upload(request: Request, file: UploadFile = File(...), title: str = Form(""), author: str = Form(""), exclude_biblio: str = Form(""),
                 exclude_quotes: str = Form(""), min_words: int = Form(8), consent_archive: str = Form(""), keep: str = Form(""), csrf: str = Form(""),
                 db: Session = Depends(get_db)):
    user = need_user(request, db)
    check_csrf(request, csrf)
    if not limiter.allow(f"upload:{user.id}", 20, 3600):
        return render(request, db, "upload.html", 429, error="Upload limit reached (20 per hour).")
    try:
        path = await save_upload(file)
        sub = services.create_submission(db, user, path, file.filename or "document", title, author, options_from(exclude_biblio, exclude_quotes, min_words),
                                         bool(consent_archive), bool(keep))
    except services.ServiceError as exc:
        return render(request, db, "upload.html", 400, error=str(exc))
    return RedirectResponse(f"/submissions/{sub.public_id}", 303)


@router.get("/submissions/{public_id}")
def submission_page(public_id: str, request: Request, db: Session = Depends(get_db)):
    user = need_user(request, db)
    sub = owned(db, user, public_id)
    return render(request, db, "submission.html", sub=sub, data=result_payload(db, sub), active=sub.status in ACTIVE)


@router.get("/history")
def history(request: Request, q: str = "", db: Session = Depends(get_db)):
    user = need_user(request, db)
    query = db.query(Submission).filter_by(user_id=user.id)
    if q.strip():
        like = f"%{q.strip()}%"
        query = query.filter((Submission.title.ilike(like)) | (Submission.file_name.ilike(like)) | (Submission.public_id.ilike(like)))
    subs = query.order_by(Submission.created_at.desc()).limit(200).all()
    rows = []
    for s in subs:
        sim, ai = (db.get(SimilarityResult, s.id), db.get(AIResult, s.id)) if s.status == "done" else (None, None)
        rows.append({"sub": s, "sim": sim, "ai": ai})
    return render(request, db, "history.html", rows=rows, q=q)


@router.post("/submissions/{public_id}/delete")
def delete_page(public_id: str, request: Request, csrf: str = Form(""), db: Session = Depends(get_db)):
    user = need_user(request, db)
    check_csrf(request, csrf)
    services.delete_submission(db, owned(db, user, public_id))
    flash(request, "Submission deleted.")
    return RedirectResponse("/history", 303)


# ------------------------------------------------------------------------------------------ verification
@router.get("/verify")
def verify_form(request: Request, db: Session = Depends(get_db)):
    return render(request, db, "verify.html", report=None, searched=False)


@router.post("/verify")
def verify_lookup(request: Request, report_id: str = Form(...), csrf: str = Form(""), db: Session = Depends(get_db)):
    check_csrf(request, csrf)
    return RedirectResponse(f"/verify/{report_id.strip()}", 303)


def _verify_data(db: Session, report_id: str) -> dict | None:
    r = db.query(Report).filter_by(report_id=report_id).first()
    if not r:
        return None
    sub = db.get(Submission, r.submission_id)
    return {"report_id": r.report_id, "type": r.type, "figure": r.figure, "checksum": r.checksum, "issued": r.created_at.strftime("%Y-%m-%d %H:%M UTC"),
            "submission_id": sub.public_id if sub else ""}


@router.get("/verify/{report_id}")
def verify_page(report_id: str, request: Request, db: Session = Depends(get_db)):
    return render(request, db, "verify.html", report=_verify_data(db, report_id), searched=True, report_id=report_id, match=None)


@router.post("/verify/{report_id}/check")
async def verify_file(report_id: str, request: Request, file: UploadFile = File(...), csrf: str = Form(""), db: Session = Depends(get_db)):
    check_csrf(request, csrf)
    data = _verify_data(db, report_id)
    h = hashlib.sha256()
    n = 0
    while chunk := await file.read(1024 * 1024):
        n += len(chunk)
        if n > 100 * 1024 * 1024:
            break
        h.update(chunk)
    match = bool(data) and h.hexdigest() == data["checksum"]
    return render(request, db, "verify.html", report=data, searched=True, report_id=report_id, match=match)


# ------------------------------------------------------------------------------------------ JSON API
@router.get("/api/csrf")
def api_csrf(request: Request, db: Session = Depends(get_db)):
    need_user(request, db)
    return {"csrf": csrf_token(request)}


@router.post("/api/submissions", status_code=202)
async def api_create(request: Request, file: UploadFile = File(...), title: str = Form(""), author: str = Form(""), exclude_biblio: bool = Form(True),
                     exclude_quotes: bool = Form(False), min_words: int = Form(8), consent_archive: bool = Form(False), keep: bool = Form(False),
                     db: Session = Depends(get_db)):
    user = need_user(request, db)
    check_csrf(request)
    if not limiter.allow(f"upload:{user.id}", 20, 3600):
        raise HTTPException(429, "Upload limit reached (20 per hour).")
    try:
        path = await save_upload(file)
        sub = services.create_submission(db, user, path, file.filename or "document", title, author, options_from(exclude_biblio, exclude_quotes, min_words), consent_archive, keep)
    except services.ServiceError as exc:
        raise HTTPException(400, str(exc))
    return {"id": sub.public_id, "status": sub.status}


@router.get("/api/submissions/{public_id}")
def api_status(public_id: str, request: Request, db: Session = Depends(get_db)):
    user = need_user(request, db)
    return result_payload(db, owned(db, user, public_id))


@router.get("/api/submissions/{public_id}/report/{rtype}")
def api_report(public_id: str, rtype: str, request: Request, db: Session = Depends(get_db)):
    user = need_user(request, db)
    sub = owned(db, user, public_id)
    if rtype not in ("similarity", "ai"):
        raise HTTPException(404, "Unknown report type")
    r = db.query(Report).filter_by(submission_id=sub.id, type=rtype).first()
    if not r or not Path(r.file_path).exists():
        raise HTTPException(404, "Report not available")
    return FileResponse(r.file_path, media_type="application/pdf", filename=f"{sub.public_id}-{rtype}-report.pdf")


@router.delete("/api/submissions/{public_id}")
def api_delete(public_id: str, request: Request, db: Session = Depends(get_db)):
    user = need_user(request, db)
    check_csrf(request)
    services.delete_submission(db, owned(db, user, public_id))
    return {"deleted": True}


@router.get("/api/verify/{report_id}")
def api_verify(report_id: str, db: Session = Depends(get_db)):
    data = _verify_data(db, report_id)
    if not data:
        raise HTTPException(404, "No report with that ID")
    return data


# ------------------------------------------------------------------------------------------ admin
@router.get("/admin")
def admin(request: Request, db: Session = Depends(get_db)):
    need_admin(request, db)
    counts = dict(db.query(Submission.status, func.count()).group_by(Submission.status).all())
    users = db.query(User).order_by(User.created_at).all()
    usage = dict(db.query(Submission.user_id, func.count()).group_by(Submission.user_id).all())
    failed = db.query(Submission).filter_by(status="failed").order_by(Submission.created_at.desc()).limit(25).all()
    stats = {"users": len(users), "submissions": sum(counts.values()), "last7": db.query(Submission).filter(Submission.created_at > utcnow().replace(microsecond=0) - __import__("datetime").timedelta(days=7)).count()}
    return render(request, db, "admin.html", counts=counts, users=users, usage=usage, failed=failed, stats=stats, brand_raw=load_brand())


@router.post("/admin/users")
def admin_create_user(request: Request, email: str = Form(...), password: str = Form(...), name: str = Form(""), quota: int = Form(10), role: str = Form("user"),
                      csrf: str = Form(""), db: Session = Depends(get_db)):
    need_admin(request, db)
    check_csrf(request, csrf)
    try:
        services.create_user(db, email, password, name, "admin" if role == "admin" else "user", quota)
        flash(request, "User created.")
    except services.ServiceError as exc:
        flash(request, str(exc), "error")
    return RedirectResponse("/admin", 303)


@router.post("/admin/users/{uid}")
def admin_update_user(uid: int, request: Request, quota: int = Form(...), role: str = Form("user"), active: str = Form(""), csrf: str = Form(""), db: Session = Depends(get_db)):
    me = need_admin(request, db)
    check_csrf(request, csrf)
    u = db.get(User, uid)
    if not u:
        raise HTTPException(404)
    u.quota = max(0, quota)
    if u.id != me.id:  # an admin cannot lock themselves out
        u.role, u.active = ("admin" if role == "admin" else "user"), bool(active)
    db.commit()
    flash(request, f"Updated {u.email}.")
    return RedirectResponse("/admin", 303)


@router.post("/admin/brand")
async def admin_brand(request: Request, brand_name: str = Form(...), organisation: str = Form(""), primary_color: str = Form(...), ai_color: str = Form(...),
                      ai_color_paraphrased: str = Form(...), page_size: str = Form("Letter"), id_prefix: str = Form(...), verify_url: str = Form(...),
                      logo: UploadFile | None = File(None), csrf: str = Form(""), db: Session = Depends(get_db)):
    need_admin(request, db)
    check_csrf(request, csrf)
    if not all(COLOR_RE.match(c) for c in (primary_color, ai_color, ai_color_paraphrased)):
        flash(request, "Colours must look like #0F4C81.", "error")
        return RedirectResponse("/admin", 303)
    if not re.match(r"^[A-Za-z0-9]{1,8}$", id_prefix) or page_size not in ("Letter", "A4") or not re.match(r"^https?://", verify_url):
        flash(request, "Check the ID prefix (letters/digits), page size and verification URL (must start with http).", "error")
        return RedirectResponse("/admin", 303)
    updates = {"brand_name": brand_name.strip()[:60], "organisation": organisation.strip()[:100], "primary_color": primary_color, "ai_color": ai_color,
               "ai_color_paraphrased": ai_color_paraphrased, "page_size": page_size, "id_prefix": id_prefix.upper(), "verify_url": verify_url.strip()}
    if logo is not None and logo.filename:
        data = await logo.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024 or not (data.startswith(b"\x89PNG\r\n\x1a\n") or data[:3] == b"\xff\xd8\xff"):
            flash(request, "The logo must be a PNG or JPEG under 2 MB.", "error")
            return RedirectResponse("/admin", 303)
        ext = "png" if data.startswith(b"\x89PNG") else "jpg"
        dest = get_settings().path("assets") / f"logo.{ext}"
        dest.write_bytes(data)
        updates["logo_png"] = str(dest)
    save_brand(updates)
    flash(request, "Brand settings saved. New reports will use them.")
    return RedirectResponse("/admin", 303)


@router.post("/admin/purge")
def admin_purge(request: Request, csrf: str = Form(""), db: Session = Depends(get_db)):
    need_admin(request, db)
    check_csrf(request, csrf)
    flash(request, f"Removed {services.purge_expired(db)} expired submissions.")
    return RedirectResponse("/admin", 303)


@router.post("/admin/submissions/{public_id}/retry")
def admin_retry(public_id: str, request: Request, csrf: str = Form(""), db: Session = Depends(get_db)):
    need_admin(request, db)
    check_csrf(request, csrf)
    sub = db.query(Submission).filter_by(public_id=public_id).first()
    upload = get_settings().data_dir / "subs" / str(sub.id) / "upload.bin" if sub else None
    if not sub or sub.status != "failed" or not upload.exists():
        flash(request, "Only failed submissions whose upload is still stored can be retried.", "error")
    else:
        u = db.get(User, sub.user_id)
        if u and u.role != "admin" and u.quota > 0:
            u.quota -= 1  # the failure refunded the credit; charge it again for the retry
        sub.status, sub.stage, sub.progress, sub.error = "queued", "Waiting in queue", 0.0, ""
        db.commit()
        jobs.enqueue(sub.id)
        flash(request, "Re-queued.")
    return RedirectResponse("/admin", 303)
