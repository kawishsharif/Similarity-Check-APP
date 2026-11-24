import re
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfReader

from app.main import app
from app.security import limiter
from conftest import ORIGINAL, OWN_WORDS, make_pdf


def csrf_of(client, path="/login"):
    html = client.get(path).text
    return re.search(r'name="csrf" value="([^"]+)"', html).group(1)


def signup(client, email=None):
    email = email or f"w{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/signup", data={"email": email, "password": "password123", "name": "Web User", "csrf": csrf_of(client, "/signup")}, follow_redirects=False)
    assert r.status_code == 303, r.text
    return email


@pytest.fixture()
def client():
    limiter.reset()
    with TestClient(app) as c:
        yield c


def wait_done(client, pid, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        d = client.get(f"/api/submissions/{pid}").json()
        if d["status"] in ("done", "failed"):
            return d
        time.sleep(0.5)
    raise AssertionError("timeout")


def test_pages_render_and_need_login(client):
    for path in ("/", "/help", "/login", "/signup", "/verify", "/healthz"):
        assert client.get(path).status_code == 200
    r = client.get("/upload", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert client.get("/api/submissions/XX-1").status_code == 401
    assert "default-src 'self'" in client.get("/").headers["content-security-policy"]


def test_login_wrong_password_and_csrf(client):
    assert client.post("/login", data={"email": "a@b.co", "password": "x", "csrf": "bad"}).status_code == 403
    r = client.post("/login", data={"email": "nobody@example.com", "password": "wrongpass1", "csrf": csrf_of(client)})
    assert r.status_code == 401 and "Incorrect email or password" in r.text


def test_upload_check_download_verify_delete(client, tmp_path):
    signup(client)
    pdf = make_pdf(tmp_path / "paper.pdf", [ORIGINAL, OWN_WORDS])
    token = csrf_of(client, "/upload")
    r = client.post("/upload", data={"csrf": token, "exclude_biblio": "1", "min_words": "8"}, files={"file": ("paper.pdf", pdf.read_bytes(), "application/pdf")}, follow_redirects=False)
    assert r.status_code == 303
    pid = r.headers["location"].rsplit("/", 1)[1]
    d = wait_done(client, pid)
    assert d["status"] == "done", d
    assert d["similarity"]["percent"] == 0 and d["ai"]["scored"] is False
    page = client.get(f"/submissions/{pid}")
    assert page.status_code == 200 and "Download Similarity Report" in page.text
    rep = client.get(d["reports"]["similarity"]["url"])
    assert rep.status_code == 200 and rep.content[:5] == b"%PDF-"
    out = tmp_path / "got.pdf"; out.write_bytes(rep.content)
    assert len(PdfReader(str(out)).pages) >= 3
    # public verification
    rid = d["reports"]["similarity"]["report_id"]
    anon = TestClient(app)
    assert anon.get(f"/api/verify/{rid}").json()["type"] == "similarity"
    tok = re.search(r'name="csrf" value="([^"]+)"', anon.get(f"/verify/{rid}").text).group(1)
    ok = anon.post(f"/verify/{rid}/check", data={"csrf": tok}, files={"file": ("r.pdf", rep.content, "application/pdf")})
    assert "identical to the report issued" in ok.text
    bad = anon.post(f"/verify/{rid}/check", data={"csrf": tok}, files={"file": ("r.pdf", rep.content + b"x", "application/pdf")})
    assert "does <b>not</b> match" in bad.text
    assert anon.get("/api/verify/NOPE").status_code == 404
    # history shows it, delete removes it
    assert pid in client.get("/history").text
    tok = csrf_of(client, f"/submissions/{pid}")
    assert client.post(f"/submissions/{pid}/delete", data={"csrf": tok}, follow_redirects=False).status_code == 303
    assert client.get(f"/api/submissions/{pid}").status_code == 404


def test_other_users_cannot_see_submission(client, tmp_path):
    signup(client)
    pdf = make_pdf(tmp_path / "p.pdf", [ORIGINAL])
    token = csrf_of(client, "/upload")
    r = client.post("/upload", data={"csrf": token}, files={"file": ("p.pdf", pdf.read_bytes(), "application/pdf")}, follow_redirects=False)
    pid = r.headers["location"].rsplit("/", 1)[1]
    other = TestClient(app)
    signup(other)
    assert other.get(f"/api/submissions/{pid}").status_code == 404
    assert other.get(f"/api/submissions/{pid}/report/ai").status_code == 404
    assert other.get(f"/submissions/{pid}").status_code == 404
    wait_done(client, pid)


def test_bad_upload_rejected(client, tmp_path):
    signup(client)
    token = csrf_of(client, "/upload")
    r = client.post("/upload", data={"csrf": token}, files={"file": ("x.pdf", b"MZ\x90\x00binary", "application/pdf")})
    assert r.status_code == 400 and "Unsupported file" in r.text
    r = client.post("/upload", data={"csrf": "wrong"}, files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 403


def test_admin_logo_upload_changes_reports(client, tmp_path):
    client.post("/login", data={"email": "admin@example.com", "password": "admin-pass-123", "csrf": csrf_of(client)})
    page = client.get("/admin")
    assert page.status_code == 200 and "Report logo" in page.text
    tok = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
    png = tmp_path / "newlogo.png"
    Image.new("RGBA", (400, 100), (200, 30, 30, 255)).save(png)
    form = {"csrf": tok, "brand_name": "Acme Check", "organisation": "Acme Univ", "primary_color": "#112233", "ai_color": "#52C7DB", "ai_color_paraphrased": "#8E6BD8",
            "page_size": "A4", "id_prefix": "ac", "verify_url": "https://acme.example/verify/"}
    r = client.post("/admin/brand", data=form, files={"logo": ("logo.png", png.read_bytes(), "image/png")}, follow_redirects=False)
    assert r.status_code == 303
    served = client.get("/brand/logo")
    assert served.status_code == 200 and served.content == png.read_bytes()
    # a non-image is refused
    r = client.post("/admin/brand", data=form, files={"logo": ("logo.png", b"<svg onload=alert(1)>", "image/png")}, follow_redirects=True)
    assert "must be a PNG or JPEG" in r.text
    bad = client.post("/admin/brand", data={**form, "primary_color": "red;}"}, follow_redirects=True)
    assert "Colours must look like" in bad.text
    # new reports use the new brand: prefix, logo, A4
    pdf = make_pdf(tmp_path / "p.pdf", [ORIGINAL])
    tok = re.search(r'name="csrf" value="([^"]+)"', client.get("/upload").text).group(1)
    r = client.post("/upload", data={"csrf": tok}, files={"file": ("p.pdf", pdf.read_bytes(), "application/pdf")}, follow_redirects=False)
    pid = r.headers["location"].rsplit("/", 1)[1]
    assert pid.startswith("AC-")
    d = wait_done(client, pid)
    assert d["status"] == "done"
    out = tmp_path / "r.pdf"; out.write_bytes(client.get(d["reports"]["ai"]["url"]).content)
    rd = PdfReader(str(out))
    assert abs(float(rd.pages[0].mediabox.width) - 595.28) < 1
    assert rd.metadata.producer == "Acme Check"
    # non-admins cannot reach settings
    user = TestClient(app); signup(user)
    assert user.get("/admin").status_code == 403


def test_signup_closed_and_login_rate_limit(client, monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "signup_open", False)
    r = client.post("/signup", data={"email": "z@example.com", "password": "password123", "csrf": csrf_of(client, "/signup")})
    assert r.status_code == 403
    codes = [client.post("/login", data={"email": "q@example.com", "password": "nopenope1", "csrf": csrf_of(client)}).status_code for _ in range(12)]
    assert codes[-1] == 429
