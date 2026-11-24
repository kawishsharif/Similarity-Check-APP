import shutil
import uuid
from pathlib import Path

import pytest
from pypdf import PdfReader

from app import services
from app.config import get_settings
from app.db import AIResult, Report, SimilarityResult, SourceRow, Submission, session_scope
from app.pipeline import process_submission
from conftest import ORIGINAL, OWN_WORDS, make_pdf
from samples import AI_TEXT, HUMAN_TEXT

S = get_settings()
CJK_FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


def _user(db, quota=5):
    return services.create_user(db, f"u{uuid.uuid4().hex[:8]}@example.com", "password123", "Test User", quota=quota)


def _submit(db, user, path: Path, name: str, **kw):
    return services.create_submission(db, user, path, name, start=False, **kw)


def test_full_pipeline_with_archive_and_consent(tmp, monkeypatch):
    monkeypatch.setattr(S, "sources", ["archive"])
    with session_scope() as db:
        a, b = _user(db), _user(db)
        text = ORIGINAL + "\n\n" + OWN_WORDS + "\n\n" + AI_TEXT[:900]
        pa = tmp / "a.txt"; pa.write_text(text)
        sa = _submit(db, a, pa, "a.txt", consent_archive=True)
        assert a.quota == 4
        process_submission(sa.id)
        db.refresh(sa)
        assert sa.status == "done" and sa.words > 100 and sa.kind == "txt"
        # the same author resubmitting does not match their own archived paper
        pa2 = tmp / "a2.txt"; pa2.write_text(text)
        sa2 = _submit(db, a, pa2, "a2.txt")
        process_submission(sa2.id)
        assert db.get(SimilarityResult, sa2.id).percent == 0
        # another user submitting a copy matches the archive
        pb = tmp / "b.txt"; pb.write_text("My essay. " + ORIGINAL + "\n\nSomething entirely different that I wrote about boats and weather.")
        sb = _submit(db, b, pb, "b.txt")
        process_submission(sb.id)
        db.refresh(sb)
        res = db.get(SimilarityResult, sb.id)
        assert sb.status == "done" and res.percent > 30
        src = db.query(SourceRow).filter_by(submission_id=sb.id).one()
        assert src.type == "submitted" and "Paper in the platform archive" in src.title
        assert db.get(AIResult, sb.id) is not None
        reports = db.query(Report).filter_by(submission_id=sb.id).all()
        assert {r.type for r in reports} == {"similarity", "ai"}
        for r in reports:
            assert len(PdfReader(r.file_path).pages) >= 3
        # deleting removes files and archive entry
        services.delete_submission(db, db.get(Submission, sa.id))
        assert not (S.data_dir / "subs" / str(sa.id)).exists()


def test_failed_job_refunds_credit_and_reports_reason(tmp):
    with session_scope() as db:
        u = _user(db, quota=2)
        p = tmp / "bad.pdf"; p.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
        sub = _submit(db, u, p, "bad.pdf")
        assert u.quota == 1
        process_submission(sub.id)
        db.refresh(sub); db.refresh(u)
        assert sub.status == "failed" and "could not be read" in sub.error and u.quota == 2


def test_upload_rejected_by_content_not_extension(tmp):
    with session_scope() as db:
        u = _user(db)
        p = tmp / "evil.pdf"; p.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00binary")
        with pytest.raises(services.ServiceError):
            _submit(db, u, p, "evil.pdf")


def test_quota_enforced(tmp):
    with session_scope() as db:
        u = _user(db, quota=0)
        p = tmp / "x.txt"; p.write_text("hello world " * 10)
        with pytest.raises(services.ServiceError, match="no checks left"):
            _submit(db, u, p, "x.txt")


def test_page_limit_enforced(tmp, monkeypatch):
    monkeypatch.setattr(S, "max_words", 50)
    with session_scope() as db:
        u = _user(db)
        p = tmp / "long.txt"; p.write_text("word " * 200)
        sub = _submit(db, u, p, "long.txt")
        process_submission(sub.id)
        db.refresh(sub)
        assert sub.status == "failed" and "limit" in sub.error


@pytest.mark.skipif(not Path(CJK_FONT).exists(), reason="CJK font not installed")
def test_chinese_document_end_to_end(tmp, monkeypatch):
    monkeypatch.setattr(S, "sources", ["archive"])
    zh = ("工业革命在十八世纪后期深刻改变了西欧的经济结构，机械化的纺织生产把工人从农村吸引到迅速发展的工厂城镇。"
          "对煤炭和钢铁的需求重塑了英格兰北部的地貌，历史学家至今仍在争论第一代工厂工人的生活水平究竟是上升还是下降。")
    other = "我小时候生活的那个安静的港口小镇教会了我耐心，这比任何课堂都更有效，每天清晨祖母都会在日出前走去面包店。"
    with session_scope() as db:
        a, b = _user(db), _user(db)
        pa = tmp / "a.txt"; pa.write_text(zh + "\n\n" + other * 2)
        sa = _submit(db, a, pa, "zh.txt", consent_archive=True)
        process_submission(sa.id); db.refresh(sa)
        assert sa.status == "done" and sa.language == "zh"
        pb = tmp / "b.txt"; pb.write_text("引言。" + zh + "\n\n另外一段完全不同的内容，讲述天气、船只和海边的生活，与前文毫无关系，只是为了增加字数。")
        sb = _submit(db, b, pb, "zh2.txt"); process_submission(sb.id); db.refresh(sb)
        assert sb.status == "done"
        assert db.get(SimilarityResult, sb.id).percent > 30
        rep = db.query(Report).filter_by(submission_id=sb.id, type="similarity").one()
        text = "\n".join(p.extract_text() for p in PdfReader(rep.file_path).pages)
        assert "overall similarity" in text
