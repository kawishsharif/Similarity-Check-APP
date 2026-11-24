"""Runtime settings (environment variables) and brand settings (brand.yaml)."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BRAND = {
    "brand_name": "Your Brand",
    "organisation": "Your Organisation",
    "logo_png": "assets/logo.png",
    "primary_color": "#0F4C81",
    "ai_color": "#52C7DB",
    "ai_color_paraphrased": "#8E6BD8",
    "page_size": "Letter",
    "id_prefix": "YB",
    "verify_url": "http://localhost:8000/verify/",
}


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    data_dir: Path
    database_url: str
    redis_url: str | None
    secret_key: str
    signup_open: bool
    admin_email: str | None
    admin_password: str | None
    brand_file: Path
    max_pages: int
    max_words: int
    max_upload_mb: int
    retention_days: int
    workers: int
    sources: list[str]
    searxng_url: str | None
    contact_email: str
    openalex_key: str | None
    core_key: str | None
    s2_key: str | None
    max_queries: int
    max_candidates: int
    min_match_words: int
    cache_days: int
    default_quota: int
    https_only: bool
    base_url: str
    smtp_host: str | None
    smtp_port: int
    smtp_user: str | None
    smtp_password: str | None
    smtp_from: str
    ai_detector: str            # "heuristic" or "binoculars"
    ai_observer_model: str
    ai_performer_model: str
    ai_classifier_model: str | None
    ai_classifier_label: str
    sim_embeddings: bool
    exclude_own_archive: bool
    sources_network: bool = True
    extra: dict = field(default_factory=dict)

    def path(self, *parts: str) -> Path:
        p = self.data_dir.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


def _load_secret(data_dir: Path) -> str:
    env = os.environ.get("SECRET_KEY")
    if env:
        return env
    f = data_dir / "secret.key"
    if f.exists():
        return f.read_text().strip()
    key = secrets.token_hex(32)
    data_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(key)
    try:
        f.chmod(0o600)
    except OSError:
        pass
    return key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    e = os.environ.get
    data_dir = Path(e("DATA_DIR", str(ROOT / "data"))).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        data_dir=data_dir,
        database_url=e("DATABASE_URL", f"sqlite:///{data_dir / 'app.db'}"),
        redis_url=e("REDIS_URL") or None,
        secret_key=_load_secret(data_dir),
        signup_open=_bool(e("SIGNUP_OPEN"), True),
        admin_email=e("ADMIN_EMAIL"),
        admin_password=e("ADMIN_PASSWORD"),
        brand_file=Path(e("BRAND_FILE", str(ROOT / "brand.yaml"))),
        max_pages=_int("MAX_PAGES", 100),
        max_words=_int("MAX_WORDS", 30000),
        max_upload_mb=_int("MAX_UPLOAD_MB", 25),
        retention_days=_int("RETENTION_DAYS", 30),
        workers=_int("WORKERS", 2),
        sources=[s.strip() for s in e("SOURCES", "openalex,crossref,arxiv,europepmc,wikipedia,searxng,archive").split(",") if s.strip()],
        searxng_url=e("SEARXNG_URL") or None,
        contact_email=e("CONTACT_EMAIL", "admin@example.com"),
        openalex_key=e("OPENALEX_API_KEY") or None,
        core_key=e("CORE_API_KEY") or None,
        s2_key=e("S2_API_KEY") or None,
        max_queries=_int("MAX_QUERIES", 80),
        max_candidates=_int("MAX_CANDIDATES", 100),
        min_match_words=_int("MIN_MATCH_WORDS", 8),
        cache_days=_int("CACHE_DAYS", 30),
        default_quota=_int("DEFAULT_QUOTA", 10),
        https_only=_bool(e("HTTPS_ONLY"), False),
        base_url=e("BASE_URL", "http://localhost:8000").rstrip("/"),
        smtp_host=e("SMTP_HOST") or None,
        smtp_port=_int("SMTP_PORT", 587),
        smtp_user=e("SMTP_USER") or None,
        smtp_password=e("SMTP_PASSWORD") or None,
        smtp_from=e("SMTP_FROM", "noreply@example.com"),
        ai_detector=e("AI_DETECTOR", "heuristic"),
        ai_observer_model=e("AI_OBSERVER_MODEL", "Qwen/Qwen2.5-1.5B"),
        ai_performer_model=e("AI_PERFORMER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        ai_classifier_model=e("AI_CLASSIFIER_MODEL") or None,
        ai_classifier_label=e("AI_CLASSIFIER_LABEL", "AI"),
        sim_embeddings=_bool(e("SIM_EMBEDDINGS"), False),
        exclude_own_archive=_bool(e("EXCLUDE_OWN_ARCHIVE"), True),
    )


def reset_settings() -> None:
    get_settings.cache_clear()


def load_brand(s: Settings | None = None) -> dict:
    s = s or get_settings()
    brand = dict(DEFAULT_BRAND)
    if s.brand_file.exists():
        brand.update(yaml.safe_load(s.brand_file.read_text(encoding="utf-8")) or {})
    logo = Path(brand["logo_png"])
    if not logo.is_absolute():
        logo = s.brand_file.parent / logo
        if not logo.exists():
            logo = ROOT / brand["logo_png"]
    brand["logo_path"] = str(logo)
    if brand["page_size"] not in ("Letter", "A4"):
        brand["page_size"] = "Letter"
    return brand


def save_brand(updates: dict, s: Settings | None = None) -> None:
    s = s or get_settings()
    cur = {}
    if s.brand_file.exists():
        cur = yaml.safe_load(s.brand_file.read_text(encoding="utf-8")) or {}
    for k, v in updates.items():
        if k in DEFAULT_BRAND:
            cur[k] = v
    s.brand_file.write_text(yaml.safe_dump(cur, allow_unicode=True, sort_keys=False), encoding="utf-8")
