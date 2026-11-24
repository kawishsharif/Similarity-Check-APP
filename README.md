# Similarity and AI-Writing Checker

A web platform where a user uploads a PDF, DOCX or TXT paper and receives two branded PDF reports:
a **Similarity Report** and an **AI-Writing Report**. Built from free and open-source parts, following
`Technical Architecture and Development Plan` v1.0.

## Run it locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
# LibreOffice (DOCX), Tesseract (OCR) and Pango (WeasyPrint) are system packages; fonts-noto-cjk for Chinese.
export ADMIN_EMAIL=you@example.com ADMIN_PASSWORD='a-long-password'
uvicorn app.main:app --reload          # http://localhost:8000
```

With no `REDIS_URL` and no `DATABASE_URL` it uses SQLite in `./data` and runs checks on an in-process
thread pool, which is enough for one small server. Sign in as the admin, open **Settings**, and:

* **upload your report logo** (PNG or JPEG), set brand name, organisation, colours, page size (Letter/A4),
  ID prefix and verification URL — new reports use them straight away (also editable in `brand.yaml`);
* create users and set their check quotas.

Command-line check without the web app: `python -m app.cli check paper.pdf --corpus ./texts --reports ./out`.

## Deploy on one server

```bash
cp .env.example .env     # edit it
docker compose up -d --build
```
Starts web, worker (RQ on Valkey/Redis), PostgreSQL (pgvector image), SearXNG and Caddy (automatic HTTPS).

## Report format

Reports follow the layout of the reference report supplied with the project: header and footer on every
page (logo, `Page X of N - label`, Submission ID), a cover page with the title, uploader and organisation,
a Document Details block and a grey counts box, an overview page with a large headline figure, a blue
"review required" box, detection/match groups and a disclaimer, then the **original pages at their own
size** with highlights drawn over them (similarity highlights carry the source number in the margin).
Measurements, fonts (Lexend Deca, Noto Sans; both OFL, bundled in `app/reports/fonts`) and colours were
taken from that file. If an uploaded page's text runs into the header/footer band, that page is shrunk to fit.
The wording, IDs (`YB-2026-00012345`) and logo are your own: no third-party name, logo, ID format or report
text is used.

## Configuration

Environment variables (see `app/config.py`): `DATA_DIR`, `DATABASE_URL`, `REDIS_URL`, `SECRET_KEY`,
`SIGNUP_OPEN`, `ADMIN_EMAIL/PASSWORD`, `MAX_PAGES`, `MAX_WORDS`, `MAX_UPLOAD_MB`, `RETENTION_DAYS`,
`DEFAULT_QUOTA`, `SOURCES` (comma list of `openalex,crossref,semanticscholar,core,arxiv,europepmc,wikipedia,searxng,archive`),
`SEARXNG_URL`, `OPENALEX_API_KEY`, `CORE_API_KEY`, `S2_API_KEY`, `CONTACT_EMAIL`, `SMTP_*`,
`AI_DETECTOR` (`heuristic` | `binoculars`), `SIM_EMBEDDINGS`.

## Tests

```bash
pytest
```
Covers ingestion (PDF/DOCX/TXT/Chinese), alignment, overlap resolution, classification, AI windowing, report
PDFs, the full pipeline with archive and consent, and the web app (auth, CSRF, access control, upload, download,
verification, admin and logo upload). Tests never use the network.

## Honest status: what is and is not verified

Verified by the automated tests and by rendering the PDFs: everything listed under *Tests*, English and Chinese.

**Not verified / known gaps — read before launch**

* **AI-writing accuracy is not established.** The default `heuristic` detector is a lightweight stylometric
  baseline with hand-set features, and its default thresholds were only checked on two short hand-written
  samples. It has *not* been measured against RAID/HC3/M4 or pre-2022 student essays, so the 1% false-positive
  target in the plan is **not met or demonstrated**. Run `scripts/calibrate.py` on real labelled data
  before showing figures to anyone, and publish the measured rates. The Binoculars + classifier path
  (`AI_DETECTOR=binoculars`, `requirements-ai.txt`) implements the plan's method but has never been run here.
* **Similarity coverage vs. the plan's acceptance test** (85% recall on PAN, <5% false matches) has not been run.
  Online connectors (OpenAlex, Crossref, arXiv, Europe PMC, Wikipedia, SearXNG) are written against their
  public APIs but were not exercised against the live services from this build environment; check each source's
  terms and rate limits (`app/similarity/sources.py`). A source that cannot be reached is reported as a warning in the
  report instead of silently counting as "no match".
* **Paraphrase matching** defaults to a lexical near-match (very similar sentences). The multilingual
  embedding method (`SIM_EMBEDDINGS=1`, BAAI bge-m3) is implemented but untested; the `archive_vectors`
  table of the plan is not built (archive search uses fingerprints only).
* **Deviations from the plan:** plain CSS and a small polling script instead of Tailwind/HTMX; a regex sentence
  splitter instead of NLTK; no QR code on the cover (the Report ID and verification URL are in the
  disclaimer; `/verify` also checks a PDF's SHA-256); MinIO/R2 storage is not implemented (local disk only).
* **Not tested here:** the Docker/Compose deployment, the RQ worker with Redis, PostgreSQL, SMTP email and OCR
  (the Tesseract output parser is covered by design, but Tesseract was not installed in the build environment).
  DOCX conversion runs LibreOffice in the worker's container but not in a network-isolated one.
* Rotated pages and PDFs whose pages have a non-zero origin are handled but lightly tested; multi-column PDFs
  rely on content-stream order for sentence order.
* Rate limiting is in-memory (single process). Retention purge keeps archive entries by design (the uploader
  consented); deleting a submission removes its archive entry.
