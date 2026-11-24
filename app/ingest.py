"""Ingestion: normalise an upload to PDF, extract tokens with page coordinates, find structure.

Every upload (PDF, DOCX, TXT) ends up as a PDF so that pages are fixed; the structured
document (`Doc`) stores one entry per token (a Latin word or a single CJK character) with its
page box, so highlights can be drawn exactly where the text sits.
"""
from __future__ import annotations

import bisect
import csv
import io
import re
import shutil
import statistics
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import textproc as tp
from .config import Settings, get_settings

FLAG_QUOTE, FLAG_CITE, FLAG_HEAD, FLAG_OCR = 1, 2, 4, 16

REF_HEADS = {"references", "referencelist", "bibliography", "workscited", "literaturecited", "referencesandnotes", "参考文献", "参考书目"}
HEADING_NUM_RE = re.compile(r"^(\d+(\.\d+)*\.?|[IVX]+\.)\s+\S")
CITE_PATTERNS = [
    re.compile(r"\((?:[^()]{0,100}?)\b(?:1[5-9]|20)\d{2}[a-z]?\b[^()]{0,60}\)"),
    re.compile(r"\[\d+(?:\s*[,–\-]\s*\d+)*\]"),
    re.compile("（[^（）]{0,100}\\d{4}[^（）]{0,40}）"),
    re.compile(r"\[[A-Z][A-Za-z\-]+(?: et al\.?)?,? (?:1[5-9]|20)\d{2}[a-z]?\]"),
]
MAX_SENTENCE_TOKENS = 80


class IngestError(Exception):
    """Raised with a message that is safe to show to the user."""


@dataclass
class Doc:
    pages: list[tuple[float, float]] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)
    norm: list[str] = field(default_factory=list)
    page: list[int] = field(default_factory=list)
    box: list[tuple[float, float, float, float]] = field(default_factory=list)  # x0, y0, x1, y1 in PDF points (origin bottom-left)
    line: list[int] = field(default_factory=list)
    flags: list[int] = field(default_factory=list)
    ends: list[bool] = field(default_factory=list)
    biblio_start: int | None = None
    language: str = "en"
    title: str = ""
    char_count: int = 0
    hidden_chars: int = 0
    replaced_chars: int = 0
    invisible_chars: int = 0
    ocr_pages: list[int] = field(default_factory=list)
    unreadable_pages: list[int] = field(default_factory=list)

    # ---- derived views -------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.norm)

    @property
    def cjk(self) -> bool:
        return self.language == "zh"

    def in_biblio(self, i: int) -> bool:
        return self.biblio_start is not None and i >= self.biblio_start

    def sentences(self) -> list[tuple[int, int]]:
        out, start = [], 0
        for i in range(self.n):
            last = i == self.n - 1
            if self.ends[i] or last or i - start + 1 >= MAX_SENTENCE_TOKENS:
                out.append((start, i + 1))
                start = i + 1
        return out

    def text(self, s: int, e: int) -> str:
        return tp.surface_text(self.raw[s:e])

    def range_boxes(self, s: int, e: int) -> dict[int, list[tuple[float, float, float, float]]]:
        """Highlight rectangles for tokens [s, e): one merged box per text line, keyed by page index."""
        out: dict[int, list[tuple[float, float, float, float]]] = {}
        cur = None  # (page, line, x0, y0, x1, y1)
        for i in range(s, e):
            p, ln = self.page[i], self.line[i]
            x0, y0, x1, y1 = self.box[i]
            if cur and cur[0] == p and cur[1] == ln:
                cur = (p, ln, min(cur[2], x0), min(cur[3], y0), max(cur[4], x1), max(cur[5], y1))
            else:
                if cur:
                    out.setdefault(cur[0], []).append(cur[2:])
                cur = (p, ln, x0, y0, x1, y1)
        if cur:
            out.setdefault(cur[0], []).append(cur[2:])
        return out

    # ---- (de)serialisation ---------------------------------------------------------------
    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["box"] = [list(b) for b in self.box]
        d["pages"] = [list(p) for p in self.pages]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Doc":
        doc = cls(**d)
        doc.box = [tuple(b) for b in doc.box]
        doc.pages = [tuple(p) for p in doc.pages]
        return doc


# --------------------------------------------------------------------------------------------
# File type, conversion to PDF
# --------------------------------------------------------------------------------------------
def sniff_type(path: Path) -> str | None:
    """Detect the file type from its content, never from the extension."""
    with open(path, "rb") as f:
        head = f.read(4096)
    if b"%PDF-" in head[:1024]:
        return "pdf"
    if head.startswith(b"PK"):
        try:
            with zipfile.ZipFile(path) as z:
                if "word/document.xml" in z.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            return None
        return None
    if b"\x00" in head:
        return None
    try:
        head.decode("utf-8")
        return "txt"
    except UnicodeDecodeError:
        try:
            path.read_bytes().decode("gb18030")
            return "txt"
        except UnicodeDecodeError:
            return None


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


_FONT_CACHE: dict[str, str] = {}


def _register_font(cjk: bool) -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    key = "cjk" if cjk else "latin"
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    cands = (
        ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"]
        if cjk
        else ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"]
    )
    for c in cands:
        if Path(c).exists():
            try:
                name = f"TxtFont-{key}"
                pdfmetrics.registerFont(TTFont(name, c, subfontIndex=0) if c.endswith(".ttc") else TTFont(name, c))
                _FONT_CACHE[key] = name
                return name
            except Exception:
                continue
    _FONT_CACHE[key] = "Helvetica"
    return "Helvetica"


def text_to_pdf(text: str, out: Path, page_size: str = "Letter") -> None:
    from xml.sax.saxutils import escape

    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    cjk = tp.cjk_ratio(text[:20000]) > 0.2
    font = _register_font(cjk)
    style = ParagraphStyle("t", fontName=font, fontSize=11, leading=15, wordWrap="CJK" if cjk else None)
    doc = SimpleDocTemplate(str(out), pagesize=A4 if page_size == "A4" else LETTER, leftMargin=64, rightMargin=64, topMargin=72, bottomMargin=72)
    story = []
    for para in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        para = para.strip("\n")
        if para.strip():
            story.append(Paragraph(escape(para).replace("\n", "<br/>"), style))
            story.append(Spacer(1, 6))
    if not story:
        raise IngestError("The file contains no text.")
    doc.build(story)


def docx_to_pdf(src: Path, out_dir: Path, timeout: int = 180) -> Path:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise IngestError("Word conversion is unavailable on this server (LibreOffice is not installed).")
    with tempfile.TemporaryDirectory(prefix="lo_") as profile:
        cmd = [soffice, f"-env:UserInstallation=file://{profile}", "--headless", "--norestore", "--convert-to", "pdf", "--outdir", str(out_dir), str(src)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise IngestError("The Word file could not be converted. Re-save it as PDF and try again.") from exc
    pdf = out_dir / (src.stem + ".pdf")
    if not pdf.exists():
        raise IngestError("The Word file could not be converted.")
    return pdf


def docx_title(path: Path) -> str:
    try:
        import docx

        d = docx.Document(str(path))
        t = (d.core_properties.title or "").strip()
        if t:
            return t
        for p in d.paragraphs:
            if p.text.strip():
                return p.text.strip()[:200]
    except Exception:
        pass
    return ""


# --------------------------------------------------------------------------------------------
# OCR fallback
# --------------------------------------------------------------------------------------------
def parse_tesseract_tsv(tsv: str, dpi: int, page_height_pt: float) -> list[dict]:
    """Convert Tesseract TSV output into word dicts in PDF points (top-origin, like pdfplumber)."""
    k = 72.0 / dpi
    words = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row.get("level") != "5" or not (row.get("text") or "").strip():
            continue
        try:
            if float(row.get("conf", -1)) < 30:
                continue
            x0, top = float(row["left"]) * k, float(row["top"]) * k
            x1, bottom = x0 + float(row["width"]) * k, top + float(row["height"]) * k
        except (KeyError, ValueError):
            continue
        words.append({"text": row["text"].strip(), "x0": x0, "x1": x1, "top": top, "bottom": bottom, "size": bottom - top, "chars": None, "hidden": False})
    return words


def ocr_page(pdf_path: Path, index: int, page_h: float) -> list[dict]:
    tess = shutil.which("tesseract")
    if not tess:
        return []
    import pypdfium2 as pdfium

    dpi = 200
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        img = pdf[index].render(scale=dpi / 72).to_pil()
    finally:
        pdf.close()
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "p.png"
        img.save(png)
        for lang in ("eng+chi_sim", "eng"):
            r = subprocess.run([tess, str(png), "stdout", "-l", lang, "--psm", "3", "tsv"], capture_output=True, text=True, timeout=180)
            if r.returncode == 0:
                return parse_tesseract_tsv(r.stdout, dpi, page_h)
    return []


# --------------------------------------------------------------------------------------------
# PDF -> Doc
# --------------------------------------------------------------------------------------------
def _is_white(col) -> bool:
    if col is None:
        return False
    if isinstance(col, (int, float)):
        return col >= 0.98
    try:
        v = [float(x) for x in col]
    except (TypeError, ValueError):
        return False
    if len(v) in (1, 3):
        return all(x >= 0.98 for x in v)
    if len(v) == 4:  # CMYK white
        return all(x <= 0.02 for x in v)
    return False


def _page_words(page, settings: Settings) -> list[dict]:
    words = []
    raw_words = page.extract_words(x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False, use_text_flow=True,
                                   extra_attrs=["size", "non_stroking_color"], return_chars=True)
    for w in raw_words:
        size = float(w.get("size") or 0)
        hidden = _is_white(w.get("non_stroking_color")) or (0 < size < 2.0)
        chars = [(c["text"], c["x0"], c["x1"], c["top"], c["bottom"]) for c in w.get("chars", [])]
        words.append({"text": w["text"], "x0": w["x0"], "x1": w["x1"], "top": w["top"], "bottom": w["bottom"], "size": size, "chars": chars, "hidden": hidden})
    return words


def _tokenise_word(w: dict, doc_state: dict) -> list[dict]:
    """Split one PDF word into tokens, with boxes, quote/sentence info. Returns token dicts."""
    text = w["text"]
    chars = w["chars"]
    # Drop zero-width characters (kept out of the text so they cannot split or disguise a word).
    if chars is not None and len(chars) == len(text):
        keep = [i for i, ch in enumerate(text) if ch not in tp.ZERO_WIDTH_CHARS]
        doc_state["invisible"] += len(text) - len(keep)
        text = "".join(text[i] for i in keep)
        chars = [chars[i] for i in keep]
    else:
        stripped = tp.strip_invisible(text)
        doc_state["invisible"] += len(text) - len(stripped)
        text, chars = stripped, None
    ms = list(tp.TOKEN_RE.finditer(text))
    out = []
    n = len(text) or 1
    for idx, m in enumerate(ms):
        a, b = m.start(), m.end()
        if chars is not None and len(chars) == len(text):
            seg = chars[a:b]
            x0, x1 = min(c[1] for c in seg), max(c[2] for c in seg)
            top, bottom = min(c[3] for c in seg), max(c[4] for c in seg)
        else:
            span = w["x1"] - w["x0"]
            x0, x1 = w["x0"] + span * a / n, w["x0"] + span * b / n
            top, bottom = w["top"], w["bottom"]
        pre = text[(ms[idx - 1].end() if idx else 0):a]
        post = text[b:(ms[idx + 1].start() if idx + 1 < len(ms) else len(text))]
        raw, replaced = tp.fix_homoglyphs(m.group(0))
        doc_state["replaced"] += replaced
        end = False
        if idx == len(ms) - 1 and tp.is_end_punct_word(text):
            end = True
        if any(c in "。！？；" for c in post):
            end = True
        out.append({"raw": raw, "x0": x0, "x1": x1, "top": top, "bottom": bottom, "pre": pre, "post": post, "end": end, "off": a, "len": b - a})
    return out


def extract_pdf(path: Path, settings: Settings | None = None, fallback_title: str = "") -> Doc:
    import pdfplumber

    settings = settings or get_settings()
    doc = Doc()
    pages_words: list[list[dict]] = []
    try:
        pdf = pdfplumber.open(str(path))
    except Exception as exc:
        raise IngestError("The PDF could not be read. It may be damaged or password-protected.") from exc
    with pdf:
        if len(pdf.pages) > settings.max_pages:
            raise IngestError(f"The document has {len(pdf.pages)} pages; the limit is {settings.max_pages}.")
        meta_title = ((pdf.metadata or {}).get("Title") or "").strip()
        for pno, page in enumerate(pdf.pages):
            doc.pages.append((float(page.width), float(page.height)))
            try:
                words = _page_words(page, settings)
            except Exception:
                words = []
            if len(words) < 15:  # probably a scan: try OCR
                ocr = ocr_page(path, pno, float(page.height))
                if len(ocr) > len(words):
                    words = ocr
                    doc.ocr_pages.append(pno)
                elif not words and page.images:
                    doc.unreadable_pages.append(pno)
            pages_words.append(words)

    # pass 1: line ids, hidden text
    flat: list[tuple[int, int, dict]] = []  # (page, line, word)
    line_id = -1
    for pno, words in enumerate(pages_words):
        prev_top = None
        for w in words:
            if w["hidden"]:
                doc.hidden_chars += len(w["text"])
                continue
            if prev_top is None or abs(w["top"] - prev_top) > max(2.0, 0.5 * (w["size"] or 10)):
                line_id += 1
            prev_top = w["top"]
            flat.append((pno, line_id, w))

    # pass 2: tokens
    state = {"invisible": 0, "replaced": 0}
    in_quote, run_start = False, 0
    stream_parts: list[str] = []
    stream_len = 0
    tok_off: list[int] = []
    tok_end: list[int] = []
    line_sizes: dict[int, list[float]] = {}
    for pno, ln, w in flat:
        toks = _tokenise_word(w, state)
        clean = tp.strip_invisible(w["text"])  # same text the tokeniser saw
        base = stream_len
        for t in toks:
            if any(c in tp.OPEN_QUOTES for c in t["pre"]):
                in_quote = True
            elif tp.STRAIGHT_QUOTE in t["pre"]:
                in_quote = not in_quote
            norm = tp.normalize_token(t["raw"])
            if not norm:
                continue
            quoted = in_quote
            i = len(doc.norm)
            if quoted and not (i and doc.flags[i - 1] & FLAG_QUOTE):
                run_start = i
            if any(c in tp.CLOSE_QUOTES for c in t["post"]):
                in_quote = False
            elif tp.STRAIGHT_QUOTE in t["post"]:
                in_quote = not in_quote
            ph = doc.pages[pno][1]
            doc.raw.append(t["raw"])
            doc.norm.append(norm)
            doc.page.append(pno)
            doc.box.append((t["x0"], ph - t["bottom"], t["x1"], ph - t["top"]))
            doc.line.append(ln)
            fl = FLAG_QUOTE if quoted else 0
            if pno in doc.ocr_pages:
                fl |= FLAG_OCR
            doc.flags.append(fl)
            doc.ends.append(t["end"])
            tok_off.append(base + t["off"])
            tok_end.append(base + t["off"] + t["len"])
            line_sizes.setdefault(ln, []).append(w["size"] or 0)
            if quoted and i - run_start > 400:  # runaway quote: treat as an unmatched mark
                for j in range(run_start, i + 1):
                    doc.flags[j] &= ~FLAG_QUOTE
                in_quote = False
        stream_parts.append(clean + " ")
        stream_len += len(clean) + 1
    doc.invisible_chars = state["invisible"]
    doc.replaced_chars = state["replaced"]
    if doc.n == 0:
        raise IngestError("No readable text was found in the document.")
    if doc.n > settings.max_words:
        raise IngestError(f"The document has about {doc.n:,} words; the limit is {settings.max_words:,}.")
    doc.char_count = sum(len(r) for r in doc.raw)
    doc.language = tp.detect_language(tp.surface_text(doc.raw[:6000]))

    # citations
    stream = "".join(stream_parts)
    for pat in CITE_PATTERNS:
        for m in pat.finditer(stream):
            lo = bisect.bisect_left(tok_off, m.start())
            hi = bisect.bisect_left(tok_off, m.end())
            for j in range(lo, min(hi, doc.n)):
                doc.flags[j] |= FLAG_CITE

    _mark_structure(doc, line_sizes, meta_title, fallback_title)
    return doc


def _mark_structure(doc: Doc, line_sizes: dict[int, list[float]], meta_title: str, fallback_title: str) -> None:
    """Headings, reference list start, sentence breaks around headings, document title."""
    lines: dict[int, list[int]] = {}
    for i, ln in enumerate(doc.line):
        lines.setdefault(ln, []).append(i)
    all_sizes = [s for v in line_sizes.values() for s in v if s > 0]
    body = statistics.median(all_sizes) if all_sizes else 10.0
    ref_candidates, heading_lines = [], []
    for ln, idxs in lines.items():
        n = len(idxs)
        sizes = [s for s in line_sizes.get(ln, []) if s > 0]
        size = statistics.mean(sizes) if sizes else body
        text = tp.surface_text([doc.raw[i] for i in idxs])
        key = re.sub(r"[\W\d_]+", "", text.lower())
        if n <= 6 and key in REF_HEADS:
            ref_candidates.append(idxs[0])
        is_head = False
        if n <= 18 and size >= body * 1.15:
            is_head = True
        elif n <= 10 and HEADING_NUM_RE.match(text) and not text.rstrip().endswith((".", ",", ";", ":")) and size >= body * 0.98 and not doc.ends[idxs[-1]]:
            is_head = True
        elif n <= 4 and key in REF_HEADS:
            is_head = True
        if is_head:
            heading_lines.append((ln, idxs, size))
    for ln, idxs, _ in heading_lines:
        for i in idxs:
            doc.flags[i] |= FLAG_HEAD
        doc.ends[idxs[-1]] = True
        if idxs[0] > 0:
            doc.ends[idxs[0] - 1] = True
    if ref_candidates:
        # Ignore a "References" line in the first third (a table of contents entry).
        late = [c for c in ref_candidates if c >= doc.n * 0.3]
        doc.biblio_start = late[-1] if late else None
    title = meta_title
    if not title or re.match(r"(?i)^(microsoft word|untitled|document\d*|\(anonymous\))", title):
        first_page_heads = [(sz, idxs) for _, idxs, sz in heading_lines if doc.page[idxs[0]] == 0]
        if first_page_heads:
            title = tp.surface_text([doc.raw[i] for i in max(first_page_heads, key=lambda x: x[0])[1]])
        else:
            title = fallback_title
    doc.title = (title or fallback_title or "Untitled document").strip()[:200]


# --------------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------------
def ingest_file(src: Path, workdir: Path, original_name: str, settings: Settings | None = None) -> tuple[Doc, Path, str]:
    """Convert `src` to a PDF inside `workdir` and extract the structured document.

    Returns (doc, pdf_path, kind) where kind is 'pdf' | 'docx' | 'txt'.
    """
    settings = settings or get_settings()
    kind = sniff_type(src)
    if kind is None:
        raise IngestError("Unsupported file. Please upload a PDF, DOCX or TXT file.")
    workdir.mkdir(parents=True, exist_ok=True)
    stem = Path(original_name).stem or "document"
    title = ""
    if kind == "pdf":
        pdf = workdir / "original.pdf"
        shutil.copyfile(src, pdf)
    elif kind == "docx":
        title = docx_title(src)
        tmp_in = workdir / "upload.docx"
        shutil.copyfile(src, tmp_in)
        conv = docx_to_pdf(tmp_in, workdir)
        pdf = workdir / "original.pdf"
        conv.replace(pdf)
        tmp_in.unlink(missing_ok=True)
    else:
        text = _decode_text(src.read_bytes())
        approx_words = len(tp.TOKEN_RE.findall(text))
        if approx_words > settings.max_words:
            raise IngestError(f"The document has about {approx_words:,} words; the limit is {settings.max_words:,}.")
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        title = first[:120] if len(first) < 120 else ""
        pdf = workdir / "original.pdf"
        from .config import load_brand

        text_to_pdf(text, pdf, load_brand(settings)["page_size"])
    doc = extract_pdf(pdf, settings, fallback_title=title or stem)
    return doc, pdf, kind
