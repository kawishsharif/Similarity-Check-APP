"""Branded PDF reports.

Layout (measured from the reference report the product follows):
  * every page carries the same header (logo, "Page X of N - label", Submission ID) and the same
    footer, stamped in the margins;
  * the cover and overview pages are designed pages (HTML/CSS -> WeasyPrint, absolute positions);
  * the document pages are the uploaded pages at their own size, highlights drawn over them.
All brand details (logo, name, colours, ID prefix, page size) come from brand.yaml.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from ..ingest import Doc
from ..similarity.engine import TYPE_LABELS

HERE = Path(__file__).parent
TEMPLATES = HERE / "templates"
FONTS = HERE / "fonts"
ENV = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]))
PAGE_SIZES = {"Letter": (612.0, 792.0), "A4": (595.28, 841.89)}
MARGIN = 36.0                  # left/right margin of header and footer
LOGO_H, LOGO_MAX_W = 16.5, 80.0
HEADER_BAND = 48.0             # original-page text inside this band triggers fit-to-frame
SOURCES_PER_PAGE = 14
HIGHLIGHT_ALPHA = 0.40
TEXT_GREY = HexColor("#191919")


def _pct(v) -> str:
    """Whole numbers; anything above zero but under one shows as '<1'."""
    if v is None:
        return "-"
    if 0 < v < 1:
        return "<1"
    return str(int(round(v)))


ENV.filters["pct"] = _pct
_font_ready = False


def _font() -> str:
    global _font_ready
    if not _font_ready:
        try:
            pdfmetrics.registerFont(TTFont("NS-Regular", str(FONTS / "NotoSans-Regular.ttf")))
        except Exception:
            return "Helvetica"
        _font_ready = True
    return "NS-Regular"


def integrity_flags(doc: Doc) -> list[str]:
    flags = []
    if doc.hidden_chars:
        flags.append(f"Hidden text: {doc.hidden_chars:,} characters in white or unreadably small type were found and ignored.")
    if doc.replaced_chars:
        flags.append(f"Replaced characters: {doc.replaced_chars:,} look-alike letters from another alphabet were mapped back before comparison.")
    if doc.invisible_chars:
        flags.append(f"Invisible characters: {doc.invisible_chars:,} zero-width characters were found inside words and ignored.")
    if doc.ocr_pages:
        flags.append("Text stored as images: page(s) " + ", ".join(str(p + 1) for p in doc.ocr_pages) + " were read with OCR and may contain recognition errors.")
    if doc.unreadable_pages:
        flags.append("Page(s) " + ", ".join(str(p + 1) for p in doc.unreadable_pages) + " contain only images and could not be read.")
    return flags


def front_pages(template: str, size: tuple[float, float], **ctx) -> list:
    from weasyprint import HTML

    html = ENV.get_template(template).render(PW=size[0], PH=size[1], css=(TEMPLATES / "report.css").read_text(), **ctx)
    return list(PdfReader(io.BytesIO(HTML(string=html, base_url=str(TEMPLATES)).write_pdf())).pages)


def _logo_box(path: str) -> tuple[float, float]:
    iw, ih = ImageReader(path).getSize()
    w = min(LOGO_H * iw / ih, LOGO_MAX_W)
    return w, w * ih / iw


def _overlay(W, H, page_no, total, label, sub_id, brand, spans=(), dx=0.0, dy=0.0, scale=1.0) -> PageObject:
    """Highlights (optional) plus the branded header and footer, on a transparent page."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(W, H))
    for sp in spans:
        colour = HexColor(sp["color"])
        c.setFillColor(colour, alpha=sp.get("alpha", HIGHLIGHT_ALPHA))
        for x0, y0, x1, y1 in sp["boxes"]:
            c.rect(dx + x0 * scale, dy + y0 * scale, (x1 - x0) * scale, (y1 - y0) * scale, stroke=0, fill=1)
        if sp.get("number"):  # numbered badge in the left margin, level with the first line of the match
            top = dy + max(b[3] for b in sp["boxes"]) * scale - 6
            bx = max(dx - 12, 13)
            c.setFillColor(colour, alpha=1)
            c.circle(bx, top, 7, stroke=0, fill=1)
            c.setFillColor(HexColor("#FFFFFF"))
            c.setFont("Helvetica-Bold", 7 if sp["number"] < 100 else 5.5)
            c.drawCentredString(bx, top - 2.5, str(sp["number"]))
    c.setFillAlpha(1)
    lw, lh = _logo_box(brand["logo_path"])
    font = _font()
    line = f"Page {page_no} of {total} - {label}"
    ident = f"Submission ID   {sub_id}"
    for logo_bottom, baseline in ((H - 24 - lh, H - 36.2), (27.5, 31.8)):
        c.drawImage(brand["logo_path"], MARGIN, logo_bottom, width=lw, height=lh, mask="auto")
        c.setFillColor(TEXT_GREY)
        c.setFont(font, 6)
        c.drawString(MARGIN + lw + 16, baseline, line)
        c.drawRightString(W - MARGIN, baseline, ident)
    c.save()
    return PdfReader(io.BytesIO(buf.getvalue())).pages[0]


def _stamp_front(page, page_no, total, label, sub_id, brand) -> PageObject:
    W, H = float(page.mediabox.width), float(page.mediabox.height)
    page.merge_page(_overlay(W, H, page_no, total, label, sub_id, brand))
    return page


def _document_page(page, spans, intrudes, page_no, total, label, sub_id, brand, size):
    """The uploaded page at its own size with highlights and the branded header/footer on top.
    If its text would run into the header or footer band, it is reduced to fit between them."""
    if page.get("/Rotate"):
        page.transfer_rotation_to_content()
    left, bottom = float(page.mediabox.left), float(page.mediabox.bottom)
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    if not intrudes:
        out = PageObject.create_blank_page(width=w, height=h)
        out.merge_transformed_page(page, Transformation().translate(-left, -bottom))
        out.merge_page(_overlay(w, h, page_no, total, label, sub_id, brand, spans))
        return out
    PW, PH = size
    top, bot, side = 54.0, 54.0, 20.0
    scale = min((PW - 2 * side) / w, (PH - top - bot) / h, 1.0)
    dx, dy = (PW - w * scale) / 2, bot + (PH - top - bot - h * scale) / 2
    out = PageObject.create_blank_page(width=PW, height=PH)
    out.merge_transformed_page(page, Transformation().translate(-left, -bottom).scale(scale).translate(dx, dy))
    out.merge_page(_overlay(PW, PH, page_no, total, label, sub_id, brand, spans, dx, dy, scale))
    return out


def _page_intrusions(doc: Doc, pages: list[tuple[float, float]]) -> list[bool]:
    flags = [False] * len(pages)
    for i in range(doc.n):
        p = doc.page[i]
        H = pages[p][1] if p < len(pages) else 792
        _, y0, _, y1 = doc.box[i]
        if y1 > H - HEADER_BAND or y0 < HEADER_BAND:
            flags[p] = True
    return flags


def _spans_from_ranges(doc: Doc, ranges) -> dict[int, list[dict]]:
    """ranges: iterable of (s0, s1, colour, number|None, alpha). Returns page index -> spans."""
    by_page: dict[int, list[dict]] = {}
    for s0, s1, colour, number, alpha in ranges:
        for pg, boxes in doc.range_boxes(s0, s1).items():
            by_page.setdefault(pg, []).append({"color": colour, "number": number, "alpha": alpha, "boxes": boxes})
    return by_page


def similarity_spans(doc: Doc, sim: dict) -> dict[int, list[dict]]:
    colour = {s["rank"]: s["color"] for s in sim["sources"]}
    return _spans_from_ranges(doc, [(m["s0"], m["s1"], colour[m["rank"]], m["rank"], 0.20 if m["kind"] == "paraphrase" else HIGHLIGHT_ALPHA) for m in sim["matches"]])


def ai_spans(doc: Doc, ai: dict, brand: dict) -> dict[int, list[dict]]:
    colours = {"ai": brand["ai_color"], "para": brand["ai_color_paraphrased"]}
    return _spans_from_ranges(doc, [(sp["s0"], sp["s1"], colours[sp["group"]], None, HIGHLIGHT_ALPHA) for sp in ai["spans"]])


def build_report(kind: str, doc: Doc, result: dict, original_pdf: Path, out_path: Path, sub: dict, report_id: str, brand: dict) -> str:
    """Build one report PDF. kind is 'similarity' or 'ai'. Returns the file's SHA-256."""
    size = PAGE_SIZES[brand["page_size"]]
    verify_link = f"{brand['verify_url']}{report_id}"
    ctx = dict(brand=brand, sub=sub, report_id=report_id, verify_link=verify_link, type_labels=TYPE_LABELS)
    if kind == "similarity":
        template, kind_label, doc_label, overview = "similarity.html", "Similarity Report", "Similarity Submission", "Similarity Overview"
        srcs = result["sources"]
        pages = [srcs[i:i + SOURCES_PER_PAGE] for i in range(0, len(srcs), SOURCES_PER_PAGE)] or [[]]
        ctx.update(sim=result, flags=integrity_flags(doc), source_pages=pages)
        front_labels = ["Cover Page", overview] + ["Sources"] * len(pages)
        spans = similarity_spans(doc, result)
    else:
        template, kind_label, doc_label, overview = "ai.html", "AI Writing Report", "AI Writing Submission", "AI Writing Overview"
        colours = {"ai": brand["ai_color"], "para": brand["ai_color_paraphrased"]}
        ai = dict(result)
        ai["groups"] = [{**g, "color": colours[g["key"]]} for g in result["groups"]]
        ctx.update(ai=ai)
        front_labels = ["Cover Page", overview]
        spans = ai_spans(doc, result, brand)
    front = front_pages(template, size, **ctx)
    original = PdfReader(str(original_pdf)).pages
    total = len(front) + len(original)
    intrusions = _page_intrusions(doc, list(doc.pages))
    writer = PdfWriter()
    for i, page in enumerate(front):
        label = front_labels[i] if i < len(front_labels) else front_labels[-1]
        writer.add_page(_stamp_front(page, i + 1, total, label, sub["id"], brand))
    for i, page in enumerate(original):
        writer.add_page(_document_page(page, spans.get(i, []), intrusions[i] if i < len(intrusions) else False, len(front) + i + 1, total, doc_label, sub["id"], brand, size))
    writer.add_metadata({"/Title": f"{kind_label} - {sub['title']}"[:250], "/Producer": brand["brand_name"], "/Author": brand["brand_name"]})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        writer.write(f)
    return hashlib.sha256(out_path.read_bytes()).hexdigest()
