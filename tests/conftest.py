import os
import sys
import tempfile
from pathlib import Path

import pytest

_tmp = tempfile.mkdtemp(prefix="simcheck_test_")
os.environ.setdefault("DATA_DIR", _tmp)
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["SOURCES"] = ""          # tests never touch the network
os.environ["ADMIN_EMAIL"] = "admin@example.com"
os.environ["ADMIN_PASSWORD"] = "admin-pass-123"
os.environ["BRAND_FILE"] = str(Path(_tmp) / "brand.yaml")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ORIGINAL = (
    "The Industrial Revolution transformed the economies of Western Europe during the late eighteenth century. "
    "Mechanised textile production moved workers from farms into rapidly growing factory towns, and the demand for coal "
    "and iron reshaped the landscape of northern England. Historians continue to debate whether living standards rose "
    "or fell for the first generation of factory workers, because wages grew slowly while urban crowding spread disease."
)
OWN_WORDS = (
    "In my own view the quiet harbour town where I grew up taught me more about patience than any classroom ever did. "
    "Every morning my grandmother walked to the bakery before sunrise, and I would follow her through the cold fog "
    "carrying a wicker basket that smelled of rye and woodsmoke."
)


@pytest.fixture()
def tmp(tmp_path):
    return tmp_path


def make_pdf(path: Path, paragraphs: list[str], heading_first: bool = False, font="Helvetica", white_text: str | None = None, tiny_text: str | None = None):
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if font != "Helvetica":
        pdfmetrics.registerFont(TTFont("Test-CJK", font, subfontIndex=0))
        font = "Test-CJK"
    c = canvas.Canvas(str(path), pagesize=LETTER)
    y = 720
    wrap = 88 if font == "Helvetica" else 36
    for k, para in enumerate(paragraphs):
        size = 18 if (heading_first and k == 0) else 11
        c.setFont(font, size)
        if font == "Helvetica":
            import textwrap
            lines = textwrap.wrap(para, wrap)
        else:
            lines = [para[i:i + wrap] for i in range(0, len(para), wrap)]
        for ln in lines:
            if y < 72:
                c.showPage(); c.setFont(font, size); y = 720
            c.drawString(72, y, ln)
            y -= size + 4
        y -= 8
    if white_text:
        c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica", 11); c.drawString(72, 100, white_text)
    if tiny_text:
        c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica", 1); c.drawString(72, 90, tiny_text)
    c.save()
    return path
