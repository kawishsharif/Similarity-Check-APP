"""Tokenisation, normalisation, language detection and sentence splitting."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
# A token is one CJK character, or a run of letters/digits (with inner apostrophes/hyphens).
TOKEN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]|[^\W_]+(?:['’\-][^\W_]+)*")
OPEN_QUOTES = "“‘「『«"
CLOSE_QUOTES = "”’」』»"
STRAIGHT_QUOTE = '"'
END_PUNCT = ".!?。！？；;"
ABBREVIATIONS = {"e.g", "i.e", "al", "fig", "figs", "dr", "mr", "mrs", "ms", "vs", "etc", "no", "pp", "vol", "eq", "cf", "st", "prof", "inc", "ltd", "approx"}

# Latin look-alikes from other alphabets (Cyrillic, Greek) used to defeat matching.
HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h", "ԛ": "q", "ԝ": "w",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    "ο": "o", "ν": "v", "ρ": "p", "α": "a", "ε": "e", "ι": "i", "κ": "k", "τ": "t", "υ": "u", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\u2060\ufeff\u00ad"
ZERO_WIDTH = dict.fromkeys(map(ord, ZERO_WIDTH_CHARS), None)
LATIN_RE = re.compile(r"[A-Za-z]")

STOPWORDS = set(
    "a an the and or but if of to in on at by for with from as is are was were be been being it its this that these those i you he she we they "
    "them his her our their not no so than then there here have has had do does did will would can could may might must shall should which who whom "
    "whose what when where why how also such more most other some any each both into over under about between through during before after above "
    "below up down out off again further once only own same too very just".split()
)


@lru_cache(maxsize=1)
def _t2s():
    try:
        from opencc import OpenCC

        return OpenCC("t2s").convert
    except Exception:  # optional dependency
        return None


def strip_invisible(text: str) -> str:
    return text.translate(ZERO_WIDTH)


def fix_homoglyphs(raw: str) -> tuple[str, int]:
    """Map look-alike letters back to Latin, only inside words that also contain real Latin letters
    (so genuine Cyrillic or Greek words are left alone). Returns (fixed word, characters replaced)."""
    n = sum(1 for ch in raw if ch in HOMOGLYPHS)
    if n == 0 or not LATIN_RE.search(raw):
        return raw, 0
    return "".join(HOMOGLYPHS.get(ch, ch) for ch in raw), n


def normalize_token(raw: str) -> str:
    t = unicodedata.normalize("NFKC", strip_invisible(raw)).lower()
    t = "".join(ch for ch in t if ch.isalnum() or ch in "'-")
    t = t.strip("'-")
    if CJK_RE.search(t):
        conv = _t2s()
        if conv:
            t = conv(t)
    return t


def cjk_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if CJK_RE.match(c)) / len(letters)


@lru_cache(maxsize=1)
def _lingua():
    try:
        from lingua import Language, LanguageDetectorBuilder

        langs = [Language.ENGLISH, Language.CHINESE, Language.SPANISH, Language.FRENCH, Language.GERMAN, Language.PORTUGUESE,
                 Language.ITALIAN, Language.RUSSIAN, Language.ARABIC, Language.JAPANESE, Language.KOREAN, Language.DUTCH]
        return LanguageDetectorBuilder.from_languages(*langs).with_low_accuracy_mode().build()
    except Exception:
        return None


def detect_language(text: str) -> str:
    """Return an ISO 639-1 code; 'zh' for Chinese-dominant text, 'en' as a safe default."""
    sample = text[:20000]
    if cjk_ratio(sample) > 0.3:
        return "zh"
    det = _lingua()
    if det is not None and len(sample.strip()) > 20:
        lang = det.detect_language_of(sample)
        if lang is not None:
            return lang.iso_code_639_1.name.lower()
    return "en"


def is_end_punct_word(word: str) -> bool:
    """True if a PDF 'word' ends a sentence (ignores common abbreviations)."""
    w = word.rstrip("\"'”’)」』]")
    if not w or w[-1] not in END_PUNCT:
        return False
    if w[-1] == ".":
        stem = w.rstrip(".").lower()
        if stem in ABBREVIATIONS or (len(stem) == 1 and stem.isalpha()):
            return False
    return True


def keyword_weight(norm_tokens: list[str]) -> float:
    """Rare-word weight: long, non-stopword tokens count more. Used to choose distinctive queries."""
    score = 0.0
    for t in norm_tokens:
        if t in STOPWORDS:
            continue
        if CJK_RE.match(t):
            score += 0.6
        else:
            score += min(len(t), 12) / 6.0
    return score


def surface_text(raws: list[str]) -> str:
    """Join raw tokens for display/search: no spaces between CJK characters."""
    out: list[str] = []
    prev_cjk = False
    for r in raws:
        cur = bool(CJK_RE.match(r))
        if out and not (cur and prev_cjk):
            out.append(" ")
        out.append(r)
        prev_cjk = cur
    return "".join(out)


def plain_to_tokens(text: str) -> tuple[list[str], list[str]]:
    """Tokenise plain text (fetched web pages etc.): returns (raw tokens, normalised tokens), blanks dropped."""
    raws, norms = [], []
    for m in TOKEN_RE.finditer(strip_invisible(text)):
        raw, _ = fix_homoglyphs(m.group(0))
        n = normalize_token(raw)
        if n:
            raws.append(raw)
            norms.append(n)
    return raws, norms


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter used for candidate texts (no external data download needed)."""
    parts: list[str] = []
    buf: list[str] = []
    words = re.split(r"(\s+)", text)
    for w in words:
        buf.append(w)
        if w.strip() and is_end_punct_word(w.strip()):
            parts.append("".join(buf).strip())
            buf = []
        elif re.search(r"[。！？]", w):
            pieces = re.split(r"(?<=[。！？])", "".join(buf))
            parts.extend(p.strip() for p in pieces[:-1] if p.strip())
            buf = [pieces[-1]]
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]
