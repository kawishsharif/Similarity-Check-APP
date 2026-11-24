"""AI-writing engine: qualifying prose, windowed scoring, two-signal agreement, figures."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from .. import textproc as tp
from ..config import Settings
from ..ingest import FLAG_HEAD, Doc
from .detectors import get_detector

WINDOW = 5
MIN_QUALIFYING_WORDS = 300
MIN_SENTENCE_TOKENS = 6
DEFAULT_THRESHOLDS = {
    "heuristic": {"stat": 0.60, "clf": 0.30, "clf_high": 0.50},
    "binoculars": {"stat": 0.50, "clf": 0.50, "clf_high": 0.80},
}
GROUP_META = {
    "ai": ("Likely AI-generated", "Text our models estimate was written by an AI tool."),
    "para": ("Likely AI-generated, then paraphrased", "Text that looks AI-written and was probably reworded afterwards."),
}


def load_thresholds(settings: Settings, detector: str) -> dict:
    """Thresholds come from data/ai_thresholds.json (written by scripts/calibrate.py) when present."""
    t = dict(DEFAULT_THRESHOLDS.get(detector, DEFAULT_THRESHOLDS["heuristic"]))
    f = settings.data_dir / "ai_thresholds.json"
    if f.exists():
        try:
            t.update(json.loads(f.read_text()).get(detector, {}))
        except (ValueError, OSError):
            pass
    return t


def display_value(percent: float | None) -> str:
    """Below 20% the figure is too unreliable to print, so it is shown as an asterisk."""
    if percent is None:
        return "--"
    if percent <= 0:
        return "0%"
    if percent < 20:
        return "*%"
    return f"{round(percent)}%"


@dataclass
class Sentence:
    s: int
    e: int
    text: str
    stat: float = 0.0
    clf: float = 0.0
    group: str | None = None


def qualifying_sentences(doc: Doc) -> list[Sentence]:
    """Full sentences in body prose: no headings, reference list, short fragments or number tables."""
    out = []
    for s, e in doc.sentences():
        if doc.in_biblio(s) or (doc.flags[s] & FLAG_HEAD) or e - s < MIN_SENTENCE_TOKENS:
            continue
        alpha = sum(1 for x in range(s, e) if any(ch.isalpha() for ch in doc.norm[x]))
        if alpha / (e - s) < 0.8:
            continue
        out.append(Sentence(s, e, doc.text(s, e)))
    return out


def run_ai_detection(doc: Doc, settings: Settings, progress: Callable[[float, str], None] | None = None, detector=None) -> dict:
    progress = progress or (lambda f, m: None)
    det = detector or get_detector(settings.ai_detector)
    th = load_thresholds(settings, det.name)
    sents = qualifying_sentences(doc)
    qualifying_words = sum(x.e - x.s for x in sents)
    base = {"detector": det.name, "qualifying_words": qualifying_words, "thresholds": th, "min_words": MIN_QUALIFYING_WORDS}
    if qualifying_words < MIN_QUALIFYING_WORDS:
        return {**base, "scored": False, "percent": None, "display": display_value(None), "flagged_words": 0, "spans": [],
                "groups": _groups({}, qualifying_words), "message": f"At least {MIN_QUALIFYING_WORDS} words of prose are needed; this document has {qualifying_words}."}

    # Score overlapping windows, then average the window scores back onto each sentence.
    n = len(sents)
    w = min(WINDOW, n)
    acc = [[0.0, 0.0, 0] for _ in range(n)]
    nwin = max(1, n - w + 1)
    for a in range(nwin):
        stat, clf = det.score([x.text for x in sents[a:a + w]], doc.language)
        for i in range(a, a + w):
            acc[i][0] += stat
            acc[i][1] += clf
            acc[i][2] += 1
        if a % 10 == 0:
            progress(a / nwin, f"Scoring text ({a}/{nwin})")
    for x, (st, cl, c) in zip(sents, acc):
        x.stat, x.clf = st / c, cl / c
        if x.stat >= th["stat"] and x.clf >= th["clf"]:
            x.group = "ai"
        elif x.clf >= th["clf_high"] and x.stat < th["stat"]:
            x.group = "para"

    # Isolated single flagged sentences are dropped.
    for i, x in enumerate(sents):
        if x.group and not ((i > 0 and sents[i - 1].group) or (i + 1 < n and sents[i + 1].group)):
            x.group = None

    words_by: dict[str, int] = {}
    counts: dict[str, int] = {}
    spans = []
    for x in sents:
        if x.group:
            words_by[x.group] = words_by.get(x.group, 0) + (x.e - x.s)
            spans.append({"s0": x.s, "s1": x.e, "group": x.group, "score": round((x.stat + x.clf) / 2, 3)})
    # count flagged runs per group (a run = consecutive sentences with the same group)
    prev = None
    for x in sents:
        if x.group and x.group != prev:
            counts[x.group] = counts.get(x.group, 0) + 1
        prev = x.group
    flagged = sum(words_by.values())
    percent = round(100.0 * flagged / qualifying_words, 2)
    return {**base, "scored": True, "percent": percent, "display": display_value(percent), "flagged_words": flagged, "spans": spans,
            "groups": _groups(words_by, qualifying_words, counts), "message": ""}


def _groups(words_by: dict[str, int], qualifying: int, counts: dict[str, int] | None = None) -> list[dict]:
    counts = counts or {}
    return [{"key": k, "name": GROUP_META[k][0], "note": GROUP_META[k][1], "count": counts.get(k, 0),
             "percent": round(100.0 * words_by.get(k, 0) / qualifying, 2) if qualifying else 0.0} for k in ("ai", "para")]
