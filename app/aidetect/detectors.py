"""AI-writing signals. Each detector scores a window of consecutive sentences with two
independent values in [0, 1]:

  stat - "how machine-like is the statistical profile" (predictability / uniformity)
  clf  - "how machine-like does a classifier / marker model think the wording is"

A sentence is flagged only when both signals pass thresholds calibrated on test data.

Two detectors are provided:

* ``HeuristicDetector`` (default) needs no model download and no GPU. It is a lightweight
  baseline built from stylometric features. It is NOT a validated detector: calibrate it with
  ``scripts/calibrate.py`` on your own human/AI samples before trusting any figure.
* ``ModelDetector`` implements the plan's method (Binoculars-style cross-perplexity with two
  small causal language models, plus an optional fine-tuned classifier) using PyTorch and
  Hugging Face transformers. It needs ``requirements-ai.txt`` and model downloads, and it has not
  been run in the development environment of this repository.
"""
from __future__ import annotations

import math
import re
import statistics
from functools import lru_cache

from ..config import Settings

Window = list[str]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


EN_MARKERS = [
    "moreover", "furthermore", "additionally", "in conclusion", "in summary", "to summarize", "overall,", "it is important to note",
    "it is worth noting", "it should be noted", "delve", "tapestry", "multifaceted", "pivotal", "crucial", "underscore", "underscores",
    "highlights the importance", "plays a crucial role", "plays a vital role", "a testament to", "in today's", "ever-evolving",
    "comprehensive", "holistic", "leverage", "foster", "navigate", "realm", "myriad", "seamless", "robust", "intricate", "nuanced",
    "significant implications", "not only", "this essay will", "in the realm of", "shed light", "paramount", "invaluable", "dynamic landscape",
    "rapidly evolving", "stakeholders", "synergy", "ethical considerations", "ultimately", "plethora", "endeavor", "whilst",
]
ZH_MARKERS = [
    "综上所述", "总而言之", "总之", "此外", "首先", "其次", "再次", "最后", "值得注意的是", "不仅", "而且", "至关重要", "密不可分", "深远的影响",
    "随着", "与此同时", "在当今", "不可忽视", "具有重要意义", "起着重要作用", "全面", "有效地", "进一步", "显著", "促进", "推动",
]
EN_MARKER_RE = re.compile("|".join(re.escape(m) if m.endswith(",") else r"\b" + re.escape(m) + r"\w*" for m in EN_MARKERS), re.I)
ZH_MARKER_RE = re.compile("|".join(map(re.escape, ZH_MARKERS)))
WORD_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
INFORMAL_RE = re.compile(r"\b\w+n't\b|\b(?:i'm|i've|i'd|i'll|we're|they're|it's|that's|can't|won't|gonna|kinda|gotta)\b|(?-i:\bI\b)|\bmy\b|\bme\b|\bmine\b", re.I)
ZH_INFORMAL_RE = re.compile(r"我|咱|俺|哈|吧|呢|啊|嘛|呀")


class HeuristicDetector:
    name = "heuristic"

    def score(self, window: Window, lang: str) -> tuple[float, float]:
        text = " ".join(window)
        if lang == "zh":
            return self._zh(window, text)
        words = WORD_RE.findall(text)
        nw = max(len(words), 1)
        lens = [len(WORD_RE.findall(s)) for s in window]
        mean = statistics.mean(lens) if lens else 0
        cv = statistics.pstdev(lens) / mean if mean else 1.0
        uniform = clamp(1.0 - cv / 0.55)
        long_ratio = sum(1 for w in words if len(w) >= 9) / nw
        abstract = clamp((long_ratio - 0.05) / 0.10)
        formal = 0.0 if INFORMAL_RE.search(text) else 1.0
        stat = clamp(0.45 * uniform + 0.35 * abstract + 0.20 * formal)
        density = 100.0 * len(EN_MARKER_RE.findall(text)) / nw
        return stat, clamp(density / 3.0)

    @staticmethod
    def _zh(window: Window, text: str) -> tuple[float, float]:
        lens = [len(s) for s in window]
        mean = statistics.mean(lens) if lens else 0
        cv = statistics.pstdev(lens) / mean if mean else 1.0
        uniform = clamp(1.0 - cv / 0.55)
        formal = 0.0 if ZH_INFORMAL_RE.search(text) else 1.0
        stat = clamp(0.6 * uniform + 0.4 * formal)
        density = 100.0 * len(ZH_MARKER_RE.findall(text)) / max(len(text), 1) * 4
        return stat, clamp(density / 3.0)


class ModelDetector:  # pragma: no cover - requires torch/transformers and model downloads
    """Binoculars-style cross-perplexity (observer/performer causal LMs) + optional classifier."""
    name = "binoculars"

    def __init__(self, settings: Settings, threshold: float = 0.9):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.torch = torch
        self.threshold = threshold
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(settings.ai_observer_model)
        self.observer = AutoModelForCausalLM.from_pretrained(settings.ai_observer_model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(settings.ai_performer_model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
        self.dev = dev
        self.clf = None
        if settings.ai_classifier_model:
            self.clf = pipeline("text-classification", model=settings.ai_classifier_model, device=0 if dev == "cuda" else -1, truncation=True, max_length=512)
            self.label = settings.ai_classifier_label

    def binoculars(self, text: str) -> float:
        torch = self.torch
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.dev)
        with torch.no_grad():
            obs = self.observer(**enc).logits[:, :-1].float()
            per = self.performer(**enc).logits[:, :-1].float()
        labels = enc.input_ids[:, 1:]
        log_ppl = torch.nn.functional.cross_entropy(obs.transpose(1, 2), labels, reduction="none").mean()
        x_ppl = -(torch.softmax(per, -1) * torch.log_softmax(obs, -1)).sum(-1).mean()
        return float(log_ppl / x_ppl)  # low = machine-like

    def score(self, window: Window, lang: str) -> tuple[float, float]:
        text = " ".join(window)
        b = self.binoculars(text)
        stat = 1.0 / (1.0 + math.exp(-(self.threshold - b) * 12))
        clf = 0.0
        if self.clf is not None:
            r = self.clf(text)[0]
            clf = float(r["score"]) if r["label"] == self.label else 1.0 - float(r["score"])
        return stat, clf


@lru_cache(maxsize=1)
def get_detector(name: str, key: str = ""):
    from ..config import get_settings

    if name == "binoculars":
        return ModelDetector(get_settings())
    return HeuristicDetector()
