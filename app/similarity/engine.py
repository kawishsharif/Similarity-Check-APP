"""Similarity engine: query selection, retrieval, alignment, overlap resolution, classification, scoring."""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from rapidfuzz import fuzz, process

from .. import textproc as tp
from ..config import Settings
from ..ingest import FLAG_CITE, FLAG_HEAD, FLAG_QUOTE, Doc
from .align import Match, align, build_index, min_match_len, sampled, shingle_hashes, shingle_size
from .sources import Candidate, Source, TextFetcher

log = logging.getLogger(__name__)

PALETTE = ["#E4572E", "#2E86AB", "#6A994E", "#F2A541", "#8E5BD9", "#17A398", "#C73E7B", "#7A6C5D", "#3D6DCC", "#B5651D", "#4F9D69", "#D1495B"]
GROUPS = [
    ("not_cited_or_quoted", "Not Cited or Quoted", "Matched text with neither an in-text citation nor quotation marks.", "#D64545"),
    ("missing_quotes", "Missing Quotation Marks", "Matched text with an in-text citation but no quotation marks.", "#E8892B"),
    ("missing_citation", "Missing Citation", "Matched text in quotation marks without an in-text citation.", "#C9A400"),
    ("cited_quoted", "Cited and Quoted", "Matched text with both quotation marks and an in-text citation.", "#3E9B5B"),
]
TYPE_LABELS = {"internet": "Internet sources", "publication": "Publications", "submitted": "Submitted works"}
CITE_LOOKAHEAD = 12


@dataclass
class SimOptions:
    exclude_biblio: bool = True
    exclude_quotes: bool = False
    min_words: int = 8


Progress = Callable[[float, str], None]


# --------------------------------------------------------------------------------------------
# 1-2. Query selection
# --------------------------------------------------------------------------------------------
def select_queries(doc: Doc, max_queries: int) -> list[str]:
    """Pick the most distinctive sentences from each ~150-word passage and turn them into search phrases."""
    cjk = doc.cjk
    min_tokens = 14 if cjk else 8
    window = 30 if cjk else 14
    passage_target = 250 if cjk else 150
    passages: list[list[tuple[float, str]]] = []
    cur: list[tuple[float, str]] = []
    cur_n = 0
    for s, e in doc.sentences():
        if doc.in_biblio(s) or (doc.flags[s] & FLAG_HEAD) or e - s < min_tokens:
            continue
        # best window inside the sentence = the stretch with the rarest words
        best, best_w = (s, min(e, s + window)), -1.0
        for a in range(s, max(s + 1, e - window + 1), max(1, window // 4)):
            w = tp.keyword_weight(doc.norm[a:a + window])
            if w > best_w:
                best, best_w = (a, min(e, a + window)), w
        cur.append((best_w, doc.text(*best)))
        cur_n += e - s
        if cur_n >= passage_target:
            passages.append(cur)
            cur, cur_n = [], 0
    if cur:
        passages.append(cur)
    picked: list[list[str]] = [[t for _, t in sorted(p, key=lambda x: -x[0])[:3]] for p in passages]
    out: list[str] = []
    seen = set()
    for rank in range(3):  # round-robin so every part of the paper is covered before the cap bites
        for p in picked:
            if rank < len(p) and p[rank] not in seen and len(out) < max_queries:
                seen.add(p[rank])
                out.append(p[rank])
    return out


# --------------------------------------------------------------------------------------------
# 3. Candidate retrieval
# --------------------------------------------------------------------------------------------
def _norm_key(c: Candidate) -> str:
    return re.sub(r"[#?].*$", "", c.key.strip().lower()).rstrip("/")


def retrieve(queries: list[str], sources: list[Source], lang: str, progress: Progress | None = None) -> list[Candidate]:
    tasks = []
    for src in sources:
        cap = getattr(src, "query_cap", None) or len(queries)
        for q in queries[:cap]:
            tasks.append((src, q))
    found: dict[str, Candidate] = {}
    done = 0

    def run(t):
        src, q = t
        src.calls = getattr(src, "calls", 0) + 1
        try:
            return src.search(q, lang)
        except Exception as exc:  # a broken connector must never fail the whole check
            src.errors = getattr(src, "errors", 0) + 1
            log.warning("%s search failed: %s", src.name, exc)
            return []

    with ThreadPoolExecutor(max_workers=max(2, min(8, len(sources) * 2))) as pool:
        for res in pool.map(run, tasks):
            done += 1
            for c in res:
                k = _norm_key(c)
                if not k:
                    continue
                if k in found:
                    found[k].hits += 1
                    if not found[k].text and c.text:
                        found[k].text = c.text
                else:
                    c.hits = 1
                    found[k] = c
            if progress and done % 10 == 0:
                progress(done / max(1, len(tasks)), f"Searching sources ({done}/{len(tasks)})")
    return list(found.values())


# --------------------------------------------------------------------------------------------
# Paraphrase / near-match pass
# --------------------------------------------------------------------------------------------
def _paraphrase_matches(doc: Doc, covered: bytearray, cand_texts: list[str], settings: Settings) -> list[tuple[int, Match]]:
    """Sentence-level near matches that exact alignment missed. Returns (candidate index, Match)."""
    min_tokens = 14 if doc.cjk else 10
    sents = [(s, e) for s, e in doc.sentences()
             if e - s >= min_tokens and not doc.in_biblio(s) and not (doc.flags[s] & FLAG_HEAD) and sum(covered[s:e]) < (e - s) * 0.3]
    if not sents:
        return []
    pool, owner = [], []
    for ci, text in enumerate(cand_texts):
        for sent in tp.split_sentences(text):
            _, norms = tp.plain_to_tokens(sent)
            if len(norms) >= min_tokens - 2:
                pool.append(" ".join(norms))
                owner.append(ci)
    if not pool:
        return []
    out: list[tuple[int, Match]] = []
    if settings.sim_embeddings:
        out = _embedding_pass(doc, sents, pool, owner)
        if out is not None:
            return out
        out = []
    for s, e in sents:
        q = " ".join(doc.norm[s:e])
        hit = process.extractOne(q, pool, scorer=fuzz.ratio, score_cutoff=86)
        if hit:
            out.append((owner[hit[2]], Match(s, e, 0, 0, "paraphrase")))
    return out


def _embedding_pass(doc, sents, pool, owner):  # pragma: no cover - needs a model download
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("BAAI/bge-m3")
        a = model.encode([" ".join(doc.norm[s:e]) for s, e in sents], normalize_embeddings=True)
        b = model.encode(pool, normalize_embeddings=True)
        sims = a @ b.T
        out = []
        for i, (s, e) in enumerate(sents):
            j = int(np.argmax(sims[i]))
            if sims[i][j] >= 0.88:
                out.append((owner[j], Match(s, e, 0, 0, "paraphrase")))
        return out
    except Exception as exc:
        log.warning("embedding backend unavailable (%s); using lexical near-match", exc)
        return None


# --------------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------------
def run_similarity(doc: Doc, opts: SimOptions, sources: list[Source], settings: Settings, fetcher: TextFetcher | None = None,
                   archive: Callable[[list[int]], list[Candidate]] | None = None, progress: Progress | None = None) -> dict:
    progress = progress or (lambda f, m: None)
    n = doc.n
    cjk = doc.cjk
    k = shingle_size(cjk)
    min_len = min_match_len(max(opts.min_words, 1), cjk)

    counted = bytearray(b"\x01") * n
    for i in range(n):
        if opts.exclude_biblio and doc.in_biblio(i):
            counted[i] = 0
        elif opts.exclude_quotes and doc.flags[i] & FLAG_QUOTE:
            counted[i] = 0
    total = sum(counted)

    queries = select_queries(doc, settings.max_queries)
    progress(0.02, f"Selected {len(queries)} search phrases")
    candidates = retrieve(queries, sources, doc.language, lambda f, m: progress(0.02 + 0.4 * f, m)) if queries and sources else []
    warnings: list[str] = []
    for src in sources:
        calls, errs = getattr(src, "calls", 0), getattr(src, "errors", 0)
        if calls and errs >= calls:
            warnings.append(f"Source '{src.name}' could not be reached; matches from it are missing.")
    candidates.sort(key=lambda c: (-c.hits, c.text is None))
    candidates = candidates[: settings.max_candidates]
    if archive:
        hashes = [h for h in shingle_hashes(doc.norm, k) if sampled(h)]
        try:
            candidates += archive(hashes)
        except Exception as exc:
            log.warning("archive lookup failed: %s", exc)
            warnings.append("The platform archive could not be searched.")
    progress(0.45, f"Downloading {len(candidates)} candidate sources")

    fetcher = fetcher or TextFetcher(settings)
    with ThreadPoolExecutor(max_workers=8) as pool:
        texts = list(pool.map(lambda c: fetcher.fetch(c) if c.type != "submitted" else c.text, candidates))
    pairs = [(c, t) for c, t in zip(candidates, texts) if t and len(t) > 80]

    progress(0.65, "Comparing text")
    index = build_index(doc.norm, k)
    cand_matches: list[list[Match]] = []
    for c, t in pairs:
        _, cn = tp.plain_to_tokens(t)
        cand_matches.append(align(doc.norm, index, cn, k, min_len))
    covered_any = bytearray(n)
    for ms in cand_matches:
        for m in ms:
            for i in range(m.s0, m.s1):
                covered_any[i] = 1

    progress(0.8, "Looking for paraphrased passages")
    para = _paraphrase_matches(doc, covered_any, [t for _, t in pairs], settings) if pairs else []
    for ci, m in para:
        cand_matches[ci].append(m)

    # 7. resolve overlaps: every counted word goes to the source with the largest total match
    cov: list[bytearray] = []
    totals: list[tuple[int, int, int]] = []
    for ci, ms in enumerate(cand_matches):
        b = bytearray(n)
        for m in ms:
            for i in range(m.s0, m.s1):
                if counted[i]:
                    b[i] = 1
        cov.append(b)
        totals.append((sum(b), -sum(m.length for m in ms if m.kind == "paraphrase"), -ci))
    order = sorted((ci for ci in range(len(cand_matches)) if totals[ci][0] > 0), key=lambda ci: totals[ci], reverse=True)
    owner = [-1] * n
    kind = [""] * n
    for ci in order:
        for m in sorted(cand_matches[ci], key=lambda m: m.kind != "exact"):  # exact before paraphrase
            for i in range(m.s0, m.s1):
                if counted[i] and owner[i] < 0:
                    owner[i], kind[i] = ci, m.kind

    segments: list[list] = []  # [s0, s1, ci, kind]
    i = 0
    while i < n:
        if owner[i] < 0:
            i += 1
            continue
        j = i
        while j < n and owner[j] == owner[i] and kind[j] == kind[i]:
            j += 1
        if j - i >= min_len:
            segments.append([i, j, owner[i], kind[i]])
        i = j

    # rank sources by the words finally assigned to them
    words_by: dict[int, int] = {}
    exact_by: dict[int, int] = {}
    for s0, s1, ci, kd in segments:
        words_by[ci] = words_by.get(ci, 0) + (s1 - s0)
        if kd == "exact":
            exact_by[ci] = exact_by.get(ci, 0) + (s1 - s0)
    ranked = sorted(words_by, key=lambda ci: (-words_by[ci], ci))
    rank_of = {ci: r + 1 for r, ci in enumerate(ranked)}
    sources_out = []
    for ci in ranked:
        c = pairs[ci][0]
        sources_out.append({"rank": rank_of[ci], "type": c.type, "title": c.title, "address": c.address, "words": words_by[ci],
                            "exact_words": exact_by.get(ci, 0), "percent": round(100.0 * words_by[ci] / total, 2) if total else 0.0,
                            "color": PALETTE[(rank_of[ci] - 1) % len(PALETTE)]})

    # 8. classify each segment
    group_words = {g[0]: 0 for g in GROUPS}
    group_count = {g[0]: 0 for g in GROUPS}
    matches_out = []
    for s0, s1, ci, kd in segments:
        length = s1 - s0
        quoted = sum(1 for x in range(s0, s1) if doc.flags[x] & FLAG_QUOTE) / length >= 0.8
        cited = any(doc.flags[x] & FLAG_CITE for x in range(max(0, s0 - 2), min(n, s1 + CITE_LOOKAHEAD)))
        cat = "cited_quoted" if quoted and cited else "missing_citation" if quoted else "missing_quotes" if cited else "not_cited_or_quoted"
        group_words[cat] += length
        group_count[cat] += 1
        alts = [rank_of[o] for o in order if o != ci and o in rank_of and cov[o][s0]][:5]
        matches_out.append({"s0": s0, "s1": s1, "rank": rank_of[ci], "category": cat, "kind": kd, "alts": alts})

    matched = sum(group_words.values())
    type_words: dict[str, int] = {}
    for src in sources_out:
        type_words[src["type"]] = type_words.get(src["type"], 0) + src["words"]
    pct = lambda w: round(100.0 * w / total, 2) if total else 0.0
    progress(0.95, "Scoring")
    return {
        "percent": pct(matched), "matched_words": matched, "total_words": total, "language": doc.language,
        "filters": {"exclude_biblio": opts.exclude_biblio, "exclude_quotes": opts.exclude_quotes, "min_words": max(opts.min_words, 1)},
        "groups": [{"key": key, "name": name, "note": note, "color": color, "count": group_count[key], "percent": pct(group_words[key])} for key, name, note, color in GROUPS],
        "source_types": [{"type": t, "label": TYPE_LABELS[t], "percent": pct(type_words.get(t, 0))} for t in ("internet", "publication", "submitted")],
        "sources": sources_out, "matches": matches_out, "warnings": warnings,
        "stats": {"queries": len(queries), "candidates": len(candidates), "compared": len(pairs), "paraphrase_words": sum(s1 - s0 for s0, s1, _, kd in segments if kd == "paraphrase")},
    }
