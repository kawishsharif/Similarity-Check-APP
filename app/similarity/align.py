"""Shingle fingerprints and text alignment (exact and near-exact shared passages)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from rapidfuzz import fuzz

MAX_POSITIONS_PER_HASH = 40  # ignore boilerplate shingles that occur everywhere
GAP_TOLERANCE = 4            # tokens that may differ inside a match (small edits)
DIAG_TOLERANCE = 3           # insertions/deletions allowed between the two texts


def shingle_size(cjk: bool) -> int:
    return 8 if cjk else 5


def min_match_len(base: int, cjk: bool) -> int:
    """Minimum match length in tokens; CJK 'words' are single characters so need more of them."""
    return int(base * 1.5) if cjk else base


def h64(tokens: list[str]) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(tokens).encode("utf-8"), digest_size=8).digest(), "big")


def shingle_hashes(norms: list[str], k: int) -> list[int]:
    return [h64(norms[i:i + k]) for i in range(len(norms) - k + 1)]


def build_index(norms: list[str], k: int) -> dict[int, list[int]]:
    idx: dict[int, list[int]] = {}
    for i, h in enumerate(shingle_hashes(norms, k)):
        idx.setdefault(h, []).append(i)
    return idx


def sampled(h: int) -> bool:
    """Archive fingerprints keep one hash in four (a simple, deterministic sample)."""
    return h % 4 == 0


@dataclass
class Match:
    s0: int  # submission token range [s0, s1)
    s1: int
    c0: int  # candidate token range [c0, c1)
    c1: int
    kind: str = "exact"

    @property
    def length(self) -> int:
        return self.s1 - self.s0


def _extend(sub: list[str], cand: list[str], s: int, c: int, step: int) -> tuple[int, int]:
    """Grow a match one token at a time while the texts agree, bridging single-word substitutions
    when the two tokens after the difference agree again."""
    while True:
        a, b = (s - 1, c - 1) if step < 0 else (s, c)
        if not (0 <= a < len(sub) and 0 <= b < len(cand)):
            return s, c
        if sub[a] == cand[b]:
            s, c = s + step, c + step
            continue
        a2, b2, a3, b3 = a + step, b + step, a + 2 * step, b + 2 * step
        if all(0 <= x < len(sub) for x in (a2, a3)) and all(0 <= y < len(cand) for y in (b2, b3)) and sub[a2] == cand[b2] and sub[a3] == cand[b3]:
            s, c = s + step, c + step
            continue
        return s, c


def align(sub: list[str], index: dict[int, list[int]], cand: list[str], k: int, min_len: int) -> list[Match]:
    """Find passages shared between the submission and one candidate text."""
    if len(cand) < k or len(sub) < k:
        return []
    hits: list[tuple[int, int, int]] = []  # (diagonal, i, j)
    for j, h in enumerate(shingle_hashes(cand, k)):
        pos = index.get(h)
        if pos and len(pos) <= MAX_POSITIONS_PER_HASH:
            for i in pos:
                hits.append((i - j, i, j))
    if not hits:
        return []
    hits.sort()
    chains: list[list[int]] = []  # [diag, s0, s1, c0, c1]
    for diag, i, j in hits:
        if chains and chains[-1][0] == diag and i <= chains[-1][2] + GAP_TOLERANCE:
            c = chains[-1]
            c[2] = max(c[2], i + k)
            c[4] = max(c[4], j + k)
        else:
            chains.append([diag, i, i + k, j, j + k])
    # Merge chains on nearby diagonals that are close together (small insertions or deletions).
    chains.sort(key=lambda c: (c[1], c[3]))
    merged: list[list[int]] = []
    for c in chains:
        if merged:
            m = merged[-1]
            if abs(c[0] - m[0]) <= DIAG_TOLERANCE and c[1] <= m[2] + GAP_TOLERANCE and c[3] <= m[4] + GAP_TOLERANCE + DIAG_TOLERANCE:
                m[2], m[4] = max(m[2], c[2]), max(m[4], c[4])
                continue
        merged.append(list(c))
    out: list[Match] = []
    for _, s0, s1, c0, c1 in merged:
        s0, c0 = _extend(sub, cand, s0, c0, -1)
        s1, c1 = _extend(sub, cand, s1, c1, +1)
        if s1 - s0 < min_len:
            continue
        a, b = " ".join(sub[s0:s1]), " ".join(cand[c0:c1])
        if fuzz.ratio(a, b) < 80:  # the chain bridged too many differences
            continue
        out.append(Match(s0, s1, c0, c1))
    return out
