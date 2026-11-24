"""Calibrate AI-writing thresholds on labelled text.

    python scripts/calibrate.py --human data/human_texts --ai data/ai_texts [--fpr 0.01]

Each folder holds .txt files (one document each, 300+ words). Use public collections (RAID, HC3, M4) and
real essays written before 2022 as the human set. The script searches for the thresholds that keep the
document-level false-positive rate at or under --fpr (default 1%) while catching as much AI text as
possible, writes them to data/ai_thresholds.json (read by the app), and prints the detection rate so it
can be published with the accuracy statement. Re-run it before each release.
"""
import argparse
import itertools
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.aidetect import engine as E  # noqa: E402
from app.aidetect.detectors import get_detector  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.ingest import ingest_file  # noqa: E402

FLAG_AT = 20.0  # a document counts as "flagged" when at least this percent of its prose is flagged


def scores(folder: Path, det, s):
    out = []
    for f in sorted(folder.glob("*.txt")):
        doc, _, _ = ingest_file(f, Path(tempfile.mkdtemp()), f.name, s)
        sents = E.qualifying_sentences(doc)
        if sum(x.e - x.s for x in sents) < E.MIN_QUALIFYING_WORDS:
            continue
        w = min(E.WINDOW, len(sents))
        acc = [[0.0, 0.0, 0] for _ in sents]
        for a in range(max(1, len(sents) - w + 1)):
            st, cl = det.score([x.text for x in sents[a:a + w]], doc.language)
            for i in range(a, a + w):
                acc[i][0] += st; acc[i][1] += cl; acc[i][2] += 1
        out.append([(x.e - x.s, a[0] / a[2], a[1] / a[2]) for x, a in zip(sents, acc)])
    return out


def doc_flagged(doc_scores, ts, tc, th):
    flags = [(st >= ts and cl >= tc) or (cl >= th and st < ts) for _, st, cl in doc_scores]
    flags = [f and ((i > 0 and flags[i - 1]) or (i + 1 < len(flags) and flags[i + 1])) for i, f in enumerate(flags)]
    words = sum(n for (n, _, _), f in zip(doc_scores, flags) if f)
    return 100.0 * words / sum(n for n, _, _ in doc_scores) >= FLAG_AT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=Path, required=True)
    ap.add_argument("--ai", type=Path, required=True)
    ap.add_argument("--fpr", type=float, default=0.01)
    a = ap.parse_args()
    s = get_settings()
    det = get_detector(s.ai_detector)
    human, ai = scores(a.human, det, s), scores(a.ai, det, s)
    if not human or not ai:
        sys.exit("need scoreable documents (300+ words of prose) in both folders")
    grid = [i / 20 for i in range(2, 20)]
    best = None
    for ts, tc in itertools.product(grid, grid):
        th = min(1.0, tc + 0.2)
        fpr = sum(doc_flagged(d, ts, tc, th) for d in human) / len(human)
        if fpr > a.fpr:
            continue
        tpr = sum(doc_flagged(d, ts, tc, th) for d in ai) / len(ai)
        if best is None or tpr > best[0]:
            best = (tpr, fpr, ts, tc, th)
    if best is None:
        sys.exit(f"no thresholds reach a false-positive rate of {a.fpr:.1%} on this data; the detector is not usable as configured")
    tpr, fpr, ts, tc, th = best
    f = s.data_dir / "ai_thresholds.json"
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur[det.name] = {"stat": ts, "clf": tc, "clf_high": th}
    f.write_text(json.dumps(cur, indent=2))
    print(f"detector={det.name} human docs={len(human)} ai docs={len(ai)}\nthresholds stat={ts} clf={tc} clf_high={th}\n"
          f"document-level false-positive rate={fpr:.1%}  detection rate={tpr:.1%}\nwritten to {f}")


if __name__ == "__main__":
    main()
