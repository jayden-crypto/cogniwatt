#!/usr/bin/env python3
"""The 6x6 colour similarity matrix (R2 Deliverable C, §6.4).

    python scripts/r2_similarity.py                              # perceptual + empirical
    python scripts/r2_similarity.py --raw data/r2_mock_responses.csv.gz
    python scripts/r2_similarity.py --embeddings                 # adds the ST arm

Output: `data/r2_color_similarity.npy` -- 6x6 over the env's COLORS tuple *in
that exact order*, symmetric, diagonal 1.0. R1 wires it into
`cogniwatt/env/tasks.py`; this script does not touch that file.

A PROBLEM WITH §6.4 THAT HAS TO BE SAID OUT LOUD
------------------------------------------------
§6.4 asks for the empirical arm to come from "when a model is wrong, which
distractor does it pick?", building a confusion matrix over candidate
attributes -- i.e. colours.

**On this dataset that matrix is flat by construction, and it must be.**
Colours here are assigned round-robin and spliced into descriptions that never
mention one; the answer is determined by commonsense inference, never by
colour. That is a deliberate property of Deliverable A -- a dataset where
colour predicted the answer would measure the wrong thing entirely (§3). The
consequence is that a model's errors cannot correlate with colour, so the
empirical *colour* confusion matrix estimates a uniform distribution.

Reporting a flat empirical matrix as if it were a measurement of colour
confusability would be backwards. So this script does three things instead:

* **Perceptual arm** (always available): CIELAB distance between the colour
  terms. This is a real, defensible statement about colour similarity that does
  not depend on our dataset at all, and it is what the env actually wants --
  a prior over which distractor a *perceptual* error lands on.
* **Empirical arm** (needs raw responses): computed and reported, with its
  flatness measured rather than assumed. If it comes out flat, that is the
  confirmation that Deliverable A has no colour shortcut -- a useful negative
  result, and it is reported as such, not as the similarity matrix.
* **Description arm** (needs raw responses): the thing that *is* measurable
  here -- do errors land on semantically nearer distractor *descriptions*?
  That is the real confusability question for this task.

The delivered matrix is the perceptual one (optionally averaged with the
embedding arm), and the report states exactly which arm produced it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniwatt.dialogue.schema import COLORS

DEFAULT_OUT = os.path.join("data", "r2_color_similarity.npy")


def _as_index(v, n: int) -> Optional[int]:
    """Coerce a display index that has round-tripped through CSV.

    Reading the raw table back from CSV turns `display_idx` into a float column
    (unparseable rows are NaN), so an `isinstance(v, int)` guard silently drops
    *every* row and yields an all-zero confusion matrix that looks like a
    finding. Parquet keeps the integer dtype, so this only bites on the CSV
    fallback path -- which is exactly the path in use when pyarrow is missing.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    i = int(f)
    return i if 0 <= i < n else None

#: sRGB for the env's five colour terms. "striped" is not a colour but a
#: pattern; the env treats it as a sixth mutually-exclusive label, so it needs
#: a defensible position in the space. Convention adopted here, and stated in
#: the report: a striped object is achromatic-patterned, so it sits between
#: black and white and is only weakly confusable with any pure hue. Modelled as
#: mid-grey with a similarity penalty applied to every chromatic term.
_SRGB: Dict[str, Tuple[float, float, float]] = {
    "red": (0.86, 0.08, 0.09),
    "blue": (0.11, 0.29, 0.78),
    "green": (0.13, 0.60, 0.24),
    "white": (0.96, 0.96, 0.96),
    "black": (0.08, 0.08, 0.08),
    "striped": (0.50, 0.50, 0.50),
}
#: How much less confusable a pattern term is with a solid colour. Applied to
#: the striped row/column only.
_PATTERN_PENALTY = 0.55


def _srgb_to_lab(rgb: Tuple[float, float, float]) -> np.ndarray:
    """sRGB -> CIELAB (D65). Perceptual distance needs a perceptual space;
    Euclidean distance in raw RGB badly misjudges e.g. green vs white."""
    c = np.array(rgb, dtype=np.float64)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = m @ c
    white = np.array([0.95047, 1.00000, 1.08883])
    t = xyz / white
    f = np.where(t > 0.008856, np.cbrt(t), (7.787 * t) + 16.0 / 116.0)
    return np.array([116.0 * f[1] - 16.0,
                     500.0 * (f[0] - f[1]),
                     200.0 * (f[1] - f[2])])


def perceptual_matrix() -> np.ndarray:
    labs = {c: _srgb_to_lab(_SRGB[c]) for c in COLORS}
    n = len(COLORS)
    d = np.zeros((n, n))
    for i, a in enumerate(COLORS):
        for j, b in enumerate(COLORS):
            d[i, j] = np.linalg.norm(labs[a] - labs[b])
    sim = 1.0 - d / d.max()          # 0..1, 1 on the diagonal
    k = COLORS.index("striped")
    sim[k, :] *= _PATTERN_PENALTY
    sim[:, k] *= _PATTERN_PENALTY
    np.fill_diagonal(sim, 1.0)
    return (sim + sim.T) / 2.0       # enforce exact symmetry


def embedding_matrix(model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    """Cosine similarity between colour terms in a referring-expression frame.

    Bare colour words embed poorly; putting each in the same noun-phrase frame
    is closer to how the term is actually used in the prompts.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    m = SentenceTransformer(model_name)
    frames = [f"a {c} mug on the shelf" for c in COLORS]
    e = m.encode(frames, normalize_embeddings=True)
    sim = np.asarray(e) @ np.asarray(e).T
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-12)
    np.fill_diagonal(sim, 1.0)
    return (sim + sim.T) / 2.0


def empirical_matrix(raw_path: str) -> Optional[Tuple[np.ndarray, Dict]]:
    """P(model picks colour j | true colour i, model is wrong), from raw responses."""
    import pandas as pd

    if not os.path.exists(raw_path):
        return None
    df = pd.read_parquet(raw_path) if raw_path.endswith(".parquet") \
        else pd.read_csv(raw_path)
    df = df[df["parse_status"].eq("ok") & df["correct"].eq(0)]
    if not len(df):
        return None

    n = len(COLORS)
    idx = {c: i for i, c in enumerate(COLORS)}
    counts = np.zeros((n, n))
    for _, r in df.iterrows():
        shown = json.loads(r["shown_colors"])       # display order
        di = _as_index(r["display_idx"], len(shown))
        if di is None:
            continue
        counts[idx[r["true_color"]], idx[shown[di]]] += 1

    row = counts.sum(axis=1, keepdims=True)
    p = np.divide(counts, np.where(row > 0, row, 1.0))

    # How far from flat is it? Under the null (colour carries no information)
    # every off-diagonal entry is 1/(k-1) in expectation. Report the deviation
    # rather than eyeballing the matrix.
    off = ~np.eye(n, dtype=bool)
    vals = p[off]
    vals = vals[vals > 0]
    cv = float(vals.std() / vals.mean()) if len(vals) and vals.mean() > 0         else float("nan")
    stats = {"n_errors": int(len(df)),
             "n_scored": int(counts.sum()),
             "offdiag_mean": float(vals.mean()) if len(vals) else float("nan"),
             "offdiag_cv": cv,
             # `None` when there is nothing to judge, so a parsing failure can
             # never masquerade as "colour is leaking".
             "flat": (None if not np.isfinite(cv) else bool(cv < 0.35))}
    sim = (p + p.T) / 2.0
    np.fill_diagonal(sim, 1.0)
    return sim, stats


def description_confusability(raw_path: str) -> Optional[Dict]:
    """Do errors land on lexically nearer distractor descriptions?

    This is the confusability question that this dataset *can* answer. Jaccard
    over content words is a deliberately transparent proxy -- it needs no model
    and cannot be accused of circularity, since the same embedder is not used
    to both generate and judge.
    """
    import pandas as pd

    if not os.path.exists(raw_path):
        return None
    from cogniwatt.dialogue.schema import load_scenarios
    spath = os.path.join("data", "r2_scenarios.jsonl")
    if not os.path.exists(spath):
        return None
    scen = {s.id: s for s in load_scenarios(spath)}

    df = pd.read_parquet(raw_path) if raw_path.endswith(".parquet") \
        else pd.read_csv(raw_path)
    df = df[df["parse_status"].eq("ok")]

    import re
    stop = set("a an the of in on at by with from and or to is was that it "
               "one already still".split())

    def toks(s: str) -> set:
        return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in stop}

    def jac(a: str, b: str) -> float:
        ta, tb = toks(a), toks(b)
        return len(ta & tb) / max(1, len(ta | tb))

    chosen, avg_distractor = [], []
    for _, r in df[df["correct"].eq(0)].iterrows():
        s = scen.get(r["scenario_id"])
        if s is None:
            continue
        perm = json.loads(r["perm"])
        di = _as_index(r["display_idx"], len(perm))
        if di is None:
            continue
        true_desc = s.candidates[s.true_idx].desc
        picked = s.candidates[perm[di]].desc
        others = [c.desc for j, c in enumerate(s.candidates) if j != s.true_idx]
        if not others:
            continue
        chosen.append(jac(true_desc, picked))
        avg_distractor.append(float(np.mean([jac(true_desc, o) for o in others])))

    if not chosen:
        return None
    return {
        "n": len(chosen),
        "similarity_of_chosen_distractor": float(np.mean(chosen)),
        "similarity_of_average_distractor": float(np.mean(avg_distractor)),
        "excess": float(np.mean(chosen) - np.mean(avg_distractor)),
        "errors_land_on_similar_distractors":
            bool(np.mean(chosen) > np.mean(avg_distractor) + 0.01),
    }


def show(name: str, m: np.ndarray) -> None:
    print(f"\n{name}")
    print("            " + "".join(f"{c:>9}" for c in COLORS))
    for i, c in enumerate(COLORS):
        print(f"  {c:<10}" + "".join(f"{m[i, j]:>9.3f}" for j in range(len(COLORS))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--raw", default=os.path.join("data", "r2_raw_responses.parquet"))
    ap.add_argument("--embeddings", action="store_true",
                    help="add the sentence-transformers arm (needs the package)")
    ap.add_argument("--report", default=os.path.join("data", "r2_similarity_report.json"))
    args = ap.parse_args()

    report: Dict[str, object] = {"colors": list(COLORS)}

    P = perceptual_matrix()
    show("PERCEPTUAL (CIELAB)  <- delivered unless an embedding arm is added", P)
    report["perceptual"] = P.tolist()
    delivered, source = P, "perceptual_cielab"

    E = embedding_matrix() if args.embeddings else None
    if args.embeddings and E is None:
        print("\n(sentence-transformers not installed; embedding arm skipped)")
    if E is not None:
        show("EMBEDDING (all-MiniLM-L6-v2)", E)
        report["embedding"] = E.tolist()
        off = ~np.eye(len(COLORS), dtype=bool)
        r = float(np.corrcoef(P[off], E[off])[0, 1])
        report["perceptual_vs_embedding_r"] = r
        print(f"\n  correlation perceptual vs embedding: r = {r:+.3f}")
        delivered = (P + E) / 2.0
        np.fill_diagonal(delivered, 1.0)
        source = "mean(perceptual_cielab, minilm_embedding)"

    raw = args.raw
    if not os.path.exists(raw):
        alt = raw.replace(".parquet", ".csv.gz")
        raw = alt if os.path.exists(alt) else raw

    emp = empirical_matrix(raw)
    if emp is None:
        print(f"\n(no raw responses at {args.raw}; empirical arm skipped)")
    else:
        M, stats = emp
        show("EMPIRICAL colour confusion (NOT delivered -- see the docstring)", M)
        report["empirical"] = M.tolist()
        report["empirical_stats"] = stats
        print(f"\n  errors={stats['n_errors']}  off-diagonal mean "
              f"{stats['offdiag_mean']:.3f}  CV {stats['offdiag_cv']:.3f}")
        if stats["flat"] is None:
            print("  -> no scorable errors; cannot judge flatness")
        elif stats["flat"]:
            print("  -> FLAT, as designed: colour carries no information in this "
                  "dataset, confirming Deliverable A has no colour shortcut. "
                  "This is a negative control, not a similarity measurement.")
        else:
            print("  -> NOT flat -- unexpected. Colour may be leaking into the "
                  "answer; re-check the dataset build before trusting any "
                  "accuracy number.")

    dc = description_confusability(raw)
    if dc:
        report["description_confusability"] = dc
        print(f"\nDESCRIPTION-LEVEL CONFUSABILITY (the measurable question here)")
        print(f"  chosen distractor similarity  {dc['similarity_of_chosen_distractor']:.4f}")
        print(f"  average distractor similarity {dc['similarity_of_average_distractor']:.4f}")
        print(f"  excess {dc['excess']:+.4f}  -> errors land on "
              f"{'semantically NEARER' if dc['errors_land_on_similar_distractors'] else 'no particular'}"
              f" distractors  (n={dc['n']})")

    assert delivered.shape == (6, 6)
    assert np.allclose(delivered, delivered.T), "matrix must be symmetric"
    assert np.allclose(np.diag(delivered), 1.0), "diagonal must be 1.0"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, delivered)
    report["delivered_source"] = source
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}  (source: {source})")
    print(f"wrote {args.report}")
    print("\nSend the .npy to Jayden; he wires it into cogniwatt/env/tasks.py. "
          "Do not edit that file here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
