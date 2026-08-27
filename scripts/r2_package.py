#!/usr/bin/env python3
"""Build the delivery archive for R1 (always from scratch).

    python scripts/r2_package.py
    python scripts/r2_package.py --out data.zip --verify

WHY THIS SCRIPT EXISTS
----------------------
The first delivery shipped a **stale** `tests/test_r2_estimators.py`: the zip
was built once, then updated by appending with a
`if name not in z.namelist(): z.write(name)` guard. `zipfile` in append mode
cannot replace an entry, so the guard silently preserved the old copy while the
corrected file sat on disk. R1 ran the suite and saw a failure that had already
been fixed here.

The lesson is that an archive built incrementally has no single moment where it
is known to match the working tree. So this script always writes a new file,
never appends, and then **verifies every entry byte-for-byte against disk**
before declaring success. A packaging step that cannot be trusted is worse than
no packaging step, because the recipient debugs code you already fixed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import zipfile
from typing import List

#: Explicit, ordered manifest. Globs are expanded but the categories are named,
#: so a new file does not silently join the delivery and a deleted one is
#: reported rather than quietly dropped.
REQUIRED = [
    # the artefact R1 trains on, and the inputs behind every number in it
    "data/dialogue_measured.npz",
    "data/r2_color_similarity.npy",
    "data/r2_rtx3060_profile.json",
    "data/r2_scenarios.jsonl",
    "data/r2_raw_responses.csv.gz",
    "data/r2_raw_offload.csv.gz",
    "data/r2_raw_q8.csv.gz",
    "data/r2_raw_reasoning.csv.gz",
    "data/r2_raw_all.csv.gz",
    "data/r2_ceilings.json",
    "data/r2_raw_responses_meta.json",
    "data/r2_analysis_real.json",
    "data/r2_scenarios_report.json",
    "data/r2_similarity_report.json",
    "data/r2_asr_curve.json",
    "data/r2_asr_results.json",
    "data/r2_asr_manifest.json",
    # evidence for the digit-label finding; not an input to any current number
    "data/r2_raw_responses_DIGITLABELS.csv.gz",
    "models/MANIFEST.json",
    "requirements-r2.txt",
    "tests/test_r2_estimators.py",
]
GLOBS = ["scripts/r2_*.py", "cogniwatt/dialogue/*.py", "docs/r2_*.md",
         "docs/r2_*.html",
         "docs/R1_*.md", "results/figures/r2_*.png"]


#: Excluded from --full, with the reason each one is not a deliverable.
FULL_SKIP_DIRS = {".venv", "__pycache__", ".git", ".idea", ".vscode"}
FULL_SKIP_EXT = {
    # 10.8 GB of weights. Re-downloadable, and models/MANIFEST.json pins the
    # exact build with its SHA-256, so identity survives without the bytes.
    ".gguf",
    ".part", ".json.part",
}
FULL_SKIP_NAMES = {"data.zip", "cogniwatt_full.zip"}


def collect_full(root: str = ".") -> List[str]:
    """Every file in the tree except environments, weights and archives.

    `.wav` clips ARE included despite being nominally regenerable: the ASR
    corpus was synthesised with whichever SAPI voices this machine has, so
    re-running `r2_asr.py --synth` elsewhere produces a *different* corpus and
    a different WER. They are measured data, not a build artefact.
    """
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in FULL_SKIP_DIRS]
        for fn in filenames:
            if fn in FULL_SKIP_NAMES:
                continue
            if os.path.splitext(fn)[1].lower() in FULL_SKIP_EXT:
                continue
            if fn.endswith(".part.json"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            out.append(rel.replace(os.sep, "/"))
    return sorted(out)


def collect() -> List[str]:
    files = list(REQUIRED)
    for g in GLOBS:
        files += sorted(glob.glob(g))
    # Normalise to forward slashes: glob returns backslashes on Windows but zip
    # entries are always "/", so an un-normalised name builds fine and then
    # fails to read back -- which is exactly the class of bug this script is
    # here to prevent.
    seen, out = set(), []
    for f in files:
        f = f.replace(os.sep, "/")
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data.zip")
    ap.add_argument("--verify", action="store_true", default=True)
    ap.add_argument("--full", action="store_true",
                    help="every file in the tree except virtualenvs, GGUF "
                         "weights and archives -- the complete handoff rather "
                         "than the curated R2 delivery")
    args = ap.parse_args()

    files = collect_full() if args.full else collect()
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print("MISSING, refusing to build a partial delivery:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    # Always a fresh file. Never append.
    if os.path.exists(args.out):
        os.remove(args.out)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f)

    if args.verify:
        bad = []
        with zipfile.ZipFile(args.out) as z:
            names = z.namelist()
            dupes = {n for n in names if names.count(n) > 1}
            if dupes:
                bad.append(f"duplicate entries: {sorted(dupes)}")
            for f in files:
                if hashlib.sha256(z.read(f)).hexdigest() != sha(f):
                    bad.append(f"CONTENT MISMATCH: {f}")
        if bad:
            print("VERIFY FAILED:", file=sys.stderr)
            for b in bad:
                print(f"  {b}", file=sys.stderr)
            return 1

    size = os.path.getsize(args.out)
    if args.full:
        skipped = []
        for dirpath, dirnames, filenames in os.walk("."):
            dirnames[:] = [d for d in dirnames if d not in {".git"}]
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() == ".gguf":
                    skipped.append(os.path.join(dirpath, fn))
        if skipped:
            gb = sum(os.path.getsize(f) for f in skipped) / 1e9
            print(f"excluded {len(skipped)} GGUF weight files ({gb:.1f} GB) -- "
                  f"models/MANIFEST.json pins each build by SHA-256; refetch with "
                  f"scripts/r2_get_models.py")
        print("excluded .venv/ (machine-specific; see requirements-r2.txt)")
    print(f"wrote {args.out}  {len(files)} entries  {size / 1e6:.1f} MB")
    print(f"verified: every entry matches the working tree byte-for-byte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
