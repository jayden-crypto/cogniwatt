"""Build the R1 -> R2 handback zip. Owner: R1.

Mirrors R2's `scripts/r2_package.py`. Ships the whole tree R2 asked for, minus
what they already have in bulk (the 721 ASR .wav clips, GGUF weights, trained
policies) -- those are theirs or re-downloadable, and including them turns a
2 MB transfer into 90 MB.

    python3 scripts/r1_package.py
"""
from __future__ import annotations

import hashlib, os, pathlib, sys, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "cogniwatt_r1_handback.zip"

#: Included wholesale.
TREES = [
    "cogniwatt",          # the package, incl. dialogue/ and R1's env/agents/analysis
    "scripts",            # r1_* and r2_*
    "tests",
    "docs",               # both handoff directions + the audit sheet
    "data/r2_audit",      # the three annotator passes  <- new, R2 has never seen these
    "data/r2_audit_quarantine",
]

#: Individual files.
FILES = [
    "data/dialogue_measured.npz",
    "data/dialogue_measured_recovery.npz",
    "data/r2_core_audit.json",
    "data/r2_analysis_real.json",
    "data/r2_analysis.json",
    "data/r2_scenarios.jsonl",
    "data/r2_scenarios_k3.jsonl",
    "data/r2_scenarios_report.json",
    "data/r2_raw_all.csv.gz",
    "data/r2_rtx3060_profile.json",
    "data/r2_asr_curve.json",
    "data/r2_similarity_report.json",
    "data/r2_color_similarity.npy",
    "data/anchor_m2_air.json",
    "data/roofline_cost_table.json",
    "data/thermal_fit_m2.json",
    "data/throttle_m2_air.json",
    "models/MANIFEST.json",
    "requirements-r2.txt",
    "README.md",
    "results/stability_runs.csv",
    "results/stability_arms.csv",
    "results/isolation.csv",
    "results/cloud_price.csv",
    "results/asr_sensitivity.json",
]

#: Never shipped: R2 already has these, or they are large and reproducible.
SKIP_DIRS = {".venv", "__pycache__", ".git", "logs", "data/r2_asr_audio",
             "results/runs", "results/runs_legacy", "results/figures"}
SKIP_SUFFIX = {".pt", ".gguf", ".wav", ".pyc"}


def skip(rel: str) -> bool:
    p = rel.replace(os.sep, "/")
    if any(p == d or p.startswith(d + "/") for d in SKIP_DIRS):
        return True
    return pathlib.Path(p).suffix in SKIP_SUFFIX


def main() -> int:
    seen, n, total = set(), 0, 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        def add(path: pathlib.Path):
            nonlocal n, total
            rel = path.relative_to(ROOT).as_posix()
            if rel in seen or skip(rel) or not path.is_file():
                return
            seen.add(rel); n += 1; total += path.stat().st_size
            z.write(path, rel)

        for t in TREES:
            base = ROOT / t
            if not base.exists():
                print(f"  (missing tree: {t})"); continue
            for p in sorted(base.rglob("*")):
                add(p)
        for f in FILES:
            p = ROOT / f
            if p.exists():
                add(p)
            else:
                print(f"  (missing file: {f})")

    h = hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]
    print(f"\n{OUT.name}: {n} files, {total/1e6:.1f} MB raw -> "
          f"{OUT.stat().st_size/1e6:.1f} MB zipped")
    print(f"sha256: {h}")
    print("\nnew to R2 (everything else they have or authored):")
    for k in sorted(seen):
        if k.startswith(("data/r2_audit", "scripts/r1_", "docs/R2_REPLY",
                         "docs/r1_", "docs/r2_audit_sheet.html", "results/")):
            print(f"  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
