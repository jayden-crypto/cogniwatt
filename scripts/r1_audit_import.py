"""Convert a copied audit result block into R2's per-annotator jsonl.

Owner: R1.

`scripts/r2_audit.py --annotate` collects answers interactively at a terminal.
The annotators here filled in the HTML sheet (`docs/r2_audit_sheet.html`) and
sent back a copied text block, which is the same information in a different
shape. This maps it onto R2's record format so `r2_audit.py --report` scores it
unchanged -- the scoring, kappa and verdict logic stays theirs.

Item N in the sheet is `build_sheet()[N-1]`, which is deterministic: the shuffle
is seeded (SHUFFLE_SEED = 611) precisely so one person's item numbers mean the
same thing in another's.

    python3 scripts/r1_audit_import.py data/audit/adit.tsv --kind human
"""
from __future__ import annotations

import argparse, json, os, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

#: Must match scripts/r2_audit.py -- its --report globs this directory.
from scripts.r2_audit import AUDIT_DIR  # noqa: E402


def parse_block(path: str):
    name, rows = None, {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        m = re.match(r"annotator:\s*(.+)$", line)
        if m:
            name = m.group(1).strip()
            continue
        m = re.match(r"^(\d+)\t(.*)$", line)
        if m:
            rows[int(m.group(1))] = m.group(2).strip()
    if not name:
        raise SystemExit(f"{path}: no 'annotator:' line")
    return name, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("block")
    ap.add_argument("--kind", default="human", choices=["human", "model"])
    ap.add_argument("--name", default=None, help="override the annotator name")
    a = ap.parse_args()

    from scripts.r2_audit import build_sheet  # noqa
    sheet = build_sheet()
    name, rows = parse_block(a.block)
    name = a.name or name

    os.makedirs(AUDIT_DIR, exist_ok=True)
    out = os.path.join(AUDIT_DIR, f"{name}.jsonl")
    n_ans = n_skip = 0
    with open(out, "w", encoding="utf-8") as f:
        for n, p in enumerate(sheet, 1):
            raw = rows.get(n, "")
            choice = None
            if raw not in ("", "?"):
                try:
                    choice = int(raw)
                except ValueError:
                    choice = None
            if choice is None:
                n_skip += 1
            else:
                n_ans += 1
            f.write(json.dumps({"core_id": p["core_id"], "annotator": name,
                                "kind": a.kind, "choice": choice,
                                "note": ""}) + "\n")
    print(f"{name}: {n_ans} answered, {n_skip} undecidable -> {out}")

    # Agreement with the authored key, for this annotator alone. Inter-annotator
    # kappa needs the second pass and is r2_audit.py --report's job.
    hit = tot = 0
    miss = []
    for n, p in enumerate(sheet, 1):
        raw = rows.get(n, "")
        if raw in ("", "?"):
            continue
        tot += 1
        if int(raw) == p["answer_shown_idx"]:
            hit += 1
        else:
            miss.append((n, p["core_id"], p["family"]))
    print(f"agreement with the authored key: {hit}/{tot} = {hit/tot:.3f}")
    if miss:
        fam = {}
        for _, _, f_ in miss:
            fam[f_] = fam.get(f_, 0) + 1
        print("disagreements by family: "
              + ", ".join(f"{k} {v}" for k, v in sorted(fam.items(), key=lambda x: -x[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
