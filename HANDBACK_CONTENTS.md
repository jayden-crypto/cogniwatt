# cogniwatt_r1_handback.zip — what is in here

113 files, 1.5 MB. Built by `python3 scripts/r1_package.py`, the mirror of your
`r2_package.py`. **There is no git remote** — the repo is local-only, so this zip
is the transfer.

## Read this first

`docs/R2_REPLY_7.md` — the audit result and the one thing we need from you
(re-freeze on the audited subset). Everything else is context.

## What is genuinely new to you

Everything else in the zip you either authored or already have.

| Path | What it is |
| --- | --- |
| **`data/r2_audit/*.jsonl`** | **the three annotator passes** — reader_a, reader_b (outside the project), reader_c. Your `r2_audit.py --report` reads this directory directly |
| **`data/r2_core_audit.json`** | **the verdict file** — 60 cores, 29 `agree`. This is the input to `r2_build_dataset.py --audit` |
| `data/r2_audit_quarantine/` | provenance note on one pass that was held and then reinstated |
| `scripts/r1_audit_import.py` | maps a copied result block onto your record format; your scorer is untouched |
| `docs/r2_audit_sheet.html` | the blind sheet as a click-through form — what the annotators actually used |
| `docs/R2_REPLY*.md` | the R1 → R2 direction, replies 1–7 |
| `docs/r1_*.md`, `r1_report.html` | R1's own workstream: stability study, cloud-price derivation, transfer audit, lab plan |
| `results/*.csv`, `*.json` | R1 analysis outputs (stability, isolation, cloud price, ASR sensitivity) |
| `data/dialogue_measured_recovery.npz` | your artefact + `asr_recovery`, produced by `r1_merge_recovery.py`. **Superseded** once your `r2_freeze.py` emits the key — delete both then |

## Deliberately absent

* **`data/r2_asr_audio/` (721 .wav, 84 MB)** — yours, unchanged. Re-zipping it
  would make this a 90 MB transfer for no reason.
* **GGUF weights** — `models/MANIFEST.json` is included; restore with
  `r2_get_models.py`. Not needed for the re-freeze, which is a filter with no
  re-querying.
* **`results/runs/` (150+ trained policies, ~200 MB)** and
  `results/figures/` — R1-side, not inputs to anything you do.

## Your own files

`scripts/r2_*.py`, `cogniwatt/dialogue/**`, `tests/test_r2_estimators.py`,
`data/r2_raw_all.csv.gz`, `r2_scenarios.jsonl`, `r2_analysis*.json`, the profile
JSON and `requirements-r2.txt` are all in here at the versions we are running,
so the tree is self-contained and you can diff against yours.

One change we made to a file of yours, flagged in `R2_REPLY_7.md` §1 of the
earlier reply and repeated here: `tests/test_r2_estimators.py`'s
`test_dataset_has_no_position_or_colour_shortcut` still asserted the old
`max − min ≤ 1` balance rule after you replaced the build guard with a
chi-square detectability test. It now calls the same chi-square logic your
`r2_build_dataset.py` uses. 17/17 green here; 13/13 on `test_env.py`.

## Verifying

```bash
python tests/test_env.py            # 13
python tests/test_r2_estimators.py  # 17
python scripts/r2_audit.py --report # reproduces 29/60, Fleiss kappa 0.592
```
