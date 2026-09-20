# Reply to R2_REPLY_8 — 321 everywhere, frozen (Adit → Jayden)

Reads cold. Your §2 is done; nothing else was asked. Retrain on this and the
measurement phase is closed on both sides.

---

## 1. The artefact

`data/dialogue_measured.npz`, `trained_on: audited_321_contested_dropped`.
Both exclusions on every tier — the three-rater audit (29 cores) and the
contested `already_satisfied` receptacle forms — so all four tiers are scored
on the same 321 items and `n_samples` is a uniform 107 × 3. `subsets` now
holds three entries: `audited_321_contested_dropped` (trained on),
`audited_348` (audit only, your reply 7 spec) and `all_660_contested_dropped`
(pre-audit). Figures regenerated on the 321.

| tier | k=2 | k=3 | k=4 | pooled | 95 % CI (29 cores) | γ | γ CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `think_tiny` | 0.495 | 0.393 | 0.327 | **0.405** | [0.346, 0.467] | **0.867** | [0.827, 0.919] |
| `think_big` | 0.607 | 0.523 | 0.449 | **0.526** | [0.415, 0.636] | **0.638** | [0.547, 0.736] |
| `offload` | 0.804 | 0.776 | 0.794 | **0.791** | [0.703, 0.873] | **0.444** | [0.319, 0.596] |
| `think_big_q8` (ablation, meta only) | 0.570 | 0.495 | 0.374 | 0.480 | [0.369, 0.592] | 0.657 | [0.573, 0.750] |

Spread **0.386**, on one item set. `offload` and `think_big_q8` are unchanged
from the 348 freeze, as they must be — they were already on these 321.
`already_satisfied` is down to 9 items (3 cores × the *away* form × 3 k) and
`docs/r2_measurements.md` §3G now says nothing may be cited from it.

Gates: `test_env.py` 13/13, `test_r2_estimators.py` 18/18. Every number
above is in §3G with the 348 and 660 columns beside it, and §3G also records
your §1 resolution (the 0.537 / 0.623 came from the two-rater verdicts) and
your §3 point about this being the fourth silent failure of the same shape —
phrased for the methods section, as you asked.

---

## 2. Delta (drop over the tree; paths repo-relative)

| file | what |
| --- | --- |
| `data/dialogue_measured.npz` | **retrain on this** |
| `data/r2_scenarios.jsonl` | 321, ids unchanged |
| `data/r2_scenarios_348_audit_only.jsonl` | the reply-7 set, kept under its real name |
| `data/r2_analysis_audited_321.json` | analysis behind the artefact |
| `data/r2_analysis_audited_348.json` | **renamed** from `r2_analysis_audited.json` — delete the old name on your side |
| `data/r2_scenarios_report.json` | shortcut guards on the 321: position/colour undetectable, target-is-longest at chance |
| `docs/r2_measurements.md` | §3G rewritten for 321 |
| `results/figures/r2_fig{1,2,3}_*.png` | regenerated |

No script changed this round. Reproduce:

```
python scripts/r2_build_dataset.py --audit data/r2_core_audit.json --drop-unaudited --drop-contested
python scripts/r2_analyze.py --raw data/r2_raw_all.csv.gz --scenarios data/r2_scenarios.jsonl --out data/r2_analysis_audited_321.json --figures
python scripts/r2_freeze.py --analysis data/r2_analysis_audited_321.json --subset-name audited_321_contested_dropped \
    --reference-analysis audited_348=data/r2_analysis_audited_348.json \
    --reference-analysis all_660_contested_dropped=data/r2_analysis_real.json \
    --audit data/r2_core_audit.json --asr-wer 0.033724146224146226
```

---

## 3. Status

| # | Item | State |
| --- | --- | --- |
| — | 321-everywhere re-freeze | **done — this delta** |
| — | R1 retrain on it | you |

Nothing outstanding on my side. And thank you for §5's last paragraph — the
corrections ran both ways, and the chapter should say that too.
