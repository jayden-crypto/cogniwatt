# Project submission — CogniWatt

Deadline: **10 November 2026** (strict; no late submissions).
One submission per group; max 3 students.

---

## Required details

| Field | Value |
| --- | --- |
| **Project title** | CogniWatt — Learning the Economics of Cognition for Edge Robots |
| **Group member 1** | Adit Uppal — 102373004 |
| **Group member 2** | Karanveer Singh — 102323049 |
| **GitHub repository** | https://github.com/jayden-crypto/cogniwatt |
| **Dataset** | https://github.com/jayden-crypto/cogniwatt/tree/main/data |
| **Project report (PDF)** | `submission/CogniWatt_Report.pdf` — 32 pages |

> Workstream split: **R1 — Karanveer Singh** (RL, environment, hardware
> calibration), **R2 — Adit Uppal** (dialogue measurement, SLM profiling,
> scenario audit). Two-person group; the course permits up to three.

---

## One-line summary

A single RL policy for a simulated household robot decides not just *what to do*
but *how much compute to spend thinking* and *when to interrupt the human*, under
an energy, latency and thermal model calibrated to real edge devices. The
headline finding inverts the project's own premise: with a cooperative human the
problem is degenerate (every agent ties at 0.968 success), but offline and alone
the learned policy beats the strongest heuristic by **19.1 points of success**.

## What the report covers

**Part I — R1 (pp. 2–19):** headline result, training-stability study, entropy
annealing, ablations, cloud-price derivation, a reproducibility finding,
hardware grounding, and four silent failures.

**Part II — R2 (pp. 20–32):** "The Saturation Ceiling" — 28,920 measured
responses over 660 scenarios and 5 model configurations, three-annotator audit,
and what each model tier can actually do.

## Dataset notes

`data/` holds 39 tracked files (~7 MB), the artefacts every number in the report
is computed from:

- `dialogue_measured.npz` — the frozen evidence artefact the policy trains on
  (`audited_321_contested_dropped`: all four tiers measured on the same 321
  human-verified scenarios)
- `r2_scenarios.jsonl` + `r2_raw_*.csv.gz` — scenario set and raw model responses
- `r2_audit/` — the three independent annotator passes
- `anchor_m2_air.json`, `thermal_fit_m2.json`, `throttle_m2_air.json`,
  `roofline_cost_table.json` — measured hardware calibration

## Reproducing the headline

```
python3 scripts/run_baselines.py 600
python3 scripts/train.py --steps 2000000 --no-recurrent --tag main
python3 scripts/evaluate_policy.py --tags main -n 1000
```
