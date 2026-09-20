# CogniWatt — R1 Lab-by-Lab Plan and Deliverable Record

**Owner:** Jayden (R1 — RL & Environment, plus the inherited edge-hardware
workstream). Mirrors `docs/R2_HANDOFF.md` §10, which is Adit's equivalent.

Course requirements this document exists to satisfy:

* progress must be demonstrable **at every lab**, not only at the end;
* the output must be **individual and attributable** to each member;
* progress must be **substantive**, not incremental bookkeeping.

Adit's schedule was written down in the handoff; R1's was not — the work was
tracked as one large report instead, which makes per-lab individual progress
hard to evidence. This reconstructs it from the artefacts actually in the
repository, and plans the remainder.

---

## Part A — completed, with the artefact that evidences each

| Lab | Milestone | Individual R1 output | Evidence in repo |
| --- | --- | --- | --- |
| 1 | Problem formalisation | POMDP definition: state, action space, observation, reward, and the argument that partial observability is *semantic* only | `docs/lab1_mdp_formalization.md` |
| 2 | Environment core | `HomeGrid-Edge` — 8 floorplans, exact BFS navigation precomputed at import, hierarchical macro-actions, task generator | `cogniwatt/env/homegrid.py`, `env/tasks.py` |
| 3 | Belief filter + interfaces | Log-space Bayesian posterior over referents; the two frozen team contracts (`ved.py`, `dialogue.py`) with working placeholders so neither workstream blocks the other | `cogniwatt/env/belief.py`, `cogniwatt/interfaces/` |
| 4 | Baselines + test suite | 9 scripted agents including `GreedyValueAgent`, an information-value-per-Joule heuristic built deliberately strong; 13-test suite with no pytest dependency | `cogniwatt/agents/baselines.py`, `tests/test_env.py` |
| 5 | PPO from scratch | Recurrent PPO written from first principles (no SB3), action masking, potential-based shaping, GAE, device conditioning | `cogniwatt/agents/ppo.py` |
| 6 | First results + failure analysis | Learned policy beats the heuristic; three documented reward-hacking failures diagnosed and fixed (paralysis under free timeout, shaping that punished being better informed, argmax evaluation fixed-point) | `docs/r1_report.html` "How it actually went" |
| 7 | Hardware calibration | M2 Air anchor via `powermetrics` — 19% prefill efficiency vs datasheet peak, showing placeholder costs were optimistic by 10–17×; measured 2.06× thermal throttle on the fanless chassis; roofline extrapolation to Jetson / Pi / Coral; radio and actuation models | `docs/r1_hardware_calibration.md`, `cogniwatt/hardware/roofline.py`, `data/*.json`, `fig5_m2_throttle.png` |
| 8 | **Training stability study** | 40-run seed sweep identifying `ent_coef` as the sole cause of the reported bimodal convergence (1/8 → 2/4 → 8/8 as it rises); early-entropy early-warning test (Welch *t* = 14.4); recurrence ablation showing the GRU buys nothing at 3.4× the cost | `docs/r1_stability.md` §1–5, `scripts/r1_sweep.py`, `scripts/r1_stability.py`, `fig6_stability.png` |
| 9 | **Entropy annealing + reproducibility audit** | `PPOConfig.ent_coef_final`: converges on every seed *and* sharpens, cutting queries 4.4 → 2.1. Audit finding that the published results were generated under a 40-dim observation, a retired device and a reward without `lambda_cloud`; legacy artefacts quarantined and a reward-mismatch guard added to prevent recurrence | `docs/r1_stability.md` §6, `cogniwatt/agents/ppo.py`, `scripts/evaluate_policy.py` |
| 10 | **Cloud-price derivation + final results** | Monetary component of `lambda_cloud` derived from token counts and the reward scale (≈0.15); privacy component argued to be a policy parameter and swept; multi-seed ablations; regenerated report | `docs/r1_cloud_price.md`, `results/cloud_price.csv`, `docs/r1_results.md` |

## Part B — how each lab's work is presentable

Each row above is defensible as a standalone lab presentation, because each
answers a question rather than reporting effort:

* **Lab 6** — *"the agent learned to do nothing, and the reward function was
  why"*. Three concrete reward-hacking failures with the fix for each. This is
  the most presentable single lab in the project.
* **Lab 7** — *"the placeholder hardware costs were wrong by 10–17×, and here is
  the measurement"*. A real thermal curve off a fanless laptop.
* **Lab 8** — *"the training instability we flagged as an open limitation is one
  hyperparameter, and here is the dose-response"*. 40 runs, Wilson intervals,
  a falsified alternative hypothesis (batch size), and a negative result
  reported as such (recurrence).
* **Lab 9** — *"our published numbers could not be reproduced from our own code,
  and here is the guard that stops it happening again"*. Uncomfortable, and the
  strongest evidence of engineering maturity in the project.
* **Lab 10** — *"we do not get to choose the price of a cloud call, so here is
  what we can derive and what we must sweep"*.

## Part C — remaining work

| # | Item | Owner | Status |
| --- | --- | --- | --- |
| 1 | Multi-seed ablations under the canonical annealed config | R1 | running |
| 2 | `lambda_cloud` sweep trained at 5 prices × 3 seeds | R1 | running |
| 3 | Regenerate `docs/r1_report.html` | R1 | pending the two above |
| 4 | Integrate R2's `dialogue_measured.npz` and retrain on measured constants | R1 | **done** — trained on `audited_321_contested_dropped`, 6 seeds per condition |
| 5 | Integrate `r2_rtx3060_profile.json` via `apply_measured_profile` | R1 | **done** — measured costs replace the roofline on that device (`think_big` 519.6 ms → 34.0 ms, 48.9 J → 1.67 J); the 7B is refused as an anchor via their `valid_device_anchor` flag |
| 6 | Wire R2's 6×6 `r2_color_similarity.npy` into `tasks.COLOR_SIM` | R1 | **done, as a deletion** — their measurement showed errors do not track colour similarity (excess −0.00006 over 6 672 errors), so `COLOR_SIM` is now uniform and their perceptual matrix stays unwired |
| 7 | Human audit of the authored cores | R1 + R2 | **done** — 3 annotators, 29/60 cores retained, Fleiss κ 0.592 |
| 8 | Final report regeneration | R1 | **done** — `r1_report.html` and `r1_results.md` are generated from the CSVs |

**The critical path is closed.** Every dialogue constant is now measured:
`TIER_ACCURACY` and `EVIDENCE_DECAY` from 19 800 queries across three model
tiers on one llama.cpp stack, `DEFAULT_ASR_WER` (and the referent-recovery rate
that replaced it) from a Whisper pipeline over 721 synthesised clips, and the
colour matrix measured and then *removed* because the assumption behind it was
false. The headline is grounded in measurement, on a scenario set three humans
independently audited.

The one remaining limitation is stated rather than fixed: the `ask` channel's
accuracy is still an R1 placeholder (0.97, scaled by measured recovery). It is
the single number the degeneracy result is most exposed to, and it is recorded
as such in `meta_json` under `ask_provenance`.

## Part D — risk register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| R2 artefact never lands | The central novelty claim stays unmeasured; the project reads as a simulation study | Interfaces are frozen and fall back to placeholders, so R1 results stand alone. If R2 slips past Lab 9, report the dialogue constants explicitly as *assumed*, with a sensitivity sweep over `TIER_ACCURACY` in place of measurement |
| Portfolio is cloud-dominated at the derived price | Weakens the "portfolio" framing that motivates the whole action space | Report conditionally across the `lambda_cloud` sweep; the agent-beats-heuristic claim is checked for invariance to it (`docs/r1_cloud_price.md` §5) |
| Reward config changes again and silently invalidates stored policies | Repeat of the Lab 9 audit finding | Guard added — `evaluate_policy.py` refuses on a mismatch |
| Tier-2 AI2-THOR port eats measurement time | Both workstreams miss their real deliverables | Already cut to a stretch goal; decide finally at Lab 8 and do not revisit |
