# CogniWatt — R1 Training Stability Study

**Owner:** Jayden (R1) · **Scripts:** `scripts/r1_sweep.py`, `scripts/r1_stability.py`
**Artefacts:** `results/stability_runs.csv`, `results/stability_arms.csv`,
`results/figures/fig6_stability.png`

---

## 1. Why this study exists

`docs/r1_report.html` reports the CogniWatt headline (PPO 0.95 success vs a 0.89
hand-engineered heuristic) **from converged runs only**, and states the reason
plainly:

> 6 of 10 non-ablation runs converge; 4 collapse to ~0.34. Collapse is
> independent of seed and of the question price, and raising the entropy bonus
> does not rescue it. […] This is the honest limitation of the current result.

A headline conditioned on discarding 40% of runs is not a defensible headline.
Either the collapse is real — in which case it needs to be characterised and
priced — or it is an artefact of a configuration that has since changed, in
which case the limitation should be retired rather than repeated. This study
decides which.

Two candidate explanations were on the table in the report:

1. **Gradient variance.** The batch is small (16 envs × 64 steps = 1024 steps
   per update). Larger batches should reduce the chance of an early bad update
   pushing the policy into the uninformed basin.
2. **Under-exploration.** Escaping the uninformed policy requires discovering a
   multi-step evidence→delivery chain whose payoff is stochastic. If entropy
   decays before that chain is found, the policy never sees the reward that
   would justify buying information, and the uninformed mode is absorbing.

A third possibility only became visible while setting the study up (§2.1).

### 1.1 The confound nobody had noticed

`results/runs/ff_2m/history.json` — the run the reported headline comes from —
was trained with `ent_coef = 0.01`. The current default in `scripts/train.py` is
**`0.12`**, raised during the hardware recalibration because realistic roofline
costs made early exploration catastrophic (a single Pi 5 inference can consume
half of both episode budgets).

So the reported "4 of 10 collapse" was measured under a configuration the
project **no longer uses**. If the collapse rate tracks `ent_coef`, then the
open limitation was already fixed as a side effect of the cost recalibration,
and nobody checked. That makes `ent_coef` an arm of this study, not a footnote.

## 2. Method

All runs: 2M env steps, feedforward unless stated, `train.py` defaults
otherwise, one OMP thread per process, 4 concurrent processes on an M2 Air.

| Arm | Configuration | Seeds | Hypothesis tested |
| --- | --- | --- | --- |
| `base` | `--n-envs 16` (current default, `ent_coef 0.12`) | 8 | what is the collapse rate now? |
| `bigbatch` | `--n-envs 48` (3× batch) | 8 | gradient variance |
| `ent03` | `--ent-coef 0.03` | 4 | intermediate exploration |
| `ent01` | `--ent-coef 0.01` (the `ff_2m` setting) | 8 | reproduce the historical collapse |
| `gru` | recurrent policy, `--n-envs 16` | 4 | does recurrence help or hurt stability? |

### 2.1 Classifying a run

A run is **collapsed** if its final success — the mean over the last 15 log
records — is below **0.60**. The two modes reported are ~0.34 and ~0.95, so the
threshold sits in empty space; `report_gap()` checks that claim directly by
counting runs that land in the 0.45–0.80 band. If that band is populated, the
outcome is a noisy continuum rather than a bimodal one and the collapse-rate
framing would be the wrong analysis.

Convergence rates are reported with **Wilson 95% intervals**, which stay correct
at 0/n and n/n where the normal approximation degenerates.

### 2.2 Early warning

For each run we record mean policy entropy over the first 300k steps and test
whether it separates the two modes (Welch *t*). If it does, collapse is
detectable at ~1/7 of the cost of a full run, which is a practical result: the
sweep driver can abandon a doomed run instead of paying for it.

## 3. Results

32 runs, 2M steps each. **9 collapsed, 23 converged, and not one landed in
between** — 0/32 in the 0.45–0.80 band. The outcome really is bimodal, so the
collapse-rate framing is the right analysis rather than an artefact of the
threshold.

| Arm | `ent_coef` | batch | converged | Wilson 95% CI | success (converged) | early entropy |
| --- | --- | --- | --- | --- | --- | --- |
| `ent01` | 0.01 | 16 | **1/8** | [0.02, 0.47] | 0.877 | 0.20 |
| `ent03` | 0.03 | 16 | **2/4** | [0.15, 0.85] | 0.893 | 0.43 |
| `base` | 0.12 | 16 | **8/8** | [0.68, 1.00] | 0.914 | 1.19 |
| `bigbatch` | 0.12 | 48 | **8/8** | [0.68, 1.00] | 0.907 | 1.34 |
| `gru` | 0.12 | 16 (recurrent) | **4/4** | [0.51, 1.00] | 0.915 | 1.19 |

### 3.1 The collapse is the entropy coefficient, and nothing else

Convergence rate is monotone in `ent_coef` — 1/8 → 2/4 → 8/8 — and every other
factor we varied left it at 8/8. The collapsed runs are not merely worse; they
are the *uninformed policy*: final success 0.35–0.37 against a `no_think`
baseline of 0.384, with final policy entropy of **0.001**. The policy has become
deterministic and stopped buying information altogether.

This reproduces the phenomenon the report describes, and identifies its cause.
`ff_2m` — the run the published headline comes from — used `ent_coef 0.01`,
where only 1 run in 8 survives. It was the lucky seed.

**The report's stated diagnosis is therefore backwards.** It says:

> raising the entropy bonus does not rescue it

Raising the entropy bonus is precisely what rescues it. The recalibration that
took `ent_coef` from 0.01 to 0.12 (because roofline-accurate costs made early
exploration catastrophic) fixed the convergence problem as a side effect, and
the limitation was never re-measured against the new default.

### 3.2 Batch size was not the problem

`bigbatch` triples the batch to 48 envs × 64 steps and changes nothing:
0.907 vs 0.914, 8/8 either way. The gradient-variance hypothesis is falsified.
Worth stating explicitly, because it is the intuitive explanation and it is
wrong — the collapse is an exploration failure, not an estimation-noise
failure. (`bigbatch` does take ~2× as many env steps to first reach 0.9, since
tripling the batch at a fixed step budget cuts the number of gradient updates
by 3×.)

### 3.3 Collapse is predictable after 15% of a run

Mean policy entropy over the first 300k steps separates the two modes almost
perfectly:

> converged **1.146** (n=23) vs collapsed **0.223** (n=9), Welch *t* = **14.37**

There is no overlap: the highest early entropy among collapsed runs (0.34) sits
far below the lowest among converged runs (0.86, and every `ent_coef 0.12` run
is above 1.15). A doomed run is identifiable after ~45 s of training rather than
~5 min. This is a practical result — a sweep driver can kill and reseed on the
early-entropy signal instead of paying for 2M steps to learn nothing.

### 3.4 Recurrence buys nothing

The GRU ablation had never been run; every reported CogniWatt number uses
`--no-recurrent`. It is now measured:

| | final success | sd | median wall time |
| --- | --- | --- | --- |
| feedforward (`base`) | 0.914 | 0.005 | **4.8 min** |
| recurrent (`gru`) | 0.915 | 0.003 | **16.2 min** |

Identical accuracy for **3.4× the compute**. This is the direct empirical
confirmation of the design claim in `agents/ppo.py`: the Bayesian belief carried
in the observation is an approximately sufficient statistic of the interaction
history, so the POMDP is nearly Markov by construction and there is no residual
history for a GRU to exploit. The belief filter does the work recurrence would
otherwise have to learn — which is the argument for having a principled belief
filter at all, now with a number attached.

*(Wall times for `base`, `bigbatch`, `gru` and `ent03` were all collected at
4 concurrent processes on the same M2 Air and are comparable. `ent01` overlapped
with a second sweep at 8-way concurrency; its wall times are not comparable and
no timing claim is made from them.)*

## 4. What changes in the report

1. **Retire the open limitation.** "6 of 10 non-ablation runs converge" is no
   longer true of the shipping configuration: it is 20/20 at `ent_coef 0.12`
   across three different architectures and batch sizes. Replace the caveat with
   this study.
2. **Report the headline unconditionally.** It no longer rests on discarding
   collapsed runs, because at the current default there are none.
3. **Correct the diagnosis.** The report says raising the entropy bonus does not
   rescue the collapse. It does; that is the whole finding.
4. **Add the recurrence ablation** (§3.4) — it was listed as future work and is
   now done, and it retroactively justifies every run being feedforward.
5. **Add the λ_Q caveat's real cause.** The report notes the question-price sweep
   is "only meaningful for λ_Q ≤ 3.5, since higher prices collapse training."
   Those sweeps ran at `ent_coef 0.01`, so the collapses were very likely the
   entropy artefact rather than the question price. The sweep is being rerun at
   the current default; if the high-λ_Q points now train, that caveat is
   retired too.

## 5. Reproducing

```bash
python3 scripts/r1_sweep.py --arms base bigbatch gru ent03 ent01 --jobs 4
python3 scripts/r1_stability.py
```

Outputs `results/stability_runs.csv` (one row per run),
`results/stability_arms.csv` (one row per arm) and
`results/figures/fig6_stability.png`.

---

## 6. Follow-on: entropy annealing, and why the old headline is not comparable

### 6.1 A single entropy coefficient cannot do both jobs

§3 establishes that `ent_coef 0.12` is required for reliable convergence. The
regenerated evaluation shows what it costs. Against the strong `greedy_value`
heuristic, on 1000 paired episodes:

| | success | return ± se | energy | queries/episode |
| --- | --- | --- | --- | --- |
| `ppo[main_s0]` (`ent 0.12`) | 0.919 | **−1.43 ± 0.20** | 182 J | 4.3 |
| `greedy_value` | 0.901 | **−0.43 ± 0.24** | 166 J | 2.2 |

The learned policy wins on success and **loses on the objective it optimises**.
It buys roughly twice the information the heuristic does; the extra energy,
latency and clarifying questions cost more reward than the accuracy is worth.
That is exactly what a permanently high entropy bonus buys — a policy that is
still deliberately stochastic at convergence and cannot sharpen.

So the two results are one trade-off:

| `ent_coef` | convergence | converged policy |
| --- | --- | --- |
| 0.01 | 1/8 | sharp, efficient |
| 0.12 | 8/8 | reliable, wasteful |

Neither endpoint is reportable. `PPOConfig.ent_coef_final` anneals linearly from
`ent_coef` to a final value, asking each phase for only the thing it is good at.
`None` preserves the old constant behaviour exactly.

### 6.2 It works

`entanneal` (0.12 → 0.01, 8 seeds): **8/8 converged**, final policy entropy
0.383 (vs 0.882 held constant, 0.090 at a constant 0.01).

| | success | return | energy | queries/episode |
| --- | --- | --- | --- | --- |
| `ppo[entanneal_s0]` | 0.901 | **−0.27** | 160 J | 2.1 |
| `ppo[main_s0]` | 0.910 | −1.55 | 180 J | 4.4 |
| `greedy_value` | 0.901 | −0.43 | 166 J | 2.2 |

Annealing keeps perfect convergence, cuts queries 4.4 → 2.1 and energy 180 J →
160 J, and moves return from −1.55 to −0.27. The policy has learned to stop
buying evidence it does not need — which is the behaviour the project is
*about*.

**But it now ties the heuristic on success** (0.901 vs 0.901 train, 0.890 vs
0.892 held out) rather than beating it. Under the corrected reward the learned
policy roughly matches a competent hand-engineered agent; it does not dominate
it. A multi-seed evaluation is required before stating this either way.

### 6.3 The old headline was measured under a different reward function

`results/runs_legacy/ff_2m/policy.pt` stores the `RewardConfig` it was trained
with. Diffing it against the current one:

| key | `ff_2m` | current |
| --- | --- | --- |
| `lambda_cloud` | *(absent)* | **0.5** |

`lambda_cloud` was added during the hardware recalibration, because once
roofline costs were realistic a cloud call strictly dominated on-device
inference and the optimal policy degenerated to "always offload". **`ff_2m` was
trained and evaluated in the regime where cloud calls were free** — and the
legacy evaluation shows it making 2.67 offload calls per episode, precisely the
behaviour now priced at 0.5 each.

Consequences, and they are not small:

1. **The 0.950 / +1.69 headline is not comparable to any current number.** The
   reward function changed, so returns are on a different scale and success
   rates reflect a different task. It is not a regression, it is a different
   experiment.
2. **`greedy_value` silently absorbed the change and the learned policy could
   not.** The heuristic reads `env.cfg.lambda_cloud` at decision time, so it
   re-optimises for whatever reward it is handed. A trained network cannot —
   it has to be retrained. Most of the apparent 0.950 → 0.910 drop is this.
3. **Any future change to `RewardConfig` invalidates every stored policy.**
   Checkpoints record their reward config; nothing checks it on load.
   `evaluate_policy.py` should refuse to evaluate a policy whose stored
   `RewardConfig` differs from the one it is being evaluated under, or at
   minimum print the diff. Recommended before the next round of results.

### 6.4 The central claim, re-established under the corrected reward

Evaluating all 8 annealed seeds (600 paired episodes each, mean ± se **over
seeds**, which is the correct unit of analysis — se over episodes within a
single seed understates the variance that matters):

| split | agent | success | return | energy | queries/ep |
| --- | --- | --- | --- | --- | --- |
| train | PPO (annealed, 8 seeds) | **0.9087 ± 0.0039** | **−0.300 ± 0.072** | 163.5 J | 2.19 |
| train | `greedy_value` | 0.8983 | −0.584 | 168.6 J | 2.17 |
| held-out | PPO (annealed, 8 seeds) | **0.8983 ± 0.0025** | **−0.465 ± 0.056** | 164.3 J | 1.83 |
| held-out | `greedy_value` | 0.8917 | −0.638 | 167.9 J | 2.00 |

**The claim survives.** The learned policy beats the strong hand-engineered
heuristic on both success and return, on training and held-out devices, with
the margin exceeding 2.5 standard errors in every case (train success
*t* ≈ 2.7, train return *t* ≈ 4.0; held-out *t* ≈ 2.6 and 3.1). Every one of
the 8 seeds converged, so this is the whole population of runs, not a
survivor-selected subset.

**The margin is ~1 point of success, not the ~6 points previously reported.**
The old margin was measured when cloud calls were free (§6.3); pricing them
removes most of it. This is the honest number and it should replace the old one.

**The shape of the win changed, and it changed for the better.** The old
learned policy won by spending more. The annealed policy wins at *equal
information spend* — 2.19 queries per episode against the heuristic's 2.17, and
slightly less energy. It is not buying more certainty; it is buying the same
amount of certainty from a better-chosen portfolio of channels, and choosing
that portfolio per device. For a project about the economics of cognition,
"same budget, better allocation" is a stronger claim than "better outcome for
more money", and it is the claim the environment was built to test.

Two supporting results, each currently on one seed and worth repeating across
seeds before they go in the report:

* device conditioning helps ~3× more on held-out hardware than on training
  hardware (ablating it costs 1.6 points held-out vs 0.5 points on train),
  which is the signature it is designed to have;
* the annealed policy shifts its channel mix by device — on `jetson_nano_2gb`,
  where the 3B model does not fit, it drops `think_big` to essentially zero and
  reallocates to cloud and dialogue.

### 6.5 Recommended guard

Checkpoints store the `RewardConfig` they were trained under, and nothing checks
it at load time — which is why a reward change silently invalidated the reported
results for weeks. `scripts/evaluate_policy.py` should diff the stored config
against the evaluation config and refuse, or at minimum warn loudly, on a
mismatch. This is a five-line change and it would have caught §6.3 immediately.

---

## 7. Open modelling issue: the portfolio is currently cloud-dominated

**Script:** `scripts/r1_cloud_price.py` · **Data:** `results/cloud_price.csv`

CogniWatt's central claim is that the optimal behaviour is a *portfolio* across
evidence channels rather than a single best channel. Under the current cost
model it is not, quite:

| split | agent | tiny | big | cloud | ask | local share |
| --- | --- | --- | --- | --- | --- | --- |
| train | `ppo[entanneal_s0]` | 0.28 | 0.05 | 1.43 | 0.34 | **16%** |
| train | `greedy_value` | 0.25 | 0.17 | 1.45 | 0.28 | 20% |
| held-out | `ppo[entanneal_s0]` | 0.08 | 0.00 | 1.35 | 0.34 | **5%** |

Both the learned and the scripted agent draw the large majority of their
evidence from the cloud. This is not a learning failure — the two agents agree,
and the heuristic re-derives it from first principles every step — it is a
property of the cost model. The roofline recalibration made local inference
genuinely expensive (`think_big` = 1591 ms / 26 J on the M2 Air) against
140 ms / 0.8 J for a cloud round trip. `lambda_cloud = 0.5` prevents the fully
degenerate "always offload" policy the report describes, but not the weaker
version of the same problem: the on-device tiers, which are the *edge* half of
an edge-AI project, carry 5–20% of the load.

Because `GreedyValueAgent` reads `env.cfg` at decision time, the price can be
swept without retraining, which maps the structure of the optimal portfolio:

| `lambda_cloud` | success | local share | cloud share | asks |
| --- | --- | --- | --- | --- |
| 0.0 | 0.890 | 8% | 81% | 0.27 |
| **0.5 (current)** | 0.902 | **18%** | 70% | 0.24 |
| 1.0 | 0.900 | 31% | 54% | 0.30 |
| 2.0 | 0.920 | **47%** | 37% | 0.44 |
| 4.0 | 0.922 | 53% | 9% | 1.04 |
| 8.0 | 0.927 | 55% | 6% | 1.04 |
| 16.0 | 0.930 | 57% | 4% | 1.05 |

**`lambda_cloud` between 1 and 2 is where the three-way portfolio actually
exists.** Below it the cloud dominates; above ~4 the cloud is priced out
entirely and the agent substitutes the human, which is a different degeneracy.

Two cautions:

1. **Returns are not comparable across rows.** Each row optimises a different
   objective, so the monotone decline in return is an artefact of adding cost,
   not evidence that a higher price is worse. Only portfolio structure and
   success can be compared across the sweep — the same trap as §6.3, and worth
   stating in the report since it is easy to misread this table.
2. **This must be argued, not tuned.** Picking `lambda_cloud` to make the
   portfolio look balanced would be assuming the conclusion. The defensible
   route is to derive it: what is the privacy and monetary cost of sending one
   scene description off-premises, expressed in the same units as the ~10 J a
   grid step of locomotion costs? The sweep says where the interesting regime
   is; the modelling argument has to say whether reality is in it.

**Recommendation:** derive `lambda_cloud` explicitly, and if the derived value
lands near 1–2, retrain and rerun. If it genuinely lands near 0.5, then the
honest finding is that under realistic 2024-class edge costs a home robot
*should* offload most of its reasoning — which is a perfectly publishable
result, but it is a different paper from the one currently written, and the
report's framing would need to change to match.
