# CogniWatt — Hardware Calibration (R1, inherited edge-sim workstream)

**Owner:** Jayden (R1). This documents the M2 Air anchor, the roofline cost
model, and the measured thermal curve that replace the placeholder Virtual Edge
Device numbers. The RTX 3060 compute anchor is Adit's (R2 §7b).

---

## 1. Why the placeholder numbers had to go

The original `_STUB_COST_TABLE` was a roofline *estimate* keyed off datasheet
peaks. Measuring the M2 Air showed it was optimistic by **10–17×** on compute,
because it implicitly assumed peak GEMM throughput. A grounding query is a
150-token prompt — a short, skinny matmul with poor occupancy — and reaches
nowhere near peak.

| Quantity | Datasheet peak | Measured (M2 Air) | Efficiency |
| --- | --- | --- | --- |
| GEMM, large square | 3.6 TFLOP/s | 2695 GFLOP/s | 75% |
| GEMM, **prefill shape** (L=150) | 3.6 TFLOP/s | **680 GFLOP/s** | **19%** |
| Streaming read bandwidth | 100 GB/s | 76.9 GB/s | 77% |

The 19% prefill efficiency is the number that matters, and using peak instead is
the single largest error in naive estimates.

**Effect on `think_big` (3B Q4) latency:**

| Device | placeholder | roofline (measured-anchored) |
| --- | --- | --- |
| M2 Air | 95 ms | **1591 ms** |
| Orin Nano | 260 ms | 4230 ms |
| Pi 5 | 900 ms | **32,400 ms** |
| RTX 3060 | 40 ms | 520 ms (until Adit measures it) |

On a Pi 5 a single 3B inference is 32 s and 218 J — over half of *both* episode
budgets. That is not a rounding change; it reshapes the optimal policy, and it
is why brute-force local inference (`always_big`) collapses from 0.74 to 0.57
success once real costs are in.

## 2. The roofline model

`cogniwatt/hardware/roofline.py`. Decoder inference splits into two phases with
different bottlenecks — conflating them is the usual failure:

```
PREFILL (compute bound):  t = 2 · P · L_prompt / (peak_FLOPS · eff_prefill)
DECODE  (memory bound):   t = L_gen · P · bytes_per_param / (peak_BW · eff_bw)
```

`eff_prefill = 0.19` and `eff_bw = 0.77` are the measured M2 efficiency factors,
applied to every device's datasheet peak. What is **measured** vs **modelled**,
stated plainly for the report:

- **Measured:** M2 Air compute + bandwidth efficiency; M2 Air thermal curve;
  (incoming) RTX 3060 per-call latency/energy from Adit.
- **Modelled:** all other devices, by applying the measured efficiency factors
  to datasheet peaks; radio energy for cloud offload (published Wi-Fi figures);
  actuation energy (explicit physical model).

## 3. Measured thermal throttle

`scripts/r1_throttle_run.py` ran a sustained 2048² GEMM on the fanless M2 Air
for 10 minutes. Throughput fell from **1421 GFLOP/s to a 690 GFLOP/s floor —
a 2.06× slowdown** — settling after ~7 minutes (`fig5_m2_throttle.png`).

This is the measurement the M2 anchor exists for: a passively cooled chassis
throttles observably, so `ThermalModel.max_slowdown` is now **2.06, measured**,
not assumed. (It happened to match the old assumed 2.0 almost exactly — a nice
independent sanity check on the original guess.) The per-burst signal was too
noisy to identify the thermal time constant precisely, so `tau_s` is set to a
defensible 300 s rather than the fragile fitted value; this is flagged in the
code.

A fan-cooled device (the RTX 3060) will barely throttle by comparison — a useful
contrast for the report, and a reason the two-anchor setup is worth having.

## 4. Two modelling fixes this surfaced

**Memory as a policy feature.** The held-out device must be able to *anticipate*
that the 3B model will not fit. That is a function of RAM, so the device feature
vector gained `memory_gb` (now 6-dim). Without it, zero-shot transfer to a
memory-limited device is asking the policy to predict something it cannot see.

**A cost for cloud offload.** Priced only on energy and latency, a cloud call
(140 ms / 0.8 J, 88% accurate) *strictly dominated* on-device inference — so the
optimal policy degenerated to "always offload" and the local tiers were dead
weight. Real home robots do not offload every scene description: it leaves the
house, and it is billed per call. Adding `lambda_cloud = 0.5` restores a
genuinely device-dependent local-vs-cloud decision.

## 5. A held-out device that is physically defensible

The previous held-out `coral_micro` profile was wrong: a Coral Edge TPU has 8 MB
of SRAM and no transformer decode support, so it cannot run a 0.5B LLM at any
speed — modelling it running one at 18 s was indefensible. Replaced with
**`jetson_nano_2gb`**, which genuinely has the property the held-out set needs:
after OS overhead ~1.2 GB is usable, so the 0.5B Q8 model (0.52 GB) fits and the
3B Q4 (1.77 GB) does not. Structurally, not just quantitatively, different from
every training device.

## 6. The RL problem got harder — and that is a result

Realistic costs made the penalty landscape far harsher: a single expensive
action can now blow half the budget, so early exploration generates
catastrophic returns and the policy collapses to the uninformed attractor unless
pushed hard to explore. The entropy coefficient had to rise from **0.01 to
0.12** to train reliably:

| ent_coef | success (500k steps) |
| --- | --- |
| 0.03 | 0.37 (collapsed) |
| 0.06 | 0.745 |
| 0.12 | 0.870 |

Two mechanisms keep it trainable: **budget-feasibility masking** (the runtime
never offers an inference the robot cannot afford to finish after — which is
what a real scheduler does), and the higher exploration pressure. This is worth
a paragraph in the report: *grounding a simulator in measured hardware costs is
not cosmetic; it materially changes the difficulty of the control problem.*

## 7. Artefacts

| File | What |
| --- | --- |
| `data/anchor_m2_air.json` | measured GEMM + bandwidth |
| `data/throttle_m2_air.json` | full throttle time series |
| `data/thermal_fit_m2.json` | fitted slowdown + time constant |
| `data/roofline_cost_table.json` | generated cost table, all devices |
| `results/figures/fig5_m2_throttle.png` | the throttle figure |
| `cogniwatt/hardware/roofline.py` | the model |
| `scripts/r1_profile_m2.py`, `r1_throttle_run.py` | reproducible measurement |

When Adit's `r2_rtx3060_profile.json` lands, `ved.apply_measured_profile()`
overrides the modelled 3060 costs with measured ones — one function call, no
other change.
