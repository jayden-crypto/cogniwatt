# CogniWatt — Audit of the Zero-Shot Transfer Claim

**Owner:** Jayden (R1) · **Data:** `results/eval/`, `results/eval_devcond16/`,
`results/eval_mask/`

---

## 1. The claim under audit

`README.md` and `docs/r1_report.html` state the project's second novelty claim as:

> …and then shows the resulting policy transfers **zero-shot to an edge device it
> has never seen, conditioned only on datasheet features.**

Two things are asserted: that transfer *happens*, and that datasheet-feature
conditioning is *why*. Both were tested. The first is true but empty; the second
is false.

## 2. Device conditioning does nothing (n = 16 per arm)

Ablating `condition_on_device` from the canonical annealed configuration, 16
seeds per arm, 400 paired episodes per seed:

| split | full method | no device conditioning | Δ | Welch *t* |
| --- | --- | --- | --- | --- |
| train | 0.8967 ± 0.0017 | 0.8962 ± 0.0017 | +0.05 pt | 0.19 |
| held-out | 0.8969 ± 0.0024 | 0.8956 ± 0.0018 | +0.13 pt | 0.42 |

**No effect on either split.** An earlier 4-seed run showed +0.67 pt held-out
(*t* = 0.91) and was suggestive enough to be worth powering properly; at 16
seeds it is gone. This is a clean example of why the ablations needed more than
one seed — the previously reported single-seed ablation numbers were noise
dressed as findings.

## 3. Why: the held-out split does not discriminate

The reason device conditioning cannot help is that there is no device-dependent
decision to make. Measured across every agent in the suite:

| agent | train | held-out | drop |
| --- | --- | --- | --- |
| `oracle` | 1.000 | 1.000 | 0.0 pt |
| `no_think` | 0.376 | 0.376 | 0.0 pt |
| `always_offload` | 0.831 | 0.831 | 0.0 pt |
| `always_big` | 0.558 | 0.556 | 0.2 pt |
| `always_tiny` | 0.496 | 0.490 | 0.6 pt |
| `always_ask` | 0.880 | 0.879 | 0.1 pt |
| `greedy_value` | 0.901 | 0.892 | 0.9 pt |
| `ppo[entanneal_s0]` | 0.911 | 0.904 | 0.7 pt |

**Nothing degrades.** `no_think` and `always_offload` are *device-blind by
construction* — they cannot adapt to hardware at all — and they score
identically on both splits. A held-out split on which a device-blind policy
loses nothing is not a test of hardware generalisation.

The learned policy's "zero-shot transfer" is therefore real in the trivial sense
(it does not degrade) and vacuous in the meaningful sense (neither does
anything else, including agents incapable of transfer).

## 4. Root cause: the cost model erased the distinction

The chain runs back to the project's most solid measurement:

1. The roofline recalibration priced on-device inference honestly — `think_big`
   at 1591 ms / 26 J on an M2 Air, 32 s on a Pi 5.
2. Against a 140 ms / 0.8 J cloud round trip, local inference is rarely the
   rational purchase. The learned policy issues **0.05 `think_big` calls per
   episode** (`docs/r1_cloud_price.md` §7.2).
3. `jetson_nano_2gb`'s distinguishing feature — the entire reason it was chosen
   as a held-out device — is that the 3B model does not fit on it.
4. **Removing a channel nobody uses changes nothing.** So the held-out device is
   behaviourally identical to a training device.
5. Hence: transfer is trivial, and device conditioning is inert.

Three separate claims — the portfolio, zero-shot transfer, and datasheet
conditioning — reduce to one root cause. That is worth stating plainly, because
each was previously reported as an independent result.

## 5. Was it ever a real test?

Partly. The *original* report showed `always_big` dropping 0.740 → 0.575 on the
old `coral_micro` held-out device, which is a genuine degradation. But that
profile was retired as physically impossible (a Coral Edge TPU cannot run a 0.5B
LLM at all), and the replacement `jetson_nano_2gb` is much closer to the
training distribution. The honest reading is that the transfer result was
already weak and became empty when the device profile was corrected — the
correction was right, and it removed the phenomenon.

## 6. What a real transfer test needs

A held-out device on which **device-blind policies measurably fail**. The
distinguishing feature must be one the optimal policy actually cares about, and
under the current cost model that is *not* whether the 3B model fits.

Candidates, in order of promise:

1. **Poor connectivity.** An offline robot has no cloud at any price, so the
   channel the policy actually relies on disappears. `net_quality_mean` is now a
   first-class parameter (`cogniwatt/interfaces/ved.py`) and a sweep is running.
   A held-out device combining low memory *and* poor connectivity would force
   `think_tiny` — the only remaining channel — and `always_offload` would
   collapse, which is exactly the discrimination the current split lacks.
2. **A much tighter energy budget.** `episode_energy_budget_mJ` is already
   per-device. A device that cannot afford both travel and reasoning forces a
   qualitatively different policy.
3. **A device where the human is unavailable** (patience 0), removing the other
   channel the policy leans on.

The test to apply before claiming transfer again: **does at least one scripted,
device-blind baseline degrade by a wide margin on the held-out split?** If not,
the split is not measuring generalisation.

## 7. Recommendation for the report

* **Withdraw** "conditioned only on datasheet features" — it is measured and
  false (§2).
* **Restate** the transfer claim as conditional: the policy does not degrade on
  unseen devices, *and neither does any baseline*, so the current split does not
  establish hardware generalisation.
* **Keep** `memory_gb` in the feature vector. It costs nothing, it is the right
  design for a device that *would* differ, and it will matter once the held-out
  split is fixed.
* **Fix the split** along §6.1 and re-run. This is a small change to
  `DEVICE_PROFILES` plus one retrain, and it converts a vacuous claim into a
  real one.

Reported as a negative result rather than quietly dropped — the same treatment
`capability_dropout` received in the original report.
