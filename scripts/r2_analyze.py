#!/usr/bin/env python3
"""Derive accuracy, evidence decay, position bias and the failure taxonomy.

    python scripts/r2_analyze.py --raw data/r2_mock_responses.csv.gz
    python scripts/r2_analyze.py --raw data/r2_raw_responses.parquet --figures

Reads only the raw-response table written by `r2_measure.py`, so it is free to
re-run and never re-pays for inference.

THE ESTIMATORS
--------------
**Accuracy (§6.1)** uses *only the first repeat* of each scenario. Pooling all N
repeats would contaminate accuracy with the repeat structure that §6.2 is about
to measure separately, and would also inflate n by a factor of N without adding
a single independent observation.

**Clustering.** Scenarios generated from one authored core are permutations of
one commonsense contrast, so they are not independent. Every interval here is a
*cluster bootstrap over cores*. The naive Wilson interval is printed alongside
purely to show how much it over-claims -- quoting the naive one would be the
easiest way to report a false precision.

**Evidence decay (§6.2)**, the headline result:

    1. correctness c[m, i] for scenario m, repeat i = 1..N
    2. one-way ANOVA intra-class correlation across repeats within a scenario
    3. effective independent draws   n_eff(n) = n / (1 + (n-1) rho)
    4. gamma by least squares against R1's geometric model:
           minimise  sum_n [ (1 - gamma^n)/(1 - gamma) - n_eff(n) ]^2

`rho -> 1` means the model always answers the same way, so repeating it buys
nothing: gamma -> 0. `rho -> 0` means repeats are independent draws:
gamma -> 1. gamma is therefore "how much genuinely new information the next
query carries", which is exactly what the env's `evidence_weight` consumes.

A NOTE ON THE EXPECTED DIRECTION OF GAMMA
-----------------------------------------
docs/R2_HANDOFF.md §6.2 says gamma should be *higher* for bigger models, but
R1's own placeholder table in `cogniwatt/interfaces/dialogue.py` has
OFFLOAD = 0.35, *below* THINK_BIG = 0.55. Those two claims disagree, and the
disagreement is substantive: a strong model may well be confidently and
consistently right, which is high rho and therefore low gamma. This script
takes no side -- it reports the measured value and flags which of the two
predictions the data supports.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_RAW = os.path.join("data", "r2_raw_responses.parquet")
DEFAULT_OUT = os.path.join("data", "r2_analysis.json")
FIG_DIR = os.path.join("results", "figures")
N_BOOT = 2000


# ---------------------------------------------------------------------------
# intervals
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Naive Wilson score interval -- assumes n independent trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((c - h) / d), float((c + h) / d))


def cluster_bootstrap(df: pd.DataFrame, stat, cluster: str = "core_id",
                      n_boot: int = N_BOOT, seed: int = 0
                      ) -> Tuple[float, float, float]:
    """(point, lo, hi) resampling whole clusters with replacement.

    Resampling rows would treat 12 permutations of one contrast as 12
    independent facts about the model, which they are not.
    """
    point = stat(df)
    groups = [g for _, g in df.groupby(cluster, sort=True)]
    if len(groups) < 2:
        return float(point), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        s = stat(pd.concat([groups[i] for i in idx], ignore_index=True))
        if s == s:  # drop NaN draws
            vals.append(float(s))
    if not vals:
        return float(point), float("nan"), float("nan")
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _acc(df: pd.DataFrame) -> float:
    """Accuracy with format violations scored as *wrong*.

    A model that cannot emit an index has failed the task. Dropping those rows
    would quietly grade the tiny model on the subset where it behaved, which is
    the most flattering possible denominator.
    """
    if len(df) == 0:
        return float("nan")
    return float((df["parse_status"].eq("ok") & df["correct"].eq(1)).mean())


# ---------------------------------------------------------------------------
# evidence decay
# ---------------------------------------------------------------------------
def icc_oneway(mat: np.ndarray) -> float:
    """ICC(1) across repeats within a scenario, by one-way ANOVA.

    `mat` is (M scenarios, N repeats) of 0/1 correctness. Returns rho in
    [0, 1); a negative estimate (repeats *more* variable than independent) is
    clipped to 0, which is the conservative direction -- it makes the model
    look more informative per repeat, not less.
    """
    M, N = mat.shape
    if M < 2 or N < 2:
        return float("nan")
    means = mat.mean(axis=1)
    msb = N * float(np.var(means, ddof=1))
    msw = float(np.mean(np.var(mat, axis=1, ddof=1)))
    denom = msb + (N - 1) * msw
    if denom <= 0:
        # Every scenario constant and identical -> perfectly deterministic.
        return 1.0 if msw == 0 else float("nan")
    return float(np.clip((msb - msw) / denom, 0.0, 1.0))


def n_eff(n: np.ndarray, rho: float) -> np.ndarray:
    return n / (1.0 + (n - 1.0) * rho)


def fit_gamma(rho: float, N: int) -> float:
    """Least-squares gamma matching R1's geometric saturation model."""
    from scipy.optimize import minimize_scalar

    if not np.isfinite(rho):
        return float("nan")
    ns = np.arange(1, N + 1, dtype=float)
    target = n_eff(ns, rho)

    def loss(g: float) -> float:
        g = float(np.clip(g, 1e-6, 1 - 1e-6))
        return float(np.sum(((1.0 - g ** ns) / (1.0 - g) - target) ** 2))

    r = minimize_scalar(loss, bounds=(1e-4, 1 - 1e-4), method="bounded")
    return float(np.clip(r.x, 1e-4, 1.0))


def correctness_matrix(df: pd.DataFrame, repeats: int) -> Optional[np.ndarray]:
    """(M, N) correctness, keeping only scenarios with the full N repeats."""
    d = df.copy()
    d["c"] = (d["parse_status"].eq("ok") & d["correct"].eq(1)).astype(float)
    piv = d.pivot_table(index="scenario_id", columns="repeat", values="c")
    piv = piv.reindex(columns=range(repeats))
    piv = piv.dropna(axis=0, how="any")
    return piv.to_numpy() if len(piv) >= 2 else None


def decay_for(df: pd.DataFrame, repeats: int, seed: int = 0
              ) -> Dict[str, object]:
    mat = correctness_matrix(df, repeats)
    if mat is None:
        return {"rho": float("nan"), "gamma": float("nan"), "n_scenarios": 0}
    rho = icc_oneway(mat)
    gamma = fit_gamma(rho, repeats)

    # Cluster bootstrap over cores: resample cores, rebuild the matrix, refit.
    cores = {cid: g for cid, g in df.groupby("core_id", sort=True)}
    keys = sorted(cores)
    rng = np.random.default_rng(seed)
    gs, rs = [], []
    for _ in range(400):  # 400 is enough for a percentile CI and refits are slow
        pick = rng.integers(0, len(keys), len(keys))
        sub = pd.concat([cores[keys[i]] for i in pick], ignore_index=True)
        # scenario_id repeats across resampled cores; make them unique first.
        sub = sub.copy()
        sub["scenario_id"] = sub["scenario_id"] + "#" + sub.groupby(
            ["scenario_id", "repeat"]).cumcount().astype(str)
        m = correctness_matrix(sub, repeats)
        if m is None:
            continue
        r = icc_oneway(m)
        if np.isfinite(r):
            rs.append(r)
            gs.append(fit_gamma(r, repeats))
    ns = np.arange(1, repeats + 1, dtype=float)
    return {
        "rho": rho,
        "rho_ci": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]
                  if rs else [float("nan")] * 2,
        "gamma": gamma,
        "gamma_ci": [float(np.percentile(gs, 2.5)), float(np.percentile(gs, 97.5))]
                    if gs else [float("nan")] * 2,
        "n_scenarios": int(mat.shape[0]),
        "n_repeats": int(mat.shape[1]),
        "n_eff_curve": [float(x) for x in n_eff(ns, rho)],
        # Two views of the same ceiling, reported separately because they come
        # from different parameterisations, and their agreement is the check
        # that the geometric fit is faithful:
        #   n_eff(n) -> 1/rho             as n -> inf   (the ICC asymptote)
        #   sum_n gamma^(n-1) = 1/(1-gamma)             (R1's geometric bound)
        # Reporting 1/(1-rho) -- as this did originally -- is neither of them.
        "n_eff_asymptote": float(1.0 / rho) if rho > 0 else float("inf"),
        "geometric_bound": (float(1.0 / (1.0 - gamma))
                            if np.isfinite(gamma) and gamma < 1 else float("inf")),
    }


# ---------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------
def classify(row) -> str:
    """One error label per response. See docs/R2_HANDOFF.md §9.

    Note what is deliberately absent: an 'attribute confusion' class. Colour in
    this dataset is assigned at random and carries no information about the
    answer, so a colour-based confusion would be indistinguishable from noise.
    Attribute confusability is measured separately, on embeddings, by
    r2_similarity.py.
    """
    st = row["parse_status"]
    if st == "unparseable":
        return "format_hedge"
    if st == "out_of_range":
        return "format_out_of_range"
    if st == "digit_answer":
        # Replied with a digit under letter labelling. Kept separate because it
        # is the signature of a model with a prior on the token, not on the
        # option -- see the note in prompts.py.
        return "format_digit"
    if row["correct"] == 1:
        return "correct"
    if row["display_idx"] == 0:
        return "position_first"
    if row["display_idx"] == row["k"] - 1:
        return "position_last"
    return "pragmatic"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--scenarios", default=None,
                    help="restrict to the scenario ids in this dataset file. "
                         "The raw table deliberately keeps responses for "
                         "dropped scenarios (they are the evidence for the "
                         "exclusion); this makes the dataset the source of "
                         "truth for what counts.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    path = args.raw
    if not os.path.exists(path):
        alt = path.replace(".parquet", ".csv.gz")
        if os.path.exists(alt):
            path = alt
        else:
            print(f"no raw responses at {args.raw}", file=sys.stderr)
            return 1
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if args.scenarios:
        from cogniwatt.dialogue.schema import load_scenarios
        keep = {sc.id for sc in load_scenarios(args.scenarios)}
        before = len(df)
        df = df[df["scenario_id"].isin(keep)]
        print(f"restricted to {len(keep)} scenarios from {args.scenarios}: "
              f"{before} -> {len(df)} rows")
    repeats = int(df["repeat"].max()) + 1
    tiers = sorted(df["tier"].unique())
    print(f"{len(df)} rows  tiers {tiers}  repeats {repeats}  <- {path}\n")

    out: Dict[str, object] = {"source": path, "n_rows": int(len(df)),
                              "tiers": tiers, "repeats": repeats}

    # -- accuracy ----------------------------------------------------------
    first = df[df["repeat"] == 0]
    ks = sorted(first["k"].unique())
    acc: Dict[str, Dict] = {}
    print("ACCURACY (first repeat only; format violations scored wrong)")
    print(f"{'tier':<12}{'k':>3}{'acc':>8}{'cluster 95% CI':>22}"
          f"{'naive Wilson':>20}{'n':>6}{'cores':>7}")
    for t in tiers:
        acc[t] = {}
        for k in list(ks) + ["pooled"]:
            sub = first[first["tier"] == t] if k == "pooled" else \
                first[(first["tier"] == t) & (first["k"] == k)]
            p, lo, hi = cluster_bootstrap(sub, _acc, seed=args.seed)
            wl, wh = wilson(int(round(p * len(sub))), len(sub))
            acc[t][str(k)] = {"acc": p, "ci": [lo, hi], "wilson": [wl, wh],
                              "n": int(len(sub)),
                              "n_cores": int(sub["core_id"].nunique())}
            print(f"{t:<12}{str(k):>3}{p:>8.3f}   [{lo:.3f}, {hi:.3f}]      "
                  f"[{wl:.3f}, {wh:.3f}]{len(sub):>6}"
                  f"{sub['core_id'].nunique():>7}")
    out["accuracy"] = acc

    # The gradient check that gates the whole workstream (§3, §11).
    pooled = {t: acc[t]["pooled"]["acc"] for t in tiers}
    spread = max(pooled.values()) - min(pooled.values())
    out["gradient"] = {"pooled": pooled, "spread": float(spread),
                       "adequate": bool(spread >= 0.10)}
    if len(tiers) < 2:
        # The gradient is a cross-tier quantity; with one tier the spread is
        # trivially 0.000, which previously printed as "TOO FLAT -- redesign
        # scenarios" and looked like a verdict on the dataset.
        print(f"\nGRADIENT  n/a -- only {tiers[0]!r} measured")
        out["gradient"]["adequate"] = None
    else:
        print(f"\nGRADIENT  spread {spread:.3f}  "
              f"({'OK' if spread >= 0.10 else 'TOO FLAT -- redesign scenarios'})")

    # -- evidence decay ----------------------------------------------------
    print(f"\nEVIDENCE DECAY (N={repeats})")
    print(f"{'tier':<12}{'rho':>7}{'rho 95% CI':>18}{'gamma':>8}"
          f"{'gamma 95% CI':>18}{'1/rho':>8}{'1/(1-g)':>9}")
    decay: Dict[str, Dict] = {}
    for t in tiers:
        d = decay_for(df[df["tier"] == t], repeats, seed=args.seed)
        decay[t] = d
        print(f"{t:<12}{d['rho']:>7.3f}  [{d['rho_ci'][0]:.3f}, {d['rho_ci'][1]:.3f}]"
              f"{d['gamma']:>8.3f}  [{d['gamma_ci'][0]:.3f}, {d['gamma_ci'][1]:.3f}]"
              f"{d['n_eff_asymptote']:>8.2f}{d['geometric_bound']:>9.2f}")
    out["decay"] = decay

    gam = {t: decay[t]["gamma"] for t in tiers}
    if len(tiers) < 2:
        print("  (single tier: the cross-tier gamma comparison needs >= 2)")
    elif all(np.isfinite(v) for v in gam.values()):
        if max(gam.values()) - min(gam.values()) < 0.05:
            note = "gamma is flat across tiers -- check the sampling temperature"
        elif "offload" not in gam:
            # Both handoff §6.2 and R1's placeholder table are claims about
            # where the *cloud* tier sits. With offload unmeasured neither can
            # be adjudicated, and saying otherwise would be inventing evidence.
            order = " > ".join(f"{t} {gam[t]:.3f}" for t in
                               sorted(gam, key=gam.get, reverse=True))
            note = (f"gamma order among measured tiers: {order}. offload not "
                    f"measured, so the handoff-vs-placeholder disagreement "
                    f"about the cloud tier stays open.")
        elif gam["offload"] == max(gam.values()):
            note = ("gamma is highest for the cloud tier: supports HANDOFF sec 6.2 "
                    "prose, contradicts R1's placeholder table")
        else:
            note = ("gamma is NOT highest for the cloud tier: supports R1's "
                    "placeholder ordering, contradicts HANDOFF sec 6.2 prose")
        out["gamma_direction_note"] = note
        print(f"  -> {note}")
    if all(v > 0.95 for v in gam.values() if np.isfinite(v)):
        print("  !! gamma ~ 1 everywhere: repeats look independent, "
              "temperature is probably too high")
    if all(v < 0.05 for v in gam.values() if np.isfinite(v)):
        print("  !! gamma ~ 0 everywhere: models are deterministic, "
              "temperature is effectively zero")

    # -- position bias -----------------------------------------------------
    print("\nPOSITION BIAS (share of parsed answers landing on each display index)")
    bias: Dict[str, Dict] = {}
    for t in tiers:
        sub = first[(first["tier"] == t) & first["parse_status"].eq("ok")]
        bias[t] = {}
        for k in ks:
            s = sub[sub["k"] == k]
            if not len(s):
                continue
            counts = s["display_idx"].value_counts(normalize=True).sort_index()
            row = [float(counts.get(j, 0.0)) for j in range(k)]
            bias[t][str(k)] = {"share": row, "chance": 1.0 / k,
                               "excess_on_0": row[0] - 1.0 / k}
            print(f"  {t:<12}k={k}  " + " ".join(f"{v:.3f}" for v in row)
                  + f"   (chance {1.0 / k:.3f}, excess on 0 {row[0] - 1.0 / k:+.3f})")
    out["position_bias"] = bias

    # -- failure taxonomy --------------------------------------------------
    print("\nFAILURE TAXONOMY (first repeat, share of all responses)")
    df_first = first.copy()
    df_first["label"] = df_first.apply(classify, axis=1)
    tax = (df_first.groupby("tier")["label"].value_counts(normalize=True)
           .unstack(fill_value=0.0))
    print(tax.round(4).to_string())
    out["taxonomy"] = {t: {c: float(v) for c, v in row.items()}
                       for t, row in tax.iterrows()}

    print("\nERROR RATE BY PRAGMATIC FAMILY (first repeat)")
    fam = (df_first.assign(err=1 - (df_first["parse_status"].eq("ok")
                                    & df_first["correct"].eq(1)).astype(int))
           .pivot_table(index="family", columns="tier", values="err"))
    print(fam.round(3).to_string())
    out["error_by_family"] = {f: {t: float(v) for t, v in row.items()}
                              for f, row in fam.iterrows()}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote analysis -> {args.out}")

    if args.figures:
        make_figures(out, df, repeats, tiers, ks)
    return 0


def make_figures(out, df, repeats, tiers, ks) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    ns = np.arange(1, repeats + 1)

    # The headline figure: marginal information vs number of queries.
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for t in tiers:
        d = out["decay"][t]
        if not np.isfinite(d["gamma"]):
            continue
        ax.plot(ns, d["n_eff_curve"], "o-",
                label=f"{t}  ρ={d['rho']:.2f}, γ={d['gamma']:.2f}")
        ax.axhline(d["n_eff_asymptote"], ls=":", lw=0.8, alpha=0.4)
    ax.plot(ns, ns, "k--", lw=1, alpha=0.5, label="independent draws (ρ=0)")
    ax.set_xlabel("queries to the tier, n")
    ax.set_ylabel("effective independent observations, $n_{eff}$")
    ax.set_title("Evidence saturation: repeating a model buys less each time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "r2_fig1_marginal_information.png")
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  figure -> {p}")

    # Accuracy vs k, with clustered intervals.
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for t in tiers:
        y = [out["accuracy"][t][str(k)]["acc"] for k in ks]
        lo = [y[i] - out["accuracy"][t][str(k)]["ci"][0] for i, k in enumerate(ks)]
        hi = [out["accuracy"][t][str(k)]["ci"][1] - y[i] for i, k in enumerate(ks)]
        ax.errorbar(ks, y, yerr=[lo, hi], marker="o", capsize=3, label=t)
    ax.plot(ks, [1.0 / k for k in ks], "k--", lw=1, alpha=0.5, label="chance")
    ax.set_xticks(list(ks)); ax.set_xlabel("candidates, k")
    ax.set_ylabel("top-1 grounding accuracy")
    ax.set_title("Grounding accuracy by candidate count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "r2_fig2_accuracy_by_k.png")
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  figure -> {p}")

    # Failure distribution per tier.
    tax = pd.DataFrame(out["taxonomy"]).T.fillna(0.0)
    order = [c for c in ["correct", "pragmatic", "position_first", "position_last",
                         "format_hedge", "format_digit", "format_out_of_range"]
             if c in tax.columns]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    tax[order].plot(kind="barh", stacked=True, ax=ax, width=0.6)
    ax.set_xlabel("share of responses"); ax.set_xlim(0, 1)
    ax.set_title("Where each tier's answers go")
    ax.legend(fontsize=7, ncol=3, loc="lower right")
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "r2_fig3_failure_taxonomy.png")
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  figure -> {p}")


if __name__ == "__main__":
    raise SystemExit(main())
