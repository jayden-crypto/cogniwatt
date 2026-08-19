"""R1 stability study: quantify and diagnose the bimodal convergence.

Owner: R1.

Background
----------
The reported CogniWatt headline (0.95 success) comes from *converged* runs.
Several seeds instead settle at ~0.34 -- almost exactly the success rate of the
`no_think` baseline, i.e. the policy stops buying information altogether and
guesses. The report calls this out as the open limitation. This script measures
it: how often does it happen, is it predictable from the early entropy
trajectory, and does tripling the batch (the gradient-variance hypothesis) or
raising the entropy bonus (the exploration hypothesis) fix it.

Reads every results/runs/<arm>_s<seed>/history.json and writes
  results/stability_runs.csv     one row per run
  results/stability_arms.csv     one row per arm
  results/figures/fig6_stability.png

    python3 scripts/r1_stability.py
"""
from __future__ import annotations

import argparse, json, math, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: A run is "collapsed" if it ends near the uninformed policy rather than near
#: the converged mode. The two modes are ~0.34 and ~0.95, so anything under this
#: is unambiguous; `report_gap` below checks the gap really is empty.
COLLAPSE_THRESHOLD = 0.60
#: Averaging window at the end of training, in log records.
TAIL = 15
#: "Early" means before this many env steps -- the window in which we ask
#: whether collapse is already predictable.
EARLY_STEPS = 300_000

TAG_RE = re.compile(r"^(?P<arm>.+?)_s(?P<seed>\d+)$")

#: Only these arms are part of the stability study. The same <arm>_s<seed> tag
#: convention is reused by the ablation and Pareto runs, which are single-seed
#: and must not be pooled into a convergence-rate estimate.
STUDY_ARMS = ("base", "bigbatch", "gru", "ent03", "ent01", "entanneal")

#: Display order for the per-arm figure: ascending exploration pressure, so the
#: dose-response in `ent_coef` reads left to right instead of alphabetically.
ARM_ORDER = ("ent01", "ent03", "base", "bigbatch", "gru", "entanneal")


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval -- correct near 0 and 1, unlike normal-approx."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load_runs(runs_dir: pathlib.Path, arms=STUDY_ARMS) -> pd.DataFrame:
    rows = []
    for d in sorted(runs_dir.iterdir()):
        f = d / "history.json"
        if not f.is_dir() and f.exists():
            blob = json.loads(f.read_text())
            h = blob.get("history") or []
            if len(h) < TAIL:
                continue
            m = TAG_RE.match(d.name)
            if not m or m["arm"] not in arms:
                continue          # ff_2m, ablations, Pareto runs are not part of this study
            succ = np.array([x["success"] for x in h], dtype=float)
            ent = np.array([x["entropy"] for x in h], dtype=float)
            step = np.array([x["step"] for x in h], dtype=float)
            early = step <= EARLY_STEPS
            final = float(np.nanmean(succ[-TAIL:]))
            rows.append({
                "arm": m["arm"], "seed": int(m["seed"]), "tag": d.name,
                "final_success": final,
                "collapsed": bool(final < COLLAPSE_THRESHOLD),
                "best_success": float(np.nanmax(succ)),
                # entropy is the diagnostic: collapse is an entropy implosion
                "entropy_early": float(np.nanmean(ent[early])) if early.any() else float("nan"),
                "entropy_final": float(np.nanmean(ent[-TAIL:])),
                # the first point where the run is already doing better than
                # guessing; never reaching it is the signature of the bad mode
                "step_to_0.6": _first_step(succ, step, 0.6),
                "step_to_0.9": _first_step(succ, step, 0.9),
                "minutes": blob.get("seconds", float("nan")) / 60.0,
                "n_envs": blob.get("args", {}).get("n_envs"),
                "ent_coef": blob.get("args", {}).get("ent_coef"),
                "recurrent": not blob.get("args", {}).get("no_recurrent", False),
                "steps": blob.get("args", {}).get("steps"),
            })
    return pd.DataFrame(rows)


def _first_step(succ, step, thr):
    i = np.flatnonzero(succ >= thr)
    return float(step[i[0]]) if i.size else float("nan")


def summarise_arms(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for arm, g in df.groupby("arm", sort=False):
        n, k = len(g), int((~g.collapsed).sum())
        p, lo, hi = wilson(k, n)
        conv = g[~g.collapsed]
        out.append({
            "arm": arm, "n_seeds": n, "n_converged": k,
            "convergence_rate": p, "ci_lo": lo, "ci_hi": hi,
            "success_all": g.final_success.mean(),
            "success_converged": conv.final_success.mean() if len(conv) else float("nan"),
            "success_sd_converged": conv.final_success.std(ddof=1) if len(conv) > 1 else float("nan"),
            "entropy_early": g.entropy_early.mean(),
            "median_minutes": g.minutes.median(),
        })
    return pd.DataFrame(out)


def report_gap(df: pd.DataFrame) -> str:
    """Check the bimodality claim: is the region around the threshold empty?"""
    s = np.sort(df.final_success.values)
    inside = s[(s > 0.45) & (s < 0.80)]
    return (f"{len(inside)}/{len(s)} runs land in the 0.45-0.80 band "
            f"(a truly bimodal outcome leaves it empty)")


def early_warning(df: pd.DataFrame) -> str:
    """Does early entropy separate the two modes? If it does, collapse is
    detectable ~7x cheaper than a full run, which is a practical result."""
    a = df[df.collapsed].entropy_early.dropna()
    b = df[~df.collapsed].entropy_early.dropna()
    if len(a) < 2 or len(b) < 2:
        return "not enough runs in one mode to test early separation"
    # Welch t on the mean early entropy of the two modes.
    t = (b.mean() - a.mean()) / math.sqrt(b.var(ddof=1) / len(b) + a.var(ddof=1) / len(a))
    return (f"mean early entropy (first {EARLY_STEPS:,} steps): "
            f"converged {b.mean():.3f} (n={len(b)}) vs collapsed {a.mean():.3f} "
            f"(n={len(a)}), Welch t = {t:.2f}")


def make_figure(df: pd.DataFrame, runs_dir: pathlib.Path, out: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(df.arm.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) every learning curve, coloured by outcome -- the bimodality itself
    ax = axes[0]
    for _, r in df.iterrows():
        h = json.loads((runs_dir / r.tag / "history.json").read_text())["history"]
        ax.plot([x["step"] / 1e6 for x in h], [x["success"] for x in h],
                lw=1.0, alpha=0.75,
                color=("tab:red" if r.collapsed else "tab:blue"))
    ax.axhline(0.384, ls=":", c="k", lw=1)
    ax.text(0.02, 0.40, "no_think baseline", fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xlabel("env steps (M)"); ax.set_ylabel("success (train)")
    ax.set_title("(a) every seed, all arms\nblue = converged, red = collapsed")

    # (b) the outcome distribution -- is the middle really empty?
    ax = axes[1]
    ax.hist(df.final_success, bins=np.linspace(0, 1, 21), color="tab:grey",
            edgecolor="k", lw=0.5)
    ax.axvline(COLLAPSE_THRESHOLD, ls="--", c="tab:red")
    ax.set_xlabel("final success (mean of last %d logs)" % TAIL)
    ax.set_ylabel("runs")
    # State what the runs actually show rather than asserting bimodality: with
    # enough exploration the second mode simply does not appear.
    n_col = int(df.collapsed.sum())
    ax.set_title("(b) outcome distribution\n"
                 + (f"bimodal: {n_col}/{len(df)} collapsed" if n_col
                    else f"single mode: 0/{len(df)} collapsed"))

    # (c) convergence rate per arm with Wilson CIs -- the actual comparison
    ax = axes[2]
    s = summarise_arms(df)
    s = s.sort_values("arm", key=lambda c: c.map(
        lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 99)).reset_index(drop=True)
    x = np.arange(len(s))
    ax.bar(x, s.convergence_rate, color="tab:blue", alpha=0.8)
    ax.errorbar(x, s.convergence_rate,
                yerr=[s.convergence_rate - s.ci_lo, s.ci_hi - s.convergence_rate],
                fmt="none", ecolor="k", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(s.arm, rotation=15)
    ax.set_ylim(0, 1.05); ax.set_ylabel("P(converged)")
    ax.set_title("(c) convergence rate, Wilson 95% CI")
    for i, r in s.iterrows():
        ax.text(i, 0.03, f"{r.n_converged}/{r.n_seeds}", ha="center", fontsize=9)
    ax.set_xlabel("ascending exploration pressure \u2192")

    fig.suptitle("CogniWatt R1 — training stability study", y=1.02, fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results/runs"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--arms", nargs="+", default=list(STUDY_ARMS))
    a = ap.parse_args()

    df = load_runs(pathlib.Path(a.runs), arms=tuple(a.arms))
    if df.empty:
        sys.exit("no <arm>_s<seed> runs found -- run scripts/r1_sweep.py first")
    df = df.sort_values(["arm", "seed"]).reset_index(drop=True)
    arms = summarise_arms(df)

    outdir = pathlib.Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "stability_runs.csv", index=False)
    arms.to_csv(outdir / "stability_arms.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== per-run ===")
    print(df[["arm", "seed", "final_success", "collapsed", "entropy_early",
              "entropy_final", "step_to_0.9", "minutes"]].to_string(index=False,
              float_format=lambda v: f"{v:.3f}"))
    print("\n=== per-arm ===")
    print(arms.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + report_gap(df))
    print(early_warning(df))

    make_figure(df, pathlib.Path(a.runs), outdir / "figures/fig6_stability.png")


if __name__ == "__main__":
    main()
