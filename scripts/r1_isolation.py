"""The 2x2 channel-isolation experiment: when is local inference rational?

Owner: R1.

docs/r1_cloud_price.md section 8.2 found that when the cloud degrades the policy
asks the human rather than reasoning locally -- `think_big` usage stays at 0.00
at every link quality. That suggested the on-device tiers are crowded out by the
*human*, not by a cheap cloud, and that local inference should only become
rational when both alternatives are gone.

This crosses connectivity with human availability and measures where the
on-device tiers actually get used.

    python3 scripts/r1_isolation.py
"""
from __future__ import annotations

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]

CELLS = [("iso_cloud_human",   True,  True),
         ("iso_cloud_nohum",   True,  False),
         ("iso_nocloud_human", False, True),
         ("iso_nocloud_nohum", False, False)]
BASELINES = ["greedy_value", "always_offload", "always_ask", "no_think", "oracle"]


def collect(split="train") -> pd.DataFrame:
    rows = []
    for arm, cloud, human in CELLS:
        f = ROOT / f"results/eval_{arm}/eval_{split}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        s = d[d.agent.str.startswith("ppo")]
        per = s.groupby("agent").success.mean()
        t, b, c, a = (s.n_tiny.mean(), s.n_big.mean(),
                      s.n_offload.mean(), s.n_ask.mean())
        tot = t + b + c + a
        row = {"arm": arm, "cloud": cloud, "human": human,
               "ppo": per.mean(),
               "ppo_se": per.std(ddof=1) / np.sqrt(len(per)),
               "n_seeds": len(per),
               "tiny": t, "big": b, "offload": c, "ask": a,
               "local_share": (t + b) / tot if tot else np.nan}
        for bl in BASELINES:
            row[bl] = d[d.agent == bl].success.mean()
        # How far does the best device-blind scripted policy fall short? On a
        # split where this is ~0, nothing is being tested (see
        # docs/r1_transfer_audit.md section 3).
        row["spread"] = row["ppo"] - min(row["always_offload"],
                                         row["always_ask"], row["no_think"])
        rows.append(row)
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame, out: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = [("cloud\n+ human", 0), ("cloud\nonly", 1), ("human\nonly", 2), ("neither", 3)]
    order = ["iso_cloud_human", "iso_cloud_nohum", "iso_nocloud_human", "iso_nocloud_nohum"]
    d = df.set_index("arm").loc[[a for a in order if a in set(df.arm)]]
    x = np.arange(len(d))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) where the evidence actually comes from
    ax = axes[0]
    bot = np.zeros(len(d))
    for col, colour, name in [("tiny", "#65a30d", "think_tiny"),
                              ("big", "#7c3aed", "think_big"),
                              ("offload", "#0891b2", "offload"),
                              ("ask", "#dc2626", "ask_human")]:
        ax.bar(x, d[col], bottom=bot, color=colour, label=name)
        bot += d[col].values
    ax.set_xticks(x); ax.set_xticklabels([l for l, _ in lab][:len(d)])
    ax.set_ylabel("queries per episode"); ax.legend(fontsize=8)
    ax.set_title("(a) which channels the policy buys")

    # (b) the on-device share -- the headline of this experiment
    ax = axes[1]
    ax.bar(x, d.local_share * 100, color="#7c3aed", alpha=.85)
    for i, v in enumerate(d.local_share * 100):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([l for l, _ in lab][:len(d)])
    ax.set_ylabel("% of evidence from on-device models")
    ax.set_ylim(0, max(70, float(d.local_share.max() * 100) + 12))
    ax.set_title("(b) on-device inference earns its place\nonly when nothing else is available")

    # (c) does the condition discriminate at all?
    ax = axes[2]
    w = 0.2
    ax.bar(x - 1.5 * w, d.ppo, w, label="PPO", color="#2563EB")
    ax.bar(x - 0.5 * w, d.greedy_value, w, label="greedy_value", color="#0891b2")
    ax.bar(x + 0.5 * w, d.always_offload, w, label="always_offload", color="#B45309")
    ax.bar(x + 1.5 * w, d.no_think, w, label="no_think", color="#94a3b8")
    ax.set_xticks(x); ax.set_xticklabels([l for l, _ in lab][:len(d)])
    ax.set_ylabel("success"); ax.set_ylim(0, 1.0); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(c) a condition that actually discriminates\n(the device split does not)")

    fig.suptitle("CogniWatt R1 — when is on-device reasoning the rational purchase?",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    df = collect()
    if df.empty:
        sys.exit("no results/eval_iso_* found -- run scripts/r1_sweep.py --arms iso_* first")
    df.to_csv(ROOT / "results/isolation.csv", index=False)
    pd.set_option("display.width", 220)
    print(df[["arm", "cloud", "human", "ppo", "ppo_se", "greedy_value",
              "always_offload", "always_ask", "no_think", "tiny", "big",
              "offload", "ask", "local_share", "spread"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    figure(df, ROOT / "results/figures/fig7_isolation.png")
