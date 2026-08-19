"""Pareto analysis and result figures. Owner: R1."""
from __future__ import annotations

from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "ppo": "#2563eb", "greedy_value": "#f59e0b", "oracle": "#111827",
    "always_ask": "#dc2626", "always_big": "#7c3aed", "always_offload": "#0891b2",
    "always_tiny": "#65a30d", "no_think": "#94a3b8", "random": "#cbd5e1",
    "random_valid": "#cbd5e1",
}


def _colour(name: str) -> str:
    for k, v in PALETTE.items():
        if name.startswith(k):
            return v
    return "#64748b"


def pareto_front(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Indices on the front: minimise x, maximise y."""
    order = np.argsort(xs)
    front, best = [], -np.inf
    for i in order:
        if ys[i] > best:
            front.append(i); best = ys[i]
    return np.array(front, dtype=int)


#: run -> (display label, colour, linestyle). Groups the runs by what they test.
RUN_STYLE = {
    "ff_2m":          ("full method (2M)",          "#2563eb", "-"),
    "main_s1":        ("full method, seed 1",       "#3b82f6", "-"),
    "main_s2":        ("full method, seed 2",       "#60a5fa", "-"),
    "capdrop":        ("+ capability dropout",      "#1d4ed8", "-"),
    "ask0.5":         ("lambda_Q = 0.5",            "#059669", "-"),
    "ask6":           ("lambda_Q = 6  (collapsed)", "#f59e0b", "-."),
    "ask12":          ("lambda_Q = 12 (collapsed)", "#d97706", "-."),
    "ablate_devcond": ("no device conditioning",    "#dc2626", "--"),
    "ablate_shaping": ("no shaping",                "#e11d48", "--"),
    "ablate_mask":    ("no action masking",         "#7f1d1d", "--"),
}


def plot_learning_curves(histories: Dict[str, List[dict]], path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=150)
    for name in [k for k in RUN_STYLE if k in histories] + \
                [k for k in histories if k not in RUN_STYLE]:
        h = histories.get(name)
        if not h:
            continue
        label, col, ls = RUN_STYLE.get(name, (name, "#94a3b8", ":"))
        st = np.array([r["step"] for r in h]); sr = np.array([r["success"] for r in h])
        w = max(1, len(sr) // 40)
        sm = np.convolve(sr, np.ones(w) / w, mode="valid")
        ax.plot(st[w - 1:], sm, label=label, lw=1.9, color=col, ls=ls, alpha=.92)
    ax.axhline(0.895, ls="--", c="#f59e0b", lw=1.3)
    ax.annotate("greedy_value heuristic (0.895)", (1.02e6, 0.865), fontsize=8, color="#b45309")
    ax.axhline(1.0, ls=":", c="#111827", lw=1.3)
    ax.annotate("oracle", (1.72e6, 0.955), fontsize=8, color="#111827")
    ax.set_xlabel("environment steps"); ax.set_ylabel("task success rate")
    ax.set_title("Training outcome is bimodal: converge to ~0.95, or collapse to the uninformed policy")
    ax.set_ylim(0, 1.05); ax.grid(alpha=.22)
    ax.legend(fontsize=7.5, loc="center right", ncol=1, framealpha=.95)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_pareto(df: pd.DataFrame, path: str, xcol="energy_J", ycol="success") -> None:
    s = df.groupby("agent").agg(x=(xcol, "mean"), y=(ycol, "mean"),
                                asks=("asks", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for _, r in s.iterrows():
        ax.scatter(r.x, r.y, s=70 + 260 * r.asks, color=_colour(r.agent),
                   edgecolor="white", lw=1.2, zorder=3)
        ax.annotate(r.agent, (r.x, r.y), textcoords="offset points",
                    xytext=(7, 5), fontsize=8)
    f = pareto_front(s.x.values, s.y.values)
    ax.plot(s.x.values[f], s.y.values[f], "--", c="#94a3b8", lw=1.2, zorder=2)
    ax.set_xlabel("mean energy per episode (J)"); ax.set_ylabel("task success rate")
    ax.set_title("Success vs energy  (marker size = clarifying questions asked)")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_action_mix(df: pd.DataFrame, path: str, agents: Sequence[str]) -> None:
    cols = ["n_tiny", "n_big", "n_offload", "n_ask"]
    labels = ["think_tiny", "think_big", "offload", "ask_human"]
    colours = ["#65a30d", "#7c3aed", "#0891b2", "#dc2626"]
    sub = df[df.agent.isin(agents)]
    devices = list(sub.device.unique())
    fig, axes = plt.subplots(1, len(agents), figsize=(4.0 * len(agents), 4.0),
                             dpi=150, sharey=True, squeeze=False)
    for ax, ag in zip(axes[0], agents):
        d = sub[sub.agent == ag].groupby("device")[cols].mean().reindex(devices)
        bottom = np.zeros(len(devices))
        for c, lab, col in zip(cols, labels, colours):
            ax.bar(devices, d[c].values, bottom=bottom, label=lab, color=col)
            bottom += d[c].values
        ax.set_title(ag, fontsize=10); ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(alpha=.2, axis="y")
    axes[0][0].set_ylabel("mean queries per episode")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Channel portfolio adapts to the device", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
