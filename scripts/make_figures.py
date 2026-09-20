"""Generate all result figures. Owner: R1."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from cogniwatt.analysis.pareto import (plot_action_mix, plot_learning_curves,
                                       plot_pareto)

FIG = pathlib.Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
RUNS = pathlib.Path("results/runs")


def load_histories(tags):
    out = {}
    for t in tags:
        f = RUNS / t / "history.json"
        if f.exists():
            out[t] = json.loads(f.read_text())["history"]
    return out


def policy_heatmap(tag: str, path: str, n_episodes: int = 4000):
    """The money figure: which action does the policy choose as a function of
    belief confidence and remaining battery?"""
    from cogniwatt.agents.ppo import ActorCritic, PPOAgent
    from cogniwatt.env.homegrid import (ACTION_NAMES, TRAIN_DEVICES, HomeGridEdge,
                                        A_ASK, A_OFFLOAD, A_THINK_BIG, A_THINK_TINY)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ck = torch.load(RUNS / tag / "policy.pt", map_location="cpu", weights_only=False)
    net = ActorCritic(recurrent=ck["recurrent"], hidden=ck["hidden"])
    net.load_state_dict(ck["state_dict"]); net.eval()
    ag = PPOAgent(net, deterministic=False, use_mask=ck.get("use_mask", True))
    env = HomeGridEdge(devices=TRAIN_DEVICES, seed=7)

    NB = 10
    grid = np.zeros((NB, NB, 5))          # conf x energy x {tiny,big,offload,ask,act}
    for ep in range(n_episodes):
        o, _ = env.reset(seed=40000 + ep); ag.reset()
        for _ in range(30):
            conf, en = env.belief.max_prob, env.ved.energy_frac_remaining
            a = ag.act(env, o)
            bi = min(NB - 1, int(conf * NB)); bj = min(NB - 1, int(en * NB))
            cat = {A_THINK_TINY: 0, A_THINK_BIG: 1, A_OFFLOAD: 2, A_ASK: 3}.get(a, 4)
            grid[bi, bj, cat] += 1
            o, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    tot = grid.sum(-1, keepdims=True)
    dom = np.where(tot[..., 0] > 0, grid.argmax(-1), -1)
    cmap = ListedColormap(["#e5e7eb", "#65a30d", "#7c3aed", "#0891b2", "#dc2626", "#2563eb"])
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    ax.imshow(dom.T + 1, origin="lower", cmap=cmap, vmin=0, vmax=5,
              extent=[0, 1, 0, 1], aspect="auto")
    ax.set_xlabel("belief confidence  max$_i$ $b_i$")
    ax.set_ylabel("battery remaining (fraction)")
    ax.set_title("Learned policy: what the agent buys, and when")
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color=c, label=l) for c, l in
                       zip(["#65a30d", "#7c3aed", "#0891b2", "#dc2626", "#2563eb"],
                           ["think_tiny", "think_big", "offload", "ask_human", "act (move/pick/place)"])],
              fontsize=8, loc="lower left", framealpha=.95)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return dom


#: The canonical policy for the single-policy figures. `entanneal_s0` is the
#: current best configuration (entropy annealed 0.12 -> 0.01: converges on every
#: seed *and* sharpens; see docs/r1_stability.md).
CANONICAL = "r2final321_s0"

#: fig1 is a learning-curve plot, so it has to stay readable. The stability
#: sweep contributes 40+ runs; those belong in fig6, not here. Show the method
#: and its ablations only.
FIG1_TAGS = ("r2final321_s0", "entanneal_s0", "ablate_devcond_s0",
             "ablate_mask_s0", "ablate_shaping_s0", "ablate_capdrop_s0")


def main():
    tags = [p.name for p in RUNS.iterdir() if (p / "history.json").exists()]
    hist = load_histories([t for t in FIG1_TAGS if t in tags] or sorted(tags)[:6])
    if hist:
        plot_learning_curves(hist, FIG / "fig1_learning_curves.png")
        print(f"  fig1_learning_curves.png  ({len(hist)} runs)")

    for split in ("train", "heldout"):
        f = pathlib.Path(f"results/eval/eval_{split}.csv")
        if not f.exists():
            continue
        df = pd.read_csv(f)
        plot_pareto(df, FIG / f"fig2_pareto_{split}.png")
        print(f"  fig2_pareto_{split}.png")
        ags = [a for a in df.agent.unique() if a.startswith("ppo")] + ["greedy_value", "always_ask"]
        plot_action_mix(df, FIG / f"fig3_action_mix_{split}.png", ags[:3])
        print(f"  fig3_action_mix_{split}.png")

    main_tag = CANONICAL if (RUNS / CANONICAL / "policy.pt").exists() else (sorted(tags)[0] if tags else None)
    if main_tag:
        policy_heatmap(main_tag, FIG / "fig4_policy_heatmap.png")
        print(f"  fig4_policy_heatmap.png  (from {main_tag})")


if __name__ == "__main__":
    main()
