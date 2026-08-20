"""Evaluate a trained policy against all baselines, on train and held-out devices.

Owner: R1. Produces the headline results table and the zero-shot transfer result.
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from cogniwatt.agents.baselines import default_baselines
from cogniwatt.agents.ppo import ActorCritic, PPOAgent
from cogniwatt.analysis.evaluate import evaluate, fmt, outcome_table, summarise
from cogniwatt.env.homegrid import (HELDOUT_DEVICES, TRAIN_DEVICES,
                                    HomeGridEdge, RewardConfig)


def load_agent(tag: str, name=None):
    ck = torch.load(f"results/runs/{tag}/policy.pt", map_location="cpu", weights_only=False)
    net = ActorCritic(recurrent=ck["recurrent"], hidden=ck["hidden"])
    net.load_state_dict(ck["state_dict"]); net.eval()
    # Stochastic evaluation is the correct protocol here, not a concession.
    # Under argmax the policy can enter a fixed point: a *saturated*
    # think_tiny query barely moves the belief, so the observation barely
    # changes, so argmax re-selects the same query forever (we measured up to
    # 10.9 tiny queries/episode, vs 2.5 when sampling). The stochasticity is
    # functional - it is what breaks the loop. Argmax understates held-out
    # success by up to 19 points and is reported only as an ablation.
    return PPOAgent(net, deterministic=False, name=name or f"ppo[{tag}]",
                    use_mask=ck.get("use_mask", True)), ck


#: Training args that describe the ENVIRONMENT rather than the optimiser, and
#: therefore have to be reproduced at evaluation time. Getting this wrong is
#: silent: the policy still runs, it is just being tested in a world it was not
#: trained for. `net_quality` was added for the connectivity study and was
#: missed here first time round -- every arm evaluated at the 0.75 default,
#: which made the scripted baselines return byte-identical numbers at every
#: link quality. That tell is what caught it.
ENV_ARGS = {"net_quality": 0.75}


def env_kwargs_from(tag: str, args: dict) -> dict:
    return {"net_quality_mean": float(args.get("net_quality", ENV_ARGS["net_quality"]))}


def check_env_args(tags, cks) -> dict:
    """All policies in one table must be evaluated in the same world."""
    seen = {t: env_kwargs_from(t, cks[t].get("args", {})) for t in tags}
    uniq = {tuple(sorted(v.items())) for v in seen.values()}
    if len(uniq) > 1:
        lines = "\n".join(f"    {t}: {v}" for t, v in seen.items())
        raise SystemExit(
            f"\nENVIRONMENT MISMATCH - refusing to evaluate these together.\n"
            f"  The policies were trained in different environments, so a shared\n"
            f"  baseline column would be meaningless:\n{lines}\n")
    return next(iter(seen.values()))


def check_reward_config(tag: str, stored: dict, active: RewardConfig) -> None:
    """Refuse to compare a policy against a reward it was not trained under.

    `lambda_cloud` was added to RewardConfig after `ff_2m` was trained. Nothing
    checked it, so a policy trained where cloud calls were free kept being
    evaluated - and reported - under a reward that priced them, for weeks. The
    stored config is right there in the checkpoint; this just reads it.
    """
    cur = active.__dict__
    diff = {k: (stored.get(k, "<absent>"), cur.get(k, "<absent>"))
            for k in sorted(set(stored) | set(cur))
            if stored.get(k, "<absent>") != cur.get(k, "<absent>")}
    if diff:
        lines = "\n".join(f"    {k}: trained={o!r}  evaluating={n!r}"
                           for k, (o, n) in diff.items())
        raise SystemExit(
            f"\nREWARD MISMATCH for {tag!r} - refusing to evaluate.\n"
            f"  The policy was trained under a different reward function, so its\n"
            f"  returns are not comparable to anything measured under this one:\n"
            f"{lines}\n"
            f"  Retrain under the current reward, or pass --allow-reward-mismatch\n"
            f"  if you genuinely intend a cross-reward comparison.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["ff_2m"])
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--allow-reward-mismatch", action="store_true",
                    help="evaluate anyway when a policy's stored RewardConfig "
                         "differs from the one being evaluated under")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    agents, cks = [], {}
    for t in a.tags:
        ag, ck = load_agent(t)
        agents.append(ag); cks[t] = ck
    rc = RewardConfig(**{k: v for k, v in cks[a.tags[0]]["reward"].items()})
    if not a.allow_reward_mismatch:
        for t in a.tags:
            check_reward_config(t, cks[t]["reward"], rc)
    env_kw = check_env_args(a.tags, cks)
    if env_kw != {"net_quality_mean": ENV_ARGS["net_quality"]}:
        print(f"[env] evaluating under {env_kw}")
    agents += default_baselines()

    for split, devs in [("train", TRAIN_DEVICES), ("heldout", HELDOUT_DEVICES)]:
        env_fn = lambda: HomeGridEdge(devices=devs, reward=rc, seed=0, **env_kw)
        df = pd.concat([evaluate(ag, env_fn, n_episodes=a.n, devices=list(devs))
                        for ag in agents], ignore_index=True)
        df.to_csv(out / f"eval_{split}.csv", index=False)
        print(f"\n{'='*100}\n=== {split.upper()} DEVICES {list(devs)}  (n={a.n} paired episodes) ===")
        print(fmt(summarise(df)))
        print(f"\n--- outcome distribution ({split}) ---")
        print(outcome_table(df).round(3).to_string())
        if split == "heldout":
            print(f"\n--- per-device breakdown ({split}) ---")
            keep = [g for g in df.agent.unique() if g.startswith("ppo") or g in ("greedy_value", "oracle")]
            print(fmt(summarise(df[df.agent.isin(keep)], by="device")))


if __name__ == "__main__":
    main()
