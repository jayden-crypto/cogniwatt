"""Rollout harness and metrics. Owner: R1."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..env.homegrid import HomeGridEdge


def run_episode(env: HomeGridEdge, agent, device: Optional[str] = None,
                seed: Optional[int] = None) -> dict:
    obs, info = env.reset(seed=seed, device=device)
    if hasattr(agent, "reset"):
        agent.reset()
    total = 0.0      # shaped return (what the agent optimises)
    env_total = 0.0  # true task return in natural units (what we report)
    steps = 0
    while True:
        a = agent.act(env, obs)
        obs, r, term, trunc, info = env.step(a)
        total += r
        env_total += info.get("env_reward", r)
        steps += 1
        if term or trunc:
            break
    q = info["queries"]
    return {
        "agent": getattr(agent, "name", "agent"),
        "device": info["device"],
        "k": info["k"],
        "outcome": info["outcome"],
        "success": float(info["outcome"] == "success"),
        "ret": env_total,
        "shaped_ret": total,
        "energy_J": info["energy_mJ"] / 1000.0,
        "latency_s": info["latency_ms"] / 1000.0,
        "asks": info["asks"],
        "moves": info["moves"],
        "invalid": info["invalid"],
        "temp_c": info["temp_c"],
        "steps": steps,
        "belief_max": info["belief_max"],
        "n_tiny": q["think_tiny"], "n_big": q["think_big"],
        "n_offload": q["offload"], "n_ask": q["ask"],
    }


def evaluate(agent, env_fn, n_episodes: int = 500,
             devices: Optional[Sequence[str]] = None,
             seed: int = 12345) -> pd.DataFrame:
    """Evaluate on a fixed episode set. The same `seed` gives every agent the
    same tasks, devices and channel noise, so comparisons are paired."""
    env = env_fn()
    rows = []
    for i in range(n_episodes):
        dev = None if devices is None else devices[i % len(devices)]
        rows.append(run_episode(env, agent, device=dev, seed=seed + i))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, by: Optional[str] = None) -> pd.DataFrame:
    keys = ["agent"] + ([by] if by else [])
    g = df.groupby(keys, sort=False)
    out = g.agg(
        n=("ret", "size"),
        success=("success", "mean"),
        ret=("ret", "mean"),
        ret_se=("ret", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        energy_J=("energy_J", "mean"),
        latency_s=("latency_s", "mean"),
        asks=("asks", "mean"),
        moves=("moves", "mean"),
        invalid=("invalid", "mean"),
        tiny=("n_tiny", "mean"), big=("n_big", "mean"),
        offload=("n_offload", "mean"), ask_q=("n_ask", "mean"),
    ).reset_index()
    return out


def outcome_table(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["agent", "outcome"]).size()
              .unstack(fill_value=0)
              .apply(lambda r: r / r.sum(), axis=1))


def fmt(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.3f}")
    return d.to_string(index=False)
