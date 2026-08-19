"""Scripted baselines for HomeGrid-Edge.

Owner: R1.

The comparison that matters is PPO vs `GreedyValueAgent` vs `OracleAgent`.
`GreedyValueAgent` is deliberately strong - it is what a competent engineer
would write without RL: estimate the information value of each channel per
Joule, buy the best one until confident enough, then deliver, while reserving
enough battery to finish. If our learned policy cannot beat it, the project has
no result, so it is built to be hard to beat rather than a strawman.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..env.homegrid import (A_ASK, A_GOTO_RECEP, A_OFFLOAD, A_PICK, A_PLACE,
                            A_THINK_BIG, A_THINK_TINY, EVIDENCE_ACTIONS,
                            N_ACTIONS)
from ..interfaces.ved import ASK, MOVE, OFFLOAD, PICK, PLACE, THINK_BIG, THINK_TINY

TIER_ACTION = {THINK_TINY: A_THINK_TINY, THINK_BIG: A_THINK_BIG,
               OFFLOAD: A_OFFLOAD, ASK: A_ASK}


class Agent:
    name = "agent"

    def reset(self) -> None:
        pass

    def act(self, env, obs) -> int:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Shared scripted "deliver the object I currently believe in" behaviour.
# --------------------------------------------------------------------------
def _deliver(env, target: Optional[int] = None) -> int:
    t = env.task
    if env.carrying >= 0:
        return A_PLACE if env.pos == int(t.recep_cell) else A_GOTO_RECEP
    tgt = env.belief.argmax if target is None else target
    if env.pos == int(t.cand_cells[tgt]):
        return A_PICK
    return int(tgt)


def _delivery_energy(env, tgt: int) -> float:
    """Energy still required to finish, if we commit to `tgt` right now."""
    t = env.task
    ved = env.ved
    mv = ved.quote(MOVE).energy_mJ
    if env.carrying >= 0:
        d = t.layout.d(env.pos, int(t.recep_cell))
        return d * mv + ved.quote(PLACE).energy_mJ
    d = (t.layout.d(env.pos, int(t.cand_cells[tgt]))
         + t.layout.d(int(t.cand_cells[tgt]), int(t.recep_cell)))
    return d * mv + ved.quote(PICK).energy_mJ + ved.quote(PLACE).energy_mJ


def _energy_left(env) -> float:
    return (env.ved.profile.episode_energy_budget_mJ - env.ved.energy_spent_mJ)


class RandomAgent(Agent):
    name = "random"

    def act(self, env, obs) -> int:
        return int(env.rng.integers(N_ACTIONS))


class RandomValidAgent(Agent):
    name = "random_valid"

    def act(self, env, obs) -> int:
        m = env.action_mask()
        idx = np.flatnonzero(m)
        return int(env.rng.choice(idx)) if idx.size else A_PLACE


class NoThinkAgent(Agent):
    """Never gathers evidence: commits to the prior. Lower bound on accuracy."""
    name = "no_think"

    def act(self, env, obs) -> int:
        return _deliver(env)


class FixedTierAgent(Agent):
    """Query one tier `n_queries` times, then deliver. Ablates channel choice."""

    def __init__(self, tier: str, n_queries: int = 3):
        self.tier, self.n = tier, int(n_queries)
        self.name = f"always_{tier.replace('think_', '')}"

    def act(self, env, obs) -> int:
        used = env.query_counts[self.tier]
        can = env.ved.quote(self.tier).available and (self.tier != ASK or env.patience > 0)
        if used < self.n and can:
            tgt = env.belief.argmax
            if _energy_left(env) > _delivery_energy(env, tgt) * 1.15:
                return TIER_ACTION[self.tier]
        return _deliver(env)


class GreedyValueAgent(Agent):
    """Strong hand-engineered heuristic: buy information until confident.

    At each decision it scores every available channel by expected reduction in
    misidentification risk per Joule spent, accounting for evidence saturation,
    and buys the best one while confidence is below `threshold` and the battery
    reserve still covers delivery. This is the non-RL system CogniWatt must beat.
    """
    name = "greedy_value"

    def __init__(self, threshold: float = 0.85, reserve: float = 1.2,
                 lambda_ask: Optional[float] = None):
        self.threshold, self.reserve = float(threshold), float(reserve)
        self.lambda_ask = lambda_ask

    def _value_per_joule(self, env, tier: str) -> float:
        c = env.ved.quote(tier)
        if not c.available:
            return -np.inf
        if tier == ASK and env.patience <= 0:
            return -np.inf
        n = env.query_counts[tier]
        w = env.oracle.evidence_weight(tier, n)
        a = env.oracle.accuracy(tier)
        k = env.task.k
        # Expected posterior on the current argmax after one weighted draw.
        p = env.belief.probs.copy()
        lik = env._likelihoods[tier]
        gain = 0.0
        for obs_idx in range(k):
            p_obs = float(np.dot(p, lik[:, obs_idx]))
            post = p * np.power(np.clip(lik[:, obs_idx], 1e-12, 1.0), w)
            post /= post.sum()
            gain += p_obs * float(post.max())
        gain -= float(p.max())                       # expected confidence gain
        lam_ask = self.cfg_lambda_ask(env)
        cost_r = (env.cfg.lambda_energy * c.energy_mJ / 1000.0
                  + env.cfg.lambda_time * c.latency_ms / 1000.0
                  + (lam_ask if tier == ASK else 0.0)
                  + (env.cfg.lambda_cloud if tier == OFFLOAD else 0.0))
        return gain / max(cost_r, 1e-6)

    def cfg_lambda_ask(self, env) -> float:
        return env.cfg.lambda_ask if self.lambda_ask is None else self.lambda_ask

    def act(self, env, obs) -> int:
        if env.carrying >= 0:
            return _deliver(env)
        tgt = env.belief.argmax
        reserve_ok = _energy_left(env) > _delivery_energy(env, tgt) * self.reserve
        if env.belief.max_prob < self.threshold and reserve_ok:
            scores = {t: self._value_per_joule(env, t) for t in EVIDENCE_ACTIONS.values()}
            best = max(scores, key=scores.get)
            if np.isfinite(scores[best]) and scores[best] > 0:
                return TIER_ACTION[best]
        return _deliver(env)


class OracleAgent(Agent):
    """Knows the referent. Upper bound: no information cost, perfect accuracy."""
    name = "oracle"

    def act(self, env, obs) -> int:
        return _deliver(env, target=env.task.true_idx)


def default_baselines():
    return [
        RandomAgent(),
        RandomValidAgent(),
        NoThinkAgent(),
        FixedTierAgent(THINK_TINY, 4),
        FixedTierAgent(THINK_BIG, 3),
        FixedTierAgent(OFFLOAD, 2),
        FixedTierAgent(ASK, 1),
        GreedyValueAgent(threshold=0.85),
        OracleAgent(),
    ]
