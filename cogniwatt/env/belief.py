"""Bayesian belief filter over the referent identity.

This is the heart of what makes CogniWatt a POMDP rather than a dressed-up
MDP: the agent cannot observe *which* object the human meant. Every
`think_*` / `offload` / `ask` action buys one draw from a noisy channel, and
the agent must decide how much evidence to purchase - and from which channel -
before committing to an irreversible placement.

Owner: R1.
"""
from __future__ import annotations

import numpy as np


class BeliefState:
    """Posterior over which of `k` candidate objects the instruction refers to."""

    def __init__(self, k: int, k_max: int):
        self.k = int(k)
        self.k_max = int(k_max)
        self.log_b = np.zeros(self.k, dtype=np.float64)  # unnormalised log-prob
        self._normalise()

    def _normalise(self) -> None:
        self.log_b -= self.log_b.max()
        p = np.exp(self.log_b)
        self.log_b = np.log(np.clip(p / p.sum(), 1e-12, 1.0))

    @property
    def probs(self) -> np.ndarray:
        return np.exp(self.log_b)

    def update(self, obs_idx: int, likelihood: np.ndarray,
               weight: float = 1.0) -> None:
        """Fold in one channel observation, discounted by `weight`.

        `likelihood[i, j] = P(obs = j | referent = i)`, so the column indexed by
        the received observation is exactly the likelihood term of Bayes' rule.
        We work in log-space: an agent may chain a dozen weak `think_tiny`
        draws, and the naive product underflows.

        `weight` implements *evidence saturation* (see
        `interfaces.dialogue.EVIDENCE_DECAY`). Repeated queries to the same
        model are not independent samples - a deterministic SLM asked twice
        returns essentially the same answer - so the n-th query to a tier
        contributes only `gamma**(n-1)` of a full observation. Without this the
        agent could reach arbitrary confidence by spamming the cheapest model,
        which is both physically wrong and would trivialise the task.
        """
        col = np.clip(likelihood[:, obs_idx], 1e-12, 1.0)
        self.log_b = self.log_b + float(weight) * np.log(col)
        self._normalise()

    @property
    def entropy(self) -> float:
        """Shannon entropy in nats."""
        p = self.probs
        return float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))

    @property
    def norm_entropy(self) -> float:
        """Entropy normalised to [0, 1] against the uniform prior over k."""
        if self.k <= 1:
            return 0.0
        return float(self.entropy / np.log(self.k))

    @property
    def argmax(self) -> int:
        return int(np.argmax(self.probs))

    @property
    def max_prob(self) -> float:
        return float(np.max(self.probs))

    def padded(self) -> np.ndarray:
        """Belief vector zero-padded to k_max for a fixed-width observation."""
        out = np.zeros(self.k_max, dtype=np.float32)
        out[:self.k] = self.probs.astype(np.float32)
        return out
