"""Dialogue / evidence-channel interface.

OWNERSHIP
---------
Interface frozen by R1 (the env's belief filter depends on it).
R2 (Conversational AI) replaces the *numbers*:

  * `TIER_ACCURACY`  <- measured top-1 referent-grounding accuracy of each SLM
                        tier on the HomeGrid-Edge referring-expression set.
  * `similarity_matrix` <- cosine similarity between candidate-object text
                        embeddings, so that errors land on *semantically
                        confusable* distractors rather than uniformly.
  * `asr_wer`        <- word error rate from the Whisper + noise-injection
                        pipeline, which degrades the `ask` channel.

CRITICAL PERFORMANCE CONTRACT
-----------------------------
`sample_evidence` is called inside the PPO rollout loop, millions of times.
It MUST NOT call a language model. R2 delivers a *precomputed* artefact:
for every (task template, candidate set, tier) triple, the SLM's output
distribution is enumerated offline once and frozen to disk. At train time this
module only samples from those frozen distributions. Live SLM calls happen
exclusively in the Tier-2 AI2-THOR demo and in final evaluation rollouts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .ved import ASK, OFFLOAD, THINK_BIG, THINK_TINY

# --------------------------------------------------------------------------
# R1 PLACEHOLDER  ->  R2 REPLACES with measured grounding accuracy.
# Probability the channel returns the *correct* referent.
# --------------------------------------------------------------------------
TIER_ACCURACY: Dict[str, float] = {
    THINK_TINY: 0.45,   # Qwen2.5-0.5B INT8 - barely better than chance at K=3
    THINK_BIG:  0.72,   # Llama-3.2-3B Q4
    OFFLOAD:    0.88,   # frontier model over the network
    ASK:        0.97,   # the human knows - degraded only by ASR
}

#: Fallback ASR word error rate. R2 replaces with the measured Whisper curve.
DEFAULT_ASR_WER: float = 0.08

# --------------------------------------------------------------------------
# EVIDENCE SATURATION  (R1 modelling contribution; R2 measures the constants)
# --------------------------------------------------------------------------
# Querying the same model twice does not yield two independent observations.
# A quantised SLM at low temperature is very nearly deterministic, and even at
# temperature its residual errors are systematically correlated: it misreads
# the *same* referring expression the *same* way. We model this as geometric
# decay of evidence weight - the n-th query to tier t contributes
# `gamma_t ** (n - 1)` of a full observation - so the total information
# obtainable from a tier saturates at 1 / (1 - gamma_t) effective draws.
#
# Consequence, and the central economic tension of the project: a cheap model
# cannot be chained up to the competence of an expensive one. It buys you a
# bounded amount of certainty, after which the only way to become more certain
# is to spend more per query, go to the cloud, or ask the human.
#
# R2 measures gamma empirically: query each SLM tier N times on the same
# referring expression at the deployment temperature and fit the decay of
# marginal information gain.
EVIDENCE_DECAY: Dict[str, float] = {
    THINK_TINY: 0.45,   # saturates at ~1.8 effective draws
    THINK_BIG:  0.55,   # ~2.2
    OFFLOAD:    0.35,   # ~1.5
    ASK:        0.15,   # ~1.2 - re-asking annoys without informing
}


@dataclass
class DialogueOracle:
    """Noisy evidence channels over the referent identity.

    Each 'thinking' action draws one observation from a confusion channel whose
    reliability depends on the tier. The env folds that observation into a
    Bayesian belief. This is what makes CogniWatt a genuine POMDP with
    information-gathering actions rather than a disguised MDP.
    """
    tier_accuracy: Dict[str, float] = field(default_factory=lambda: dict(TIER_ACCURACY))
    evidence_decay: Dict[str, float] = field(default_factory=lambda: dict(EVIDENCE_DECAY))
    asr_wer: float = DEFAULT_ASR_WER
    #: Referent recovery rate: how often the pipeline recovers the *correct
    #: referent* from a noisy transcript. Takes precedence over `asr_wer` when
    #: present. R2 measured both and showed they come apart badly under noise --
    #: at 0 dB, WER 0.426 but recovery 0.917 (docs/R1_REPLY_2.md section 3.2).
    #: The env does not consume a word error rate; it consumes "did the `ask`
    #: action deliver the right referent", so recovery is the correct term.
    #: Using (1 - wer) systematically understates the channel in noise, which is
    #: exactly the regime the degraded-condition held-out axis lives in.
    asr_recovery: Optional[float] = None
    patience_max: int = 3
    #: Optional per-candidate-count accuracy, {tier: {k: acc}}. Grounding gets
    #: harder as the candidate set grows, and R2 measures it per k. When absent
    #: we fall back to the pooled scalar in `tier_accuracy`.
    accuracy_by_k: Dict[str, Dict[int, float]] = field(default_factory=dict)
    #: Free-form provenance from R2's measurement run (model ids, quantisation,
    #: temperature, sample counts, date). Carried so results stay traceable.
    meta: Dict[str, object] = field(default_factory=dict)

    def evidence_weight(self, tier: str, n_prior_queries: int) -> float:
        """Weight of the next observation from `tier`, given prior query count."""
        return float(self.evidence_decay[tier] ** max(0, int(n_prior_queries)))

    def accuracy(self, tier: str, k: Optional[int] = None) -> float:
        """Effective channel accuracy. The `ask` channel is gated by ASR quality.

        Uses R2's per-k measurement when available, since disambiguating one of
        four candidates is strictly harder than one of two.
        """
        by_k = self.accuracy_by_k.get(tier)
        if by_k and k is not None and k in by_k:
            a = by_k[k]
        else:
            a = self.tier_accuracy[tier]
        if tier == ASK:
            # A misrecognised answer is worse than no answer; scale toward chance.
            # Prefer measured referent recovery over (1 - WER); see `asr_recovery`.
            a = a * (self.asr_recovery if self.asr_recovery is not None
                     else (1.0 - self.asr_wer))
        return float(np.clip(a, 1e-3, 1.0 - 1e-3))

    def likelihood(self, tier: str, k: int,
                   similarity: Optional[np.ndarray] = None) -> np.ndarray:
        """P(observation = j | true referent = i) as a (k, k) row-stochastic matrix.

        Diagonal is the channel accuracy; off-diagonal mass is distributed over
        distractors proportionally to semantic similarity (uniform if R2 has not
        yet supplied embeddings).
        """
        a = self.accuracy(tier, k)
        if k == 1:
            return np.ones((1, 1), dtype=np.float64)
        if similarity is None:
            off = np.full((k, k), 1.0 / (k - 1), dtype=np.float64)
            np.fill_diagonal(off, 0.0)
        else:
            off = np.array(similarity, dtype=np.float64).copy()
            np.fill_diagonal(off, 0.0)
            rows = off.sum(axis=1, keepdims=True)
            rows[rows <= 0] = 1.0
            off = off / rows
        mat = (1.0 - a) * off
        np.fill_diagonal(mat, a)
        return mat

    def sample_evidence(self, tier: str, true_idx: int, k: int,
                        rng: np.random.Generator,
                        similarity: Optional[np.ndarray] = None) -> int:
        """Draw one observation index from the channel. Hot path - no LM calls."""
        mat = self.likelihood(tier, k, similarity)
        return int(rng.choice(k, p=mat[true_idx]))


#: Keys R2's artefact must provide. See docs/R2_HANDOFF.md for the full spec.
REQUIRED_KEYS = ("tiers", "accuracy", "evidence_decay", "asr_wer")
OPTIONAL_KEYS = ("k_values", "accuracy_by_k", "n_samples", "meta_json",
                 "asr_recovery")


def load_precomputed(path: str, strict: bool = False) -> DialogueOracle:
    """Load R2's frozen SLM measurement artefact.

    Schema (npz)
    ------------
    tiers          (T,)      str    tier names, must match THINK_TINY/THINK_BIG/OFFLOAD/ASK
    accuracy       (T,)      float  top-1 grounding accuracy, pooled over k
    evidence_decay (T,)      float  gamma per tier, in (0, 1]
    asr_wer        scalar    float  measured word error rate of the ASR pipeline
    asr_recovery   scalar    float  optional, referent recovery rate; preferred
                                    over asr_wer because the env consumes
                                    "was the right referent recovered", not WER
    k_values       (Kn,)     int    optional, e.g. [2, 3, 4]
    accuracy_by_k  (T, Kn)   float  optional, accuracy per tier per candidate count
    n_samples      (T, Kn)   int    optional, provenance for confidence intervals
    meta_json      scalar    str    optional, JSON: model ids, quantisation, temperature

    Falls back to R1 placeholders when the file is absent, so the RL workstream
    is never blocked on the dialogue workstream. Pass `strict=True` in R2's own
    tests to assert the artefact is complete.
    """
    import json
    import os
    if not os.path.exists(path):
        if strict:
            raise FileNotFoundError(f"dialogue artefact missing: {path}")
        return DialogueOracle()

    d = np.load(path, allow_pickle=True)
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        raise KeyError(f"{path} missing required keys {missing}; "
                       f"see docs/R2_HANDOFF.md for the schema")

    tiers = [str(t) for t in d["tiers"]]
    known = {THINK_TINY, THINK_BIG, OFFLOAD, ASK}
    unknown = set(tiers) - known
    if unknown:
        raise ValueError(f"unknown tier names {unknown}; expected subset of {known}")

    acc = {t: float(a) for t, a in zip(tiers, d["accuracy"])}
    decay = {t: float(g) for t, g in zip(tiers, d["evidence_decay"])}
    for t, g in decay.items():
        if not (0.0 < g <= 1.0):
            raise ValueError(f"evidence_decay[{t}]={g} outside (0, 1]")

    by_k: Dict[str, Dict[int, float]] = {}
    if "accuracy_by_k" in d and "k_values" in d:
        ks = [int(k) for k in d["k_values"]]
        for t, row in zip(tiers, np.atleast_2d(d["accuracy_by_k"])):
            by_k[t] = {k: float(v) for k, v in zip(ks, row)}

    meta = json.loads(str(d["meta_json"])) if "meta_json" in d else {}
    recovery = float(d["asr_recovery"]) if "asr_recovery" in d else None

    # Placeholders survive for any tier R2 has not yet measured.
    full_acc = dict(TIER_ACCURACY); full_acc.update(acc)
    full_decay = dict(EVIDENCE_DECAY); full_decay.update(decay)
    return DialogueOracle(tier_accuracy=full_acc, evidence_decay=full_decay,
                          asr_wer=float(d["asr_wer"]), asr_recovery=recovery,
                          accuracy_by_k=by_k, meta=meta)
