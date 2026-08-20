"""HomeGrid-Edge: the CogniWatt Tier-1 environment.

Owner: R1.

A household robot receives an *underspecified* spoken instruction ("put the mug
on the shelf") in a house containing several objects consistent with it. It must
decide, at every decision point, between acting in the world and spending
compute or human patience to reduce its uncertainty - under a hardware-accurate
energy, latency and thermal budget supplied by the Virtual Edge Device.

Hierarchical action design: low-level navigation is solved exactly by
precomputed BFS (`tasks._all_pairs_bfs`) and exposed as `goto` macro-actions
that are billed for every grid step they traverse. The learned policy therefore
operates at the semantic/resource level, which is where this project's
contribution lies. This cuts episode length from ~60 primitive steps to ~10
decisions and is what makes training feasible on a laptop.

API follows Gymnasium v0.29+ (`reset -> (obs, info)`,
`step -> (obs, reward, terminated, truncated, info)`) without importing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..interfaces.dialogue import DialogueOracle
from ..interfaces.ved import (ASK, FEATURE_DIM, MOVE, OFFLOAD, PICK, PLACE,
                              THINK_BIG, THINK_TINY, VirtualEdgeDevice)
from ..spaces import Box, Discrete
from .belief import BeliefState
from .tasks import GRID, MAX_DIST, Task, sample_task

K_MAX = 4

# ---- action layout -------------------------------------------------------
A_GOTO_CAND = 0                      # 0 .. K_MAX-1
A_GOTO_RECEP = K_MAX                 # 4
A_PICK = K_MAX + 1                   # 5
A_PLACE = K_MAX + 2                  # 6
A_THINK_TINY = K_MAX + 3             # 7
A_THINK_BIG = K_MAX + 4              # 8
A_OFFLOAD = K_MAX + 5                # 9
A_ASK = K_MAX + 6                    # 10
N_ACTIONS = K_MAX + 7                # 11

EVIDENCE_ACTIONS: Dict[int, str] = {
    A_THINK_TINY: THINK_TINY,
    A_THINK_BIG: THINK_BIG,
    A_OFFLOAD: OFFLOAD,
    A_ASK: ASK,
}
ACTION_NAMES = ([f"goto_cand{i}" for i in range(K_MAX)] +
                ["goto_recep", "pick", "place",
                 "think_tiny", "think_big", "offload", "ask"])

OBS_DIM = 41

TRAIN_DEVICES = ("orin_nano", "pi5", "orin_nx", "m2_air")
HELDOUT_DEVICES = ("jetson_nano_2gb", "rtx3060")


@dataclass
class RewardConfig:
    """Reward weights. These are the Pareto knobs swept in the results study.

    Scaled so that exhausting the whole energy or latency budget costs about as
    much as a task failure, which keeps the terms commensurable.
    """
    r_success: float = 10.0
    r_wrong: float = -10.0
    r_depleted: float = -10.0
    r_timeout: float = -8.0
    r_incomplete: float = -15.0  # ran out of decisions without delivering
    lambda_energy: float = 0.025      # per Joule
    lambda_time: float = 0.1667       # per second
    lambda_ask: float = 2.0           # per clarifying question (human patience)
    lambda_cloud: float = 0.5         # per cloud call (privacy + monetary)
    r_invalid: float = -0.1

    # NOTE (R1): `lambda_cloud` closes a modelling gap that only became visible
    # once the roofline replaced the placeholder compute costs. Priced purely on
    # energy and latency, a cloud call (140 ms / 0.8 J, 88% accurate) strictly
    # dominates on-device inference (1591 ms / 26 J, 72% accurate on the M2) -
    # so the optimal policy is simply "always offload" and the local tiers are
    # dead weight. That is an artefact of omitting the two real costs of
    # offloading a home robot's scene description: it leaves the house, and it
    # is billed per call. With that priced in, the local-vs-cloud decision
    # becomes genuinely device-dependent again.

    # Potential-based shaping (Ng et al., 1999). Total shaping over an episode
    # telescopes to gamma^T * Phi(s_T) - Phi(s_0) with Phi(terminal) := 0, i.e.
    # a constant, so the optimal policy is provably unchanged. It exists purely
    # to make partial progress visible to exploration. Ablated in the results.
    shaping: bool = True
    w_info: float = 1.0      # potential on belief confidence
    w_progress: float = 2.0  # potential on progress toward delivery
    gamma: float = 0.99

    # NOTE (R1): `r_incomplete` is load-bearing, not cosmetic. Without it the
    # decision budget expiring is free, and the optimal policy is *paralysis* -
    # spam the cheapest action for 30 steps rather than attempt the task and
    # risk -10 for a wrong delivery. Two training runs collapsed to exactly
    # this: the policy emitted `think_tiny` 30x per episode, entropy -> 0,
    # success -> 0, and its return (-8.24) genuinely beat the no-information
    # baseline (-9.52). Non-delivery is now priced as the worst outcome: for an
    # assistant, unbounded latency is a total failure, whereas a wrong
    # placement is at least recoverable. Kept in the report as a clean worked
    # example of reward hacking.


class HomeGridEdge:
    """Resource-constrained referential grounding POMDP."""

    metadata = {"name": "HomeGrid-Edge-v0"}

    def __init__(self,
                 devices: Sequence[str] = TRAIN_DEVICES,
                 reward: Optional[RewardConfig] = None,
                 oracle: Optional[DialogueOracle] = None,
                 max_steps: int = 30,
                 k_min: int = 2, k_max: int = 4,
                 condition_on_device: bool = True,
                 capability_dropout: float = 0.0,
                 net_quality_mean: float = 0.75,
                 patience_max: Optional[int] = None,
                 seed: Optional[int] = None):
        self.devices = list(devices)
        self.cfg = reward or RewardConfig()
        self.oracle = oracle or DialogueOracle()
        # Human availability. `patience_max = 0` means nobody is home to ask -
        # the condition that, together with a failed link, leaves on-device
        # inference as the only surviving evidence channel. See
        # docs/r1_cloud_price.md section 8.3.
        if patience_max is not None:
            self.oracle = replace(self.oracle, patience_max=int(patience_max))
        self.max_steps = int(max_steps)
        self.k_min, self.k_max = int(k_min), int(k_max)
        self.condition_on_device = bool(condition_on_device)
        # Domain randomisation over *model availability*. Our first policy
        # transferred well to an unseen device that was quantitatively
        # different (rtx3060: 0.943) but failed on one that was structurally
        # different (coral_micro: 0.700), where `think_big` does not fit at
        # all - it substituted the cheapest remaining channel without
        # accounting for that channel's saturation ceiling. Randomly removing
        # the big model during training exposes the policy to that regime.
        self.capability_dropout = float(capability_dropout)
        # Mean wireless link quality. Poor connectivity is what makes local
        # inference genuinely necessary rather than merely priced-in; see
        # docs/r1_cloud_price.md section 7.2.
        self.net_quality_mean = float(net_quality_mean)

        self.action_space = Discrete(N_ACTIONS)
        self.observation_space = Box(-1.0, 1.0, (OBS_DIM,), np.float32)
        self.rng = np.random.default_rng(seed)
        self._t: Optional[Task] = None

    # -- lifecycle ---------------------------------------------------------
    def reset(self, seed: Optional[int] = None,
              device: Optional[str] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.task = sample_task(self.rng, self.k_min, self.k_max)
        self.device_name = device or str(self.rng.choice(self.devices))
        self.ved = VirtualEdgeDevice(self.device_name, self.rng,
                                     net_quality_mean=self.net_quality_mean)
        if self.capability_dropout > 0 and self.rng.random() < self.capability_dropout:
            self.ved.reset(disabled_ops={THINK_BIG})
        self.belief = BeliefState(self.task.k, K_MAX)

        self.pos = int(self.task.start_cell)
        self.carrying = -1
        self.steps = 0
        self.patience = int(self.oracle.patience_max)
        self.query_counts: Dict[str, int] = {t: 0 for t in EVIDENCE_ACTIONS.values()}
        self.n_moves = 0
        self.n_invalid = 0
        self._likelihoods = {
            tier: self.oracle.likelihood(tier, self.task.k, self.task.similarity)
            for tier in EVIDENCE_ACTIONS.values()
        }
        self._phi = self._potential()
        return self._obs(), self._info()

    # -- observation -------------------------------------------------------
    def _obs(self) -> np.ndarray:
        t, k = self.task, self.task.k
        o = np.zeros(OBS_DIM, dtype=np.float32)
        i = 0
        o[i:i + K_MAX] = self.belief.padded();                       i += K_MAX
        o[i] = self.belief.norm_entropy;                             i += 1
        o[i] = self.belief.max_prob;                                 i += 1
        for j in range(K_MAX):                                        # travel cost
            o[i + j] = (t.layout.d(self.pos, int(t.cand_cells[j])) / MAX_DIST) if j < k else 0.0
        i += K_MAX
        o[i:i + k] = 1.0;                                             i += K_MAX  # valid mask
        o[i] = t.layout.d(self.pos, int(t.recep_cell)) / MAX_DIST;   i += 1
        o[i] = 1.0 if self.carrying >= 0 else 0.0;                   i += 1
        if self.carrying >= 0:
            o[i + self.carrying] = 1.0
        i += K_MAX
        o[i] = self.ved.energy_frac_remaining;                       i += 1
        o[i] = self.ved.time_frac_remaining;                         i += 1
        o[i] = self.ved.thermal.norm_temp;                           i += 1
        o[i] = self.ved.net_quality;                                 i += 1
        o[i] = self.patience / max(1, self.oracle.patience_max);     i += 1
        if self.condition_on_device:
            o[i:i + FEATURE_DIM] = self.ved.profile.feature_vector()
        i += FEATURE_DIM
        for j, tier in enumerate(EVIDENCE_ACTIONS.values()):
            o[i + j] = min(1.0, self.query_counts[tier] / 5.0)
        i += 4
        o[i] = self.steps / self.max_steps;                          i += 1
        for j in range(K_MAX):
            o[i + j] = 1.0 if (j < k and self.pos == int(t.cand_cells[j])) else 0.0
        i += K_MAX
        o[i] = 1.0 if self.pos == int(t.recep_cell) else 0.0;        i += 1
        assert i == OBS_DIM, f"obs layout mismatch: {i} != {OBS_DIM}"
        return o

    def _potential(self) -> float:
        """Phi(s): higher when the agent is confident and close to finishing."""
        if not self.cfg.shaping:
            return 0.0
        t = self.task
        info_term = self.cfg.w_info * (1.0 - self.belief.norm_entropy)
        # The progress potential must NOT depend on the belief. An earlier
        # version measured distance to `belief.argmax`; acquiring evidence that
        # moved the argmax to a farther candidate then produced *negative*
        # shaping, so the agent was actively penalised for becoming better
        # informed and plateaued at exactly the no-information success rate.
        # Remaining work is now a belief-independent lower bound.
        if self.carrying >= 0:
            d = t.layout.d(self.pos, int(t.recep_cell))
        else:
            d = min(t.layout.d(self.pos, int(t.cand_cells[j]))
                    + t.layout.d(int(t.cand_cells[j]), int(t.recep_cell))
                    for j in range(t.k))
        prog_term = self.cfg.w_progress * (1.0 - min(1.0, d / (2.0 * MAX_DIST)))
        return float(info_term + prog_term)

    def _info(self, **kw) -> dict:
        d = {
            "device": self.device_name,
            "k": self.task.k,
            "true_idx": self.task.true_idx,
            "belief_argmax": self.belief.argmax,
            "belief_max": self.belief.max_prob,
            "energy_mJ": self.ved.energy_spent_mJ,
            "latency_ms": self.ved.time_spent_ms,
            "temp_c": self.ved.thermal.temp_c,
            "asks": self.oracle.patience_max - self.patience,
            "moves": self.n_moves,
            "invalid": self.n_invalid,
            "queries": dict(self.query_counts),
        }
        d.update(kw)
        return d

    # -- dynamics ----------------------------------------------------------
    def _spend(self, op: str, repeat: int = 1) -> Tuple[float, bool]:
        """Charge `repeat` instances of `op`. Returns (reward_cost, ok)."""
        c = self.ved.quote(op)
        if not c.available:
            return 0.0, False
        cost_r = 0.0
        for _ in range(repeat):
            self.ved.commit(op, c)
            cost_r += (self.cfg.lambda_energy * c.energy_mJ / 1000.0
                       + self.cfg.lambda_time * c.latency_ms / 1000.0)
        return cost_r, True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        a = int(action)
        t = self.task
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False
        outcome = "ongoing"

        # ---- navigation macro-actions ------------------------------------
        if a < A_GOTO_RECEP or a == A_GOTO_RECEP:
            if a == A_GOTO_RECEP:
                target = int(t.recep_cell)
                valid = True
            else:
                valid = a < t.k
                target = int(t.cand_cells[a]) if valid else self.pos
            if not valid:
                reward += self.cfg.r_invalid
                self.n_invalid += 1
            else:
                d = t.layout.d(self.pos, target)
                if d <= 0:
                    reward += self.cfg.r_invalid
                    self.n_invalid += 1
                else:
                    cost, _ = self._spend(MOVE, repeat=d)
                    reward -= cost
                    self.n_moves += d
                    self.pos = target

        # ---- manipulation -------------------------------------------------
        elif a == A_PICK:
            here = [j for j in range(t.k) if self.pos == int(t.cand_cells[j])]
            if self.carrying >= 0 or not here:
                reward += self.cfg.r_invalid
                self.n_invalid += 1
            else:
                cost, _ = self._spend(PICK)
                reward -= cost
                self.carrying = here[0]

        elif a == A_PLACE:
            if self.carrying < 0 or self.pos != int(t.recep_cell):
                reward += self.cfg.r_invalid
                self.n_invalid += 1
            else:
                cost, _ = self._spend(PLACE)
                reward -= cost
                terminated = True
                if self.carrying == t.true_idx:
                    reward += self.cfg.r_success
                    outcome = "success"
                else:
                    reward += self.cfg.r_wrong
                    outcome = "wrong_object"

        # ---- cognition -----------------------------------------------------
        else:
            tier = EVIDENCE_ACTIONS[a]
            if tier == ASK and self.patience <= 0:
                # The human has stopped answering. Pay the social cost anyway.
                reward += self.cfg.r_invalid - self.cfg.lambda_ask
                self.n_invalid += 1
            else:
                cost, ok = self._spend(tier)
                if not ok:                       # e.g. 3B model does not fit on Coral
                    reward += self.cfg.r_invalid
                    self.n_invalid += 1
                else:
                    reward -= cost
                    got_evidence = True
                    if tier == OFFLOAD and not self.ved.offload_succeeds():
                        got_evidence = False     # paid the round trip, link dropped
                    if tier == ASK:
                        self.patience -= 1
                        reward -= self.cfg.lambda_ask
                    elif tier == OFFLOAD:
                        reward -= self.cfg.lambda_cloud
                    if got_evidence:
                        w = self.oracle.evidence_weight(tier, self.query_counts[tier])
                        obs_idx = self.oracle.sample_evidence(
                            tier, t.true_idx, t.k, self.rng, t.similarity)
                        self.belief.update(obs_idx, self._likelihoods[tier], weight=w)
                    self.query_counts[tier] += 1

        # ---- budget termination -------------------------------------------
        if not terminated:
            if self.ved.depleted:
                reward += self.cfg.r_depleted
                terminated, outcome = True, "battery_dead"
            elif self.ved.out_of_time:
                reward += self.cfg.r_timeout
                terminated, outcome = True, "out_of_time"
            elif self.steps >= self.max_steps:
                # Terminal, not truncated: exceeding the decision budget is a
                # task failure inside the MDP, not an artificial time limit, so
                # there is no value to bootstrap.
                reward += self.cfg.r_incomplete
                terminated, outcome = True, "incomplete"

        # Potential-based shaping; Phi := 0 in terminal states so the sum
        # telescopes to a constant and optimality is preserved.
        env_reward = reward
        if self.cfg.shaping:
            phi_next = 0.0 if terminated else self._potential()
            reward += self.cfg.gamma * phi_next - self._phi
            self._phi = phi_next
        return (self._obs(), float(reward), terminated, truncated,
                self._info(outcome=outcome, env_reward=env_reward))

    # -- convenience --------------------------------------------------------
    def _delivery_reserve(self) -> Tuple[float, float]:
        """Lower bound on the (energy, time) still needed to finish the task.

        Belief-independent: the cheapest candidate, so it never depends on what
        the agent currently believes. Used to keep the robot from spending its
        way out of being able to complete the job at all.
        """
        t = self.task
        mv = self.ved.quote(MOVE)
        if self.carrying >= 0:
            d = t.layout.d(self.pos, int(t.recep_cell))
            fin = self.ved.quote(PLACE)
            return d * mv.energy_mJ + fin.energy_mJ, d * mv.latency_ms + fin.latency_ms
        d = min(t.layout.d(self.pos, int(t.cand_cells[j]))
                + t.layout.d(int(t.cand_cells[j]), int(t.recep_cell))
                for j in range(t.k))
        pk, pl = self.ved.quote(PICK), self.ved.quote(PLACE)
        return (d * mv.energy_mJ + pk.energy_mJ + pl.energy_mJ,
                d * mv.latency_ms + pk.latency_ms + pl.latency_ms)

    def _affordable(self, op: str) -> bool:
        """Can the robot run `op` and still finish the task afterwards?

        Once the roofline replaced the placeholder compute costs, a single 3B
        inference on a Pi 5 came to 218 J and 32 s - more than half of both
        episode budgets in one action. Leaving such actions available made early
        exploration catastrophic: entropy collapsed within 200k steps and the
        policy abandoned information-gathering entirely, exactly the failure
        mode seen at high lambda_Q. Masking them is not a training trick, it is
        what a real runtime does - you do not schedule an inference you lack the
        battery to survive.
        """
        c = self.ved.quote(op)
        if not c.available:
            return False
        e_res, t_res = self._delivery_reserve()
        e_left = self.ved.profile.episode_energy_budget_mJ - self.ved.energy_spent_mJ
        t_left = self.ved.profile.episode_time_budget_ms - self.ved.time_spent_ms
        return (c.energy_mJ + e_res <= e_left) and (c.latency_ms + t_res <= t_left)

    def action_mask(self) -> np.ndarray:
        """Which actions are currently well-formed. Used by baselines and for
        optional masked-PPO ablations; the main agent does *not* receive it, so
        that learning to avoid invalid actions remains part of the problem."""
        t = self.task
        m = np.ones(N_ACTIONS, dtype=bool)
        for j in range(K_MAX):
            m[j] = j < t.k and t.layout.d(self.pos, int(t.cand_cells[j])) > 0
        m[A_GOTO_RECEP] = t.layout.d(self.pos, int(t.recep_cell)) > 0
        m[A_PICK] = self.carrying < 0 and any(
            self.pos == int(t.cand_cells[j]) for j in range(t.k))
        m[A_PLACE] = self.carrying >= 0 and self.pos == int(t.recep_cell)
        m[A_THINK_TINY] = self._affordable(THINK_TINY)
        m[A_THINK_BIG] = self._affordable(THINK_BIG)
        m[A_OFFLOAD] = self._affordable(OFFLOAD)
        m[A_ASK] = self.patience > 0 and self._affordable(ASK)
        if not m.any():                      # never strand the agent
            m[A_GOTO_RECEP if self.carrying >= 0 else 0] = True
        return m
