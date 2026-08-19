# CogniWatt — Problem Formalization (Lab 1, R1 deliverable)

**Workstream:** R1 — RL & Environment
**Artefact:** `cogniwatt/env/`, `cogniwatt/agents/`, `tests/test_env.py`

---

## 1. Problem statement

A domestic robot running on an embedded compute platform receives a spoken
instruction that is *referentially underspecified*: "put the mug on the shelf",
in a house containing several mugs. Resolving the ambiguity requires evidence,
and every source of evidence has a price:

| Evidence source | Pays in |
| --- | --- |
| On-device small language model (0.5B, INT8) | energy, latency |
| On-device larger model (3B, Q4) | more energy, more latency, thermal headroom |
| Offloading to a cloud model | radio energy, network latency, failure risk |
| Asking the human a clarifying question | the human's patience |

Acting on a wrong referent is expensive too: the robot walks to the wrong room,
picks up the wrong object, and delivers it — burning an order of magnitude more
energy than any amount of reasoning would have cost.

**The decision problem is therefore economic:** how much cognition, and of which
kind, should be purchased before committing to an irreversible physical act?

## 2. Formal model

CogniWatt is the POMDP $\langle S, A, T, R, \Omega, O, \gamma \rangle$:

**State** $s = (\ell, \text{pos}, c, i^\*, \mathbf{e}, \tau, \eta, \rho, n)$ —
floorplan $\ell$, robot cell, carried object $c$, **true referent $i^\*$
(unobserved)**, device resource state $\mathbf{e}$ (energy spent, latency spent),
die temperature $\tau$, link quality $\eta$, remaining human patience $\rho$,
per-tier query counts $n$.

**Observation.** The agent never sees $i^\*$. It maintains a Bayesian posterior
$b \in \Delta^{k-1}$ over the $k$ candidates, updated in log-space
(`env/belief.py`). Spatial structure is fully observed — a domestic robot has a
map — so *all* partial observability is semantic. This is a deliberate
modelling choice: it concentrates the difficulty in the part of the problem the
project is about, and keeps the POMDP solvable on a laptop.

**Action space** $|A| = 11$, hierarchical:

- $\texttt{goto\_cand}_i$, $i \in \{0..3\}$, and $\texttt{goto\_recep}$ —
  macro-actions; navigation is solved exactly by precomputed BFS and **billed
  per traversed grid cell**
- $\texttt{pick}$, $\texttt{place}$ — $\texttt{place}$ is the irreversible commitment
- $\texttt{think\_tiny}$, $\texttt{think\_big}$, $\texttt{offload}$, $\texttt{ask}$

Physical actions and cognitive actions occupy **one action space**. This is the
core structural claim of the project.

**Reward.**

$$R = \underbrace{R_{\text{succ}}\mathbb{1}[\text{correct}] + R_{\text{wrong}}\mathbb{1}[\text{incorrect}]}_{\text{task}} - \underbrace{\lambda_E E - \lambda_T T}_{\text{hardware}} - \underbrace{\lambda_Q q}_{\text{human}}$$

with $R_{\text{succ}}=+10$, $R_{\text{wrong}}=-10$, and $\lambda$ scaled so that
exhausting the whole energy or latency budget costs about as much as failing.
The $\lambda$s are the Pareto knobs swept in the results study.

## 3. Two modelling contributions

### 3.1 Evidence saturation

Querying the same model twice does not give two independent observations: a
quantised SLM at low temperature is nearly deterministic, and its residual
errors are systematically correlated. We model the $n$-th query to tier $t$ as
contributing $\gamma_t^{\,n-1}$ of a full observation, so information from a tier
saturates at $1/(1-\gamma_t)$ effective draws.

Measured consequence (K=3, `think_tiny` at 45% accuracy):

| tier | 1 query | 2 | 3 | ∞ |
| --- | --- | --- | --- | --- |
| `think_tiny` | 0.450 | 0.505 | 0.530 | **0.550** |
| `think_big` | 0.720 | 0.864 | 0.912 | **0.950** |
| `offload` | 0.880 | 0.949 | 0.963 | **0.969** |
| `ask` | 0.892 | 0.927 | 0.931 | **0.932** |

**A cheap model cannot be chained up to the competence of an expensive one.**
Without this the problem is degenerate: the optimal policy is simply to spam
whichever channel has the best information-per-Joule. With it, the optimal
policy is a *portfolio* across channels — which is what both our heuristic and
our learned agent discover.

### 3.2 Locomotion dominance

At 0.5 m/s and 10 W drive power, one grid step costs ~10 J; a 3B inference on
Orin Nano costs ~3.6 J. Compute is cheap relative to motion. The *energy*
constraint therefore binds mainly on small-battery NPU-class devices, whereas
*latency* and *patience* bind everywhere. We keep the three cost terms separate
and sweep them rather than collapsing them into one scalar a priori.

## 4. Devices

Policies are conditioned on a 5-dim **datasheet** feature vector — peak INT8
TOPS, memory bandwidth, TDP, battery capacity, NPU presence. Deliberately *not*
the cost table, otherwise zero-shot transfer to an unseen device would be
vacuous.

| | train | | | | held out | |
| --- | --- | --- | --- | --- | --- | --- |
| | `orin_nano` | `pi5` | `orin_nx` | `m2_air` | `jetson_nano_2gb` | `rtx3060` |

`jetson_nano_2gb` cannot fit the 3B model at all — `think_big` is *unavailable* —
so a policy that has only ever seen devices where it works must adapt from
datasheet features alone.

## 5. Reward-hacking incident (kept deliberately)

The first two training runs collapsed to **paralysis**: entropy → 0, success →
0, the policy emitting `think_tiny` 30 times per episode and absorbing the
end-of-episode penalty. This was correct behaviour under a flawed reward — at
the time, stalling scored −8.24 against −9.52 for attempting the task from the
prior. Non-delivery is now priced as the *worst* outcome ($-15$): for an
assistant, unbounded latency is total failure, while a wrong placement is at
least recoverable.

A second, subtler failure followed. The shaping potential measured distance to
`belief.argmax`, so evidence that moved the argmax to a farther candidate
produced *negative* shaping — the agent was penalised for becoming better
informed, and plateaued at exactly the no-information success rate (0.33 vs the
0.36 uninformed baseline). Making the potential belief-independent moved
success to 0.93 within 400k steps.

Both are retained in the report as worked examples of reward misspecification.

## 6. Verification

`tests/test_env.py` — 12/12 passing. Notably `test_shaping_preserves_task_return`
confirms numerically that $\sum_t \gamma^t F(s_t,s_{t+1}) = -\Phi(s_0)$
(Ng et al., 1999), so shaping provably does not change the optimal policy, and
`test_evidence_saturation_is_bounded` pins the saturation property.
