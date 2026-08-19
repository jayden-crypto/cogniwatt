"""Self-contained test suite for HomeGrid-Edge. Owner: R1.

Run: python3 tests/test_env.py
No pytest dependency, so it runs identically on both team machines.
"""
import pathlib, sys, traceback
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from cogniwatt.agents.baselines import NoThinkAgent, OracleAgent
from cogniwatt.analysis.evaluate import evaluate
from cogniwatt.env.belief import BeliefState
from cogniwatt.env.homegrid import (A_ASK, A_PLACE, A_THINK_BIG, A_THINK_TINY,
                                    HELDOUT_DEVICES, N_ACTIONS, OBS_DIM,
                                    TRAIN_DEVICES, HomeGridEdge, RewardConfig)
from cogniwatt.env.tasks import LAYOUTS, sample_task
from cogniwatt.interfaces.dialogue import DialogueOracle
from cogniwatt.interfaces.ved import THINK_BIG, VirtualEdgeDevice

TESTS = []
def test(fn):
    TESTS.append(fn); return fn


@test
def test_spaces_and_obs():
    env = HomeGridEdge(seed=0); o, _ = env.reset()
    assert o.shape == (OBS_DIM,) and o.dtype == np.float32
    assert env.action_space.n == N_ACTIONS
    assert np.all(np.isfinite(o)) and o.min() >= -1e-6 and o.max() <= 1.0 + 1e-6


@test
def test_seed_determinism():
    a = HomeGridEdge(seed=0); b = HomeGridEdge(seed=0)
    o1, _ = a.reset(seed=7); o2, _ = b.reset(seed=7)
    assert np.allclose(o1, o2)
    for act in [A_THINK_TINY, A_THINK_BIG, 0, A_ASK]:
        r1 = a.step(act); r2 = b.step(act)
        assert np.allclose(r1[0], r2[0]) and abs(r1[1] - r2[1]) < 1e-9


@test
def test_belief_is_a_distribution():
    env = HomeGridEdge(seed=1); env.reset(seed=3)
    for _ in range(25):
        env.step(int(env.rng.integers(N_ACTIONS)))
        p = env.belief.probs
        assert abs(p.sum() - 1.0) < 1e-9 and np.all(p >= 0)


@test
def test_evidence_saturation_is_bounded():
    """Chaining the cheap model must not reach arbitrary confidence."""
    o = DialogueOracle(); b = BeliefState(3, 4); L = o.likelihood(THINK_BIG, 3)
    bt = BeliefState(3, 4); Lt = o.likelihood("think_tiny", 3)
    for n in range(200):
        bt.update(2, Lt, weight=o.evidence_weight("think_tiny", n))
        b.update(2, L, weight=o.evidence_weight(THINK_BIG, n))
    assert bt.max_prob < 0.60, f"tiny saturated too high: {bt.max_prob}"
    assert b.max_prob > bt.max_prob, "big must dominate tiny at saturation"


@test
def test_resources_are_monotone():
    env = HomeGridEdge(seed=2); env.reset(seed=5)
    e, t = 0.0, 0.0
    for _ in range(20):
        _, _, term, trunc, info = env.step(int(env.rng.integers(N_ACTIONS)))
        assert info["energy_mJ"] >= e - 1e-9 and info["latency_ms"] >= t - 1e-9
        e, t = info["energy_mJ"], info["latency_ms"]
        if term or trunc:
            break


@test
def test_shaping_preserves_task_return():
    """env_reward must be shaping-free, and shaping must telescope to a constant."""
    kw = dict(devices=TRAIN_DEVICES, seed=0)
    on = HomeGridEdge(reward=RewardConfig(shaping=True), **kw)
    off = HomeGridEdge(reward=RewardConfig(shaping=False), **kw)
    # The action sequence must come from an INDEPENDENT stream: drawing it from
    # on.rng would advance one env's random state and not the other's, and the
    # two runs would silently diverge.
    gamma = RewardConfig().gamma
    for ep in range(30):
        on.reset(seed=100 + ep); off.reset(seed=100 + ep)
        phi0 = on._phi
        arng = np.random.default_rng(9000 + ep)
        se = so = disc_shaping = 0.0
        for t in range(30):
            a = int(arng.integers(N_ACTIONS))
            _, r1, t1, u1, i1 = on.step(a); _, r2, t2, u2, i2 = off.step(a)
            se += i1["env_reward"]; so += i2["env_reward"]
            disc_shaping += (gamma ** t) * (r1 - i1["env_reward"])
            assert (t1, u1) == (t2, u2), "shaping changed the termination structure"
            if t1 or u1:
                break
        assert abs(se - so) < 1e-9, "shaping leaked into env_reward"
        # Ng et al. (1999): sum_t gamma^t (gamma*Phi_{t+1} - Phi_t) = gamma^T*Phi_T - Phi_0,
        # and Phi_T := 0 in terminal states, so the discounted shaping is exactly -Phi_0.
        assert abs(disc_shaping + phi0) < 1e-6, \
            f"shaping did not telescope: {disc_shaping} vs {-phi0}"


@test
def test_action_mask_never_empty():
    env = HomeGridEdge(seed=4)
    for ep in range(40):
        env.reset(seed=200 + ep)
        for _ in range(30):
            m = env.action_mask()
            assert m.any(), "no legal action available"
            _, _, term, trunc, _ = env.step(int(env.rng.choice(np.flatnonzero(m))))
            if term or trunc:
                break


@test
def test_memory_limited_device_cannot_run_the_big_model():
    env = HomeGridEdge(devices=("jetson_nano_2gb",), seed=0)
    env.reset(device="jetson_nano_2gb")
    assert not env.ved.quote(THINK_BIG).available
    assert not env.action_mask()[A_THINK_BIG]
    before = env.ved.energy_spent_mJ
    _, r, _, _, _ = env.step(A_THINK_BIG)
    assert env.ved.energy_spent_mJ == before, "charged for an unavailable model"
    assert r < 0


@test
def test_patience_is_finite():
    env = HomeGridEdge(seed=0); env.reset(seed=11)
    n = env.oracle.patience_max
    for _ in range(n):
        env.step(A_ASK)
    assert env.patience == 0 and not env.action_mask()[A_ASK]


@test
def test_oracle_and_nothink_bracket_the_problem():
    env_fn = lambda: HomeGridEdge(devices=TRAIN_DEVICES, seed=0)
    o = evaluate(OracleAgent(), env_fn, n_episodes=200, devices=list(TRAIN_DEVICES))
    n = evaluate(NoThinkAgent(), env_fn, n_episodes=200, devices=list(TRAIN_DEVICES))
    assert o.success.mean() == 1.0, f"oracle must be perfect, got {o.success.mean()}"
    assert 0.25 < n.success.mean() < 0.50, f"no_think should be ~1/k, got {n.success.mean()}"
    assert o.ret.mean() > n.ret.mean()


@test
def test_all_layout_cells_reachable():
    for L in LAYOUTS:
        d = L.dist[np.ix_(L.free_cells, L.free_cells)]
        assert (d >= 0).all(), "disconnected floorplan"


@test
def test_dialogue_artefact_roundtrip():
    """The exact contract R2 delivers against. See docs/R2_HANDOFF.md.

    Writes a synthetic artefact in R2's schema, loads it, and checks the
    numbers reach the environment's belief filter intact.
    """
    import json, tempfile, os
    from cogniwatt.interfaces.dialogue import load_precomputed, REQUIRED_KEYS
    from cogniwatt.interfaces.ved import ASK, OFFLOAD, THINK_BIG, THINK_TINY

    tiers = [THINK_TINY, THINK_BIG, OFFLOAD, ASK]
    ks = [2, 3, 4]
    acc_by_k = np.array([[.62, .45, .38], [.86, .72, .64],
                         [.95, .88, .82], [.98, .97, .96]])
    path = os.path.join(tempfile.mkdtemp(), "dialogue_measured.npz")
    np.savez(path, tiers=np.array(tiers), accuracy=acc_by_k.mean(axis=1),
             evidence_decay=np.array([.41, .58, .33, .12]), asr_wer=0.113,
             k_values=np.array(ks), accuracy_by_k=acc_by_k,
             n_samples=np.full((4, 3), 500),
             meta_json=json.dumps({"tiny": "qwen2.5-0.5b-q8", "temp": 0.7}))

    o = load_precomputed(path, strict=True)
    assert abs(o.evidence_decay[THINK_TINY] - 0.41) < 1e-9, "decay was dropped"
    assert abs(o.accuracy(THINK_BIG, k=2) - 0.86) < 1e-9, "per-k accuracy ignored"
    assert abs(o.accuracy(THINK_BIG, k=4) - 0.64) < 1e-9
    assert o.accuracy(THINK_BIG, k=4) < o.accuracy(THINK_BIG, k=2), "k must make it harder"
    assert o.meta["tiny"] == "qwen2.5-0.5b-q8", "provenance lost"
    # ASR must degrade the human channel, not pass through untouched.
    assert o.accuracy(ASK, k=3) < 0.97

    # And it must actually drive the environment.
    env = HomeGridEdge(devices=TRAIN_DEVICES, oracle=o, seed=0)
    env.reset(seed=1)
    L = env._likelihoods[THINK_BIG]
    assert abs(L.sum(axis=1)[0] - 1.0) < 1e-9, "likelihood not row-stochastic"

    # A malformed artefact must fail loudly, never silently fall back.
    bad = os.path.join(tempfile.mkdtemp(), "bad.npz")
    np.savez(bad, tiers=np.array(tiers), accuracy=np.ones(4) * .5, asr_wer=0.1)
    try:
        load_precomputed(bad); assert False, "missing evidence_decay must raise"
    except KeyError:
        pass


@test
def test_heldout_devices_are_disjoint():
    assert not (set(TRAIN_DEVICES) & set(HELDOUT_DEVICES)), "transfer claim would be invalid"


if __name__ == "__main__":
    ok = fail = 0
    for t in TESTS:
        try:
            t(); print(f"  PASS  {t.__name__}"); ok += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}"); traceback.print_exc(); fail += 1
    print(f"\n{ok} passed, {fail} failed")
    sys.exit(1 if fail else 0)
