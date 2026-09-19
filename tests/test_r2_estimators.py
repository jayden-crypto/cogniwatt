#!/usr/bin/env python3
"""R2 estimator tests. Run: python tests/test_r2_estimators.py

`gamma` is the headline result of the R2 workstream, and it is not directly
observable -- it is fitted from an ICC that is itself estimated. So the ICC
estimator is validated against a generator whose rho is known in closed form
before any of it is pointed at a real model. If it cannot recover a known rho
from synthetic data, its gamma on real data means nothing.

These are additional to `tests/test_env.py`, which stays untouched (R1 owns it).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


# ---------------------------------------------------------------------------
@test
def test_icc_recovers_known_rho():
    """Beta-binomial data with rho = 1/(1+kappa) must come back out.

    This is the load-bearing test of the whole decay analysis.
    """
    from r2_analyze import icc_oneway

    for rho_true in (0.15, 0.35, 0.55, 0.75):
        kappa = (1 - rho_true) / rho_true
        p = 0.6
        rng = np.random.default_rng(7)
        M, N = 4000, 10
        theta = rng.beta(p * kappa, (1 - p) * kappa, size=M)
        mat = (rng.random((M, N)) < theta[:, None]).astype(float)
        rho_hat = icc_oneway(mat)
        assert abs(rho_hat - rho_true) < 0.03, \
            f"rho {rho_true}: estimated {rho_hat:.4f}"


@test
def test_icc_endpoints():
    """A deterministic model gives rho=1; independent repeats give rho~0."""
    from r2_analyze import icc_oneway

    rng = np.random.default_rng(0)
    M, N = 2000, 10
    # Perfectly consistent: each scenario is always right or always wrong.
    fixed = (rng.random(M) < 0.6).astype(float)
    assert icc_oneway(np.repeat(fixed[:, None], N, axis=1)) > 0.98

    # Fully independent draws at one shared rate.
    indep = (rng.random((M, N)) < 0.6).astype(float)
    assert icc_oneway(indep) < 0.03


@test
def test_gamma_is_monotone_decreasing_in_rho():
    """More correlated repeats must mean less information per repeat.

    This is the direction the env depends on: `evidence_weight` uses
    gamma**n, so a higher gamma has to mean a *more* informative next query.
    """
    from r2_analyze import fit_gamma

    gs = [fit_gamma(r, 10) for r in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)]
    assert all(a > b for a, b in zip(gs, gs[1:])), gs
    assert gs[0] > 0.8 and gs[-1] < 0.2, gs


@test
def test_gamma_stays_in_the_env_domain():
    """`load_precomputed` rejects gamma outside (0, 1]; the fit must never emit one."""
    from r2_analyze import fit_gamma

    for rho in np.linspace(0.001, 0.999, 60):
        g = fit_gamma(float(rho), 10)
        assert 0.0 < g <= 1.0, (rho, g)


@test
def test_neff_saturates_at_the_geometric_bound():
    """n_eff must approach 1/rho, matching R1's 1/(1-gamma) saturation story."""
    from r2_analyze import n_eff

    for rho in (0.2, 0.5, 0.8):
        v = n_eff(np.array([1e6]), rho)[0]
        assert abs(v - 1.0 / rho) / (1.0 / rho) < 0.01, (rho, v)


@test
def test_mock_backend_hits_its_targets():
    """The simulator's realised accuracy and rho must match what it was asked for."""
    from cogniwatt.dialogue.backends import MockBackend, Query
    from r2_analyze import icc_oneway

    be = MockBackend("t", accuracy=0.62, rho=0.40, position_bias=0.0,
                     format_violation=0.0, seed=3)
    M, N, k = 1500, 10, 3
    mat = np.zeros((M, N))
    for m in range(M):
        true_idx = m % k
        for i in range(N):
            r = be.answer(Query(f"q{m}_{i}", "", k, f"s{m}", true_idx, i))
            mat[m, i] = int(r.text == str(true_idx))
    assert abs(mat.mean() - 0.62) < 0.02, mat.mean()
    assert abs(icc_oneway(mat) - 0.40) < 0.04, icc_oneway(mat)


@test
def test_cluster_bootstrap_is_wider_than_wilson():
    """Correlated clusters must widen the interval, not narrow it.

    If it did not, the clustering would be decorative and every CI in the
    report would be overconfident.
    """
    import pandas as pd
    from r2_analyze import _acc, cluster_bootstrap, wilson

    rng = np.random.default_rng(1)
    rows = []
    # 40 cores, 12 near-identical scenarios each: heavy within-core correlation.
    for c in range(40):
        p = rng.beta(2, 2)
        for j in range(12):
            ok = rng.random() < p
            rows.append({"core_id": f"c{c}", "parse_status": "ok",
                         "correct": int(ok)})
    df = pd.DataFrame(rows)
    p, lo, hi = cluster_bootstrap(df, _acc, seed=0)
    wl, wh = wilson(int(df["correct"].sum()), len(df))
    assert (hi - lo) > 1.5 * (wh - wl), \
        f"clustered {hi - lo:.4f} vs wilson {wh - wl:.4f}"


@test
def test_parser_separates_format_failure_from_wrong_answer():
    from cogniwatt.dialogue.prompts import parse_choice

    assert parse_choice("C", 3) == (2, "ok")
    assert parse_choice(' {"choice": "B"} ', 3) == (1, "ok")
    assert parse_choice("B. the blue mug", 3) == (1, "ok")
    assert parse_choice("The answer is A", 3) == (0, "ok")
    assert parse_choice(" (C) ", 3) == (2, "ok")
    assert parse_choice("D", 3) == (3, "out_of_range")
    assert parse_choice("I cannot tell which one.", 3) == (None, "unparseable")
    assert parse_choice(None, 3) == (None, "unparseable")
    # A bare digit under letter labelling must NOT be coerced into an index --
    # that is the degenerate behaviour the letter scheme exists to expose.
    assert parse_choice("1", 3) == (None, "digit_answer")


@test
def test_prompt_uses_letters_not_digits():
    """Regression guard on the instrument itself.

    Digit labels made a 0.5B model answer "1" reflexively at 237/240, which
    scored as a plausible-looking 0.358 with a 99.8% parse rate. Reverting the
    template would silently reintroduce that.
    """
    from cogniwatt.dialogue.prompts import build_prompt

    p = build_prompt("put the mug away", ["a red mug", "a blue mug", "a white mug"])
    assert "A. a red mug" in p and "B. a blue mug" in p and "C. a white mug" in p
    assert "letter only" in p
    assert "0. " not in p and "1. " not in p


@test
def test_shuffle_maps_back_to_the_dataset_index():
    """The permutation round-trip -- getting this wrong silently ruins accuracy."""
    from cogniwatt.dialogue.prompts import shuffled_view

    rng = np.random.default_rng(0)
    descs = ["a", "b", "c", "d"]
    for _ in range(200):
        true_idx = int(rng.integers(0, 4))
        shown, perm, shown_true = shuffled_view(descs, true_idx, rng)
        assert shown[shown_true] == descs[true_idx]
        assert perm[shown_true] == true_idx
        assert sorted(perm) == [0, 1, 2, 3]


@test
def test_seed_is_stable_across_processes():
    """The paired-comparison guarantee depends on this.

    `hash()` is randomised per process for str (PEP 456), so seeding the
    candidate shuffle with it made tiers measured in separate runs see
    different permutations -- while the docstring claimed they saw the same
    one. A 4 GB card forces one model resident at a time, so separate runs are
    the normal case, not the exception.
    """
    import subprocess

    from cogniwatt.dialogue.prompts import stable_seed

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("import sys; sys.path.insert(0, %r);"
            "from cogniwatt.dialogue.prompts import stable_seed;"
            "print(stable_seed('sc_0001', 0))" % here)
    seen = {subprocess.check_output([sys.executable, "-c", code], text=True).strip()
            for _ in range(3)}
    assert len(seen) == 1, f"seed varies across processes: {seen}"
    assert seen.pop() == str(stable_seed("sc_0001", 0))


@test
def test_dataset_has_no_position_or_colour_shortcut():
    """The built dataset must be beatable only by reasoning, not by a prior."""
    from collections import Counter

    from cogniwatt.dialogue.schema import load_scenarios

    from r2_build_dataset import build

    # The handoff's >= 600 / >= 200-per-k quota is a property of the *authored
    # pool*: the build emits 720 so that the human audit can drop cores without
    # the survivors falling under it. The shipped file is what survives the
    # audit (348 from 29 cores after R1's reply 7), so the quota is checked on
    # the unfiltered build and the shipped file is held to the shortcut checks
    # plus a floor that keeps the chi-square guard meaningful.
    full = build()
    assert len(full) >= 600, f"authored pool builds only {len(full)}; handoff asks for >= 600"
    for k in (2, 3, 4):
        assert sum(1 for s in full if s.k == k) >= 200, f"k={k}: pool under quota"

    path = os.path.join("data", "r2_scenarios.jsonl")
    if not os.path.exists(path):
        print("    (skipped: run scripts/r2_build_dataset.py first)")
        return
    ss = load_scenarios(path)
    for k in (2, 3, 4):
        sub = [s for s in ss if s.k == k]
        assert len(sub) >= 100, f"k={k}: only {len(sub)} shipped scenarios"
        c = Counter(s.true_idx for s in sub)
        _assert_no_detectable_prior(c, f"k={k} position")
    cols = Counter(s.candidates[s.true_idx].color for s in ss)
    _assert_no_detectable_prior(cols, "colour")


@test
def test_audit_filter_keeps_ids_aligned_with_raw_table():
    """Dropping cores must never renumber the survivors.

    Ids are positional, and the raw response table is keyed by them. The first
    audit filter ran before id assignment; with 29 of 60 cores kept, every one
    of the 348 surviving ids then named a different scenario from the one
    measured under that id, and the analysis would have scored dropped-core
    responses as the audited set without an error. This pins the fix.
    """
    from r2_build_dataset import apply_audit, audited_core_ids, build
    from cogniwatt.dialogue.authored import CORES

    full = build()
    full_map = {s.id: s.core_id for s in full}

    # Synthetic verdicts: keep every other core. Independent of the real audit
    # file so the test does not change meaning when the audit is re-run.
    import json, tempfile
    kept_ids = {c.id for i, c in enumerate(CORES) if i % 2 == 0}
    verdicts = {c.id: {"status": "agree" if c.id in kept_ids else "disagree"}
                for c in CORES}
    vp = os.path.join(tempfile.mkdtemp(), "audit.json")
    with open(vp, "w", encoding="utf-8") as f:
        json.dump({"verdicts": verdicts}, f)

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        kept = audited_core_ids(list(CORES), vp, drop_unaudited=True)
        aud = apply_audit(full, kept)

    assert kept == kept_ids
    assert {s.core_id for s in aud} == kept_ids, "filter kept the wrong cores"
    assert all(full_map[s.id] == s.core_id for s in aud), \
        "audit filter renumbered scenarios; raw-table join is broken"
    assert all(s.core_id in kept_ids for s in aud)

    # And against the real raw table when it is present.
    raw = os.path.join("data", "r2_raw_all.csv.gz")
    if os.path.exists(raw):
        import pandas as pd
        rm = (pd.read_csv(raw, usecols=["scenario_id", "core_id"])
                .drop_duplicates("scenario_id").set_index("scenario_id")["core_id"])
        bad = [s.id for s in aud if s.id in rm.index and rm[s.id] != s.core_id]
        assert not bad, f"{len(bad)} audited ids disagree with the raw table, e.g. {bad[:3]}"


def _assert_no_detectable_prior(counts, what, alpha=0.01):
    """Mirror of `assert_no_shortcuts` in scripts/r2_build_dataset.py.

    R1 edit: this test previously asserted `max - min <= 1`, which the build
    script itself stopped using when `--drop-contested` landed -- dropping a
    non-random subset makes exact balance unattainable (k=2 comes out 108/112).
    R1_REPLY_2.md section 1 replaced the build guard with a chi-square test of
    *detectability*, which is the property that actually matters, but the test
    file was not updated with it. 108/112 gives chi2 p = 0.787.

    Kept identical to the build guard so there is one definition of "no
    shortcut" rather than two that can drift apart.
    """
    from scipy.stats import chisquare
    obs = np.array(list(counts.values()), dtype=float)
    assert obs.sum() > 0, f"{what}: no observations"
    p = float(chisquare(obs).pvalue)
    assert p > alpha, f"{what}: detectable prior (chi2 p={p:.4f}) {dict(counts)}"


@test
def test_clarify_never_hallucinates_an_object():
    """Deliverable E asks for a 0% hallucination rate. Assert it, don't hope."""
    from cogniwatt.dialogue.clarify import best_question, groundedness
    from cogniwatt.dialogue.schema import load_scenarios

    path = os.path.join("data", "r2_scenarios.jsonl")
    if not os.path.exists(path):
        print("    (skipped: run scripts/r2_build_dataset.py first)")
        return
    bad = []
    for s in load_scenarios(path):
        descs = [c.desc for c in s.candidates]
        ok, offending = groundedness(best_question(descs).text, descs)
        if not ok:
            bad.append((s.id, offending))
    assert not bad, f"ungrounded questions: {bad[:5]}"


@test
def test_clarify_never_leaks_the_colour():
    """A question naming the colour would resolve the referent without the
    human doing the disambiguating work -- that is not the `ask` channel."""
    from cogniwatt.dialogue.clarify import best_question
    from cogniwatt.dialogue.schema import COLORS, load_scenarios

    path = os.path.join("data", "r2_scenarios.jsonl")
    if not os.path.exists(path):
        print("    (skipped: run scripts/r2_build_dataset.py first)")
        return
    import re
    for s in load_scenarios(path)[::7]:
        q = best_question([c.desc for c in s.candidates]).text.lower()
        for col in COLORS:
            assert not re.search(rf"\b{col}\b", q), f"{s.id}: question leaks {col!r}: {q}"


@test
def test_clarify_respects_the_polar_ceiling():
    """No yes/no question can carry more than 1 - H(epsilon) bits."""
    from cogniwatt.dialogue.clarify import (best_question, polar_ceiling,
                                            expected_information_gain)

    descs = [f"a red mug on shelf {i}" for i in range(4)]
    for eps in (0.0001, 0.02, 0.08, 0.2, 0.4):
        cap = polar_ceiling(eps)
        q = best_question(descs, epsilon=eps)
        assert q.expected_ig <= cap + 1e-9, (eps, q.expected_ig, cap)
        # With k=4 uniform, a 2-vs-2 split hits the ceiling exactly.
        assert abs(q.expected_ig - cap) < 1e-6, (eps, q.expected_ig, cap)


@test
def test_clarify_beats_random_once_the_belief_is_skewed():
    """The Deliverable-E claim: selecting the split is worth something.

    Under a *uniform* belief at k<=3 every question is equivalent by symmetry,
    so the honest claim is conditional -- selection pays once prior thinking
    has skewed the belief. Assert the conditional form, not a blanket win.
    """
    from cogniwatt.dialogue.clarify import best_question, expected_information_gain

    rng = np.random.default_rng(0)
    for k in (3, 4):
        descs = [f"a red mug in position {i}" for i in range(k)]
        sel, rnd = [], []
        for _ in range(400):
            b = rng.dirichlet(np.full(k, 0.7))
            sel.append(best_question(descs, b).expected_ig)
            i = int(rng.integers(0, k))
            rnd.append(expected_information_gain(b, (i,), 0.08))
        assert np.mean(sel) > np.mean(rnd) * 1.2, \
            f"k={k}: selection {np.mean(sel):.4f} vs random {np.mean(rnd):.4f}"


@test
def test_posterior_is_a_distribution_and_moves_the_right_way():
    from cogniwatt.dialogue.clarify import posterior

    b = np.array([0.25, 0.25, 0.25, 0.25])
    for eps in (0.01, 0.08, 0.3):
        p_yes = posterior(b, (0, 1), True, eps)
        p_no = posterior(b, (0, 1), False, eps)
        for p in (p_yes, p_no):
            assert abs(p.sum() - 1.0) < 1e-9
            assert (p >= 0).all()
        # "yes" must raise the subset's mass, "no" must lower it.
        assert p_yes[:2].sum() > 0.5 > p_no[:2].sum(), (eps, p_yes, p_no)


# ---------------------------------------------------------------------------
def main() -> int:
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(_TESTS) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
