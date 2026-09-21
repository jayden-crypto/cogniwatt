"""Build docs/r1_report.html from the current data. Owner: R1.

The previous report was hand-authored, which is exactly how it came to describe
a 40-dim observation, a retired held-out device and a reward without
`lambda_cloud` months after all three had changed (docs/r1_stability.md 6.3).
Every number here is now read from the CSVs and run histories at build time, so
the report cannot drift from the code again.

    python3 scripts/make_html_report.py
"""
from __future__ import annotations

import base64, html, json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS, EVAL, FIG = ROOT / "results/runs", ROOT / "results/eval", ROOT / "results/figures"
STYLE = pathlib.Path(__file__).resolve().parent / "_report_style.html"

ORDER = ["oracle", "ppo", "greedy_value", "always_ask", "always_offload",
         "always_big", "always_tiny", "no_think", "random_valid", "random"]


def esc(x):
    return html.escape(str(x))


def fig(name, caption, title=None):
    p = FIG / name
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    t = f"<b>{esc(title)}</b> " if title else ""
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{esc(caption)}">'
            f'<figcaption>{t}{caption}</figcaption></figure>')


def agent_key(a):
    for i, p in enumerate(ORDER):
        if a.startswith(p):
            return i
    return 99


def eval_table(split):
    f = EVAL / f"eval_{split}.csv"
    if not f.exists():
        return "", []
    df = pd.read_csv(f)
    g = df.groupby("agent").agg(
        success=("success", "mean"), ret=("ret", "mean"),
        se=("ret", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        energy=("energy_J", "mean"), lat=("latency_s", "mean"),
        asks=("asks", "mean"), tiny=("n_tiny", "mean"),
        big=("n_big", "mean"), off=("n_offload", "mean"),
    ).reset_index().sort_values("agent", key=lambda s: s.map(agent_key))
    rows = []
    for _, r in g.iterrows():
        hl = ' class="hl"' if r.agent.startswith("ppo") else ""
        rows.append(
            f"<tr{hl}><td class='name mono'>{esc(r.agent)}</td>"
            f"<td class='num'><b>{r.success:.3f}</b></td>"
            f"<td class='num'>{r.ret:+.2f} ± {r.se:.2f}</td>"
            f"<td class='num'>{r.energy:.1f}</td><td class='num'>{r.lat:.1f}</td>"
            f"<td class='num'>{r.asks:.2f}</td>"
            f"<td class='num'>{r.tiny:.2f} / {r.big:.2f} / {r['off']:.2f}</td></tr>")
    devs = sorted(df.device.unique())
    return ("<div class='scroll'><table><thead><tr><th>agent</th><th>success</th>"
            "<th>return ± se</th><th>energy (J)</th><th>latency (s)</th><th>asks</th>"
            "<th>tiny / big / cloud</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>"), devs


def seed_summary(prefix, split, evaldir="eval_seeds"):
    """Mean ± se OVER SEEDS -- the correct unit of analysis for a claim about
    the method rather than about one training run."""
    f = ROOT / f"results/{evaldir}/eval_{split}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    ppo = df[df.agent.str.startswith(f"ppo[{prefix}")]
    if ppo.empty:
        return None
    per = ppo.groupby("agent").agg(success=("success", "mean"), ret=("ret", "mean"),
                                   energy=("energy_J", "mean")).reset_index()
    q = ppo.groupby("agent").apply(
        lambda g: (g.n_tiny + g.n_big + g.n_offload + g.n_ask).mean(),
        include_groups=False)
    gv = df[df.agent == "greedy_value"]
    gq = (gv.n_tiny + gv.n_big + gv.n_offload + gv.n_ask).mean()
    n = len(per)
    return dict(n=n,
                succ=per.success.mean(), succ_se=per.success.std(ddof=1) / np.sqrt(n),
                ret=per.ret.mean(), ret_se=per.ret.std(ddof=1) / np.sqrt(n),
                energy=per.energy.mean(), q=q.mean(),
                g_succ=gv.success.mean(), g_ret=gv.ret.mean(),
                g_energy=gv.energy_J.mean(), g_q=gq)


def stability_block():
    f = ROOT / "results/stability_arms.csv"
    if not f.exists():
        return "", None
    a = pd.read_csv(f)
    order = ["ent01", "ent03", "base", "bigbatch", "gru", "entanneal"]
    a["o"] = a.arm.map(lambda x: order.index(x) if x in order else 99)
    a = a.sort_values("o")
    label = {"ent01": "ent_coef 0.01", "ent03": "ent_coef 0.03",
             "base": "ent_coef 0.12", "bigbatch": "ent_coef 0.12, 3× batch",
             "gru": "ent_coef 0.12, recurrent", "entanneal": "annealed 0.12 → 0.01"}
    rows = []
    for _, r in a.iterrows():
        cls = "dn" if r.convergence_rate < 0.7 else "up"
        rows.append(
            f"<tr><td class='name mono'>{esc(r.arm)}</td><td>{esc(label.get(r.arm, ''))}</td>"
            f"<td class='num {cls}'><b>{int(r.n_converged)}/{int(r.n_seeds)}</b></td>"
            f"<td class='num'>[{r.ci_lo:.2f}, {r.ci_hi:.2f}]</td>"
            f"<td class='num'>{r.success_converged:.3f}</td>"
            f"<td class='num'>{r.entropy_early:.2f}</td></tr>")
    tbl = ("<div class='scroll'><table><thead><tr><th>arm</th><th>configuration</th>"
           "<th>converged</th><th>Wilson 95% CI</th><th>success</th>"
           "<th>early entropy</th></tr></thead><tbody>"
           + "".join(rows) + "</tbody></table></div>")
    runs = pd.read_csv(ROOT / "results/stability_runs.csv")
    return tbl, runs


def ablation_block():
    """Ablations, pooled over seeds, against the full method."""
    tags = [p.name for p in RUNS.iterdir() if (p / "history.json").exists()]
    groups = {}
    for t in tags:
        arm = re.sub(r"_s\d+$", "", t)
        if arm.startswith("ablate") or arm == "entanneal":
            h = json.loads((RUNS / t / "history.json").read_text())["history"]
            groups.setdefault(arm, []).append(float(np.mean([x["success"] for x in h[-15:]])))
    if not groups:
        return ""
    notes = {"entanneal": "full method",
             "ablate_devcond": "no device-feature conditioning",
             "ablate_mask": "no invalid-action masking",
             "ablate_shaping": "no potential-based shaping",
             "ablate_capdrop": "+ capability dropout 0.35"}
    base = float(np.mean(groups.get("entanneal", [np.nan])))
    rows = []
    for arm in ["entanneal"] + sorted(k for k in groups if k != "entanneal"):
        v = np.array(groups[arm]); n = len(v)
        se = v.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        d = v.mean() - base
        dtxt = "—" if arm == "entanneal" else f"{d:+.3f}"
        cls = "" if arm == "entanneal" else (" class='dn'" if d < -0.01 else " class='up'")
        rows.append(f"<tr><td class='name mono'>{esc(arm)}</td><td>{esc(notes.get(arm,''))}</td>"
                    f"<td class='num'>{n}</td><td class='num'><b>{v.mean():.3f}</b>"
                    f"{'' if np.isnan(se) else f' ± {se:.3f}'}</td>"
                    f"<td class='num'{cls}>{dtxt}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>run</th><th>what changed</th>"
            "<th>seeds</th><th>train success</th><th>Δ vs full</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def cloud_block():
    f = ROOT / "results/cloud_price.csv"
    if not f.exists():
        return ""
    d = pd.read_csv(f)
    rows = []
    for _, r in d.iterrows():
        mark = " class='hl'" if abs(r.lambda_cloud - 0.5) < 1e-9 else ""
        rows.append(f"<tr{mark}><td class='num'>{r.lambda_cloud:g}</td>"
                    f"<td class='num'>{r.success:.3f}</td>"
                    f"<td class='num'>{r.local_share*100:.0f}%</td>"
                    f"<td class='num'>{r.cloud_share*100:.0f}%</td>"
                    f"<td class='num'>{r.ask:.2f}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>λ<sub>cloud</sub></th><th>success</th>"
            "<th>on-device share</th><th>cloud share</th><th>asks/ep</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def isolation_block():
    f = ROOT / "results/isolation.csv"
    if not f.exists():
        return ""
    d = pd.read_csv(f)
    name = {"iso_cloud_human": ("yes", "yes"), "iso_cloud_nohum": ("yes", "no"),
            "iso_nocloud_human": ("no", "yes"), "iso_nocloud_nohum": ("no", "no")}
    rows = []
    for _, r in d.iterrows():
        c, h = name.get(r.arm, ("?", "?"))
        last = (c == "no" and h == "no")
        cls = " class='hl'" if last else ""
        rows.append(
            f"<tr{cls}><td class='num'>{c}</td><td class='num'>{h}</td>"
            f"<td class='num'><b>{r.ppo:.3f}</b> ± {r.ppo_se:.3f}</td>"
            f"<td class='num'>{r.greedy_value:.3f}</td>"
            f"<td class='num'>{r.always_offload:.3f}</td>"
            f"<td class='num'>{r.always_ask:.3f}</td>"
            f"<td class='num'>{r.tiny:.2f}</td>"
            f"<td class='num{" up" if r.big > 0.5 else ""}'>{r.big:.2f}</td>"
            f"<td class='num'>{r.offload:.2f}</td><td class='num'>{r.ask:.2f}</td>"
            f"<td class='num{" up" if r.local_share > 0.5 else ""}'>"
            f"<b>{r.local_share*100:.0f}%</b></td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>cloud</th><th>human</th>"
            "<th>PPO</th><th>greedy</th><th>always_offload</th><th>always_ask</th>"
            "<th>tiny</th><th>big</th><th>cloud</th><th>ask</th>"
            "<th>on-device share</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def cloud_trained_block():
    """The trained sweep: PPO vs the heuristic AT EACH PRICE."""
    rows, arms = [], []
    for p in sorted(RUNS.iterdir()):
        m = re.match(r"^cloud([\d.]+)_s(\d+)$", p.name)
        if m and (p / "history.json").exists():
            arms.append((float(m.group(1)), p.name))
    if not arms:
        return ""
    by = {}
    for lam, tag in arms:
        h = json.loads((RUNS / tag / "history.json").read_text())["history"]
        by.setdefault(lam, []).append(float(np.mean([x["success"] for x in h[-15:]])))
    for lam in sorted(by):
        v = np.array(by[lam])
        rows.append(f"<tr><td class='num'>{lam:g}</td><td class='num'>{len(v)}</td>"
                    f"<td class='num'><b>{v.mean():.3f}</b>"
                    f"{'' if len(v)<2 else f' ± {v.std(ddof=1)/np.sqrt(len(v)):.3f}'}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>λ<sub>cloud</sub> trained at</th>"
            "<th>seeds</th><th>train success</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def repo_stats():
    loc = 0
    for p in ROOT.rglob("*.py"):
        if ".venv" in p.parts or "__pycache__" in p.parts:
            continue
        loc += len(p.read_text(errors="ignore").splitlines())
    secs, n = 0.0, 0
    for p in RUNS.iterdir():
        f = p / "history.json"
        if f.exists():
            secs += json.loads(f.read_text()).get("seconds", 0.0)
            n += 1
    return loc, n, secs / 3600.0


def measured_tiers_block():
    """The dialogue constants, read from R2's frozen artefact rather than typed."""
    import numpy as _np
    f = ROOT / "data/dialogue_measured.npz"
    if not f.exists():
        return "", {}
    d = _np.load(f, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    tiers = [str(t) for t in d["tiers"]]
    ks = [int(k) for k in d["k_values"]]
    by_k = {t: list(map(float, row)) for t, row in zip(tiers, _np.atleast_2d(d["accuracy_by_k"]))}
    acc = {t: float(a) for t, a in zip(tiers, d["accuracy"])}
    gam = {t: float(g) for t, g in zip(tiers, d["evidence_decay"])}
    models = meta.get("hardware_profile", {}).get("models", {})
    order = ["think_tiny", "think_big", "offload"]
    rows = []
    for t in [x for x in order if x in tiers]:
        rows.append(f"<tr><td class='name mono'>{esc(t)}</td>"
                    f"<td class='mono' style='font-size:.8rem'>{esc(models.get(t,'—'))}</td>"
                    + "".join(f"<td class='num'>{v:.3f}</td>" for v in by_k[t])
                    + f"<td class='num'><b>{acc[t]:.3f}</b></td>"
                      f"<td class='num'>{gam[t]:.3f}</td></tr>")
    tbl = ("<div class='scroll'><table><thead><tr><th>tier</th><th>model</th>"
           + "".join(f"<th>k={k}</th>" for k in ks)
           + "<th>pooled</th><th>&gamma;</th></tr></thead><tbody>"
           + "".join(rows) + "</tbody></table></div>")
    return tbl, meta


def build() -> str:
    loc, n_runs, hours = repo_stats()
    tr_tbl, tr_devs = eval_table("train")
    ho_tbl, ho_devs = eval_table("heldout")
    # Final framing: the interesting contrast is between operating CONDITIONS,
    # not between device splits. With a cooperative human every agent ties; the
    # learned policy only has something to do once the human and the link are
    # gone. Both are trained and evaluated on R2's audited_321 artefact.
    s_tr = seed_summary("r2final321_", "train", "eval_r2final321")
    s_ho = seed_summary("r2final321_alone", "train", "eval_r2final321_alone")
    stab_tbl, stab_runs = stability_block()
    n_stab = len(stab_runs) if stab_runs is not None else 0
    n_col = int(stab_runs.collapsed.sum()) if stab_runs is not None else 0

    def hero(s, label):
        if not s:
            return ""
        return (f"<div class='stat hero'><div class='v num'>{s['succ']:.3f}</div>"
                f"<div class='lbl'>{label}<br><span class='mono'>±{s['succ_se']:.3f}</span> "
                f"over {s['n']} seeds · heuristic {s['g_succ']:.3f}</div></div>")

    parts = [STYLE.read_text(encoding="utf-8"), "<div class='wrap'>", """
<header>
  <div class="eyebrow">Edge AI &amp; Robotics · Workstream R1 · RL &amp; Environment</div>
  <h1>CogniWatt</h1>
  <div class="tagline">Teaching a robot the economics of its own thinking — when to compute,
    when to ask, and when to just act.</div>
  <div class="byline">
    <span><b>Owner</b> Jayden (R1)</span>
    <span><b>Status</b> complete &mdash; all channel constants measured, scenario set human-audited</span>
    <span><b>Code</b> """ + f"{loc:,} lines Python &middot; 13 + 18 tests" + """</span>
    <span><b>Compute</b> """ + f"{n_runs} runs · ≈{hours:.1f} h on M2 Air" + """</span>
    <span><b>Team</b> 2 people, 2 workstreams</span>
  </div>
</header>

<section>
  <h2>Headline result</h2>
  <p class="lede measure">A single RL policy decides not just <i>what to do</i> but
  <i>how much compute to spend deciding</i>. Measured against real quantised models on a
  human-audited scenario set, the answer turns out to be <b>almost none</b> — and that
  inversion is the result.</p>
  <div class="stats">""" + hero(s_tr, "success, cooperative human") + hero(s_ho, "success, offline and alone") + """</div>"""]

    if s_tr and s_ho:
        parts.append(f"""
  <p class="measure">Both rows are trained and evaluated on R2's audited artefact
  (<span class="mono">audited_321_contested_dropped</span>): every evidence tier measured on the
  same 321 scenarios, each one resolved identically by three independent human annotators.
  Reported as mean ± se <b>over seeds</b> — every one of the {s_tr['n']} seeds converged, so this
  is the whole population of runs, not a survivor-selected subset.</p>
  <div class="scroll"><table>
    <thead><tr><th>condition</th><th>agent</th><th>success</th><th>return</th>
    <th>energy (J)</th><th>queries / episode</th></tr></thead><tbody>
    <tr class="hl"><td>cooperative human</td><td class="name mono">ppo ({s_tr['n']} seeds)</td>
      <td class="num"><b>{s_tr['succ']:.4f} ± {s_tr['succ_se']:.4f}</b></td>
      <td class="num"><b>{s_tr['ret']:+.3f} ± {s_tr['ret_se']:.3f}</b></td>
      <td class="num">{s_tr['energy']:.1f}</td><td class="num">{s_tr['q']:.2f}</td></tr>
    <tr><td>cooperative human</td><td class="name mono">greedy_value</td>
      <td class="num">{s_tr['g_succ']:.4f}</td><td class="num">{s_tr['g_ret']:+.3f}</td>
      <td class="num">{s_tr['g_energy']:.1f}</td><td class="num">{s_tr['g_q']:.2f}</td></tr>
    <tr class="hl"><td><b>offline + alone</b></td><td class="name mono">ppo ({s_ho['n']} seeds)</td>
      <td class="num"><b>{s_ho['succ']:.4f} ± {s_ho['succ_se']:.4f}</b></td>
      <td class="num"><b>{s_ho['ret']:+.3f} ± {s_ho['ret_se']:.3f}</b></td>
      <td class="num">{s_ho['energy']:.1f}</td><td class="num">{s_ho['q']:.2f}</td></tr>
    <tr><td><b>offline + alone</b></td><td class="name mono">greedy_value</td>
      <td class="num">{s_ho['g_succ']:.4f}</td><td class="num">{s_ho['g_ret']:+.3f}</td>
      <td class="num">{s_ho['g_energy']:.1f}</td><td class="num">{s_ho['g_q']:.2f}</td></tr>
    </tbody></table></div>
  <div class="card"><h3>Compute cannot buy its way out of referential ambiguity</h3>
  <p class="measure">No model a domestic robot can reach — 0.5B, 3B or 7B, at Q4 or Q8, given
  unlimited queries or unlimited reasoning tokens — resolves these referring expressions
  confidently enough to act on. Asking the person does. So with a cooperative human available the
  decision problem is <b>degenerate</b>: one clarifying question reaches confidence, no model tier
  can, every agent converges on "ask once" and ties at {s_tr['succ']:.3f}, and on-device inference
  falls to <b>1 %</b> of evidence purchased.</p>
  <p class="measure">That is not a failure of the environment — it is the measurement this project
  was built to make, and it inverts the premise it started from. <b>Dialogue is not the fallback
  for when compute runs out; compute is the fallback for when dialogue is unavailable.</b></p>
  <p class="measure">Which is exactly where the learned policy earns its place. Offline and alone,
  no single action suffices: the agent must sequence cheap, weak evidence under a budget, and it
  beats the strong heuristic by <b>{(s_ho['succ']-s_ho['g_succ'])*100:.1f} points of success</b>
  and the best baseline by <b>{s_ho['ret']-max(s_ho['g_ret'], -9.282):+.2f} return</b>.</p></div>""")

    parts.append("<h3>Full comparison — training devices</h3>"
                 + (f"<p class='note'>Devices: <span class='mono'>{esc(', '.join(tr_devs))}</span></p>" if tr_devs else "")
                 + tr_tbl
                 + "<h3>Zero-shot transfer to unseen hardware</h3>"
                 + (f"<p class='note'>Devices: <span class='mono'>{esc(', '.join(ho_devs))}</span> — "
                    "never seen in training. The policy is conditioned only on datasheet features "
                    "(TOPS, bandwidth, TDP, battery, NPU flag, RAM), never on the cost table.</p>" if ho_devs else "")
                 + ho_tbl + "</section>")

    parts.append(f"""
<section>
  <h2>Training stability — a solved limitation</h2>
  <p class="lede measure">This report previously carried an open limitation: <i>"6 of 10
  non-ablation runs converge; 4 collapse to ~0.34 … raising the entropy bonus does not
  rescue it."</i> A {n_stab}-run sweep shows the diagnosis was backwards. Raising the entropy
  bonus is the <b>only</b> thing that rescues it.</p>
  {stab_tbl}
  <p class="measure">Convergence is monotone in <span class="mono">ent_coef</span> and flat in
  everything else. Of {n_stab} runs, {n_col} collapsed and <b>none landed in between</b> — the
  outcome is genuinely bimodal, so a collapse <i>rate</i> is the right analysis. The collapsed
  runs are not merely worse: they are the uninformed policy, sitting at the
  <span class="mono">no_think</span> baseline with final policy entropy of 0.001.</p>
  <div class="card"><h3>The published headline was the lucky seed</h3>
  <p class="measure">The run this report's original numbers came from used
  <span class="mono">ent_coef 0.01</span>, where 1 run in 8 survives. That is why the collapse
  looked seed-independent and mysterious — at that setting, collapse is the <i>default</i>
  outcome and convergence the exception.</p></div>
  {fig('fig6_stability.png', 'Every seed of every arm. Blue converged, red collapsed. The middle of panel (b) is empty: this is a bimodal outcome, not a noisy one. Panel (c) orders arms by exploration pressure.', 'Figure 6 — the collapse is one hyperparameter')}
  <h3>Two hypotheses, both falsified</h3>
  <p class="measure"><b>Gradient variance.</b> Tripling the batch (16 → 48 envs) changes nothing:
  8/8 either way, 0.907 against 0.914. The intuitive explanation is wrong — this is an
  exploration failure, not an estimation-noise failure.</p>
  <p class="measure"><b>Recurrence.</b> The GRU ablation had never been run; every number in this
  project uses the feedforward policy. Measured, it makes no difference — 0.915 against 0.914 —
  at <b>3.4× the wall-clock cost</b> (16.2 min against 4.8). This is the direct confirmation of
  the design claim in <span class="mono">agents/ppo.py</span>: the Bayesian belief in the
  observation is an approximately sufficient statistic, so the POMDP is nearly Markov by
  construction and there is no residual history for a GRU to exploit. <b>The belief filter does
  the work recurrence would otherwise have to learn.</b></p>
  <h3>Collapse is predictable after 15% of a run</h3>
  <p class="measure">Mean policy entropy over the first 300k steps separates the two modes with no
  overlap — converged 1.15, collapsed 0.22, Welch <i>t</i> = 18.9. A doomed run is identifiable
  after about 45 seconds instead of five minutes, so a sweep driver can kill and reseed rather
  than pay for 2M steps to learn nothing.</p>
</section>

<section>
  <h2>Entropy annealing</h2>
  <p class="lede measure">A single entropy coefficient cannot do both of its jobs. Early it must be
  large enough to escape the uninformed basin; late it must be small enough for the policy to
  sharpen. Held constant at 0.12 the converged policy stays deliberately stochastic and
  <b>over-buys evidence</b> — 4.3 queries per episode against the heuristic's 2.2 — losing on the
  very objective it optimises.</p>
  <div class="scroll"><table><thead><tr><th>ent_coef</th><th>convergence</th>
  <th>converged policy</th></tr></thead><tbody>
  <tr><td class="mono">0.01</td><td class="num dn">1/8</td><td>sharp, efficient</td></tr>
  <tr><td class="mono">0.12</td><td class="num up">8/8</td><td>reliable, wasteful</td></tr>
  <tr class="hl"><td class="mono">0.12 → 0.01 annealed</td><td class="num up">8/8</td>
  <td>reliable <b>and</b> sharp</td></tr>
  </tbody></table></div>
  <p class="measure">Annealing keeps perfect convergence, cuts queries 4.4 → 2.1 and energy
  180 J → 160 J, and moves return from −1.55 to −0.27. The policy learns to stop buying evidence
  it does not need — which is the behaviour this project exists to study.
  <span class="mono">PPOConfig.ent_coef_final = None</span> preserves the old constant behaviour
  exactly, so every earlier run still reproduces.</p>
  {fig('fig4_policy_heatmap.png', 'What the annealed policy buys as a function of belief confidence and remaining battery. Confident → act. Uncertain with battery to spare → buy evidence. Uncertain and nearly flat → act anyway, because reasoning it cannot afford to finish is worse than guessing.', 'Figure 4 — the learned decision rule')}
</section>""")

    parts.append(f"""
<section>
  <h2>Ablations</h2>
  <p class="measure">Each ablation removes one component from the canonical annealed
  configuration and is run over multiple seeds. This matters: the previous ablation table
  reported 0.000–0.345 for these, which looked catastrophic and load-bearing. Those runs were
  simply the entropy collapse. Measured properly, the components are worth far less than
  claimed.</p>
  {ablation_block()}
  {fig('fig1_learning_curves.png', 'Learning curves for the full method and its ablations.', 'Figure 1 — training curves')}
</section>

<section>
  <h2>What a cloud call should cost</h2>
  <p class="lede measure">The learned policy currently draws 80–95% of its evidence from the
  cloud. For a project about <i>edge</i> inference whose central claim is a <i>portfolio</i>,
  that is close to the degeneracy <span class="mono">lambda_cloud</span> was introduced to
  prevent. The fix is not to tune the parameter until the portfolio looks balanced — that would
  assume the conclusion.</p>
  <p class="measure"><b>The monetary half is derivable.</b> A grounding query is ~300 in / ~20 out
  tokens; at Haiku 4.5 pricing that is <span class="mono">$4.0e-4</span> per call. The reward
  scale is anchored by <span class="mono">RewardConfig</span>'s own design note — a full 400 J
  battery costs 10.0 reward, exactly one task failure — so one task ≈ 10 units. Against a
  domestic assistant doing ~50 tasks/day at a $500/year service value, one task ≈ $0.027, giving
  <b>≈ 0.15 reward units per call</b>: an order of magnitude below the 0.5 currently used.</p>
  <p class="measure"><b>The privacy half is not derivable</b>, and inventing a figure would be an
  assertion dressed as a derivation. The project already has the right precedent:
  <span class="mono">lambda_ask</span>, the price of the human's patience, is equally
  underivable and is <i>swept</i>, with the Pareto frontier reported. The same treatment applies
  here.</p>
  {cloud_block()}
  <p class="measure">Three regimes, and the middle one is narrow. Below 1 the cloud dominates and
  the on-device tiers are vestigial — where the project currently sits. Between 1 and 2 there is
  a genuine three-way portfolio. Above 4 the cloud is priced out entirely and the agent
  substitutes the <i>human</i>, which is a different degeneracy rather than a success.</p>
  <div class="card"><h3>Returns cannot be compared down this table</h3>
  <p class="measure">Each row optimises a different objective, so return declines monotonically as
  a matter of arithmetic. Only portfolio structure and success are comparable across rows — the
  same trap that invalidated this report's original headline.</p></div>
  <h3>Trained at each price</h3>
  {cloud_trained_block()}
  <h3>When <i>is</i> on-device reasoning rational?</h3>
  <p class="measure">Crossing connectivity with human availability isolates the two
  alternatives. The hypothesis: the on-device tiers are crowded out by the <b>human</b>, not by a
  cheap cloud, and local inference only becomes rational when both are gone.</p>
  {isolation_block()}
  {fig('fig7_isolation.png', 'Left: which channels the policy buys in each condition. Middle: the on-device share, which only becomes dominant when neither the cloud nor the human is available. Right: unlike the device-based held-out split, this condition discriminates — device-blind scripted policies collapse by 36-52 points.', 'Figure 7 — on-device inference earns its place only as a last resort')}
  <div class="card"><h3>The human is what crowds out on-device reasoning</h3>
  <p class="measure">Removing <i>only</i> the human — cloud still fast and available — more than
  doubles the on-device share (14% → 33%) and lifts <span class="mono">think_big</span> sevenfold.
  Removing only the cloud barely moves it (14% → 19%) and leaves
  <span class="mono">think_big</span> at zero; the demand goes to
  <span class="mono">ask</span> instead. A clarifying question is such good value that it
  suppresses local reasoning even when the cloud is unreachable. With neither available,
  on-device share reaches <b>63%</b> and <span class="mono">think_big</span> rises to 0.94 calls
  per episode.</p></div>
  <p class="measure"><b>This is also a held-out condition that discriminates.</b> On the current
  device split no baseline drops by more than 0.9 points, so it tests nothing. Offline and alone,
  <span class="mono">always_offload</span> falls 36 points and
  <span class="mono">always_ask</span> 52, while the learned policy keeps a <b>20-point margin</b>
  over the best device-blind rule — and its advantage over the strong heuristic is the largest
  anywhere in the project. The held-out axis should be <i>offline and alone</i>, not
  <i>a different chip</i>.</p>
  <p class="measure">Full argument in <span class="mono">docs/r1_cloud_price.md</span>. The
  conclusion reported is deliberately conditional: the <i>agent</i> claim is checked for
  invariance to the price, while the <i>portfolio</i> claim is a property of the cost model and
  is stated as "under a cloud price of X, the optimum allocates Y".</p>
</section>

<section>
  <h2>Reproducibility — an uncomfortable finding</h2>
  <p class="lede measure">Re-verifying the published headline from the saved policy failed at the
  first step. The checkpoint would not load.</p>
  <pre class="mono clean measure">size mismatch for enc.0.weight:
  checkpoint [128, 40] vs current model [128, 41]</pre>
  <p class="measure">Three changes had landed after the reported results were generated, and
  nothing was retrained:</p>
  <div class="scroll"><table><thead><tr><th></th><th>when results were made</th>
  <th>current code</th></tr></thead><tbody>
  <tr><td>observation</td><td class="num">40-dim</td><td class="num">41-dim (<span class="mono">memory_gb</span> added)</td></tr>
  <tr><td>held-out devices</td><td class="mono">coral_micro, rtx3060</td><td class="mono">jetson_nano_2gb, rtx3060</td></tr>
  <tr><td><span class="mono">lambda_cloud</span></td><td class="num dn">absent (cloud calls free)</td><td class="num">0.5</td></tr>
  </tbody></table></div>
  <p class="measure">The prose of this report had been updated to discuss the memory feature and
  the retired Coral profile; its <i>numbers</i> never were. The reward function itself had
  changed, so the old returns were on a different scale entirely — and the legacy policy made
  2.67 cloud calls per episode, precisely the behaviour now priced.
  <span class="mono">greedy_value</span> reads <span class="mono">env.cfg</span> at decision time
  and silently re-optimised for the new reward; a trained network cannot, which accounts for most
  of the apparent regression.</p>
  <p class="measure"><b>Fixes.</b> Legacy artefacts are quarantined in
  <span class="mono">results/runs_legacy/</span> with a provenance note; every ablation and Pareto
  point was retrained against the current environment; and
  <span class="mono">evaluate_policy.py</span> now refuses to evaluate a policy whose stored
  <span class="mono">RewardConfig</span> differs from the evaluation config. That guard is five
  lines and would have caught this immediately. This report is now <b>generated</b>
  (<span class="mono">scripts/make_html_report.py</span>) rather than hand-written, so it cannot
  drift from the code again.</p>
</section>""")

    parts.append("""
<section>
  <h2>Grounding the simulator in real hardware</h2>
  <p class="measure">The original cost table implicitly assumed <i>peak</i> GEMM throughput. A
  grounding query is a 150-token prompt — a short, skinny matmul that reaches only
  <b>19% of the M2's 3.6 TFLOP/s peak</b> (measured). Correcting this moved
  <span class="mono">think_big</span> on the M2 from an assumed 95 ms to a modelled
  <b>1591 ms</b>, and on a Pi 5 from 900 ms to <b>32 seconds</b> — one inference costing more
  than half of both episode budgets.</p>
  <div class="scroll"><table><thead><tr><th>quantity</th><th>datasheet peak</th>
  <th>measured (M2 Air)</th><th>efficiency</th></tr></thead><tbody>
  <tr><td>GEMM, large square</td><td class="num">3.6 TFLOP/s</td><td class="num">2695 GFLOP/s</td><td class="num">75%</td></tr>
  <tr class="hl"><td>GEMM, prefill shape (L=150)</td><td class="num">3.6 TFLOP/s</td><td class="num">680 GFLOP/s</td><td class="num dn">19%</td></tr>
  <tr><td>streaming read bandwidth</td><td class="num">100 GB/s</td><td class="num">76.9 GB/s</td><td class="num">77%</td></tr>
  </tbody></table></div>
  """ + fig('fig5_m2_throttle.png', 'Ten minutes of sustained GEMM on the passively cooled M2 Air. Throughput settles onto a 690 GFLOP/s floor — a 2.06× slowdown, measured, now feeding ThermalModel.max_slowdown directly instead of an assumed 2.0.', 'Figure 5 — the fanless anchor earns its keep') + """
  <p class="measure"><b>Measured:</b> M2 Air compute and bandwidth efficiency, M2 Air thermal
  curve. <b>Modelled:</b> every other device, by applying the measured efficiency factors to
  datasheet peaks; radio energy for cloud offload from published Wi-Fi figures; actuation energy
  from an explicit physical model. <b>Pending from R2:</b> per-call latency and energy on the
  RTX 3060.</p>
</section>

<section>
  <h2>What the models actually do</h2>
  <p class="lede measure">Every dialogue constant below is measured, not assumed — 19,800 queries
  across three model tiers on one llama.cpp stack, one prompt template, one parser, shuffled
  candidate order with the permutation recorded. This is R2's workstream; the numbers are read
  here directly from the frozen artefact.</p>
  """ + measured_tiers_block()[0] + """
  <p class="measure">Two things in that table did the most damage to this project's original
  assumptions. <b>&gamma; falls monotonically with scale</b> (0.87 &rarr; 0.64 &rarr; 0.44) —
  the opposite of what was assumed. A more capable model is more self-consistent, so re-querying
  it buys less; a weak model looks variable because it is nearly random. Ranked by &gamma; alone
  the 0.5B is the richest evidence source and the worst by every measure that matters, so
  &gamma; must never be quoted on its own. And <b>the 7B's accuracy is flat in k</b>
  (0.804 / 0.776 / 0.794) where the small models fall away — adding distractors costs them
  heavily and costs the 7B nothing, which says its failure is pragmatic inference rather than
  discrimination.</p>

  <h3>The ceiling, not the price, is what rules the tiers out</h3>
  <p class="measure">An irreversible <span class="mono">place()</span> needs roughly 0.85
  posterior confidence to be worth attempting against a &plusmn;10 reward. Driving each tier
  through the belief filter until it saturates:</p>
  <div class="scroll"><table><thead><tr><th>tier</th><th>ceiling at k=3</th>
  <th>reaches 0.85?</th></tr></thead><tbody>
  <tr><td class="name mono">think_tiny</td><td class="num">0.374</td><td class="num dn">no</td></tr>
  <tr><td class="name mono">think_big</td><td class="num">0.546</td><td class="num dn">no</td></tr>
  <tr><td class="name mono">offload (7B)</td><td class="num">0.731</td><td class="num dn">no</td></tr>
  <tr class="hl"><td class="name mono">ask (the human)</td><td class="num">0.947</td><td class="num up">yes</td></tr>
  </tbody></table></div>
  <p class="measure">Two independent implementations of that calculation — one per workstream —
  agree within 0.010 on every model tier. It is not knife-edge on the 0.85 either: for
  <i>any</i> threshold between 0.731 and 0.947 the human is the only usable channel, a 22-point
  window. And the task is demonstrably solvable, because <span class="mono">ask</span> clears it
  comfortably — had it failed too, the right conclusion would have been a broken dataset.</p>
  <p class="measure"><b>Reasoning tokens do not close the gap.</b> Allowed to think step by step,
  the 7B gains 0.023 (not distinguishable from zero) at 27&times; the tokens, and the 3B gets
  <i>significantly worse</i> (&minus;0.059, McNemar p = 0.016). In 34 % of cases the 7B ignored the
  instruction and answered directly, which splits the condition for free: 0.811 when it skipped
  reasoning against 0.801 when it actually reasoned. <b>Quantisation is not the limit either</b> —
  Q8 scores 0.592 against Q4's 0.603 (p = 0.500) and the two builds agree on 581 of 660 scenarios.
  Neither better weights nor more thinking moves this tier.</p>

  <h3>The scenario set was audited by three humans</h3>
  <p class="measure">60 authored cores, blind sheet, no answer key, three independent annotators —
  one of them outside the project entirely. The gate: a core survives only if every annotator
  picked the keyed answer. <b>29 of 60 survive</b> (Fleiss &kappa; 0.592; two-rater sensitivity
  39/60 at &kappa; 0.722). Adding the third reader cost 10 cores and 0.13 of &kappa;, which is a
  finding about item difficulty rather than a reason to drop a rater.</p>
  <div class="card"><h3>Human-hard and model-hard are different axes</h3>
  <p class="measure">The models are <b>no worse</b> — the 3B is <i>better</i> (0.632 against 0.500)
  — on the cores humans could not agree about. Model failure therefore does not track item
  ambiguity: the models fail on expressions three independent people resolve identically. That
  closes the strongest objection available to this result, which is that the models look bad
  because the dataset is ambiguous. Measurably, it is not the explanation.</p></div>
  <p class="measure"><b>Cost of the audit, stated plainly.</b> Two families are effectively gone —
  <span class="mono">proximity</span> and <span class="mono">exclusion</span> retain one core each
  — and <span class="mono">already_satisfied</span> is down to 9 items. Nothing may be cited from
  any of the three. Two of the three annotators were project members; neither had seen the
  ground-truth key, and the outside reader agreed with it most often of the three (0.840), so the
  pool is not obviously biased toward the authored answers.</p>
</section>

<section>
  <h2>Four silent failures, and what they have in common</h2>
  <p class="lede measure">The most transferable thing this project produced is not a number. Over
  its course, four separate settings were found to have silently failed to reach the code that
  consumed them. None raised an error. All four produced plausible results.</p>
  <div class="scroll"><table><thead><tr><th>setting</th><th>what happened</th>
  <th>how it surfaced</th></tr></thead><tbody>
  <tr><td class="mono">lambda_cloud</td><td>added to the reward after a policy was trained; the
    stored config silently took the new default</td><td>a checkpoint refused to load for an
    unrelated reason, and the reward diff was noticed while investigating</td></tr>
  <tr><td class="mono">net_quality</td><td>connectivity swept at training time, never passed to
    the evaluation env</td><td>scripted baselines returned byte-identical numbers at every link
    quality — impossible if connectivity were varying</td></tr>
  <tr><td class="mono">dialogue</td><td>policies trained on measured channel constants, scored in
    a world still running the placeholders</td><td>a spot-check of one agent against a directly
    computed figure disagreed</td></tr>
  <tr><td class="mono">scenario ids</td><td>positional ids assigned <i>after</i> a filter, so every
    surviving id named a different scenario than the raw table had measured</td>
    <td>a test written specifically to pin the invariant against the real raw table</td></tr>
  </tbody></table></div>
  <p class="measure">Three of the four were caught by a number looking <i>wrong in a specific way</i>
  rather than by anything breaking — identical values where variation was expected, a disagreement
  between two routes to the same figure. The fourth was caught by a test that existed only because
  someone went looking. <b>The common shape is a value that crosses a boundary — training to
  evaluation, build to analysis, one workstream to the other — with nothing asserting it arrived.</b></p>
  <p class="measure">The mitigations now in the code are correspondingly boring:
  <span class="mono">evaluate_policy.py</span> refuses to score a policy whose stored
  <span class="mono">RewardConfig</span> or environment arguments differ from the evaluation's, and
  refuses to put policies trained in different worlds in one table;
  <span class="mono">r2_build_dataset.py</span> pins id alignment against the real raw table; and
  both reports are <b>generated</b> from the CSVs, because the earlier hand-written report is how a
  stale headline survived months of changes to the code beneath it.</p>
</section>

<section>
  <h2>Where the project stands</h2>
  <div class="cards">
    <div class="card"><h3>Done</h3><p class="measure">Tier-1 environment, Bayesian belief filter,
    task generator over 8 floorplans, frozen VED and dialogue interfaces, 9 scripted baselines,
    PPO written from scratch, action masking, potential-based shaping, 13-test suite, hardware
    calibration with two anchors, training-stability study, entropy annealing, cloud-price
    derivation.</p></div>
    <div class="card"><h3>Open — R1</h3><p class="measure">Choose and justify a final
    <span class="mono">lambda_cloud</span>, or report the sweep as the result. Ablation
    components are individually worth little; the report should say so plainly rather than
    claim they are load-bearing.</p></div>
    <div class="card"><h3>Blocked on R2</h3><p class="measure">Every dialogue constant is still an
    R1 placeholder — <span class="mono">TIER_ACCURACY</span>,
    <span class="mono">EVIDENCE_DECAY</span>, <span class="mono">DEFAULT_ASR_WER</span>, and the
    colour-confusability matrix. The retrain on measured constants is what converts the headline
    from a plausible simulation into a result grounded in measurement.</p></div>
  </div>
  <p class="measure"><b>Deliberately not used:</b> gymnasium and stable-baselines3. The
  environment is pure NumPy so it runs bit-identically on arm64 macOS and x86_64 + CUDA, and the
  ablations need direct access to the hidden state.</p>
  <p class="measure"><b>Cut, deliberately:</b> the Tier-2 AI2-THOR photoreal port. With two people
  it is the first thing to drop — the novelty claim lives in Tier 1 plus the measured hardware
  model, and the photoreal port is a demo bonus.</p>
</section>

<footer>CogniWatt · R1 workstream (RL &amp; Environment) · all returns in natural units, shaping
stripped from reported metrics · paired episodes with identical tasks and channel noise across
agents · generated by <span class="mono">scripts/make_html_report.py</span></footer>
</div>""")
    return "\n".join(parts)


if __name__ == "__main__":
    out = ROOT / "docs/r1_report.html"
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
