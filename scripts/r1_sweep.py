"""Seed sweep driver for the R1 stability study. Owner: R1.

The headline R1 result is reported from *converged* runs, because 4 of 10
non-ablation runs collapse to ~0.34 (the uninformed policy). That is the honest
limitation in docs/r1_report.html and this script is what turns it from an
anecdote into a measurement: run each arm over many seeds, keep every history,
and let scripts/r1_stability.py classify and test them.

Arms are declared here rather than in a shell script so the study is
reproducible on either team machine and the exact flags are version-controlled.

    python3 scripts/r1_sweep.py --arms base bigbatch --jobs 4
"""
import argparse, os, pathlib, subprocess, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The canonical training configuration: feedforward, 16 envs, entropy annealed
#: 0.12 -> 0.01 (converges on every seed AND sharpens; docs/r1_stability.md).
ANNEAL = ["--no-recurrent", "--n-envs", "16", "--ent-coef", "0.12",
          "--ent-coef-final", "0.01"]

#: Canonical config + R2's fully-measured artefact. The base for every
#: experiment run after the three-tier measurement landed.
MEAS = ANNEAL + ["--dialogue", "data/dialogue_measured.npz"]

#: arm -> (extra train.py flags, seeds).  Everything else is train.py defaults.
ARMS = {
    # the configuration the reported results were produced with
    "base":     (["--no-recurrent", "--n-envs", "16"], list(range(8))),
    # 3x the batch: the variance-reduction hypothesis for the collapse
    "bigbatch": (["--no-recurrent", "--n-envs", "48"], list(range(8))),
    # the recurrence ablation that was never run: the belief in the observation
    # is an approximately sufficient statistic, so a GRU may buy nothing
    "gru":      (["--n-envs", "16"], list(range(4))),
    # entropy is the other candidate explanation for the collapse. ff_2m --
    # the run the reported headline comes from -- used ent_coef 0.01, whereas
    # train.py's default is now 0.12. If the collapse rate tracks this, the
    # "bimodal training" limitation is really an under-exploration limitation
    # that the roofline recalibration already fixed, and these two arms say so.
    "ent03":    (["--no-recurrent", "--n-envs", "16", "--ent-coef", "0.03"], list(range(4))),
    "ent01":    (["--no-recurrent", "--n-envs", "16", "--ent-coef", "0.01"], list(range(8))),
    # The synthesis of the two findings: start where runs reliably escape the
    # uninformed basin (0.12) and end where the policy can sharpen (0.01).
    "entanneal": (["--no-recurrent", "--n-envs", "16",
                   "--ent-coef", "0.12", "--ent-coef-final", "0.01"], list(range(16))),

    # --- regeneration arms -------------------------------------------------
    # The shipped results/eval CSVs were produced against a 40-dim observation
    # and a `coral_micro` held-out device, both of which the hardware
    # recalibration removed (memory_gb entered the feature vector, and
    # coral_micro became jetson_nano_2gb). The ff_2m checkpoint therefore no
    # longer loads, and every ablation / Pareto history was lost. These arms
    # rebuild the report against the environment that actually ships.
    "main":            (["--no-recurrent", "--n-envs", "16"], [0]),
    # Ablations run the CANONICAL config (annealed) minus one component, over 4
    # seeds. Run against constant ent_coef instead they are not comparable to
    # the full method, and at one seed each the differences (0.896-0.918) sit
    # inside seed noise and establish nothing.
    # 16 seeds: device conditioning is a *headline* claim (zero-shot transfer
    # from datasheet features alone) and its effect is small relative to seed
    # variance -- at 4 seeds the held-out difference is the right sign but only
    # t = 0.91, which establishes nothing.
    "ablate_devcond":  (ANNEAL + ["--no-device-cond"], list(range(16))),
    "ablate_shaping":  (ANNEAL + ["--no-shaping"], list(range(4))),
    "ablate_mask":     (ANNEAL + ["--no-mask"], list(range(4))),
    "ablate_capdrop":  (ANNEAL + ["--cap-dropout", "0.35"], list(range(4))),

    # --- price of a cloud call (docs/r1_stability.md section 7) -------------
    # lambda_cloud prices the two real costs of offloading a home robot's scene
    # description: it leaves the house, and it is billed. Swept because the
    # portfolio composition depends on it and the monetary half is derivable
    # while the privacy half is a policy choice.
    "cloud0.15": (ANNEAL + ["--lambda-cloud", "0.15"], list(range(3))),
    "cloud0.5":  (ANNEAL + ["--lambda-cloud", "0.5"],  list(range(3))),
    "cloud1":    (ANNEAL + ["--lambda-cloud", "1.0"],  list(range(3))),
    "cloud2":    (ANNEAL + ["--lambda-cloud", "2.0"],  list(range(3))),
    "cloud4":    (ANNEAL + ["--lambda-cloud", "4.0"],  list(range(3))),

    # --- connectivity (docs/r1_cloud_price.md section 7.3) ------------------
    # Price is not the physically honest reason to reason locally -- an offline
    # robot has no cloud at any price. Sweeping link quality tests whether the
    # portfolio claim survives for a *principled* reason rather than a tuned one.
    "net0.9":   (ANNEAL + ["--net-quality", "0.9"],  list(range(3))),
    "net0.5":   (ANNEAL + ["--net-quality", "0.5"],  list(range(3))),
    "net0.25":  (ANNEAL + ["--net-quality", "0.25"], list(range(3))),
    "net0.1":   (ANNEAL + ["--net-quality", "0.1"],  list(range(3))),

    # --- 2x2: is on-device inference EVER the rational purchase? ------------
    # Section 8.2 found that when the cloud fails the policy asks the human
    # rather than reasoning locally. This isolates the two alternatives: local
    # inference should only become necessary when BOTH are gone, which is the
    # robot alone in the house and offline - the canonical edge-AI deployment,
    # and the one that justifies on-device models existing at all.
    "iso_cloud_human":   (ANNEAL + ["--net-quality", "0.9", "--patience", "3"], list(range(3))),
    "iso_cloud_nohum":   (ANNEAL + ["--net-quality", "0.9", "--patience", "0"], list(range(3))),
    "iso_nocloud_human": (ANNEAL + ["--net-quality", "0.1", "--patience", "3"], list(range(3))),
    "iso_nocloud_nohum": (ANNEAL + ["--net-quality", "0.1", "--patience", "0"], list(range(3))),

    # --- R2 integration ----------------------------------------------------
    # Trains on the measured local tiers with offload/ask still at R1
    # placeholders. This is the configuration R2 warns about in handback
    # section 3.2: think_big measured at 0.568 against an unmeasured offload at
    # 0.88 doubles the apparent big->offload competence gap. Run precisely to
    # find out whether the action mix collapses onto the cloud, which decides
    # whether R1 retrains now or waits for the cloud measurement.
    "r2prov": (ANNEAL + ["--dialogue", "data/dialogue_provisional.npz"], list(range(3))),
    # The real thing: R2's delivered artefact, 660 scenarios after the
    # contested-family drop, measured asr_wer. offload/ask accuracy are still
    # R1 placeholders and meta_json says so.
    "r2real": (ANNEAL + ["--dialogue", "data/dialogue_measured.npz"], list(range(4))),
    # Canonical after the two corrections R2's measurement forced: uniform
    # COLOR_SIM (their section 4 finding) and referent recovery in place of
    # (1 - WER) for the ask channel (their section 3.2). Still measured-local
    # against placeholder offload/ask accuracy.
    "r2final": (ANNEAL + ["--dialogue", "data/dialogue_measured.npz"],
                list(range(4))),
    # All three model tiers measured (offload = Qwen2.5-7B Q4_K_M hosted
    # locally), ask still an R1 placeholder scaled by measured referent
    # recovery. This is the artefact the report's headline should be built on.
    "r2full": (ANNEAL + ["--dialogue", "data/dialogue_measured.npz"],
               list(range(6))),

    # --- where does the problem stop being degenerate? ----------------------
    # With all three model tiers measured, none reaches the confidence an
    # irreversible place() needs, so the optimum collapses to "ask once" and the
    # learned policy becomes indistinguishable from the always_ask baseline
    # (0.887 vs 0.886). That is an honest finding but it leaves nothing for RL
    # to allocate. lambda_ask is an R1 modelling choice, not a measurement, and
    # the project already treats it as a Pareto knob -- so sweep it under
    # MEASURED constants and find the price at which cognition has to be
    # budgeted again.
    "m_ask2":  (MEAS + ["--lambda-ask", "2.0"],  list(range(3))),
    "m_ask4":  (MEAS + ["--lambda-ask", "4.0"],  list(range(3))),
    "m_ask6":  (MEAS + ["--lambda-ask", "6.0"],  list(range(3))),
    "m_ask10": (MEAS + ["--lambda-ask", "10.0"], list(range(3))),
    # patience is the other lever: a human who answers once is a different
    # problem from one who answers three times.
    "m_pat1":  (MEAS + ["--patience", "1"], list(range(3))),
    # The decisive test for the RL contribution under measured constants. In the
    # nominal condition the problem is degenerate (ask once) and greedy_value
    # wins, because one-step lookahead with the true likelihoods is optimal when
    # the optimum is one action deep. Offline and alone, no single action
    # suffices and the agent must sequence cheap evidence under a budget --
    # which is where a learned policy should earn its place.
    "m_alone": (MEAS + ["--net-quality", "0.1", "--patience", "0"], list(range(4))),

    # --- the audited artefact ----------------------------------------------
    # R2's re-freeze on the 29 human-verified cores. asr_recovery is now emitted
    # upstream, so r1_merge_recovery.py is deleted and this trains on their file
    # directly.
    "r2audited":       (MEAS, list(range(4))),
    "r2audited_alone": (MEAS + ["--net-quality", "0.1", "--patience", "0"], list(range(4))),

    # --- final: both exclusions on every tier, 321 items ---------------------
    # R2's audited_321_contested_dropped freeze. All four tiers scored on one
    # item set (n_samples uniform 107x3), so the gradient spread is a like-for-
    # like comparison. This is the artefact the report's numbers come from.
    "r2final321":       (MEAS, list(range(6))),
    "r2final321_alone": (MEAS + ["--net-quality", "0.1", "--patience", "0"], list(range(6))),
    "ask0.5":          (["--no-recurrent", "--n-envs", "16", "--lambda-ask", "0.5"], [0]),
    "ask3.5":          (["--no-recurrent", "--n-envs", "16", "--lambda-ask", "3.5"], [0]),
    "ask6":            (["--no-recurrent", "--n-envs", "16", "--lambda-ask", "6.0"], [0]),
    "ask12":           (["--no-recurrent", "--n-envs", "16", "--lambda-ask", "12.0"], [0]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["base", "bigbatch"])
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not re-run a tag that already has a history.json")
    a = ap.parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    jobs = []
    for arm in a.arms:
        if arm not in ARMS:
            sys.exit(f"unknown arm {arm!r}; have {sorted(ARMS)}")
        flags, seeds = ARMS[arm]
        for s in seeds:
            tag = f"{arm}_s{s}"
            if a.skip_existing and (ROOT / "results/runs" / tag / "history.json").exists():
                continue
            jobs.append((tag, [a.python, "-u", str(ROOT / "scripts/train.py"),
                               "--steps", str(a.steps), "--seed", str(s),
                               *flags, "--tag", tag]))

    print(f"{len(jobs)} runs, {a.jobs} at a time, {a.steps:,} steps each")
    if a.dry_run:
        for tag, cmd in jobs:
            print(" ", tag, " ".join(cmd[3:]))
        return

    # One thread per process: these are tiny MLPs, so intra-op parallelism only
    # makes the processes fight over the same 4 performance cores.
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running, t0 = [], time.time()
    while jobs or running:
        while jobs and len(running) < a.jobs:
            tag, cmd = jobs.pop(0)
            log = open(ROOT / "logs" / f"{tag}.log", "w")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=ROOT, env=env)
            running.append((tag, p, log))
            print(f"[{time.time()-t0:7.0f}s] start {tag}", flush=True)
        time.sleep(2.0)
        for item in list(running):
            tag, p, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                print(f"[{time.time()-t0:7.0f}s] done  {tag}  rc={p.returncode}", flush=True)
    print(f"sweep complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
