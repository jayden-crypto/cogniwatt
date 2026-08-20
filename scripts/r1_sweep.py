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
