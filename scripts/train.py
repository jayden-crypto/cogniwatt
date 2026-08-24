"""Train a CogniWatt meta-policy. Owner: R1.

Examples
--------
python3 scripts/train.py --steps 1500000 --tag main
python3 scripts/train.py --steps 1500000 --no-recurrent   --tag ablate_gru
python3 scripts/train.py --steps 1500000 --no-device-cond --tag ablate_devcond
python3 scripts/train.py --steps 1500000 --lambda-ask 6.0 --tag ask_costly
"""
import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from cogniwatt.agents.ppo import PPOConfig, train_ppo
from cogniwatt.env.homegrid import RewardConfig, TRAIN_DEVICES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_500_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.12)
    p.add_argument("--ent-coef-final", type=float, default=None,
                   help="anneal ent-coef linearly to this value; omit to hold constant")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambda-ask", type=float, default=2.0)
    p.add_argument("--lambda-energy", type=float, default=0.025)
    p.add_argument("--lambda-time", type=float, default=0.1667)
    p.add_argument("--lambda-cloud", type=float, default=0.5)
    p.add_argument("--no-recurrent", action="store_true")
    p.add_argument("--no-mask", action="store_true")
    p.add_argument("--cap-dropout", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=3,
                   help="clarifying questions the human will answer; 0 = nobody home")
    p.add_argument("--dialogue", default=None,
                   help="path to R2's frozen dialogue artefact (.npz); omit to "
                        "use the R1 placeholder constants")
    p.add_argument("--net-quality", type=float, default=0.75,
                   help="mean wireless link quality; low values make cloud "
                        "offload unreliable and force on-device reasoning")
    p.add_argument("--no-shaping", action="store_true")
    p.add_argument("--no-device-cond", action="store_true")
    p.add_argument("--torch-device", default="auto")
    p.add_argument("--tag", default="main")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    rc = RewardConfig(lambda_ask=a.lambda_ask,
                      lambda_energy=a.lambda_energy,
                      lambda_time=a.lambda_time,
                      lambda_cloud=a.lambda_cloud,
                      shaping=not a.no_shaping)
    oracle = None
    if a.dialogue:
        from cogniwatt.interfaces.dialogue import load_precomputed
        oracle = load_precomputed(a.dialogue, strict=True)
        print(f"dialogue artefact: {a.dialogue}")
        print(f"  measured tiers: {oracle.meta.get('tiers_measured', '?')}")
        print(f"  placeholders:   {oracle.meta.get('tiers_inherited_from_r1_placeholders', '?')}")
    env_kwargs = dict(devices=TRAIN_DEVICES, reward=rc, oracle=oracle,
                      condition_on_device=not a.no_device_cond,
                      capability_dropout=a.cap_dropout,
                      net_quality_mean=a.net_quality,
                      patience_max=a.patience)
    cfg = PPOConfig(total_steps=a.steps, n_envs=a.n_envs, horizon=a.horizon,
                    lr=a.lr, ent_coef=a.ent_coef,
                    ent_coef_final=a.ent_coef_final, seed=a.seed,
                    recurrent=not a.no_recurrent, use_mask=not a.no_mask)

    out = pathlib.Path("results/runs") / a.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    net, hist = train_ppo(cfg, env_kwargs, devices=list(TRAIN_DEVICES),
                          device_pref=a.torch_device, verbose=not a.quiet)
    dur = time.time() - t0

    torch.save({"state_dict": net.state_dict(),
                "recurrent": cfg.recurrent,
                "hidden": cfg.hidden,
                "condition_on_device": not a.no_device_cond,
                "use_mask": not a.no_mask,
                "reward": rc.__dict__,
                "args": vars(a)}, out / "policy.pt")
    (out / "history.json").write_text(json.dumps(
        {"history": hist, "args": vars(a), "seconds": dur}, indent=2))
    print(f"\nsaved -> {out}/policy.pt   ({dur/60:.1f} min)")


if __name__ == "__main__":
    main()
