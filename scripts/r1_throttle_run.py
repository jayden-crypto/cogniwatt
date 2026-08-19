"""Sustained-load thermal throttle curve for the fanless M2 Air.

Owner: R1. This is the measurement the M2 anchor exists for: a passively
cooled chassis throttles measurably under sustained load, and the resulting
curve is a *measured* input to `interfaces.ved.ThermalModel` rather than an
assumed one. A fan-cooled box (Adit's 3060) will not show this nearly as well.

No sudo required: we do not need die temperature, only the observable the model
actually consumes - the latency multiplier over time. Temperature is a latent
variable; sustained throughput degradation is the thing being modelled.
"""
import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--out", default="data/throttle_m2_air.json")
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.randn(a.n, a.n, device=dev)
    y = torch.randn(a.n, a.n, device=dev)
    flop = 2.0 * a.n ** 3

    samples, t0, nxt = [], time.perf_counter(), 0.0
    print(f"sustained {a.n}x{a.n} GEMM on {dev} for {a.minutes} min", flush=True)
    while True:
        el = time.perf_counter() - t0
        if el > a.minutes * 60:
            break
        # one timed burst
        iters = 12
        ts = time.perf_counter()
        for _ in range(iters):
            z = x @ y
        if dev == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter() - ts) / iters
        g = flop / dt / 1e9
        if el >= nxt:
            samples.append({"t_s": round(el, 1), "gflops": round(g, 1)})
            print(f"  t={el:7.1f}s  {g:7.1f} GFLOP/s", flush=True)
            nxt += a.interval

    base = max(s["gflops"] for s in samples[:3])
    fin = sum(s["gflops"] for s in samples[-5:]) / min(5, len(samples))
    out = {"device": "m2_air", "n": a.n, "backend": dev,
           "samples": samples,
           "gflops_initial": base, "gflops_sustained": round(fin, 1),
           "slowdown": round(base / fin, 3)}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\ninitial {base:.0f} -> sustained {fin:.0f} GFLOP/s  "
          f"(slowdown x{base/fin:.3f})\nwrote {a.out}")


if __name__ == "__main__":
    main()
