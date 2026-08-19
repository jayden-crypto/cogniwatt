"""M2 Air hardware anchor: achievable compute and memory bandwidth.

Owner: R1 (inherited from the dissolved R3 workstream).

Measures the two roofline parameters that actually govern transformer inference:

  * achievable GEMM throughput (GFLOP/s)  -> governs PREFILL (compute bound)
  * achievable memory bandwidth (GB/s)    -> governs DECODE  (weight streaming)

Peak datasheet numbers are useless here; what matters is what the machine
sustains on LLM-shaped operations. Everything runs without sudo so it is
reproducible on any team machine. Power is measured separately - see
`scripts/r1_power_capture.sh`, which does need sudo.
"""
import argparse, json, platform, time, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def gemm_gflops(m, k, n, dev, dtype=torch.float32, iters=8):
    a = torch.randn(m, k, dtype=dtype, device=dev)
    b = torch.randn(k, n, dtype=dtype, device=dev)
    for _ in range(3):
        c = a @ b
    if dev == "mps":
        torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    if dev == "mps":
        torch.mps.synchronize()
    dt = (time.perf_counter() - t) / iters
    return 2.0 * m * k * n / dt / 1e9


def read_bandwidth(dev, mb=512, iters=10):
    """Streaming-read bandwidth: the decode-phase bottleneck.

    A reduction over a large tensor touches every byte exactly once, which is
    the same access pattern as streaming model weights for one decoded token.
    """
    x = torch.randn(mb * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    for _ in range(3):
        s = x.sum()
    if dev == "mps":
        torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        s = x.sum()
    if dev == "mps":
        torch.mps.synchronize()
    dt = (time.perf_counter() - t) / iters
    return mb / 1024.0 / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/anchor_m2_air.json")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    res = {"machine": platform.machine(), "processor": platform.processor(),
           "torch": torch.__version__, "threads": a.threads, "devices": {}}

    # Shapes chosen to mirror transformer prefill GEMMs rather than square
    # benchmark matrices: (prompt_tokens x hidden) @ (hidden x ffn).
    shapes = [(150, 896, 4864), (150, 2048, 8192), (512, 2048, 8192), (2048, 2048, 2048)]
    for dev in devs:
        d = {"gemm": {}, "bandwidth_gb_s": None}
        for (m, k, n) in shapes:
            try:
                d["gemm"][f"{m}x{k}x{n}"] = round(gemm_gflops(m, k, n, dev), 1)
            except Exception as e:
                d["gemm"][f"{m}x{k}x{n}"] = None
        d["gemm_peak_gflops"] = round(max(v for v in d["gemm"].values() if v), 1)
        d["bandwidth_gb_s"] = round(read_bandwidth(dev), 1)
        res["devices"][dev] = d
        print(f"[{dev}] peak GEMM {d['gemm_peak_gflops']:.0f} GFLOP/s | "
              f"read BW {d['bandwidth_gb_s']:.1f} GB/s")
        for s, v in d["gemm"].items():
            print(f"       {s:>18s}  {v}")

    best = max(res["devices"].values(), key=lambda d: d["gemm_peak_gflops"])
    res["anchor"] = {"gflops": best["gemm_peak_gflops"], "bw_gb_s": best["bandwidth_gb_s"]}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nanchor -> {res['anchor']}\nwrote {a.out}")


if __name__ == "__main__":
    main()
