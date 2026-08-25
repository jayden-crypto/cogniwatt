#!/usr/bin/env python3
"""Per-inference latency and energy on the discrete GPU (R2 Deliverable D, §7b).

    python scripts/r2_profile.py --idle-only            # P_idle, run this first
    python scripts/r2_profile.py --tiers think_tiny think_big
    python scripts/r2_profile.py --tiers offload --live-cloud   # latency only

Output: data/r2_rtx3060_profile.json  (filename fixed by the handoff contract;
the *measured* part is named honestly inside -- see the device note below).

THE SAMPLING-RATE TRAP
----------------------
`nvidia-smi` refreshes power at roughly 10 Hz. A `think_tiny` call can be
shorter than one sample period, so timing it directly returns whatever the
counter happened to hold -- a number that looks plausible and means nothing.
The fix, applied automatically here: run the call in a tight loop and divide
total energy by the loop count. `--auto-loop` measures one call first, then
picks a loop count that makes each measured burst at least `--min-burst-ms`
long (default 2000 ms, i.e. ~20 power samples per burst).

VALID DEVICE ANCHOR vs VALID ACCURACY
-------------------------------------
For accuracy, partial CPU offload is fine -- the outputs are identical, just
slower. For a hardware anchor it is not: a partially-offloaded run
characterises "part of a GPU plus a CPU", which is not a device. This script
therefore refuses to profile unless the model is fully GPU-resident
(`n_gpu_layers=-1`), unless you pass `--allow-partial-offload`, which stamps
`valid_device_anchor: false` into the output so the number can never be quoted
as a device measurement by accident.

DEVICE NAMING
-------------
The handoff names the file `r2_rtx3060_profile.json` and the sim's device
profile `rtx3060`. Whatever card is actually in the machine is recorded in
`device_measured` / `gpu_name` / `vram_gb` from `nvidia-smi`, not assumed.
Reconciling the measured part against the sim's reference part is R1's call;
this script's job is to be precise about what was measured.

MAINS vs BATTERY
----------------
A laptop GPU behaves very differently on battery. The script records the power
source and refuses to run on battery without `--allow-battery`.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OUT = os.path.join("data", "r2_rtx3060_profile.json")
SAMPLE_MS = 100          # nvidia-smi refreshes power at ~10 Hz
WARMUP_DISCARD = 5


# ---------------------------------------------------------------------------
# nvidia-smi
# ---------------------------------------------------------------------------
def _smi(*args: str) -> str:
    exe = shutil.which("nvidia-smi") or "nvidia-smi"
    return subprocess.check_output([exe, *args], text=True, stderr=subprocess.STDOUT)


def _f(x: str) -> Optional[float]:
    """nvidia-smi reports unsupported fields as '[N/A]'. On this laptop part
    `power.limit` is one of them; `power.draw` is not, which is what matters."""
    try:
        return float(x)
    except ValueError:
        return None


def gpu_info() -> Dict[str, object]:
    out = _smi("--query-gpu=name,memory.total,driver_version,power.limit",
               "--format=csv,noheader,nounits").strip().splitlines()[0]
    name, mem, drv, plim = [x.strip() for x in out.split(",")]
    return {"gpu_name": name, "vram_mib": int(float(mem)),
            "vram_gb": round(float(mem) / 1024.0, 2),
            "driver_version": drv, "power_limit_w": _f(plim)}


class PowerSampler:
    """Background `nvidia-smi -lms` reader.

    Streams power / clock / temperature samples for as long as it is running.
    One long-lived subprocess rather than one per call: spawning nvidia-smi
    costs ~50 ms, which is larger than the thing being measured.
    """

    def __init__(self, period_ms: int = SAMPLE_MS):
        self.period_ms = int(period_ms)
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.samples: List[Tuple[float, float, float, float]] = []  # t, W, MHz, C
        self._stop = threading.Event()

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                w, mhz, c = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue  # header or "[N/A]"
            with self._lock:
                self.samples.append((time.perf_counter(), w, mhz, c))

    def start(self) -> "PowerSampler":
        exe = shutil.which("nvidia-smi") or "nvidia-smi"
        self._proc = subprocess.Popen(
            [exe, "--query-gpu=power.draw,clocks.sm,temperature.gpu",
             "--format=csv,noheader,nounits", "-lms", str(self.period_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            bufsize=1)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        time.sleep(0.3)  # let the first samples land
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def mark(self) -> int:
        with self._lock:
            return len(self.samples)

    def window(self, i0: int, i1: Optional[int] = None):
        with self._lock:
            return list(self.samples[i0:i1])

    def energy_mJ(self, win, p_idle: float) -> float:
        """Trapezoidal integral of (P - P_idle) dt, in millijoules.

        Energy *above idle* is the marginal cost of the inference. The idle
        draw is what the device spends existing, and charging it to the model
        would make a fast model look expensive purely for being measured on a
        machine that was already on.
        """
        if len(win) < 2:
            return float("nan")
        t = np.array([s[0] for s in win])
        p = np.array([s[1] for s in win]) - p_idle
        return float(np.trapezoid(p, t) * 1000.0)


def summarize(vals: List[float]) -> Dict[str, object]:
    """Median and IQR. First-call outliers are large and unrepresentative, so
    the mean would be reporting the warm-up, not the steady state."""
    a = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"median": None, "iqr": [None, None], "n": 0}
    return {"median": float(np.median(a)),
            "iqr": [float(np.percentile(a, 25)), float(np.percentile(a, 75))],
            "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


# ---------------------------------------------------------------------------
def measure_idle(seconds: float, period_ms: int) -> Dict[str, object]:
    print(f"idle baseline: sampling {seconds:.0f}s with nothing running...")
    s = PowerSampler(period_ms).start()
    time.sleep(seconds)
    win = s.window(0)
    s.stop()
    if len(win) < 10:
        raise RuntimeError(f"only {len(win)} power samples; is nvidia-smi working?")
    p = np.array([x[1] for x in win])
    mhz = np.array([x[2] for x in win])
    c = np.array([x[3] for x in win])
    out = {"p_idle_w": float(np.median(p)),
           "p_idle_iqr": [float(np.percentile(p, 25)), float(np.percentile(p, 75))],
           "clocks_sm_mhz_median": float(np.median(mhz)),
           "temp_c_median": float(np.median(c)),
           "n_samples": int(len(win)),
           "effective_hz": round(len(win) / seconds, 2)}
    print(f"  P_idle {out['p_idle_w']:.2f} W  (IQR {out['p_idle_iqr'][0]:.2f}-"
          f"{out['p_idle_iqr'][1]:.2f})   SM {out['clocks_sm_mhz_median']:.0f} MHz  "
          f"{out['temp_c_median']:.0f} C   {out['effective_hz']} Hz effective")
    return out


def on_mains() -> Optional[bool]:
    """True on mains, False on battery, None if undetectable."""
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance -ClassName Win32_Battery).BatteryStatus"],
                text=True, stderr=subprocess.DEVNULL).strip()
            if not out:
                return True   # no battery reported -> desktop
            # 1 = discharging; 2 = on AC. Other codes are charging states (AC).
            return int(out.splitlines()[0]) != 1
        except Exception:
            return None
    for p in ("/sys/class/power_supply/AC/online", "/sys/class/power_supply/ADP1/online"):
        if os.path.exists(p):
            with open(p) as f:
                return f.read().strip() == "1"
    return None


def profile_local_tier(tier: str, prompts: List[str], p_idle: float, args
                       ) -> Dict[str, object]:
    from cogniwatt.dialogue.backends import Query, make_backend

    if args.n_gpu_layers != -1 and not args.allow_partial_offload:
        raise SystemExit(
            f"refusing to profile {tier} with n_gpu_layers={args.n_gpu_layers}: a "
            f"partial-CPU-offload run is not a device anchor. Pass "
            f"--allow-partial-offload to record it as an invalid anchor anyway.")

    be = make_backend(tier, models_dir=args.models_dir, temperature=args.temperature,
                      n_gpu_layers=args.n_gpu_layers, seed=args.seed)
    print(f"  {be.meta()}")
    qs = [Query(f"prof|{tier}|{i}", p, 3, f"prof{i}", 0, 0)
          for i, p in enumerate(prompts)]

    # Warm-up: weight paging and the first CUDA context cost seconds and are
    # not steady-state behaviour.
    for i in range(WARMUP_DISCARD):
        be.answer(qs[i % len(qs)])

    # How many calls per measured burst? One call must not be shorter than a
    # power sample period, or the energy figure is noise.
    t0 = time.perf_counter()
    be.answer(qs[0])
    one_ms = (time.perf_counter() - t0) * 1000.0
    loop = 1
    if args.auto_loop and one_ms < args.min_burst_ms:
        loop = int(np.ceil(args.min_burst_ms / max(one_ms, 0.05)))
    print(f"  single call ~{one_ms:.1f} ms -> {loop} call(s) per measured burst")

    sampler = PowerSampler(args.period_ms).start()
    lat: List[float] = []
    energy: List[float] = []
    try:
        for r in range(args.reps):
            q = qs[r % len(qs)]
            i0 = sampler.mark()
            t0 = time.perf_counter()
            for _ in range(loop):
                be.answer(q)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            time.sleep(args.period_ms / 1000.0 * 2)  # let the tail samples land
            win = sampler.window(i0)
            e = sampler.energy_mJ(win, p_idle)
            lat.append(dt_ms / loop)
            energy.append(e / loop)
            if (r + 1) % 10 == 0:
                print(f"    {r + 1}/{args.reps}  "
                      f"{np.median(lat):.1f} ms  {np.median(energy):.0f} mJ",
                      flush=True)
    finally:
        sampler.stop()
        be.close()

    win_all = sampler.window(0)
    return {
        "latency_ms": summarize(lat),
        "energy_mJ": summarize(energy),
        "calls_per_burst": loop,
        "reps": args.reps,
        "warmup_discarded": WARMUP_DISCARD,
        "clocks_sm_mhz_median": float(np.median([s[2] for s in win_all]))
                                if win_all else None,
        "temp_c_max": float(np.max([s[3] for s in win_all])) if win_all else None,
        "valid_device_anchor": args.n_gpu_layers == -1,
        "model": be.meta().get("model_id"),
    }


def profile_offload(prompts: List[str], args) -> Dict[str, object]:
    """Latency only. A cloud call spends *radio* energy on the robot, not GPU
    energy, and a desktop 3060 has no radio -- so energy here is deliberately
    null and R1 models the radio side from published Wi-Fi transmit figures."""
    from cogniwatt.dialogue.backends import AnthropicLiveBackend, Query

    be = AnthropicLiveBackend(temperature=args.temperature)
    lat, fails = [], 0
    for i in range(args.offload_reps):
        r = be.answer(Query(f"off|{i}", prompts[i % len(prompts)], 3, f"o{i}", 0, 0))
        lat.append(r.latency_ms)
        if r.error:
            fails += 1
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{args.offload_reps}  "
                  f"median {np.median(lat):.0f} ms  failures {fails}", flush=True)
    s = summarize(lat)
    s.update({"failure_rate": fails / max(1, args.offload_reps),
              "energy_mJ_median": None,
              "note": "energy is radio-side and modelled by R1, not measured here"})
    return {"latency_ms": s, "energy_mJ": {"median": None},
            "reps": args.offload_reps, "valid_device_anchor": False,
            "model": be.model_id}


def sample_prompts(path: str, n: int) -> List[str]:
    """Real prompts from the dataset -- profiling on a synthetic string would
    measure the wrong prompt length, and prompt length drives both latency and
    energy."""
    from cogniwatt.dialogue.prompts import build_prompt
    from cogniwatt.dialogue.schema import load_scenarios

    ss = load_scenarios(path)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ss), size=min(n, len(ss)), replace=False)
    return [build_prompt(ss[i].instruction, [c.desc for c in ss[i].candidates])
            for i in pick]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tiers", nargs="*", default=["think_tiny", "think_big"])
    ap.add_argument("--scenarios", default=os.path.join("data", "r2_scenarios.jsonl"))
    ap.add_argument("--idle-seconds", type=float, default=60.0)
    ap.add_argument("--period-ms", type=int, default=SAMPLE_MS)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--offload-reps", type=int, default=200)
    ap.add_argument("--min-burst-ms", type=float, default=2000.0)
    ap.add_argument("--auto-loop", action="store_true", default=True)
    ap.add_argument("--no-auto-loop", dest="auto_loop", action="store_false")
    ap.add_argument("--models-dir", default=os.path.join("models"))
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--allow-partial-offload", action="store_true")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ambient-c", type=float, default=None,
                    help="ambient temperature; recorded, not measured")
    ap.add_argument("--fan-curve", default="default")
    ap.add_argument("--idle-only", action="store_true")
    args = ap.parse_args()

    info = gpu_info()
    mains = on_mains()
    print(f"{info['gpu_name']}  {info['vram_gb']} GB  driver {info['driver_version']}"
          f"  power limit {info['power_limit_w']} W")
    print(f"power source: {'mains' if mains else 'BATTERY' if mains is False else 'unknown'}")
    if mains is False and not args.allow_battery:
        raise SystemExit("on battery: a laptop GPU behaves very differently. "
                         "Plug in, or pass --allow-battery to record it as such.")

    idle = measure_idle(args.idle_seconds, args.period_ms)

    prof: Dict[str, object] = {
        "device": "rtx3060",                 # the sim's device-profile name
        "device_measured": info["gpu_name"], # what was actually under the sampler
        **info,
        "on_mains": mains,
        "ambient_c": args.ambient_c,
        "fan_curve": args.fan_curve,
        "sampler_hz": round(1000.0 / args.period_ms, 2),
        "sampler_effective_hz": idle["effective_hz"],
        "p_idle_w": idle["p_idle_w"],
        "idle": idle,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ops": {},
        "models": {},
        "notes": ("energy is measured above idle; offload energy is radio-side "
                  "and modelled by R1, not measured here. The file name follows "
                  "the handoff contract; `device_measured` is the real part."),
    }

    if args.idle_only:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(prof, f, indent=2)
        print(f"\nwrote idle baseline only -> {args.out}")
        return 0

    # The handoff assumes nvidia-smi delivers a clean 10 Hz. On this part it
    # emits 100 ms bursts separated by ~0.9 s stalls, so the *effective* rate is
    # ~3 Hz. Sizing bursts off the nominal rate would leave ~6 samples per
    # measurement, which is the "nonsense numbers" failure §7b warns about.
    # Size them off what was actually observed instead, targeting >= 20 samples.
    need_ms = 20_000.0 / max(idle["effective_hz"], 0.5)
    if need_ms > args.min_burst_ms:
        print(f"\nsampler runs at {idle['effective_hz']} Hz effective, not "
              f"{prof['sampler_hz']} Hz nominal -> raising burst target "
              f"{args.min_burst_ms:.0f} -> {need_ms:.0f} ms for >= 20 samples/burst")
        args.min_burst_ms = need_ms
    prof["min_burst_ms"] = args.min_burst_ms

    prompts = sample_prompts(args.scenarios, 32)
    for tier in args.tiers:
        print(f"\n[{tier}]")
        if tier == "offload":
            prof["ops"][tier] = profile_offload(prompts, args)  # type: ignore[index]
        else:
            prof["ops"][tier] = profile_local_tier(  # type: ignore[index]
                tier, prompts, idle["p_idle_w"], args)
        prof["models"][tier] = prof["ops"][tier].get("model")  # type: ignore[index]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=2)
    print(f"\nwrote profile -> {args.out}")
    for t, o in prof["ops"].items():  # type: ignore[union-attr]
        lm, em = o["latency_ms"], o["energy_mJ"]
        print(f"  {t:<11} latency {lm['median']:.1f} ms "
              f"[{lm['iqr'][0]:.1f}, {lm['iqr'][1]:.1f}]  energy "
              + (f"{em['median']:.0f} mJ" if em["median"] is not None else "n/a")
              + ("" if o["valid_device_anchor"] else "   (NOT a valid device anchor)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
