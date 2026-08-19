"""Virtual Edge Device (VED) - cost model interface.

OWNERSHIP
---------
The *interface* in this file (Op, Cost, DeviceProfile, VirtualEdgeDevice) is
frozen by R1 (RL & Environment) because the environment's reward function
depends on it.

The *numbers* in DEVICE_PROFILES below start as R1 placeholders derived from a
roofline estimate + published vendor figures. They are replaced with values
calibrated against our two anchor machines. On a two-person team the rule is
"whoever runs the model profiles it":

  * R2 (Adit, RTX 3060) instruments the SLM measurement passes it is already
    running for grounding accuracy, and returns per-inference latency and
    energy for THINK_TINY / THINK_BIG / OFFLOAD. That is the rtx3060 anchor.
  * R1 (Jayden, M2 Air) owns the m2_air anchor via `powermetrics`, the thermal
    fit on the fanless chassis, the actuation model, the network model, and the
    roofline extrapolation to the Jetson / Pi / Coral profiles.

Contract either way: keep the class/method signatures identical, replace
`_STUB_COST_TABLE` and `ThermalModel`. Nothing in cogniwatt/env/ may change.

MODELLING NOTE (important, goes in the report)
----------------------------------------------
Locomotion energy dominates compute energy on a mobile base by roughly an
order of magnitude (a 1 m/s drive step costs ~10 J; a 3B-parameter inference
costs ~3.6 J on Orin Nano). Consequently the *energy* constraint binds mainly
on small-battery, NPU-class devices, while *latency* (the human is standing
there waiting) and *patience* (finite tolerance for clarifying questions) bind
on every device. We therefore expose all three as separately weighted cost
terms and sweep them in the Pareto study rather than collapsing them into one
scalar a priori.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Operations the robot can spend resources on.
# --------------------------------------------------------------------------
MOVE = "move"
PICK = "pick"
PLACE = "place"
THINK_TINY = "think_tiny"
THINK_BIG = "think_big"
OFFLOAD = "offload"
ASK = "ask"

ALL_OPS = (MOVE, PICK, PLACE, THINK_TINY, THINK_BIG, OFFLOAD, ASK)
COMPUTE_OPS = (THINK_TINY, THINK_BIG, OFFLOAD, ASK)


@dataclass(frozen=True)
class Cost:
    """Resource cost of a single operation."""
    latency_ms: float
    energy_mJ: float
    available: bool = True


# --------------------------------------------------------------------------
# COST TABLE - derived from the roofline model, anchored on real measurement.
# --------------------------------------------------------------------------
# Built at import time by `cogniwatt.hardware.roofline`, which models prefill as
# compute-bound and decode as memory-bound, using efficiency factors measured on
# the M2 Air (scripts/r1_profile_m2.py):
#     prefill-shape GEMM  680 GFLOP/s  = 19% of the 3.6 TFLOP/s datasheet peak
#     streaming read BW  76.9 GB/s     = 77% of the 100 GB/s datasheet peak
#
# The hand-written placeholder table this replaced was optimistic by 10-17x,
# because it implicitly assumed peak GEMM throughput. A 150-token prompt is a
# short, skinny matmul with poor occupancy; using peak is the classic error.
# Concretely: think_big on the M2 Air went from an assumed 95 ms to a modelled
# 1591 ms, and on a Pi 5 from 900 ms to 32 s - which is why the Pi 5 policy has
# to stop reasoning locally and start offloading.
#
# R2 (Adit) delivers measured RTX 3060 latency/energy; `apply_measured_profile`
# below is where those numbers override the model.
from ..hardware.roofline import build_cost_table

_STUB_COST_TABLE: Dict[str, Dict[str, Tuple[float, float, bool]]] = build_cost_table()


def apply_measured_profile(path: str) -> bool:
    """Override modelled costs with measured ones for a single device.

    Consumes the schema in `data/r2_rtx3060_profile.json` (see docs/R2_HANDOFF.md
    section 7b). Returns True if anything was applied. Measurements always win
    over the roofline; every device without a measurement stays modelled, and
    the report must say which is which.
    """
    import json
    import os
    if not os.path.exists(path):
        return False
    d = json.loads(open(path).read())
    dev = d["device"]
    if dev not in _STUB_COST_TABLE:
        raise KeyError(f"measured profile names unknown device {dev!r}")
    for op, m in d.get("ops", {}).items():
        lat = m.get("latency_ms_median")
        en = m.get("energy_mJ_median")
        if lat is None:
            continue
        old = _STUB_COST_TABLE[dev][op]
        _STUB_COST_TABLE[dev][op] = (float(lat),
                                     float(en) if en is not None else old[1],
                                     old[2])
    return True


@dataclass
class ThermalModel:
    """First-order lumped thermal model -> clock throttling.

    T_{k+1} = T_k + dt/tau * (T_amb + R_th * P - T_k)

    Above `t_throttle` the device sheds clock, which we expose as a latency
    multiplier applied to *compute* ops only (actuation is unaffected).
    R1 fits the parameters to a sustained-load run on the fanless M2 Air, which
    throttles measurably within ~10 minutes - the fanless chassis is the reason
    this anchor is worth having.
    """
    # `max_slowdown` is MEASURED: a 10-minute sustained-GEMM run on the fanless
    # M2 Air degraded steady-state throughput by 2.06x (data/throttle_m2_air.json,
    # data/thermal_fit_m2.json). The remaining parameters are a plausible
    # lumped-RC parameterisation reproducing that endpoint; the per-burst signal
    # was too noisy to identify the time constant precisely (the fit landed at
    # ~1000s but with wide error bars), so tau_s is set to a defensible 300s
    # rather than the fragile fitted value. A fan-cooled device (the RTX 3060)
    # shows far less throttling and uses a higher t_throttle.
    t_ambient: float = 25.0
    t_throttle: float = 70.0
    t_max: float = 95.0
    r_th: float = 2.2          # degC per Watt
    tau_s: float = 300.0       # thermal time constant (see note above)
    max_slowdown: float = 2.06 # MEASURED on M2 Air sustained load

    temp_c: float = field(default=25.0, init=False)

    def reset(self) -> None:
        self.temp_c = self.t_ambient

    def step(self, energy_mJ: float, latency_ms: float) -> None:
        dt = max(latency_ms, 1e-6) / 1000.0
        power_w = (energy_mJ / 1000.0) / dt if dt > 0 else 0.0
        target = self.t_ambient + self.r_th * power_w
        alpha = 1.0 - np.exp(-dt / self.tau_s)
        self.temp_c += alpha * (target - self.temp_c)
        self.temp_c = float(np.clip(self.temp_c, self.t_ambient, self.t_max))

    @property
    def throttle_factor(self) -> float:
        if self.temp_c <= self.t_throttle:
            return 1.0
        frac = (self.temp_c - self.t_throttle) / (self.t_max - self.t_throttle)
        return float(1.0 + (self.max_slowdown - 1.0) * np.clip(frac, 0.0, 1.0))

    @property
    def norm_temp(self) -> float:
        return float((self.temp_c - self.t_ambient) / (self.t_max - self.t_ambient))


@dataclass
class DeviceProfile:
    """Static description of an edge device.

    `feature_vector` is what the RL policy is *conditioned* on. It must contain
    only quantities a deployment engineer would know from a datasheet - never
    the cost table itself - otherwise zero-shot transfer to an unseen device is
    not a meaningful claim.
    """
    name: str
    tops_int8: float           # peak INT8 throughput
    mem_bw_gbps: float         # memory bandwidth
    tdp_w: float               # thermal design power
    battery_wh: float          # onboard battery
    has_npu: float             # 1.0 / 0.0
    memory_gb: float = 8.0     # usable RAM for model weights

    episode_energy_budget_mJ: float = 400_000.0
    episode_time_budget_ms: float = 60_000.0

    def feature_vector(self) -> np.ndarray:
        """Normalised datasheet features fed to the policy. Shape (6,).

        `memory_gb` is load-bearing, not padding: whether a model fits is a
        function of RAM, so without it a policy has no way to anticipate that
        `think_big` will be unavailable on an unseen device. Everything here is
        a datasheet figure a deployment engineer would know in advance - never
        the cost table itself, or zero-shot transfer would be vacuous.
        """
        return np.array([
            np.log1p(self.tops_int8) / np.log1p(300.0),
            np.log1p(self.mem_bw_gbps) / np.log1p(400.0),
            self.tdp_w / 200.0,
            self.battery_wh / 100.0,
            self.has_npu,
            np.log1p(self.memory_gb) / np.log1p(32.0),
        ], dtype=np.float32)


DEVICE_PROFILES: Dict[str, DeviceProfile] = {
    "orin_nano": DeviceProfile("orin_nano", 40.0, 68.0, 15.0, 30.0, 1.0, memory_gb=8.0),
    "pi5":       DeviceProfile("pi5", 0.0, 17.0, 8.0, 25.0, 0.0, memory_gb=8.0),
    "orin_nx":   DeviceProfile("orin_nx", 100.0, 102.0, 25.0, 40.0, 1.0, memory_gb=16.0),
    "m2_air":    DeviceProfile("m2_air", 15.8, 100.0, 20.0, 52.6, 1.0, memory_gb=16.0),
    # ---- held out ----
    # Memory-limited: the 3B model does not fit, so the policy must shift its
    # channel portfolio using datasheet features alone.
    "jetson_nano_2gb": DeviceProfile("jetson_nano_2gb", 1.0, 25.6, 10.0, 12.0, 1.0,
                                     memory_gb=1.2,
                                     episode_energy_budget_mJ=300_000.0),
    "rtx3060":   DeviceProfile("rtx3060", 102.0, 336.0, 115.0, 60.0, 0.0, memory_gb=6.0),
}

FEATURE_DIM = 6


class VirtualEdgeDevice:
    """Stateful edge-device emulator: costs, battery, thermals, network.

    Usage (from the env):
        ved = VirtualEdgeDevice("orin_nano", rng)
        ved.reset()
        c = ved.quote(THINK_BIG)      # what would it cost?
        if c.available:
            ved.commit(THINK_BIG, c)  # actually spend it
    """

    def __init__(self, device: str, rng: np.random.Generator | None = None,
                 thermal: ThermalModel | None = None):
        if device not in DEVICE_PROFILES:
            raise KeyError(f"unknown device {device!r}; have {sorted(DEVICE_PROFILES)}")
        self.device = device
        self.profile = DEVICE_PROFILES[device]
        self.rng = rng or np.random.default_rng()
        self.thermal = thermal or ThermalModel()
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self, disabled_ops=None) -> None:
        """`disabled_ops` marks models that do not fit on this device this
        episode. Used both for genuine hardware limits and for capability
        dropout during training (see HomeGridEdge.capability_dropout)."""
        self.disabled_ops = set(disabled_ops or ())
        self.thermal.reset()
        self.energy_spent_mJ = 0.0
        self.time_spent_ms = 0.0
        # AR(1) link quality in [0, 1]; low quality -> offload failures
        self.net_quality = float(np.clip(self.rng.normal(0.75, 0.15), 0.05, 1.0))

    def step_network(self) -> None:
        """Advance the wireless link. R1 may replace with a netem-derived trace."""
        self.net_quality = float(np.clip(
            0.9 * self.net_quality + 0.1 * self.rng.normal(0.75, 0.12)
            + self.rng.normal(0.0, 0.04), 0.05, 1.0))

    # -- cost --------------------------------------------------------------
    def quote(self, op: str) -> Cost:
        """Cost of `op` right now, including thermal throttling. No side effects."""
        lat, en, avail = _STUB_COST_TABLE[self.device][op]
        if not avail or op in self.disabled_ops:
            return Cost(0.0, 0.0, False)
        if op in COMPUTE_OPS:
            lat = lat * self.thermal.throttle_factor
        if op == OFFLOAD:
            # Poor link inflates round-trip time (retransmissions).
            lat = lat * (1.0 + 2.0 * (1.0 - self.net_quality))
            en = en * (1.0 + 1.0 * (1.0 - self.net_quality))
        return Cost(float(lat), float(en), True)

    def offload_succeeds(self) -> bool:
        """Did the cloud round trip come back in time? Costs are paid either way."""
        return bool(self.rng.random() < self.net_quality)

    def commit(self, op: str, cost: Cost) -> None:
        """Actually spend the resources and advance device state."""
        self.energy_spent_mJ += cost.energy_mJ
        self.time_spent_ms += cost.latency_ms
        self.thermal.step(cost.energy_mJ, cost.latency_ms)
        self.step_network()

    # -- budget state ------------------------------------------------------
    @property
    def energy_frac_remaining(self) -> float:
        return float(np.clip(
            1.0 - self.energy_spent_mJ / self.profile.episode_energy_budget_mJ, 0.0, 1.0))

    @property
    def time_frac_remaining(self) -> float:
        return float(np.clip(
            1.0 - self.time_spent_ms / self.profile.episode_time_budget_ms, 0.0, 1.0))

    @property
    def depleted(self) -> bool:
        return self.energy_frac_remaining <= 0.0

    @property
    def out_of_time(self) -> bool:
        return self.time_frac_remaining <= 0.0
