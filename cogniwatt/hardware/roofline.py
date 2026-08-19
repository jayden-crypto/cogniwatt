"""Roofline cost model for transformer inference on edge devices.

Owner: R1 (inherited from the dissolved R3 workstream).

Replaces the hand-written placeholder numbers in `interfaces.ved` with values
derived from first principles and anchored to a real measurement.

THE MODEL
---------
Decoder-only transformer inference splits into two phases with different
bottlenecks, and conflating them is the usual way these estimates go wrong:

  PREFILL  - process the prompt. Compute bound.
             FLOPs ~= 2 * P * L_prompt        (P = parameter count)
             latency = FLOPs / achievable_FLOPS

  DECODE   - generate each output token. *Memory* bound: every weight is read
             once per token, and at batch size 1 there is no arithmetic
             intensity to hide it behind.
             bytes = L_gen * P * bytes_per_param
             latency = bytes / achievable_bandwidth

CALIBRATION
-----------
`achievable` is not the datasheet peak. Measured on the M2 Air anchor
(`scripts/r1_profile_m2.py`):

    peak GEMM (large, square)        2695 GFLOP/s   ->  75% of 3.6 TFLOP/s spec
    GEMM at prefill shape (L=150)     680 GFLOP/s   ->  19% of spec
    streaming read bandwidth         76.9 GB/s      ->  77% of 100 GB/s spec

The prefill-shape number is the one that matters and it is 4x below peak: a
150-token prompt is a short, skinny GEMM with poor occupancy. Using peak here
is the single biggest error in naive estimates - it is what made the original
placeholder table optimistic by 10-17x.

WHAT IS MEASURED vs MODELLED (state this in the report)
------------------------------------------------------
  measured   : M2 Air compute + bandwidth efficiency, M2 Air thermal curve
  measured   : RTX 3060 per-call latency and energy (R2/Adit delivers)
  modelled   : all other devices, by applying the measured efficiency factors
               to datasheet peaks
  modelled   : radio energy for cloud offload, from published Wi-Fi figures
  modelled   : actuation energy, from an explicit physical model
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# --------------------------------------------------------------------------
# Efficiency factors, calibrated on the M2 Air anchor.
# --------------------------------------------------------------------------
#: Fraction of datasheet peak achieved by a short-prompt prefill GEMM.
#: Measured: 680 / 3600 on the M2 Air at (150 x 896 x 4864).
EFF_PREFILL = 0.19
#: Fraction of datasheet peak memory bandwidth achieved by streaming reads.
#: Measured: 76.9 / 100 on the M2 Air.
EFF_BANDWIDTH = 0.77
#: CPU-only backends lose more: no wide SIMT, and llama.cpp's CPU kernels are
#: further from peak than its Metal/CUDA ones.
EFF_PREFILL_CPU = 0.40

#: Bytes per parameter by GGUF quantisation (includes scale/zero-point overhead).
BYTES_PER_PARAM = {"q8_0": 1.06, "q4_k_m": 0.55, "fp16": 2.0}


@dataclass(frozen=True)
class ModelSpec:
    """A language model as the roofline sees it."""
    name: str
    params_b: float          # billions of parameters
    quant: str               # key into BYTES_PER_PARAM

    @property
    def bytes(self) -> float:
        return self.params_b * 1e9 * BYTES_PER_PARAM[self.quant]


@dataclass(frozen=True)
class HwSpec:
    """Datasheet peaks plus power. Peaks are fp16/fp32 dense figures."""
    name: str
    peak_gflops: float       # dense fp16/fp32 GFLOP/s
    peak_bw_gb_s: float
    tdp_w: float
    idle_w: float
    is_cpu_only: bool = False
    max_model_gb: float = 1e9   # what physically fits in memory

    @property
    def eff_prefill(self) -> float:
        return EFF_PREFILL_CPU if self.is_cpu_only else EFF_PREFILL


#: Datasheet figures. Only `m2_air` is anchored by our own measurement;
#: `rtx3060` will be anchored by R2's profile when it lands.
HW: Dict[str, HwSpec] = {
    "m2_air":      HwSpec("m2_air",      3600.0, 100.0,  20.0, 3.0, max_model_gb=12.0),
    "orin_nano":   HwSpec("orin_nano",   1280.0,  68.0,  15.0, 2.5, max_model_gb=6.0),
    "orin_nx":     HwSpec("orin_nx",     3400.0, 102.0,  25.0, 4.0, max_model_gb=12.0),
    "pi5":         HwSpec("pi5",           76.8,  17.0,   8.0, 2.0, is_cpu_only=True,
                          max_model_gb=6.0),
    # Jetson Nano 2GB. Replaces an earlier `coral_micro` profile, which was
    # physically indefensible: a Coral Edge TPU has 8 MB of SRAM and no
    # transformer decode support, so it cannot run a 0.5B LLM at any speed.
    # The Nano 2GB genuinely has the property the held-out set needs - after
    # OS overhead roughly 1.2 GB is usable, so the 0.5B Q8 model (0.52 GB)
    # fits and the 3B Q4 (1.77 GB) does not. Structurally, not just
    # quantitatively, different from every training device.
    "jetson_nano_2gb": HwSpec("jetson_nano_2gb", 472.0, 25.6, 10.0, 1.8,
                              max_model_gb=1.2),
    "rtx3060":     HwSpec("rtx3060",    10900.0, 336.0, 115.0, 14.2, max_model_gb=5.0),
}

MODELS: Dict[str, ModelSpec] = {
    "tiny": ModelSpec("qwen2.5-0.5b-instruct", 0.49, "q8_0"),
    "big":  ModelSpec("llama-3.2-3b-instruct", 3.21, "q4_k_m"),
}

#: Workload shape for the grounding query: the instruction plus k candidate
#: descriptions, answered with a single index under structured output.
PROMPT_TOKENS = 150
GEN_TOKENS = 8

#: Utilisation while active, as a fraction of TDP. Compute-bound prefill pushes
#: the part harder than memory-bound decode does.
UTIL_PREFILL = 0.85
UTIL_DECODE = 0.55


def fits(device: str, model: str) -> bool:
    """Does the model physically fit in this device's memory?"""
    return MODELS[model].bytes / 1e9 <= HW[device].max_model_gb


def inference_cost(device: str, model: str,
                   prompt_tokens: int = PROMPT_TOKENS,
                   gen_tokens: int = GEN_TOKENS,
                   throttle: float = 1.0) -> Optional[Dict[str, float]]:
    """Latency (ms) and energy (mJ) for one grounding query.

    Returns None when the model does not fit - which is a real constraint, not
    an edge case: it is why `coral_micro` cannot run the 3B model at all.
    """
    hw, m = HW[device], MODELS[model]
    if not fits(device, model):
        return None

    prefill_flops = 2.0 * m.params_b * 1e9 * prompt_tokens
    t_prefill = prefill_flops / (hw.peak_gflops * 1e9 * hw.eff_prefill) * throttle
    decode_bytes = gen_tokens * m.bytes
    t_decode = decode_bytes / (hw.peak_bw_gb_s * 1e9 * EFF_BANDWIDTH) * throttle

    e_prefill = t_prefill * hw.tdp_w * UTIL_PREFILL
    e_decode = t_decode * hw.tdp_w * UTIL_DECODE
    return {
        "latency_ms": (t_prefill + t_decode) * 1e3,
        "energy_mJ": (e_prefill + e_decode) * 1e3,
        "prefill_ms": t_prefill * 1e3,
        "decode_ms": t_decode * 1e3,
    }


# --------------------------------------------------------------------------
# Non-inference costs: actuation and radio. Modelled, not measured.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RobotSpec:
    """Physical parameters of the mobile base. One grid cell = `cell_m`."""
    cell_m: float = 0.5
    speed_m_s: float = 0.5
    drive_power_w: float = 10.0
    manip_time_s: float = 1.5
    manip_power_w: float = 6.0
    #: Wi-Fi round trip for a short request/response, and radio power while active.
    radio_rtt_ms: float = 140.0
    radio_power_w: float = 2.5
    #: Speak-then-listen round trip for a clarifying question.
    dialogue_rtt_ms: float = 900.0
    dialogue_power_w: float = 0.45


ROBOT = RobotSpec()


def actuation_cost(kind: str, robot: RobotSpec = ROBOT) -> Dict[str, float]:
    if kind == "move":
        t = robot.cell_m / robot.speed_m_s
        return {"latency_ms": t * 1e3, "energy_mJ": t * robot.drive_power_w * 1e3}
    if kind in ("pick", "place"):
        t = robot.manip_time_s
        return {"latency_ms": t * 1e3, "energy_mJ": t * robot.manip_power_w * 1e3}
    raise KeyError(kind)


def offload_cost(device: str, robot: RobotSpec = ROBOT) -> Dict[str, float]:
    """Cloud call: latency is the network round trip; energy is *radio*, not
    compute - the inference happens elsewhere. Adit measures the real latency
    distribution on the 3060; the radio energy stays modelled."""
    t = robot.radio_rtt_ms / 1e3
    return {"latency_ms": robot.radio_rtt_ms,
            "energy_mJ": t * robot.radio_power_w * 1e3 + HW[device].idle_w * t * 1e3}


def ask_cost(device: str, robot: RobotSpec = ROBOT) -> Dict[str, float]:
    t = robot.dialogue_rtt_ms / 1e3
    return {"latency_ms": robot.dialogue_rtt_ms,
            "energy_mJ": t * robot.dialogue_power_w * 1e3 + HW[device].idle_w * t * 1e3}


def build_cost_table() -> Dict[str, Dict[str, tuple]]:
    """Regenerate `interfaces.ved._STUB_COST_TABLE` from the roofline."""
    out: Dict[str, Dict[str, tuple]] = {}
    for dev in HW:
        row = {}
        for kind in ("move", "pick", "place"):
            c = actuation_cost(kind)
            row[kind] = (round(c["latency_ms"], 1), round(c["energy_mJ"], 1), True)
        for op, mdl in (("think_tiny", "tiny"), ("think_big", "big")):
            c = inference_cost(dev, mdl)
            row[op] = ((0.0, 0.0, False) if c is None
                       else (round(c["latency_ms"], 1), round(c["energy_mJ"], 1), True))
        for op, fn in (("offload", offload_cost), ("ask", ask_cost)):
            c = fn(dev)
            row[op] = (round(c["latency_ms"], 1), round(c["energy_mJ"], 1), True)
        out[dev] = row
    return out
