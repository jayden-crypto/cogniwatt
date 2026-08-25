"""Model backends for the three tiers (R2 Deliverable B).

PERFORMANCE CONTRACT
--------------------
Nothing in this module may be imported from `cogniwatt/env/**` or
`cogniwatt/agents/**`. Every heavyweight dependency (`llama_cpp`, `anthropic`)
is imported lazily inside the backend that needs it, so importing this module
costs nothing and a machine without CUDA can still run the mock path.
See docs/R2_HANDOFF.md §1.1.

THE FOUR BACKENDS
-----------------
* `MockBackend`          -- a *generative* tier simulator with a known accuracy
                            and a known intra-class correlation. Its purpose is
                            to validate the estimators in `r2_analyze.py`
                            against ground truth before any real inference is
                            paid for. See the note on gamma recovery below.
* `LlamaCppBackend`      -- `think_tiny` / `think_big`, GGUF via llama.cpp.
* `AnthropicBatchBackend`-- `offload` accuracy, via the Message Batches API
                            (half price). Gives answers but NOT latency.
* `AnthropicLiveBackend` -- `offload` latency, via ordinary synchronous calls.

WHY OFFLOAD NEEDS TWO BACKENDS
------------------------------
The handoff asks for offload accuracy cheaply (Batch API, ~$4) *and* for the
real wall-clock round-trip latency distribution (§7b). Those are incompatible
in one pass: a batch request is queued and returns in minutes-to-hours, so its
elapsed time measures Anthropic's queue, not a robot's network round trip.
So accuracy comes from the batch pass and latency from a separate small
synchronous pass (n ~ 200, a few cents). Both are recorded; the report says
which number came from which.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .prompts import CHOICE_SCHEMA, stable_seed


@dataclass
class Query:
    """One prompt plus the bookkeeping a backend may need.

    Real backends read `prompt` and nothing else. `shown_true_idx` is present
    for the mock simulator and for provenance -- a backend that peeked at it
    would be cheating, which is why the real ones do not take it.
    """
    qid: str
    prompt: str
    k: int
    scenario_id: str
    shown_true_idx: int
    repeat: int


@dataclass
class TierResponse:
    qid: str
    text: Optional[str]
    latency_ms: float = float("nan")
    error: Optional[str] = None
    #: Token accounting where the backend reports it (cloud tier).
    usage: Dict[str, int] = field(default_factory=dict)


class Backend:
    """Interface every tier is measured through."""

    name: str = "backend"
    model_id: str = "unknown"

    def answer(self, q: Query) -> TierResponse:
        raise NotImplementedError

    def answer_batch(self, qs: Sequence[Query]) -> List[TierResponse]:
        """Default: sequential. Overridden where a real batch API exists."""
        return [self.answer(q) for q in qs]

    def meta(self) -> Dict[str, object]:
        return {"backend": self.name, "model_id": self.model_id}

    def close(self) -> None:
        """Free VRAM. On a 4 GB card this is mandatory, not a nicety -- two
        resident models push `think_big` into partial CPU offload, which
        silently invalidates its hardware anchor (docs/R2_HANDOFF.md §5.1)."""


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------
class MockBackend(Backend):
    """A tier simulator with a *known* accuracy and a *known* repeat correlation.

    The generative model is beta-binomial, which is the simplest thing that
    reproduces the phenomenon the project is about:

        theta_m ~ Beta(alpha, beta)      per scenario, mean = `accuracy`
        correct_i | theta_m ~ Bernoulli(theta_m)     for each repeat i

    A model's competence on a *particular* referring expression is fixed; it
    misreads the same expression the same way. `theta_m` is that per-scenario
    competence, and re-querying only resamples the Bernoulli, not `theta_m`.

    For this model the intra-class correlation is exactly

        rho = 1 / (1 + alpha + beta) = 1 / (1 + kappa)

    so setting `kappa = (1 - rho) / rho` gives an exact target rho. That makes
    this backend a *test oracle*: run `r2_analyze.py` over mock data and it must
    recover the rho it was given. If the estimator cannot recover a known rho,
    its gamma on real data means nothing -- and gamma is the headline result.

    Errors are not uniform. When the model is wrong it prefers index 0
    (`position_bias`), which is the dominant failure mode of small instruct
    models and the one §6.1 asks to be reported.
    """

    name = "mock"

    def __init__(self, tier: str, accuracy: float, rho: float,
                 position_bias: float = 0.35, format_violation: float = 0.0,
                 seed: int = 0):
        if not 0.0 < rho < 1.0:
            raise ValueError("rho must be in (0, 1)")
        self.model_id = f"mock:{tier}"
        self.tier = tier
        self.accuracy = float(accuracy)
        self.rho = float(rho)
        self.position_bias = float(position_bias)
        self.format_violation = float(format_violation)
        self._seed = int(seed)
        kappa = (1.0 - self.rho) / self.rho
        self._alpha = self.accuracy * kappa
        self._beta = (1.0 - self.accuracy) * kappa
        self._theta: Dict[str, float] = {}

    def _theta_for(self, scenario_id: str) -> float:
        """Per-scenario competence, drawn once and cached.

        Keyed off the scenario id and the backend seed, so the same scenario
        gets the same latent competence no matter what order it is measured in.
        """
        if scenario_id not in self._theta:
            rng = np.random.default_rng(
                stable_seed(self._seed, self.tier, scenario_id))
            self._theta[scenario_id] = float(rng.beta(self._alpha, self._beta))
        return self._theta[scenario_id]

    def answer(self, q: Query) -> TierResponse:
        rng = np.random.default_rng(
            stable_seed(self._seed, self.tier, q.scenario_id, q.repeat))
        theta = self._theta_for(q.scenario_id)

        if rng.random() < self.format_violation:
            return TierResponse(q.qid, "I'm not sure which one you mean.",
                                latency_ms=1.0)

        if rng.random() < theta:
            choice = q.shown_true_idx
        else:
            wrong = [j for j in range(q.k) if j != q.shown_true_idx]
            if 0 in wrong and rng.random() < self.position_bias:
                choice = 0
            else:
                choice = int(rng.choice(wrong))
        return TierResponse(q.qid, str(choice), latency_ms=1.0)

    def meta(self) -> Dict[str, object]:
        d = super().meta()
        d.update({"accuracy_target": self.accuracy, "rho_target": self.rho,
                  "position_bias": self.position_bias,
                  "format_violation": self.format_violation,
                  "note": "SYNTHETIC -- never ship these numbers"})
        return d


# ---------------------------------------------------------------------------
# Local GGUF tiers
# ---------------------------------------------------------------------------
def _ensure_cuda_dlls() -> List[str]:
    """Make the pip-installed CUDA runtime visible to `ggml-cuda.dll` on Windows.

    The cu124 llama-cpp-python wheel ships `ggml-cuda.dll` but *not* the CUDA
    runtime it links against (`cudart64_12`, `cublas64_12`, `cublasLt64_12`).
    Those come from `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12`, which
    install into `site-packages/nvidia/*/bin`.

    Both `os.add_dll_directory` *and* `PATH` are needed, and neither alone is
    enough -- which is why this looks redundant:

    * Python 3.8+ normally loads extension DLLs with
      `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`, which consults `add_dll_directory`
      but **not** `PATH`.
    * llama-cpp-python, however, passes `winmode=ctypes.RTLD_GLOBAL`, and
      `ctypes.RTLD_GLOBAL` is **0** on Windows. `LoadLibraryExW(path, 0, 0)`
      falls back to the *legacy* search order, which consults `PATH` but
      **ignores** every `add_dll_directory` entry.

    So the directory has to be in both places to survive either loader path.
    Without it the import fails with "Could not find module llama.dll (or one
    of its dependencies)" -- naming the wrong DLL entirely, since the one that
    actually failed is `ggml-cuda.dll`, three links down the chain.

    No-op on non-Windows, where the loader honours RPATH/LD_LIBRARY_PATH.
    """
    added: List[str] = []
    if os.name != "nt":
        return added
    import site

    roots = list(site.getsitepackages())
    sp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots.append(os.path.join(os.path.dirname(sp), ".venv", "Lib", "site-packages"))
    seen = set()
    for root in roots:
        nvidia = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia) or nvidia in seen:
            continue
        seen.add(nvidia)
        for pkg in sorted(os.listdir(nvidia)):
            b = os.path.join(nvidia, pkg, "bin")
            if os.path.isdir(b):
                try:
                    os.add_dll_directory(b)
                except OSError:
                    pass
                if b not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")
                added.append(b)
    return added


class LlamaCppBackend(Backend):
    """`think_tiny` / `think_big` through llama.cpp.

    llama.cpp rather than raw `transformers` on purpose: the project's claim is
    about *quantised on-device* models, and GGUF Q8/Q4 is what actually ships
    on edge hardware. fp16 transformers weights would measure a model nobody
    deploys (docs/R2_HANDOFF.md §5.1).

    `n_gpu_layers=-1` puts every layer on the GPU. Dial it down only if the
    model will not load -- and note that a partial-offload run is still valid
    for *accuracy* (identical outputs, just slower) but invalid as a *hardware*
    anchor, because it characterises half a GPU plus a CPU.
    """

    name = "llama_cpp"

    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 1024,
                 max_tokens: int = 64, seed: int = 0, verbose: bool = False):
        _ensure_cuda_dlls()
        from llama_cpp import Llama  # lazy: keeps CUDA off the import path

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.model_path = model_path
        self.model_id = os.path.basename(model_path)
        self.max_tokens = int(max_tokens)
        self.n_gpu_layers = int(n_gpu_layers)
        self._llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers,
                          n_ctx=n_ctx, seed=seed, verbose=verbose, logits_all=False)

    def answer(self, q: Query) -> TierResponse:
        t0 = time.perf_counter()
        try:
            out = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": q.prompt}],
                temperature=self.temperature, max_tokens=self.max_tokens,
                seed=self._call_seed(q),
            )
            dt = (time.perf_counter() - t0) * 1000.0
            text = out["choices"][0]["message"]["content"]
            return TierResponse(q.qid, text, latency_ms=dt)
        except Exception as e:  # a dead call must not kill a 20k-call campaign
            dt = (time.perf_counter() - t0) * 1000.0
            return TierResponse(q.qid, None, latency_ms=dt, error=repr(e))

    #: Set by the measurement script before the pass; kept as an attribute so
    #: the deployment temperature is recorded in one place.
    temperature: float = 0.7

    def _call_seed(self, q: Query) -> int:
        # Distinct seed per repeat, or llama.cpp returns the identical sample
        # every time and gamma collapses to a measurement of nothing. Stable
        # across processes so a re-run reproduces the same samples.
        return stable_seed(q.scenario_id, q.repeat) % (2 ** 31)

    def meta(self) -> Dict[str, object]:
        d = super().meta()
        d.update({"model_path": self.model_path, "n_gpu_layers": self.n_gpu_layers,
                  "max_tokens": self.max_tokens, "temperature": self.temperature,
                  "fully_on_gpu": self.n_gpu_layers == -1})
        return d

    def close(self) -> None:
        try:
            self._llm.close()
        except Exception:
            pass
        self._llm = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Cloud tier
# ---------------------------------------------------------------------------
#: Structured output config. `output_config.format` needs the json_schema
#: wrapper -- the bare `{"choice": ...}` schema shown in the handoff is not the
#: accepted shape. Prefill is not used: it returns 400 on all current models.
OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": CHOICE_SCHEMA}}

#: Haiku 4.5 rather than Opus 5 / Sonnet 5 because the decay protocol needs a
#: controlled sampling temperature, and `temperature` was removed from the
#: frontier models (400). Haiku 4.5 still accepts it, so it is the tier where
#: the measurement is actually well-defined (docs/R2_HANDOFF.md §5).
DEFAULT_CLOUD_MODEL = "claude-haiku-4-5"


def _params(model: str, prompt: str, temperature: float, max_tokens: int) -> Dict:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": OUTPUT_CONFIG,
    }


class AnthropicBatchBackend(Backend):
    """`offload` accuracy through the Message Batches API (half price).

    Results come back in arbitrary order, so they are keyed by `custom_id`
    throughout -- never by position. Latency is deliberately left as NaN here:
    see the module docstring.
    """

    name = "anthropic_batch"

    def __init__(self, model: str = DEFAULT_CLOUD_MODEL, temperature: float = 0.7,
                 max_tokens: int = 256, poll_s: float = 30.0):
        import anthropic  # lazy

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model_id = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.poll_s = float(poll_s)
        self.batch_ids: List[str] = []

    def answer(self, q: Query) -> TierResponse:
        return self.answer_batch([q])[0]

    def answer_batch(self, qs: Sequence[Query]) -> List[TierResponse]:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        reqs = [Request(custom_id=q.qid,
                        params=MessageCreateParamsNonStreaming(
                            **_params(self.model_id, q.prompt,
                                      self.temperature, self.max_tokens)))
                for q in qs]
        batch = self._client.messages.batches.create(requests=reqs)
        self.batch_ids.append(batch.id)

        while True:
            b = self._client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            time.sleep(self.poll_s)

        by_id: Dict[str, TierResponse] = {}
        for r in self._client.messages.batches.results(batch.id):
            rt = r.result.type
            if rt == "succeeded":
                msg = r.result.message
                text = next((blk.text for blk in msg.content if blk.type == "text"), None)
                by_id[r.custom_id] = TierResponse(
                    r.custom_id, text,
                    usage={"input": msg.usage.input_tokens,
                           "output": msg.usage.output_tokens})
            else:
                by_id[r.custom_id] = TierResponse(r.custom_id, None, error=rt)
        # Any id the batch never reported is an explicit failure, not a silent gap.
        return [by_id.get(q.qid, TierResponse(q.qid, None, error="missing_from_batch"))
                for q in qs]

    def meta(self) -> Dict[str, object]:
        d = super().meta()
        d.update({"temperature": self.temperature, "max_tokens": self.max_tokens,
                  "batch_ids": self.batch_ids, "pricing": "batch (50%)",
                  "latency_valid": False})
        return d


class AnthropicLiveBackend(Backend):
    """`offload` latency through ordinary synchronous calls.

    This is the pass that produces a genuine wall-clock round-trip
    distribution for the env's offload latency. Run it small (n ~ 200) --
    it is at full price and its purpose is timing, not accuracy.
    """

    name = "anthropic_live"

    def __init__(self, model: str = DEFAULT_CLOUD_MODEL, temperature: float = 0.7,
                 max_tokens: int = 256, timeout_s: float = 30.0):
        import anthropic  # lazy

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(timeout=timeout_s, max_retries=0)
        self.model_id = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_s = float(timeout_s)

    def answer(self, q: Query) -> TierResponse:
        t0 = time.perf_counter()
        try:
            msg = self._client.messages.create(
                **_params(self.model_id, q.prompt, self.temperature, self.max_tokens))
            dt = (time.perf_counter() - t0) * 1000.0
            text = next((b.text for b in msg.content if b.type == "text"), None)
            return TierResponse(q.qid, text, latency_ms=dt,
                                usage={"input": msg.usage.input_tokens,
                                       "output": msg.usage.output_tokens})
        except Exception as e:
            # Timeouts and 429s are part of the latency story, not noise to
            # discard -- §7b asks for the failure/timeout rate explicitly.
            dt = (time.perf_counter() - t0) * 1000.0
            return TierResponse(q.qid, None, latency_ms=dt, error=repr(e))

    def meta(self) -> Dict[str, object]:
        d = super().meta()
        d.update({"temperature": self.temperature, "max_tokens": self.max_tokens,
                  "timeout_s": self.timeout_s, "pricing": "standard",
                  "latency_valid": True})
        return d


# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------
#: Where the GGUF files are expected. Override with --models-dir.
DEFAULT_MODELS_DIR = os.path.join("models")

#: Filenames the download step should produce. Quantisation is part of the
#: identity of the measurement: it changes both accuracy and energy, and both
#: must trace to the same build (docs/R2_HANDOFF.md §5.1).
TIER_MODEL_FILES: Dict[str, str] = {
    "think_tiny": "qwen2.5-0.5b-instruct-q8_0.gguf",
    "think_big": "llama-3.2-3b-instruct-q4_k_m.gguf",
}

#: Mock calibration. Deliberately set to the handoff's target gradient
#: (0.45 / 0.70 / 0.88) so a mock end-to-end run reproduces the shape the real
#: campaign is aiming for -- and so a real result that misses it is obvious.
MOCK_PROFILE: Dict[str, Dict[str, float]] = {
    "think_tiny": {"accuracy": 0.45, "rho": 0.55, "position_bias": 0.45,
                   "format_violation": 0.06},
    "think_big": {"accuracy": 0.70, "rho": 0.42, "position_bias": 0.30,
                  "format_violation": 0.01},
    "offload": {"accuracy": 0.88, "rho": 0.68, "position_bias": 0.15,
                "format_violation": 0.0},
}


def make_backend(tier: str, mock: bool = False, models_dir: str = DEFAULT_MODELS_DIR,
                 temperature: float = 0.7, n_gpu_layers: int = -1,
                 seed: int = 0, live_cloud: bool = False) -> Backend:
    """Build the backend for `tier`. The only place a tier maps to a model."""
    if mock:
        if tier not in MOCK_PROFILE:
            raise ValueError(f"no mock profile for tier {tier!r}")
        return MockBackend(tier, seed=seed, **MOCK_PROFILE[tier])

    if tier in TIER_MODEL_FILES:
        path = os.path.join(models_dir, TIER_MODEL_FILES[tier])
        be = LlamaCppBackend(path, n_gpu_layers=n_gpu_layers, seed=seed)
        be.temperature = temperature
        return be

    if tier == "offload":
        cls = AnthropicLiveBackend if live_cloud else AnthropicBatchBackend
        return cls(temperature=temperature)

    raise ValueError(f"unknown tier {tier!r}")
