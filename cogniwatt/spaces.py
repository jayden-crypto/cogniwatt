"""Minimal Gymnasium-compatible space objects.

CogniWatt's Tier-1 environment deliberately has *no* hard dependency on
gymnasium: the env must run bit-identically on arm64 macOS (M2, dev/inference
rig) and x86_64 + CUDA (RTX 3060, training rig). Pinning gymnasium across both
platforms buys nothing, because we implement PPO ourselves.

We therefore reproduce the exact subset of the Gymnasium API surface we use.
`cogniwatt.env.homegrid.HomeGridEdge` follows the Gymnasium v0.29+ contract
(`reset(seed=...) -> (obs, info)`, `step() -> (obs, r, terminated, truncated,
info)`), so `to_gymnasium()` can wrap it for SB3 later with zero changes.
"""
from __future__ import annotations

import numpy as np


class Box:
    """Continuous box space. Mirrors gymnasium.spaces.Box for our usage."""

    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.asarray(low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.asarray(high, dtype=dtype)
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x) -> bool:
        x = np.asarray(x)
        return x.shape == self.shape and bool(np.all(x >= self.low - 1e-6) and np.all(x <= self.high + 1e-6))

    def __repr__(self) -> str:
        return f"Box({self.shape}, {self.dtype.__name__ if hasattr(self.dtype,'__name__') else self.dtype})"


class Discrete:
    """Discrete space over {0, ..., n-1}. Mirrors gymnasium.spaces.Discrete."""

    def __init__(self, n: int):
        self.n = int(n)
        self.shape = ()
        self.dtype = np.int64

    def sample(self, rng: np.random.Generator | None = None) -> int:
        rng = rng or np.random.default_rng()
        return int(rng.integers(self.n))

    def contains(self, x) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    def __repr__(self) -> str:
        return f"Discrete({self.n})"
