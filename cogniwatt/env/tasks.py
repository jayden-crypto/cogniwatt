"""HomeGrid-Edge world generation: floorplans, referring expressions, distances.

Owner: R1.

Design decision (justify in the report): the *spatial* problem is fully
observable - a domestic robot has a map of the house and knows where objects
are. The uncertainty is *semantic*: which of several map-consistent objects did
the human actually mean? Restricting partial observability to the semantic
channel keeps the POMDP tractable on a laptop while placing all the difficulty
in the part of the problem this project is actually about.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

GRID = 9
N_LAYOUTS = 8

NOUNS = ("mug", "book", "bottle", "bowl", "remote", "cushion")
COLORS = ("red", "blue", "green", "white", "black", "striped")
RECEPTACLES = ("shelf", "sink", "table", "drawer", "bin")

# Colour confusability -- now UNIFORM, on measurement.
#
# This was a hand-written matrix (base 0.15 with six pairs set 0.25-0.55, e.g.
# red/striped 0.55) encoding the assumption that a model's errors land on
# semantically similar distractors. R2 tested that assumption directly and it
# does not hold on this task: over 6 672 measured errors, the similarity of the
# chosen distractor was 0.1025 against 0.1025 for the average distractor -- an
# excess of -0.00006 (docs/R1_HANDBACK.md section 4).
#
# R2's colour-confusion matrix is also flat by construction, because colours are
# assigned round-robin and never mentioned in the descriptions, so the referent
# is never recoverable from colour. That is a successful check that the dataset
# has no colour shortcut rather than a measurement of confusability.
#
# A uniform off-diagonal is therefore the empirically supported choice, and the
# hand-written structure was adding signal the errors do not have. R2 also
# delivered a perceptual CIELAB matrix; it stays unwired, because finding (2)
# above says model errors will not follow it either.
#
# Kept as a constant rather than deleted so `Task.similarity` keeps its type and
# a future measured matrix can be dropped straight in.
_C = len(COLORS)
COLOR_SIM = np.full((_C, _C), 0.15, dtype=np.float64)
np.fill_diagonal(COLOR_SIM, 1.0)


# --------------------------------------------------------------------------
# Floorplans
# --------------------------------------------------------------------------
def _build_layout(door_seed: int) -> np.ndarray:
    """4-room floorplan; doorway positions vary with `door_seed`.

    Multiple layouts stop the policy from memorising one geometry, which would
    make the travel-cost trade-off degenerate.
    """
    rng = np.random.default_rng(door_seed)
    walkable = np.ones((GRID, GRID), dtype=bool)
    mid = GRID // 2
    walkable[mid, :] = False
    walkable[:, mid] = False
    for c in rng.choice([1, 2, 3], size=1, replace=False):
        walkable[mid, c] = True
    for c in rng.choice([5, 6, 7], size=1, replace=False):
        walkable[mid, c] = True
    for r in rng.choice([1, 2, 3], size=1, replace=False):
        walkable[r, mid] = True
    for r in rng.choice([5, 6, 7], size=1, replace=False):
        walkable[r, mid] = True
    return walkable


def _all_pairs_bfs(walkable: np.ndarray) -> np.ndarray:
    """Exact shortest-path lengths between every pair of free cells.

    The floorplan family is fixed and tiny, so we solve navigation *exactly*,
    once, at import time. Rollouts then look up path costs in O(1). This is the
    'A* low-level controller' of the hierarchical design, precomputed.
    """
    n = GRID * GRID
    dist = np.full((n, n), -1, dtype=np.int32)
    for src in range(n):
        r0, c0 = divmod(src, GRID)
        if not walkable[r0, c0]:
            continue
        d = dist[src]
        d[src] = 0
        q = deque([(r0, c0)])
        while q:
            r, c = q.popleft()
            base = d[r * GRID + c]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID and 0 <= nc < GRID and walkable[nr, nc]:
                    idx = nr * GRID + nc
                    if d[idx] < 0:
                        d[idx] = base + 1
                        q.append((nr, nc))
    return dist


@dataclass(frozen=True)
class Layout:
    walkable: np.ndarray
    dist: np.ndarray
    free_cells: np.ndarray
    room_of: np.ndarray

    def d(self, a: int, b: int) -> int:
        return int(self.dist[a, b])


def _room_id(r: int, c: int) -> int:
    mid = GRID // 2
    if r == mid or c == mid:
        return -1
    return (0 if r < mid else 2) + (0 if c < mid else 1)


def _make_layouts() -> List[Layout]:
    out = []
    for s in range(N_LAYOUTS):
        w = _build_layout(s)
        dist = _all_pairs_bfs(w)
        free = np.array([r * GRID + c for r in range(GRID) for c in range(GRID) if w[r, c]],
                        dtype=np.int32)
        room = np.array([_room_id(*divmod(int(i), GRID)) for i in range(GRID * GRID)],
                        dtype=np.int32)
        out.append(Layout(w, dist, free, room))
    return out


LAYOUTS: List[Layout] = _make_layouts()
MAX_DIST = max(int(l.dist[l.dist >= 0].max()) for l in LAYOUTS)


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
@dataclass
class Task:
    """One episode's world state and referring expression."""
    layout: Layout
    k: int
    cand_cells: np.ndarray        # (k,) flat cell indices
    cand_colors: np.ndarray       # (k,) indices into COLORS
    noun: str
    receptacle: str
    recep_cell: int
    start_cell: int
    true_idx: int
    similarity: np.ndarray        # (k, k) semantic confusability

    @property
    def instruction(self) -> str:
        """The *underspecified* command the human actually utters."""
        return f"put the {self.noun} on the {self.receptacle}"

    @property
    def disambiguated(self) -> str:
        """What they should have said. Used for the oracle baseline and demos."""
        c = COLORS[self.cand_colors[self.true_idx]]
        return f"put the {c} {self.noun} on the {self.receptacle}"


def sample_task(rng: np.random.Generator, k_min: int = 2, k_max: int = 4) -> Task:
    """Generate one episode. Candidates are spread across rooms so that
    committing to the wrong referent costs real travel."""
    layout = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    k = int(rng.integers(k_min, k_max + 1))

    rooms = list(rng.permutation(4))[:k]
    cells = []
    for room in rooms:
        pool = [int(i) for i in layout.free_cells if layout.room_of[i] == room]
        cells.append(int(rng.choice(pool)))
    cand_cells = np.array(cells, dtype=np.int32)

    colors = rng.choice(len(COLORS), size=k, replace=False)
    noun = str(rng.choice(NOUNS))
    receptacle = str(rng.choice(RECEPTACLES))

    others = [int(i) for i in layout.free_cells if int(i) not in set(cells)]
    recep_cell = int(rng.choice(others))
    start_pool = [i for i in others if i != recep_cell]
    start_cell = int(rng.choice(start_pool))

    sim = COLOR_SIM[np.ix_(colors, colors)].copy()
    return Task(layout=layout, k=k, cand_cells=cand_cells, cand_colors=colors,
                noun=noun, receptacle=receptacle, recep_cell=recep_cell,
                start_cell=start_cell, true_idx=int(rng.integers(k)),
                similarity=sim)
