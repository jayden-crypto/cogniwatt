"""Scenario schema for the referring-expression grounding set (R2 Deliverable A).

The vocabulary here is *mirrored* from `cogniwatt/env/tasks.py` and asserted
equal at import time, so a change on R1's side breaks loudly here instead of
silently decorrelating the dataset from the environment.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

# --- mirrored from cogniwatt/env/tasks.py -------------------------------------
NOUNS: Tuple[str, ...] = ("mug", "book", "bottle", "bowl", "remote", "cushion")
COLORS: Tuple[str, ...] = ("red", "blue", "green", "white", "black", "striped")
RECEPTACLES: Tuple[str, ...] = ("shelf", "sink", "table", "drawer", "bin")
K_VALUES: Tuple[int, ...] = (2, 3, 4)

#: The env's 4-room floorplan is unnamed (room ids 0-3). We name them here for
#: scene descriptions only; R1's generator assigns cells independently, so the
#: room field is language context, not a spatial constraint.
ROOMS: Tuple[str, ...] = ("kitchen", "living_room", "bedroom", "hallway")

#: The commonsense inference each scenario turns on. Recorded per scenario so
#: accuracy and the failure taxonomy (Deliverable F) can be reported per family
#: -- "the 0.5B model fails at pragmatics, not perception" needs this column.
PRAGMATIC_FAMILIES: Tuple[str, ...] = (
    "already_satisfied",   # the other candidates are already where they belong
    "affordance",          # only one candidate can serve the stated purpose
    "state_blocks",        # one candidate is in use / full / dirty / occupied
    "ownership",           # social convention picks the referent out
    "recency",             # the one just used / just put down
    "proximity",           # the one near the speaker or the stated destination
    "condition",           # broken, empty, expired -> the one to deal with
    "purpose_fit",         # matches the *stated* destination's function
    "exclusion",           # everything else is explicitly ruled out
    "salience",            # the odd one out in an otherwise uniform set
)


@dataclass
class Candidate:
    color: str
    desc: str
    room: str

    def validate(self) -> None:
        assert self.color in COLORS, f"color {self.color!r} not in env COLORS"
        assert self.room in ROOMS, f"room {self.room!r} not in ROOMS"
        assert self.desc.strip(), "empty description"


@dataclass
class Scenario:
    id: str
    k: int
    noun: str
    receptacle: str
    instruction: str
    candidates: List[Candidate]
    true_idx: int
    rationale: str
    difficulty: str = "medium"          # easy | medium | hard
    family: str = "already_satisfied"
    audit: str = "unaudited"            # unaudited | agree | disagree
    #: Which authored core this was expanded from. Scenarios sharing a core_id
    #: are colour/position/phrasing permutations of one commonsense contrast and
    #: are therefore NOT independent -- confidence intervals must cluster on it.
    core_id: str = ""
    #: Index into the core's instruction forms, so a phrasing effect is testable.
    form_idx: int = 0

    def to_json(self) -> Dict[str, object]:
        d = asdict(self)
        d["candidates"] = [asdict(c) if not isinstance(c, dict) else c
                           for c in self.candidates]
        return d

    @staticmethod
    def from_json(d: Dict[str, object]) -> "Scenario":
        d = dict(d)
        d["candidates"] = [Candidate(**c) for c in d["candidates"]]  # type: ignore[arg-type]
        return Scenario(**d)  # type: ignore[arg-type]


def validate_scenario(s: Scenario) -> None:
    """Raise AssertionError on anything that would corrupt a measurement."""
    assert s.k in K_VALUES, f"{s.id}: k={s.k} not in {K_VALUES}"
    assert len(s.candidates) == s.k, f"{s.id}: {len(s.candidates)} candidates, k={s.k}"
    assert s.noun in NOUNS, f"{s.id}: noun {s.noun!r} not in env NOUNS"
    assert s.receptacle in RECEPTACLES, f"{s.id}: receptacle {s.receptacle!r} unknown"
    assert 0 <= s.true_idx < s.k, f"{s.id}: true_idx {s.true_idx} out of range"
    assert s.difficulty in ("easy", "medium", "hard"), f"{s.id}: bad difficulty"
    assert s.family in PRAGMATIC_FAMILIES, f"{s.id}: unknown family {s.family!r}"
    assert s.rationale.strip(), f"{s.id}: rationale is required (it is the audit trail)"
    for c in s.candidates:
        c.validate()
    colors = [c.color for c in s.candidates]
    # The env draws candidate colors without replacement; the dataset must match
    # or the similarity sub-matrix R1 slices out is not the one we measured.
    assert len(set(colors)) == s.k, f"{s.id}: duplicate colors {colors}"
    # The instruction must stay underspecified -- naming the colour leaks the answer.
    lowered = s.instruction.lower()
    for c in s.candidates:
        assert c.color not in lowered, (
            f"{s.id}: instruction names the colour {c.color!r}; that is the "
            f"disambiguated form, not the underspecified one")
    assert s.noun in lowered, f"{s.id}: instruction must mention the noun {s.noun!r}"


def save_scenarios(path: str, scenarios: Iterable[Scenario]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(s.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_scenarios(path: str) -> List[Scenario]:
    out: List[Scenario] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Scenario.from_json(json.loads(line)))
    return out
