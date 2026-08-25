"""The one prompt template, the one parser (R2 Deliverable B).

Every tier -- 0.5B local, 3B local, cloud -- sees byte-identical prompt text.
Any per-tier prompt tuning would make the accuracy comparison meaningless
(docs/R2_HANDOFF.md §5, §11), so there is deliberately no `tier` argument here.

Candidate order is shuffled per query and the permutation returned, so position
bias is measurable rather than silently baked into the accuracy numbers.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: Options are labelled A, B, C, D -- NOT 0, 1, 2, 3 as docs/R2_HANDOFF.md §5
#: suggests. This was changed on evidence, and the evidence is worth recording
#: because the digit form silently destroys the measurement:
#:
#:   Qwen2.5-0.5B-Q8_0, temperature 0.0, 240 scenarios, one prompt per scheme
#:     0-based digits : accuracy 0.358, answered "1" on 237/240
#:     1-based digits : accuracy 0.375, answered "1" on 168/240
#:     letters        : accuracy 0.483, answers spread A 107 / B 87 / C 35 / D 11
#:
#: Under 0-based labels "1" is the second option; under 1-based labels "1" is
#: the first. The model emitted the token "1" at the same rate either way, so it
#: was not choosing an option at all -- it has a prior on the *digit*. Any
#: digit-labelled multiple choice therefore measures that prior rather than
#: grounding ability, and it does so while looking perfectly well-formed: the
#: parse rate was 99.8%.
#:
#: This is not per-tier prompt tuning, which would invalidate the cross-tier
#: comparison (§5, §11). One template is still used for every tier; it is the
#: template itself that was wrong.
PROMPT_TEMPLATE = (
    'You are a home robot. The person said: "{instruction}"\n'
    "Objects that could match:\n"
    "{options}\n"
    "Which one did they mean? Answer with the letter only."
)

#: Labels in display order. k <= 4 throughout, so four suffice.
LABELS = ("A", "B", "C", "D")

#: JSON schema for the cloud tier's structured output. Keeps the cloud tier on
#: the same parse path as the local ones -- it just never emits prose.
CHOICE_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": "string", "enum": ["A", "B", "C", "D"]}},
    "required": ["choice"],
    "additionalProperties": False,
}


def stable_seed(*parts: object) -> int:
    """A seed that is identical across processes and runs.

    Python's built-in `hash()` is randomised per process for `str` (PEP 456),
    so `hash(("sc_0001", 0))` returns a different value every invocation. Using
    it to seed the candidate shuffle silently broke the guarantee that every
    tier sees the *same* permutation: it held within one process, but tiers
    measured in separate runs -- which is exactly how a 4 GB card forces you to
    work, one model resident at a time -- each drew a different shuffle. The
    cross-tier accuracy comparison then stops being paired, and a tier that
    happened to draw easier positions looks better at grounding.

    blake2b is deterministic across processes, machines and Python versions.
    """
    h = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2 ** 32)


def shuffled_view(descs: Sequence[str], true_idx: int,
                  rng: np.random.Generator) -> Tuple[List[str], List[int], int]:
    """Shuffle candidates for one query.

    Returns `(shown, perm, shown_true_idx)` where `shown[j] == descs[perm[j]]`,
    so a parsed display index `j` maps back to the dataset index via `perm[j]`.
    """
    perm = [int(i) for i in rng.permutation(len(descs))]
    shown = [descs[i] for i in perm]
    shown_true_idx = perm.index(int(true_idx))
    return shown, perm, shown_true_idx


def build_prompt(instruction: str, shown: Sequence[str]) -> str:
    options = "\n".join(f"{LABELS[j]}. {d}" for j, d in enumerate(shown))
    return PROMPT_TEMPLATE.format(instruction=instruction, options=options)


# Ordered most- to least-specific. Each yields a display index.
_LETTER_PATTERNS = (
    re.compile(r'"choice"\s*:\s*"?([A-D])"?', re.I),     # structured output
    re.compile(r'^\s*\(?([A-D])\)?\s*$', re.I),          # bare "B" / "(B)"
    re.compile(r'^\s*\(?([A-D])\)?\s*[.):\-]', re.I),    # "B." / "B)" / "B:"
    re.compile(r'\b(?:answer|choice|option|letter)\b\W{0,12}?\(?([A-D])\)?\b', re.I),
    re.compile(r'\b([A-D])\b'),                          # a lone capital letter
)
#: Structured output may still deliver an integer index (the cloud tier's JSON
#: schema uses one), so an explicit `"choice": 2` stays parseable.
_JSON_INT = re.compile(r'"choice"\s*:\s*(\d+)')


def parse_choice(text: str, k: int) -> Tuple[Optional[int], str]:
    """Parse a model response into a display index.

    Returns `(index, status)`. `status` feeds the failure taxonomy directly:

    * ``ok``            -- a clean in-range choice
    * ``out_of_range``  -- a valid label, but not one of this scenario's k
    * ``digit_answer``  -- replied with a bare digit under letter labelling
    * ``unparseable``   -- nothing recoverable (hedging, refusal, prose)

    A failed parse is *not* silently converted into a wrong answer; callers
    record it so format violations stay distinguishable from grounding errors.

    Bare digits are deliberately **not** coerced to an index. Under letter
    labelling "1" is ambiguous -- it could mean the first option or a leftover
    0-indexed habit -- and the model that motivated this change emits "1"
    reflexively. Guessing would silently manufacture answers out of exactly the
    degenerate behaviour we are trying to detect, so it is recorded as its own
    failure class instead.
    """
    if text is None:
        return None, "unparseable"
    s = str(text).strip()

    m = _JSON_INT.search(s)
    if m:
        v = int(m.group(1))
        return (v, "ok") if 0 <= v < k else (v, "out_of_range")

    for pat in _LETTER_PATTERNS:
        m = pat.search(s)
        if m:
            v = ord(m.group(1).upper()) - 65
            return (v, "ok") if 0 <= v < k else (v, "out_of_range")

    if re.search(r'\d', s):
        return None, "digit_answer"
    return None, "unparseable"
