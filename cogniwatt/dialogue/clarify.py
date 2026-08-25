"""Clarification-question generation for the `ask` action (R2 Deliverable E).

In the env, `ask` costs reward and returns an abstract observation -- it never
generates language. For the demo and the qualitative results it has to produce
a real question: *"Do you mean the one on the counter, or the one in the
cupboard?"*

WHAT IS BEING OPTIMISED
-----------------------
A question partitions the candidate set. Asking it and getting a truthful
answer moves the belief; the best question is the one that moves it most, i.e.
maximises expected information gain

    IG(S) = H(b) - E_answer[ H(b | answer) ]

For a yes/no question "is the referent in S?" answered with error rate epsilon,
IG is maximised by the split that comes closest to halving the *belief mass* --
not the candidate count. Splitting 3-vs-1 candidates is the right question when
that 1 holds half the probability.

GROUNDEDNESS IS STRUCTURAL, NOT HOPED FOR
-----------------------------------------
The handoff asks for a hallucination rate of 0%. Rather than generating text
and checking it afterwards, questions here are *assembled from spans taken
verbatim out of the candidate descriptions*. There is no free-text generation
step, so there is no mechanism by which an object that is not in the scene
could appear in a question. `groundedness` re-verifies this on the produced
string anyway, because a structural guarantee that is never checked has a habit
of quietly becoming untrue.

This is the entropy-maximising arm of the Deliverable-E comparison. The other
arm -- asking an SLM to write the question -- goes through
`cogniwatt/dialogue/backends.py` and is measured against this one on the same
information-gain metric.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Leading article + colour + noun, e.g. "a red mug on the counter" -> the span
#: "on the counter". The tail is what actually distinguishes one candidate from
#: another; the colour is an arbitrary label the human may not share.
_HEAD = re.compile(r"^a\s+\w+\s+\w+\s*,?\s*", re.I)


def distinguishing_span(desc: str) -> str:
    """The part of a description that identifies it, minus colour and noun.

    Colour is deliberately dropped. In this environment colour is an arbitrary
    label, and a question that says "do you mean the red one?" would resolve
    the referent without the human doing any of the disambiguating work -- which
    is not the channel being modelled.
    """
    tail = _HEAD.sub("", desc.strip()).strip()
    return tail or desc.strip()


@dataclass
class Question:
    text: str
    #: Candidate indices the question groups together (the "yes" side).
    subset: Tuple[int, ...]
    expected_ig: float
    kind: str  # "alternative" | "polar"

    def answer_partition(self, true_idx: int) -> bool:
        return true_idx in self.subset


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def posterior(belief: np.ndarray, subset: Sequence[int], said_yes: bool,
              epsilon: float) -> np.ndarray:
    """Belief after a yes/no answer with error rate `epsilon`.

    `epsilon` folds together the human misspeaking and ASR mistranscribing --
    it is where `DEFAULT_ASR_WER` enters the `ask` channel.
    """
    b = np.asarray(belief, dtype=np.float64)
    inside = np.zeros_like(b, dtype=bool)
    inside[list(subset)] = True
    # P(hear "yes" | referent inside) = 1 - epsilon; outside = epsilon.
    like = np.where(inside, 1.0 - epsilon, epsilon)
    if not said_yes:
        like = 1.0 - like
    post = b * like
    s = post.sum()
    return post / s if s > 0 else b.copy()


def expected_information_gain(belief: np.ndarray, subset: Sequence[int],
                              epsilon: float) -> float:
    b = np.asarray(belief, dtype=np.float64)
    inside = np.zeros(len(b), dtype=bool)
    inside[list(subset)] = True
    mass = float(b[inside].sum())
    p_yes = mass * (1 - epsilon) + (1 - mass) * epsilon
    h_yes = entropy(posterior(b, subset, True, epsilon))
    h_no = entropy(posterior(b, subset, False, epsilon))
    return entropy(b) - (p_yes * h_yes + (1 - p_yes) * h_no)


# ---------------------------------------------------------------------------
def _phrase(span: str) -> str:
    """Turn a description span into a referring phrase: 'the one <span>'."""
    span = span.strip().rstrip(".")
    if span.startswith(("on ", "in ", "under ", "beside ", "by ", "behind ",
                        "at ", "with ", "from ", "near ", "inside ")):
        return f"the one {span}"
    return f"the one that is {span}" if not span.startswith("that ") \
        else f"the one {span}"


def build_question(descs: Sequence[str], subset: Sequence[int],
                   ig: float) -> Question:
    """Assemble the surface form from verbatim description spans."""
    others = [j for j in range(len(descs)) if j not in subset]
    spans = [distinguishing_span(descs[i]) for i in subset]

    # A two-way split reads far more naturally as an alternative question, and
    # it is also the one a person can answer without a yes/no ambiguity.
    if len(subset) == 1 and len(others) == 1:
        a = _phrase(distinguishing_span(descs[subset[0]]))
        b = _phrase(distinguishing_span(descs[others[0]]))
        return Question(f"Do you mean {a}, or {b}?", tuple(subset), ig,
                        "alternative")

    if len(subset) == 1:
        return Question(f"Do you mean {_phrase(spans[0])}?", tuple(subset), ig,
                        "polar")

    # A multi-element subset is a *set-membership* question, and "Do you mean A,
    # or B?" is the wrong surface form for it -- that reads as an alternative
    # between A and B, so a "yes" would be meaningless. The scoring in
    # `posterior` has always treated it as membership; only the wording was
    # ambiguous. "Is it one of these" makes yes/no well defined.
    joined = ", or ".join(_phrase(s) for s in spans)
    return Question(f"Is it one of these: {joined}?", tuple(subset), ig, "polar")


def best_question(descs: Sequence[str], belief: Optional[np.ndarray] = None,
                  epsilon: float = 0.08,
                  max_subset: Optional[int] = None) -> Question:
    """The grounded question with the highest expected information gain.

    Searches every non-trivial subset. k <= 4 here, so that is at most 7
    partitions -- exhaustive search is cheap and exact, and this never runs on
    the PPO hot path.
    """
    k = len(descs)
    if k < 2:
        raise ValueError("need at least two candidates to disambiguate")
    b = np.full(k, 1.0 / k) if belief is None else np.asarray(belief, float)
    b = b / b.sum()

    cap = max_subset or k - 1
    best: Optional[Question] = None
    seen = set()
    for size in range(1, min(cap, k - 1) + 1):
        for subset in combinations(range(k), size):
            # A subset and its complement are the same question asked the other
            # way round; score each partition once.
            comp = tuple(j for j in range(k) if j not in subset)
            key = frozenset((subset, comp))
            if key in seen:
                continue
            seen.add(key)
            ig = expected_information_gain(b, subset, epsilon)
            if best is None or ig > best.expected_ig:
                best = build_question(descs, subset, ig)
    assert best is not None
    return best


def random_question(descs: Sequence[str], rng: np.random.Generator,
                    epsilon: float = 0.08) -> Question:
    """Baseline: a grounded question chosen at random.

    The comparison that matters for Deliverable E is not "grounded vs
    hallucinated" -- it is "does *choosing* the split actually buy anything over
    asking about an arbitrary candidate". Without this baseline, a good IG
    number could just be an artefact of small k.
    """
    k = len(descs)
    i = int(rng.integers(0, k))
    b = np.full(k, 1.0 / k)
    return build_question(descs, (i,), expected_information_gain(b, (i,), epsilon))


# ---------------------------------------------------------------------------
def scene_vocabulary(descs: Sequence[str]) -> set:
    words = set()
    for d in descs:
        words |= set(re.findall(r"[a-z']+", d.lower()))
    return words

#: Function words and the fixed question frame -- these are not scene content,
#: so their presence is not a hallucination.
_FRAME = set("""do you mean the one that is or a an of in on at by with from
under beside behind near inside and is are it these""".split())


def polar_ceiling(epsilon: float) -> float:
    """Most bits a single yes/no answer can carry, given error rate `epsilon`.

    This is 1 - H(epsilon), and it is the reason `ask` cannot be a one-shot
    solution at k=4: two candidates hold 1 bit of uncertainty, four hold 2, and
    no single polar question can deliver 2 bits. At the placeholder ASR error of
    0.08 the ceiling is ~0.60 bits, so a k=4 scene needs at least four questions
    to be resolved by asking alone -- which is precisely the pressure that makes
    the agent's spend/ask trade-off non-trivial rather than "always ask".
    """
    e = float(np.clip(epsilon, 1e-9, 1 - 1e-9))
    return 1.0 - float(-(e * np.log2(e) + (1 - e) * np.log2(1 - e)))


def groundedness(question: str, descs: Sequence[str]) -> Tuple[bool, List[str]]:
    """Is every content word in the question traceable to the scene?

    Returns (ok, offending_words). Because questions are assembled from verbatim
    spans this should never fail -- which is exactly why it is worth asserting:
    it turns a structural claim into a checked one, and it is the metric the
    handoff asks to be reported as 0%.
    """
    vocab = scene_vocabulary(descs) | _FRAME
    bad = [w for w in re.findall(r"[a-z']+", question.lower()) if w not in vocab]
    return (not bad), bad


def evaluate(descs: Sequence[str], true_idx: int, belief: Optional[np.ndarray] = None,
             epsilon: float = 0.08) -> Dict[str, object]:
    """One scenario's worth of Deliverable-E metrics."""
    k = len(descs)
    b = np.full(k, 1.0 / k) if belief is None else np.asarray(belief, float)
    b = b / b.sum()
    q = best_question(descs, b, epsilon)
    said_yes = q.answer_partition(true_idx)
    post = posterior(b, q.subset, said_yes, epsilon)
    ok, bad = groundedness(q.text, descs)
    return {
        "question": q.text, "kind": q.kind, "subset": list(q.subset),
        "expected_ig": q.expected_ig,
        "realised_ig": entropy(b) - entropy(post),
        "entropy_before": entropy(b), "entropy_after": entropy(post),
        "posterior_on_truth": float(post[true_idx]),
        "resolves": bool(post.argmax() == true_idx),
        "grounded": ok, "ungrounded_words": bad,
    }
