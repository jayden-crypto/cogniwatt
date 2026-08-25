"""R2 conversational-AI workstream: scenarios, model tiers, measurement.

Nothing in this package may be imported from `cogniwatt/env/**` or
`cogniwatt/agents/**`. Model backends live behind `backends.py` and are only
touched by `scripts/r2_*.py`; the training loop reads frozen numbers from
`data/dialogue_measured.npz` instead. See docs/R2_HANDOFF.md §1.1.
"""
from .schema import (COLORS, NOUNS, RECEPTACLES, ROOMS, K_VALUES, PRAGMATIC_FAMILIES,
                     Candidate, Scenario, load_scenarios, save_scenarios, validate_scenario)
from .prompts import PROMPT_TEMPLATE, build_prompt, parse_choice, shuffled_view

__all__ = ["COLORS", "NOUNS", "RECEPTACLES", "ROOMS", "K_VALUES", "PRAGMATIC_FAMILIES",
           "Candidate", "Scenario", "load_scenarios", "save_scenarios",
           "validate_scenario", "PROMPT_TEMPLATE", "build_prompt", "parse_choice",
           "shuffled_view"]
