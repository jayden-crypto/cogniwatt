"""Hand-authored referring-expression cores (R2 Deliverable A, source of truth).

WHY CORES RATHER THAN SCENARIOS
-------------------------------
A `Core` is one commonsense contrast: a *target* description plus a pool of
distractor descriptions, every one of which is individually a defensible
*non*-answer for the same instruction. `scripts/r2_build_dataset.py` expands a
core into scenarios by choosing k-1 distractors, assigning distinct colours and
shuffling positions.

This buys three things the flat-scenario format could not:

1. **Auditability.** The audit unit is the core, not the scenario. Auditing 60
   cores certifies all generated scenarios, because every generated scenario is
   a colour/position/phrasing permutation of an audited contrast.
2. **k-comparability.** The k=2, k=3 and k=4 forms of a core share a target and
   a rationale, so "accuracy falls with k" is measured on the *same* inference
   rather than on three separately-authored sets of unknown relative difficulty.
3. **Clustered statistics.** Scenarios from one core are not independent. Every
   generated scenario carries `core_id`, so confidence intervals can be
   clustered on it instead of over-counting n.

AUTHORING RULES (enforced by `validate_cores` and by `validate_scenario`)
-------------------------------------------------------------------------
1. The answer is determined by *commonsense inference*, never by the colour.
   Colours are arbitrary labels in the env; a scenario solvable from the colour
   alone would measure the wrong thing. Descriptions therefore never mention a
   colour -- the generator splices one in.
2. Every description begins literally with ``a {noun}`` so the colour splice is
   mechanical and cannot change relative description length.
3. Descriptions are kept to comparable length. If the correct candidate were
   reliably the longest or most detailed, we would be measuring a length prior.
4. The instruction stays underspecified -- it never names the colour.
5. Every core carries a `rationale`: the one-line argument for its answer. If a
   human auditor cannot accept the rationale, the core is dropped and with it
   every scenario generated from it.

COVERAGE
--------
60 cores = 6 nouns x 10 pragmatic families, one core per cell. The noun x family
grid is therefore exactly balanced by construction, which is what lets the
failure taxonomy (Deliverable F) say "the 0.5B model fails at pragmatics, not
perception" without confounding noun with family.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .schema import COLORS, NOUNS, PRAGMATIC_FAMILIES, RECEPTACLES, ROOMS

#: Preposition per receptacle, so generated instructions are grammatical.
PREP: Dict[str, str] = {"shelf": "on", "table": "on",
                        "sink": "in", "drawer": "in", "bin": "in"}

#: Instruction phrasings. All are underspecified and all mention the noun.
#: Varying them stops any tier from overfitting one surface form, and the form
#: index is recorded per scenario so a phrasing effect stays visible.
INSTRUCTION_FORMS: Tuple[str, ...] = (
    "put the {noun} {prep} the {receptacle}",
    "could you put the {noun} {prep} the {receptacle}",
    "the {noun} needs to go {prep} the {receptacle}",
)
#: Only valid where the destination is not load-bearing for the inference.
AWAY_FORM = "put the {noun} away"


@dataclass(frozen=True)
class Core:
    id: str
    family: str
    noun: str
    receptacle: str
    #: (description, room) -- the correct referent.
    target: Tuple[str, str]
    #: >= 3 (description, room) pairs, each a defensible non-answer on its own.
    distractors: Tuple[Tuple[str, str], ...]
    difficulty: str
    rationale: str
    #: True when "put the {noun} away" is also a fair phrasing for this core.
    away_ok: bool = False

    def forms(self) -> Tuple[str, ...]:
        return INSTRUCTION_FORMS + ((AWAY_FORM,) if self.away_ok else ())

    def instruction(self, form: str) -> str:
        return form.format(noun=self.noun, prep=PREP[self.receptacle],
                           receptacle=self.receptacle)


def _c(cid, family, noun, recept, target, distractors, difficulty, rationale,
       away_ok=False) -> Core:
    return Core(cid, family, noun, recept, target, tuple(distractors),
                difficulty, rationale, away_ok)


CORES: List[Core] = [
    # ---------------------------------------------------------------- mug ---
    _c("mug_already", "already_satisfied", "mug", "shelf",
       ("a mug on the kitchen counter, half full of coffee", "kitchen"),
       [("a mug already sitting inside the cupboard", "kitchen"),
        ("a mug lined up with the others on the shelf", "kitchen"),
        ("a mug hanging on its hook above the counter", "kitchen")],
       "easy", "the others are already put away", away_ok=True),
    _c("mug_afford", "affordance", "mug", "drawer",
       ("a mug, squat and handleless, that clears the drawer lip", "kitchen"),
       [("a mug, tall, with a wide flared rim", "kitchen"),
        ("a mug with a broad saucer fused to its base", "kitchen"),
        ("a mug in a padded travel sleeve twice its width", "kitchen")],
       "medium", "only the squat one will actually fit in a drawer"),
    _c("mug_state", "state_blocks", "mug", "sink",
       ("a mug pushed aside, drained, a dried ring inside", "kitchen"),
       [("a mug someone is still drinking from at the table", "kitchen"),
        ("a mug steaming, just poured, on the desk", "bedroom"),
        ("a mug with a teabag brewing under a saucer", "kitchen")],
       "medium", "you clear the finished one, not one still in use"),
    _c("mug_own", "ownership", "mug", "shelf",
       ("a mug from the shared kitchen cupboard", "kitchen"),
       # "colleague" is office-coded and jars with a household scene; a
       # housemate's named mug carries the same ownership inference in-setting.
       [("a mug with a housemate's name painted across it", "kitchen"),
        ("a mug the guest brought and is taking home", "living_room"),
        ("a mug kept on the child's own tray", "kitchen")],
       "medium", "you put away the household one, not someone's personal mug",
       away_ok=True),
    _c("mug_recency", "recency", "mug", "sink",
       ("a mug still warm, set down a moment ago", "living_room"),
       [("a mug that has stood on the shelf all week", "kitchen"),
        ("a mug unpacked this morning and never filled", "kitchen"),
        ("a mug kept in the cabinet for visitors", "living_room")],
       "medium", "the one just used is the one to clear"),
    _c("mug_prox", "proximity", "mug", "table",
       ("a mug on the chair pulled up to the table", "kitchen"),
       [("a mug two rooms away on the bedroom windowsill", "bedroom"),
        ("a mug at the far end of the hallway shelf", "hallway"),
        ("a mug left by the sofa in the next room", "living_room")],
       "medium", "the nearest one is meant when a destination is named"),
    _c("mug_cond", "condition", "mug", "bin",
       ("a mug with a clean break through the handle", "kitchen"),
       [("a mug chipped slightly at the rim but usable", "kitchen"),
        ("a mug stained inside but perfectly sound", "kitchen"),
        ("a mug with a faded print and no damage", "kitchen")],
       "medium", "the broken one is the one to throw out"),
    _c("mug_purpose", "purpose_fit", "mug", "shelf",
       ("a mug that came with the dinner set", "kitchen"),
       [("a mug holding pens on the desk", "bedroom"),
        ("a mug with a seedling growing in it", "hallway"),
        ("a mug used to scoop dry cat food", "kitchen")],
       "hard", "only the one still used as crockery belongs with the crockery"),
    _c("mug_excl", "exclusion", "mug", "drawer",
       ("a mug from the everyday set, in the rack", "kitchen"),
       [("a mug the landlord asked to be left out", "kitchen"),
        ("a mug kept out for the morning routine", "kitchen"),
        ("a mug set aside for the charity box", "hallway")],
       "hard", "every other mug has been kept out for a stated reason"),
    _c("mug_salience", "salience", "mug", "shelf",
       ("a mug matching the four already lined up on the shelf", "kitchen"),
       [("a mug from the mismatched camping set", "hallway"),
        ("a mug belonging to the espresso tray", "kitchen"),
        ("a mug that lives on the bedside table", "bedroom")],
       "medium", "the one that completes the set already on the shelf"),

    # --------------------------------------------------------------- book ---
    _c("book_already", "already_satisfied", "book", "shelf",
       # Not "lying open": an open book reads as *in use*, which is the
       # state_blocks inference and would compete with this core's own one.
       ("a book left face-down on the arm of the sofa", "living_room"),
       [("a book already standing upright on the shelf", "living_room"),
        ("a book slotted into the bookcase spine-out", "bedroom"),
        ("a book filed in the shelving unit with the atlases", "living_room")],
       "easy", "the others are already shelved", away_ok=True),
    _c("book_afford", "affordance", "book", "sink",
       ("a book of laminated recipe cards, splashed with sauce", "kitchen"),
       [("a book with a soft paper cover", "living_room"),
        ("a book with gilt edges and a cloth spine", "bedroom"),
        ("a book of loose photographs in sleeves", "living_room")],
       "hard", "only the wipeable one can be taken to the sink"),
    _c("book_state", "state_blocks", "book", "drawer",
       ("a book finished last month, the bookmark taken out", "bedroom"),
       [("a book someone is reading right now on the sofa", "living_room"),
        ("a book held open with notes being copied out", "bedroom"),
        ("a book propped on the stand for tonight's recipe", "kitchen")],
       "medium", "you do not put away a book still being used"),
    _c("book_own", "ownership", "book", "shelf",
       ("a book that belongs to the house", "living_room"),
       [("a book with a library return slip tucked in", "hallway"),
        ("a book borrowed from a neighbour, to go back", "hallway"),
        ("a book the visitor left with a train ticket in it", "living_room")],
       "hard", "the ones headed elsewhere are staged for return, not shelving"),
    _c("book_recency", "recency", "book", "table",
       ("a book set down mid-chapter a minute ago", "living_room"),
       [("a book that has sat in the hall for months", "hallway"),
        ("a book still in its delivery wrapper", "hallway"),
        ("a book stored flat under the bed", "bedroom")],
       "medium", "the one being read is the one to keep within reach"),
    _c("book_prox", "proximity", "book", "drawer",
       ("a book resting on top of the chest of drawers", "bedroom"),
       [("a book two rooms away on the kitchen counter", "kitchen"),
        ("a book on the hallway windowsill by the door", "hallway"),
        ("a book down the side of the sofa", "living_room")],
       "medium", "the one already at the destination is meant"),
    _c("book_cond", "condition", "book", "bin",
       ("a book water-damaged and swollen shut", "hallway"),
       [("a book with its spine intact, borrowed from the library", "bedroom"),
        ("a book with a creased cover but readable", "living_room"),
        ("a book missing only its dust jacket", "living_room")],
       "medium", "the ruined one gets binned; the rest are still readable"),
    _c("book_purpose", "purpose_fit", "book", "table",
       ("a book of takeaway menus", "kitchen"),
       [("a book of pressed flowers, fragile", "bedroom"),
        ("a book of tax records in a ring binder", "bedroom"),
        ("a book of baby photographs", "living_room")],
       "medium", "the menus are what the table is being laid for"),
    _c("book_excl", "exclusion", "book", "bin",
       ("a book of expired coupons from two years ago", "kitchen"),
       [("a book signed by its author", "living_room"),
        ("a book still under a reading-group deadline", "living_room"),
        ("a book that came with the flat's inventory", "hallway")],
       "hard", "every other book has a stated reason to be kept"),
    _c("book_salience", "salience", "book", "shelf",
       ("a book matching the numbered series already on the shelf", "living_room"),
       [("a book from the pile of magazines", "living_room"),
        ("a book of children's stories kept in the nursery", "bedroom"),
        ("a book of road maps kept in the car", "hallway")],
       "medium", "the one that completes the run already shelved"),

    # ------------------------------------------------------------- bottle ---
    _c("bottle_already", "already_satisfied", "bottle", "bin",
       ("a bottle, drained, standing on the desk", "bedroom"),
       [("a bottle already dropped in the recycling", "kitchen"),
        ("a bottle sitting in the bin bag by the door", "hallway"),
        ("a bottle crushed flat in the waste caddy", "kitchen")],
       "easy", "the others have already been binned"),
    _c("bottle_afford", "affordance", "bottle", "shelf",
       ("a bottle of oil that keeps at room temperature", "kitchen"),
       [("a bottle that must be refrigerated once opened", "kitchen"),
        ("a bottle of milk already sweating in the warm", "kitchen"),
        ("a bottle of yeast starter that has to stay cold", "kitchen")],
       "hard", "only the shelf-stable one can go on a shelf"),
    _c("bottle_state", "state_blocks", "bottle", "sink",
       ("a bottle emptied, crusted with dried smoothie", "kitchen"),
       [("a bottle someone is drinking from on the sofa", "living_room"),
        ("a bottle filled and capped for the school run", "hallway"),
        ("a bottle half poured into a jug and still needed", "kitchen")],
       "easy", "only the finished one is ready for washing"),
    _c("bottle_own", "ownership", "bottle", "drawer",
       ("a bottle from the household spares", "kitchen"),
       [("a bottle of someone's prescription drops", "bedroom"),
        ("a bottle the guest brought as a gift", "living_room"),
        ("a bottle labelled for the baby's feeds", "kitchen")],
       "medium", "the household one goes to storage; the rest belong to a person"),
    _c("bottle_recency", "recency", "bottle", "bin",
       ("a bottle finished a moment ago, cap on the side", "living_room"),
       [("a bottle unopened at the back of the cupboard", "kitchen"),
        ("a bottle still cold from the fridge", "kitchen"),
        ("a bottle kept refilled for the plants", "hallway")],
       "medium", "the one just emptied is the one to throw out"),
    _c("bottle_prox", "proximity", "bottle", "sink",
       ("a bottle standing on the draining board beside the tap", "kitchen"),
       [("a bottle on the bedroom windowsill upstairs", "bedroom"),
        ("a bottle at the far end of the hall", "hallway"),
        ("a bottle behind the sofa in the next room", "living_room")],
       "medium", "the one already at the sink is meant"),
    _c("bottle_cond", "condition", "bottle", "sink",
       ("a bottle with sour milk congealed in the neck", "kitchen"),
       [("a bottle rinsed clean and drying upside down", "kitchen"),
        ("a bottle sealed and untouched in its wrapper", "kitchen"),
        ("a bottle wiped down and back in the fridge door", "kitchen")],
       "medium", "the spoiled one is what the sink is for"),
    _c("bottle_purpose", "purpose_fit", "bottle", "table",
       ("a bottle of sparkling water, chilled", "kitchen"),
       [("a bottle of washing-up liquid by the tap", "kitchen"),
        ("a bottle of turpentine from the cupboard", "hallway"),
        ("a bottle of engine oil from the garage shelf", "hallway")],
       "easy", "the dining table gets the drink, not the chemicals"),
    _c("bottle_excl", "exclusion", "bottle", "shelf",
       ("a bottle of cordial from the back of the cupboard", "kitchen"),
       [("a bottle set aside for the recycling run", "hallway"),
        ("a bottle being kept chilled for tonight", "kitchen"),
        ("a bottle promised to the neighbour tomorrow", "hallway")],
       "hard", "every other bottle is already spoken for"),
    _c("bottle_salience", "salience", "bottle", "drawer",
       ("a bottle of tablets matching the others in the medicine drawer", "bedroom"),
       [("a bottle of squash from the drinks crate", "kitchen"),
        ("a bottle of bleach kept high up", "kitchen"),
        ("a bottle of wine from the rack", "kitchen")],
       "medium", "the one that belongs with what the drawer already holds"),

    # --------------------------------------------------------------- bowl ---
    _c("bowl_already", "already_satisfied", "bowl", "sink",
       ("a bowl with cereal dregs drying on the table", "kitchen"),
       [("a bowl already soaking under the tap", "kitchen"),
        ("a bowl stacked in the washing-up water", "kitchen"),
        ("a bowl sitting in the sink waiting its turn", "kitchen")],
       "easy", "the others are already in the sink"),
    _c("bowl_afford", "affordance", "bowl", "shelf",
       ("a bowl light enough to lift onto the high shelf", "kitchen"),
       [("a bowl of cast iron, too heavy to lift overhead", "kitchen"),
        ("a bowl too tall to stand under that shelf", "kitchen"),
        ("a bowl with a cracked base that would not sit flat", "kitchen")],
       "medium", "only one of them can physically go up there"),
    _c("bowl_state", "state_blocks", "bowl", "drawer",
       ("a bowl washed and dried on the rack", "kitchen"),
       [("a bowl with dough rising under a cloth", "kitchen"),
        ("a bowl of batter resting for later", "kitchen"),
        ("a bowl catching a drip under the pipe", "kitchen")],
       "hard", "only the empty, dry one is free to be put away"),
    _c("bowl_own", "ownership", "bowl", "table",
       ("a bowl from the shared cupboard", "kitchen"),
       [("a bowl the child decorated and keeps for themselves", "bedroom"),
        ("a bowl on loan from next door", "hallway"),
        ("a bowl the guest brought their salad in", "living_room")],
       "hard", "only the household one is ours to lay out"),
    _c("bowl_recency", "recency", "bowl", "table",
       ("a bowl just filled with soup, still steaming", "kitchen"),
       [("a bowl that has been in the cupboard since spring", "kitchen"),
        ("a bowl unpacked from the move and never used", "hallway"),
        ("a bowl put away clean last night", "kitchen")],
       "medium", "the one just served is the one to bring to the table"),
    _c("bowl_prox", "proximity", "bowl", "bin",
       ("a bowl of scrapings standing right next to the bin", "kitchen"),
       [("a bowl of scrapings up in the bedroom", "bedroom"),
        ("a bowl left in the hallway by the coats", "hallway"),
        ("a bowl on the far side of the living room", "living_room")],
       "easy", "the one already at the destination is meant"),
    _c("bowl_cond", "condition", "bowl", "bin",
       ("a bowl holding fruit gone soft and brown", "kitchen"),
       [("a bowl of clementines, all firm", "kitchen"),
        ("a bowl of apples bought yesterday", "kitchen"),
        ("a bowl of nuts still in their shells", "living_room")],
       "medium", "the spoiled contents are what needs binning"),
    _c("bowl_purpose", "purpose_fit", "bowl", "table",
       ("a bowl of salad, dressed and ready", "kitchen"),
       [("a bowl of cat biscuits", "hallway"),
        ("a bowl of loose screws from the toolbox", "hallway"),
        ("a bowl of soil with cuttings taking root", "living_room")],
       "easy", "the table is being laid for a meal"),
    _c("bowl_excl", "exclusion", "bowl", "drawer",
       ("a bowl from the everyday stack", "kitchen"),
       [("a bowl needed for tomorrow's baking", "kitchen"),
        ("a bowl set out to catch the leak", "hallway"),
        ("a bowl reserved for the party at the weekend", "living_room")],
       "hard", "every other bowl has been kept out for a stated reason"),
    _c("bowl_salience", "salience", "bowl", "shelf",
       ("a bowl matching the four already stacked on the shelf", "kitchen"),
       [("a bowl from the mixing set in the cupboard", "kitchen"),
        ("a bowl from the picnic basket", "hallway"),
        ("a bowl that lives on the dressing table", "bedroom")],
       "medium", "the one that completes the stack already up there"),

    # ------------------------------------------------------------- remote ---
    _c("remote_already", "already_satisfied", "remote", "drawer",
       ("a remote face down on the sofa cushion", "living_room"),
       [("a remote already lying in the open drawer", "living_room"),
        ("a remote clipped into its holder in the cabinet", "living_room"),
        ("a remote zipped into the console bag", "living_room")],
       "easy", "the others are already stowed", away_ok=True),
    _c("remote_afford", "affordance", "remote", "table",
       ("a remote that works the television", "living_room"),
       [("a remote with no batteries and the cover missing", "living_room"),
        ("a remote for a player that was sold years ago", "living_room"),
        ("a remote whose buttons no longer respond", "living_room")],
       "medium", "only the working one is worth having to hand"),
    _c("remote_state", "state_blocks", "remote", "drawer",
       ("a remote for the old player nobody uses", "living_room"),
       [("a remote in someone's hand, mid-programme", "living_room"),
        ("a remote paired to the speaker currently playing", "living_room"),
        ("a remote set to a timer that is counting down", "bedroom")],
       "medium", "the ones in use cannot be put away"),
    _c("remote_own", "ownership", "remote", "shelf",
       ("a remote belonging to the living-room set", "living_room"),
       [("a remote from the neighbour's projector, to return", "hallway"),
        ("a remote the landlord left with the inventory", "hallway"),
        ("a remote from a car the household no longer owns", "bedroom")],
       "hard", "only the household one belongs on the shelf here"),
    _c("remote_recency", "recency", "remote", "table",
       ("a remote set down on the sofa a moment ago", "living_room"),
       [("a remote taped inside the cabinet as a spare", "living_room"),
        ("a remote still in its factory bag", "hallway"),
        ("a remote kept in the drawer since the move", "bedroom")],
       "hard", "the stored ones are deliberate; the loose one is the stray"),
    _c("remote_prox", "proximity", "remote", "table",
       ("a remote on the arm of the chair pulled up to the table", "living_room"),
       [("a remote in the bedroom upstairs", "bedroom"),
        ("a remote at the far end of the hall shelf", "hallway"),
        ("a remote in the kitchen by the radio", "kitchen")],
       "medium", "the nearest one is meant"),
    _c("remote_cond", "condition", "remote", "bin",
       ("a remote leaking corrosion from the battery bay", "living_room"),
       [("a remote that has simply lost its pairing", "living_room"),
        ("a remote with one sticky button", "living_room"),
        ("a remote missing only its battery cover", "living_room")],
       "hard", "the corroded one is scrap; the rest can be fixed"),
    _c("remote_purpose", "purpose_fit", "remote", "drawer",
       ("a remote used once a year for the projector", "living_room"),
       [("a remote used every evening for the television", "living_room"),
        ("a remote that opens the garage door on the way out", "hallway"),
        ("a remote for the bedside lamp", "bedroom")],
       "hard", "only the rarely used one belongs out of reach in a drawer"),
    _c("remote_excl", "exclusion", "remote", "shelf",
       ("a remote for the spare set-top box in the cupboard", "living_room"),
       [("a remote needed for the film later", "living_room"),
        ("a remote set aside for electrical recycling", "hallway"),
        ("a remote promised to the neighbour", "hallway")],
       "hard", "every other remote is already spoken for"),
    _c("remote_salience", "salience", "remote", "drawer",
       ("a remote matching the others already in the drawer", "living_room"),
       [("a remote from the car keys tray", "hallway"),
        ("a remote kept taped to the bedhead", "bedroom"),
        ("a remote that lives on the kitchen radio", "kitchen")],
       "medium", "the one that belongs with what the drawer already holds"),

    # ------------------------------------------------------------ cushion ---
    _c("cushion_already", "already_satisfied", "cushion", "shelf",
       ("a cushion on the floor beside the front door", "hallway"),
       [("a cushion squared up at the end of the sofa", "living_room"),
        ("a cushion already stacked on the shelf", "living_room"),
        ("a cushion zipped into the storage bag", "bedroom")],
       "medium", "the others are already in their place", away_ok=True),
    _c("cushion_afford", "affordance", "cushion", "table",
       ("a cushion, small, used as a laptop rest", "bedroom"),
       [("a cushion, full size, from the sofa", "living_room"),
        ("a cushion the length of the window seat", "living_room"),
        ("a cushion in a bulky quilted cover", "bedroom")],
       "hard", "only the small one makes sense on a table"),
    _c("cushion_state", "state_blocks", "cushion", "drawer",
       ("a cushion fallen down behind the armchair", "living_room"),
       [("a cushion with the cat asleep on it", "living_room"),
        ("a cushion propped behind someone's back", "living_room"),
        ("a cushion under the sleeping baby's head", "bedroom")],
       "hard", "you do not move the ones in use"),
    _c("cushion_own", "ownership", "cushion", "drawer",
       ("a cushion from the spare throw set", "living_room"),
       [("a cushion from the baby's cot", "bedroom"),
        ("a cushion the guest brought for their back", "living_room"),
        ("a cushion the dog sleeps on", "hallway")],
       "hard", "only the spare belongs in storage"),
    _c("cushion_recency", "recency", "cushion", "shelf",
       ("a cushion pulled off the sofa a minute ago", "living_room"),
       [("a cushion that has been in the hall for months", "hallway"),
        ("a cushion still in its shop wrapping", "bedroom"),
        ("a cushion stored flat since the summer", "bedroom")],
       "medium", "the one just displaced is the one to put back"),
    _c("cushion_prox", "proximity", "cushion", "table",
       ("a cushion on the chair pushed up to the table", "living_room"),
       [("a cushion in the bedroom two doors along", "bedroom"),
        ("a cushion at the far end of the hallway", "hallway"),
        ("a cushion behind the sofa in the corner", "living_room")],
       "medium", "the nearest one is meant"),
    _c("cushion_cond", "condition", "cushion", "bin",
       ("a cushion split open with the stuffing coming out", "living_room"),
       [("a cushion with a stain that washes out", "living_room"),
        ("a cushion faded on one side only", "bedroom"),
        ("a cushion with a loose seam, easily sewn", "bedroom")],
       "medium", "only the ruined one is past saving"),
    _c("cushion_purpose", "purpose_fit", "cushion", "sink",
       ("a cushion with a washable cover, soaked with juice", "living_room"),
       [("a cushion of feather down that must be dry cleaned", "bedroom"),
        ("a cushion glued onto a solid foam base", "living_room"),
        ("a cushion with an electric heat pad inside", "bedroom")],
       "hard", "only the washable one can go anywhere near water"),
    _c("cushion_excl", "exclusion", "cushion", "shelf",
       ("a cushion from the everyday sofa set", "living_room"),
       [("a cushion needed for the guest bed tonight", "bedroom"),
        ("a cushion set aside for the charity bag", "hallway"),
        ("a cushion promised to the neighbour", "hallway")],
       "hard", "every other cushion is already spoken for"),
    _c("cushion_salience", "salience", "cushion", "drawer",
       ("a cushion matching the covers already folded in the drawer", "bedroom"),
       [("a cushion from the garden furniture set", "hallway"),
        ("a cushion from the car's back seat", "hallway"),
        ("a cushion that lives on the reading chair", "living_room")],
       "medium", "the one that belongs with what the drawer already holds"),
]


def colorize(desc: str, noun: str, color: str) -> str:
    """Splice a colour in: 'a mug on the ...' -> 'a red mug on the ...'.

    Descriptions are authored to start with a bare ``a {noun}`` precisely so
    this is mechanical -- it cannot reorder anything or lengthen a description
    except by the colour word, which every candidate receives.
    """
    prefix = f"a {noun}"
    if not desc.startswith(prefix):
        raise ValueError(f"description must start with {prefix!r}: {desc!r}")
    return f"a {color} {noun}" + desc[len(prefix):]


def validate_cores(cores: Optional[List[Core]] = None) -> None:
    """Structural audit of the core set. Run by the build script before expansion."""
    cores = CORES if cores is None else cores
    ids = [c.id for c in cores]
    assert len(set(ids)) == len(ids), "duplicate core ids"
    for c in cores:
        assert c.noun in NOUNS, f"{c.id}: noun {c.noun!r}"
        assert c.receptacle in RECEPTACLES, f"{c.id}: receptacle {c.receptacle!r}"
        assert c.family in PRAGMATIC_FAMILIES, f"{c.id}: family {c.family!r}"
        assert c.difficulty in ("easy", "medium", "hard"), f"{c.id}: difficulty"
        assert c.rationale.strip(), f"{c.id}: rationale required"
        assert len(c.distractors) >= 3, f"{c.id}: need >=3 distractors for k=4"
        for desc, room in (c.target,) + tuple(c.distractors):
            assert room in ROOMS, f"{c.id}: room {room!r}"
            assert desc.startswith(f"a {c.noun}"), \
                f"{c.id}: description must start with 'a {c.noun}': {desc!r}"
            for col in COLORS:
                # Word-boundary, not substring: "flared" contains "red".
                assert not re.search(rf"\b{col}\b", desc, re.I), \
                    f"{c.id}: description leaks a colour {col!r}: {desc!r}"
        descs = [c.target[0]] + [d[0] for d in c.distractors]
        assert len(set(descs)) == len(descs), f"{c.id}: duplicate descriptions"
