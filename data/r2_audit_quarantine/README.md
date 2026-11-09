# Annotation-pass quarantine log — raised, resolved, closed

**This directory is intentionally empty of annotation passes.** All three live
in `data/r2_audit/`. The note below is kept as the record of a provenance
challenge that was raised against one of them and then resolved in its favour.

## What was raised

Mid-audit, `reader_c.jsonl` was pulled from `data/r2_audit/` and held here over a
provenance doubt: part of the sheet may have been filled in by the second
annotator rather than by that annotator. Independence is the only property the audit
measures, so a shared sheet would have invalidated the pass regardless of what
its numbers said.

## How it was resolved

The pass was **redone from scratch** rather than partially salvaged, because
there was no way to tell which items would have been whose. The redo reproduced
the original at **0.978 on co-committed items**, which is not a rate a shared
sheet produces. The pass was therefore **reinstated** to `data/r2_audit/` and
the quarantine closed. The decision is recorded in `docs/R2_REPLY_7.md` §5.

## What the live basis is

The three-annotator result stands as the basis for the frozen artefact:

| basis | cores retained | Fleiss κ | raw agreement | file |
| --- | --- | --- | --- | --- |
| **3 annotators (live)** | **29 / 60** | **0.592** | **0.617** | `data/r2_core_audit.json` |
| 2 annotators (sensitivity) | 39 / 60 | 0.722 | 0.800 | `data/r2_core_audit_2rater.json` |

`data/r2_scenarios.jsonl` (321 scenarios, 29 cores) and the
`audited_321_contested_dropped` artefact are built from the 3-annotator row.

The 2-annotator row is a **sensitivity analysis computed before the third pass
existed**, not a post-hoc exclusion. It is reported alongside the headline
figure in the dialogue chapter and in `docs/r2_measurements.md` §3G, and the
10-core / 0.13-κ gap between the rows is treated as a finding about item
difficulty rather than as a reason to drop a rater. Pairwise agreement is
A↔B 0.800, A↔C 0.625, B↔C 0.638: one careful reader
resolves roughly a third of these differently from the other two, reproduced
across two separate sittings.

## The distinction this log exists to preserve

* Dropping a pass because its numbers are inconvenient is data dredging and
  would invalidate the audit.
* Dropping a pass that was not produced by one person working independently is
  required, because independence is the only property the audit measures.

The doubt here was of the second kind, it was tested rather than assumed, and it
did not survive the test. No pass has been excluded from this audit on any
ground.
