# Quarantined annotation passes

`annotator3.jsonl` is held here, not in `data/r2_audit/`, because its provenance is
in doubt: part of the sheet may have been filled in by the second annotator
rather than by that annotator.

This is a **provenance** exclusion, not a results-based one. The distinction
matters and should survive into the report:

* Dropping a pass because its numbers are inconvenient is data dredging and
  would invalidate the audit.
* Dropping a pass that was not produced by one person working independently is
  required, because independence is the only property the audit measures.

If the pass turns out to be genuinely one person's, move it back and re-run
`python scripts/r2_audit.py --report`. If not, it should be redone from scratch
rather than partially salvaged -- there is no way to tell which items are whose.

Open risk if it was shared: the two passes are then not independent of each
other, which would also compromise `annotator2` (the outside annotator). In that
case both need redoing, and `adit.jsonl` is the only surviving clean pass.
