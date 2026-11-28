# QUARANTINED TRAJECTORY DATA — DO NOT USE FOR FINE-TUNING

**Quarantined:** 2026-11-28
**By:** BUILD SPEC v1, workstream A part 1 (blind-mode persistence integrity)
**Related audit:** `docs/audit-2026-11-28.md` (Defect #7 — highest severity)

## Files

- `portswigger_trajectories_raw.jsonl`
- `portswigger_trajectories_chatml.jsonl`

(Moved verbatim from `data/` — contents are **unmodified** from when they were written.)

## Why these were quarantined

Blind-mode scrubbing never reached `autonomous_execution_history`. In
`agents/base_agent.py`'s autonomous loop the raw execution results were stored
into history **before** the blind-scrub block, and that block only ever
reassigned the local `stdout`/`stderr` used to build the `Observation` — it
never touched the stored history copy. Both training-data writers in
`run_portswigger_lab.py` (`_write_raw_trajectory` and `_build_chatml_record`)
read directly from that unscrubbed history, so every `--blind` run wrote the
real PortSwigger lab answer-key text into both files.

The audit **confirmed real lab titles / answer-key markers on lines 32–33** of
both files (reproduced byte-for-byte, not inferred): the literal
`<title>Exploiting XXE to perform SSRF attacks</title>` and the
`<!--LAB_HEAD_START-->` lab-head marker are present in the ChatML training text
of a record whose entire stated purpose (`--blind`) was to prevent exactly that.

## Consequences / rules

- **These files MUST NOT be used for fine-tuning.** Any `--blind` line that
  carries the leak would train the model directly on partial answer-key text —
  defeating the benchmark the blind mode exists to produce.
- They are **retained for forensic reference** (so the leak can be re-examined)
  and **must not be silently reintroduced** into any training pipeline or merged
  back into `data/`.
- Do **not** delete, modify, or overwrite the original contents.

## What changed

The persistence boundary is fixed in the current checkout: results are now
scrubbed (in `--blind` runs) via `BaseAgent._history_results_for_store` **before**
they enter `autonomous_execution_history`, so the planner's recent-history prompt
and both trajectory writers all receive scrubbed data. Fresh (clean) trajectory
files will be written to `data/` by subsequent runs; these quarantined originals
stay here untouched.
