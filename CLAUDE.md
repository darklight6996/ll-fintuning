# CLAUDE.md

## Cross-model coordination (REQUIRED)
- **Before doing anything, read `work.md` at the repo root** — it is the shared ledger of what every model is doing, the locked design principles, and the MUST-PRESERVE invariants.
- **After every meaningful change, append a Log entry to `work.md`** (format is defined in that file). Never change code without logging it there.
- If a task would require weakening a MUST-PRESERVE item (rigor gate, suspicion gate, blind scrubbing, human-in-the-loop), STOP and log it as `blocked / needs-human`.
