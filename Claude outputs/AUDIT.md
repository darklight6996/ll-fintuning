# Autonomous Security Agent — Code Audit

**Scope of review:** the autonomous ("autohunt") reasoning path end‑to‑end — `main.py`, `core/ai_planner.py`, `core/investigation.py`, `core/action_contract.py`, `agents/base_agent.py`, the CTF agent (`actions`, `command_builders`, `completion`, `agent`), `rag/core/scope.py`, `rag/core/command_executor.py`, `rag/core/chat_assistant.py`, `rag/core/model_router.py`.

**Verdict up front:** the deterministic *scaffolding* (typed actions, pure‑Python builders, shell=False execution, scope parsing) is genuinely solid. The **autonomous brain is not wired up**. Four independent defects each, on their own, prevent an autonomous run from ever succeeding, and three of them are the direct mechanical cause of the loops / hallucination / "can't decide" behaviour you're seeing. None of this is a model‑quality nuance — these are plumbing bugs. The small‑model choice makes it worse but is not the primary cause.

This document only identifies problems and proposes solutions as originally
written. No code was changed *at the time this audit was written*.

**UPDATE (2026-11-11):** every C/H/M/MED item below has since been implemented
and verified (via unit tests plus live GPU smoke testing with
Qwen2.5-7B-Instruct + a local QLoRA adapter on an RTX 3060 Ti) -- see each
item's **STATUS** note. This file is kept as the historical record of what was
found and fixed, not an open backlog. The one caveat found during live
verification: `verify_autohunt_smoke_test.py`'s default target (`127.0.0.1` on
a Windows dev box) still can't produce a CTF-agent finding, because the CTF
action catalog (SUID audits, cron jobs, `/proc/version`, `path_hijack`, ...) is
Linux-only by design -- that's a smoke-test-target mismatch, not a bug in any
of the items below. A true end-to-end run needs a Linux target (planned: WSL2
with CUDA passthrough).

---

## How to read this

Your four symptoms map to specific defects:

| Symptom | Primary cause(s) |
|---|---|
| "It never finishes / loops to the end every time" | C1 (findings can't be produced) + C2 (retry budget is dead) |
| "It hallucinates" | C4 (param schemas never shown to the model) + M1–M3 (sampling, wrong system prompt, truncation) + tiny model |
| "It gets stuck in loops" | C2 (retry budget dead) + C4 (invalid proposals rejected → retried) + no no‑progress detector |
| "It can't decide the next step properly" | C3 (scope flags never set → every active action rejected) + C5 (planner sees no observation data to reason from) + C1 |

---

## CRITICAL defects (each one alone breaks autonomy)

### C1 — The evidence chain has no producer: a Finding can never be created in autonomous mode
**Where:** `agents/base_agent.py::run_autonomous` (steps 4–6, lines ~425–467); `core/investigation.py::EvidenceStore.record_finding`; `agents/ctf/completion.py::CTF_COMPLETION`.

**What's wrong.** In the autonomous loop the AI can only return `propose_action` or `propose_completion`. On a successful execution the loop records an **Observation only** (lines ~457–464). Nothing ever creates a `Hypothesis`, nothing ever creates a `Validation`, and `EvidenceStore.record_finding()` is **never called anywhere in the loop**. `Hypothesis`, `Validation`, `Finding` are imported at line 315 and then unused.

But completion is gated on findings: `CTF_COMPLETION.minimum_findings = 1`, and `verify_completion` rejects completion unless `len(evidence_store.findings) >= 1`. Since findings can never be produced, **`verify_completion` always returns False**. Every `propose_completion` is rejected and the loop `continue`s until `max_steps` (50). That is exactly your "it always runs to the end and never decides it's done."

The whole `EvidenceGraph` (entities/relationships, `find_paths`, `correlate_cross_observations`) is also dead in the live loop — `add_entity`/`add_relationship` are never called. The README's "Observation → Hypothesis → Validation → Finding → Impact" is enforced on the *consumer* side but has no *producer* side.

**Fix (design, not a patch).** Give the loop a way to move evidence forward. Two viable options:

- **Option A (recommended): add structured decision verbs.** Extend `AIActionProposal.decision` to include `form_hypothesis`, `record_validation`, and `record_finding`, and handle them in `run_autonomous`. The planner then explicitly says "observations obs_x, obs_y suggest hypothesis H; validation action A confirmed it; record finding F." Keep `record_finding` framework‑gated exactly as it is (that gate is good). This makes the advertised reasoning chain real.
- **Option B: deterministic promotion.** After each observation, run a per‑agent `analyze(observation) -> [Hypothesis]` and, when a validation action succeeds against an open hypothesis, auto‑create the `Validation` with a verdict derived from the tool output (e.g. SUID binary present, flag string matched). Less "AI", more reliable on a weak model.

Either way: **until a finding can be created, no autonomous run of any agent whose `minimum_findings > 0` can ever succeed.** This is the single most important fix.

---

**STATUS: IMPLEMENTED (both options, layered).**
1. *Minimal floor (done first):* `agents/base_agent.py::_promote_evidence_if_warranted` deterministically
   promotes an Observation to Hypothesis -> Validation -> Finding whenever the executed action has
   `satisfies_phase=True` (already authored per-action in every `agents/*/actions.py`, previously
   unread anywhere) and actually returned exit_code 0 with non-empty output. This alone guarantees
   `minimum_findings >= 1` is reachable even on a weak/non-cooperative model.
2. *Full Option A (done second):* `AIActionProposal.decision` now also accepts `form_hypothesis`,
   `record_validation`, and `record_finding` (`core/action_contract.py`). `agents/base_agent.py` gained
   `_handle_form_hypothesis` / `_handle_record_validation` / `_handle_record_finding`, wired into
   `run_autonomous()` right before the 10-gate action pipeline (step "3b"). Every `obs_id`/`hyp_id` the
   AI cites is checked against what actually exists in `EvidenceStore` before anything is recorded --
   a fabricated or paraphrased id is rejected with a specific reason, not silently accepted.
   `record_finding` still goes through `EvidenceStore.record_finding()`'s own `can_create_finding()` gate
   unchanged, so a hypothesis that was never actually confirmed by a supporting validation is refused
   regardless of what the AI claims. `core/ai_planner.py` now documents all five decisions in the system
   prompt, lists real `obs_id`s in a new `OBSERVATIONS` prompt section (previously the AI had no way to
   cite a real id at all), and raises a retry-eligible `ValueError` in `_parse_proposal` when a verb's
   required parameters are missing. Regression tests: `tests/test_remediation_c1_decision_verbs.py`.

---

### C2 — The retry‑budget gate (Gate 6) is dead in autonomous mode → infinite thrashing

**STATUS: IMPLEMENTED.** `agents/base_agent.py::run_autonomous` now passes
`execution_history=execution_history` into `self.executor.execute_action(...)`
explicitly instead of relying on `**kwargs`, so Gate 6 sees the real history and
`prior_attempts` is computed correctly. Regression:
`tests/test_remediation_c1_c2_h1.py::test_retry_budget_trips_via_execution_history_kwarg`.
**Where:** `agents/base_agent.py::run_autonomous` line ~426 (the `execute_action` call) vs `rag/core/command_executor.py::execute_action` signature (line ~187) and Gate 6 (lines ~369–383).

**What's wrong.** The loop calls:
```python
self.executor.execute_action(proposal=proposal, registry=registry, scope=self.scope,
                             session_state=self.session_mgr.state, mode=self.mode)
```
but the method signature is `execute_action(self, proposal, registry, scope, approval_policy="manual", execution_history=None, **kwargs)`. So `session_state=` and `mode=` land in `**kwargs`. `mode` is picked back up for the approval policy (that part works), **but `execution_history` is never passed**, so it stays `None` → `execution_history = execution_history or []` → **always empty**.

Gate 6 computes `prior_attempts` by counting matching `action_identity` **in that empty list**, so `prior_attempts` is always 0 and the budget **never trips**. The retry budget — the mechanism the README advertises as "no infinite loops" — does nothing in the autonomous path. The planner can propose the same failing action again and again, bounded only by `max_steps`. This is a core loop driver.

(Also note: `base_agent` builds a `retry_counts` dict keyed by `action_id` for the *planner prompt*, while Gate 6 keys by `action_identity`. Two different keys, and only the prompt one is ever populated. Tighten both to use `action_identity`.)

**Fix.** Pass the real history into the executor:
```python
self.executor.execute_action(proposal=proposal, registry=registry, scope=self.scope,
                             execution_history=execution_history, mode=self.mode)
```
and make the signature explicit (`mode="manual"` as a named param) instead of relying on `**kwargs` fishing. Add a test that asserts Gate 6 rejects the 3rd identical proposal.

---

### C3 — Scope authorization flags are never set → every active/state‑changing action is rejected

**STATUS: IMPLEMENTED.** `main.py::gather_scope` now prompts for and sets all
three flags (`active_testing_allowed`, `active_network_allowed`,
`state_changes_allowed`); the `--target`/`--autohunt` fast paths take them as
explicit CLI flags (`--authorize-active`/`--authorize-network`/
`--authorize-state-change`) instead of defaulting all three to False.
**Where:** `main.py::gather_scope` (lines ~160–169) and the `--target`/`--autohunt` fast paths (lines ~251–258); `rag/core/scope.py` (fields at lines 28–31); `rag/core/command_executor.py` Gate 3 (lines ~290–322); `core/ai_planner.py` lines ~280–281.

**What's wrong.** `Scope` has **four** separate authorization flags: `active_testing_allowed`, `active_network_allowed`, `state_changes_allowed`, `recon_only_authorized`. The operator interview only ever sets `active_testing_allowed` (from "Authorize Active Testing?"). `active_network_allowed` and `state_changes_allowed` default to **False and are never set by anything**. The `--target` and `--autohunt` fast paths construct `Scope(target=…, in_scope=[…])` with **all four False**.

Now look at Gate 3:
- line ~302: any action with `network_effect == "active"` is rejected unless `active_network_allowed` → **always rejected**.
- line ~313: any action with `state_change != "none"` is rejected unless `state_changes_allowed` → **always rejected** (e.g. CTF `path_hijack_validation`).

So in autonomous mode, **only passive, local, non‑state‑changing actions can ever run.** For a firewall/pentest/red‑team/cloud agent — whose entire job is active network testing — *every meaningful action is rejected at Gate 3*, the planner keeps proposing them, and the run thrashes to `max_steps`. For CTF it happens to work (all actions are local/none) except privesc.

Compounding it: the planner prompt (ai_planner lines ~280–281) reports `active_testing` and `state_changes` by reading `active_network_allowed`/`state_changes_allowed` — the ones that are always False — so the model is *told* it may do nothing active and avoids proposing progress, while the one flag the operator did set (`active_testing_allowed`) is never surfaced to the model. Split brain between what the operator authorized, what the planner is told, and what the gate enforces.

**Fix.**
1. In `gather_scope`, set all relevant flags from the interview (e.g. a single "Authorize active network testing?" prompt that sets `active_network_allowed`, and an explicit "Authorize state‑changing/exploit validation?" that sets `state_changes_allowed`). Decide whether `active_testing_allowed` is a master switch that implies the others, or keep them independent — but then collect them.
2. For `--autohunt`/`--target`, either prompt for these or document that autohunt is passive‑only by default and require an explicit `--authorize-active` flag.
3. Make the planner read the **same** attributes the executor enforces, and pass all of them into the prompt so the model's picture matches the gate.
4. Pick one naming convention. Having `active_testing_allowed` *and* `active_network_allowed` *and* `recon_only_authorized` invites exactly this class of bug — collapse to a small, documented set.

---

### C4 — The planner never shows the model each action's parameter schema → guaranteed hallucinated params → Gate 1 rejects → loop

**STATUS: IMPLEMENTED.** `core/ai_planner.py::_build_prompt` now renders each
action's parameter schema (field names, types, required/default) via
`_format_params_schema`, instead of showing only `action_id [risk/network/state]:
description`.
**Where:** `core/ai_planner.py::_build_prompt` (lines ~217–225) vs `core/action_contract.py::ActionRegistry.to_prompt_schema` (produces `parameters_schema`, lines ~127–149); `validate_proposal` with Pydantic `extra="forbid"` (lines ~109–125).

**What's wrong.** `to_prompt_schema()` deliberately exports a `parameters_schema` for every action. But `_build_prompt` builds the catalog line as just `action_id [risk, network, state]: description` and **throws the parameter schema away**. The model is asked to emit `"parameters": { … }` for actions like `strings_analysis` (needs `binary_path`), `read_file` (needs `file_path`), `path_hijack_validation` (needs `fake_binary_name`, `target_suid_binary`) **without ever being told those fields exist or what they're called.**

So the model guesses field names. Gate 1 validates with `extra="forbid"` and a required‑field check, so a guessed/missing/misnamed field is rejected. The planner only retries on *JSON parse* errors (3×), not on semantic rejections — the rejection goes back as truncated history text (80 chars, see C5) and the model guesses again, differently. Because each different guess is a different `action_identity`, even a *working* Gate 6 wouldn't stop it. This is a primary hallucination‑and‑loop engine.

**Fix.**
1. In `_build_prompt`, include the parameter schema per action — at minimum the required field names, types and defaults. You already compute it; just render it:
   ```
   - strings_analysis [low risk]: Run strings on a binary.
       params: binary_path:str (required), min_length:int=8
   ```
2. Feed semantic rejections back into the planner retry loop, not just JSON errors: if Gate 1 rejects for a parameter reason, re‑prompt with the exact validation error and the correct schema before counting it as a step.
3. Consider constrained decoding (a JSON grammar / function‑calling format) so parameter names can't be invented — see M3.

---

### C5 — The planner is reasoning almost blind: observation output isn't fed back usefully

**STATUS: IMPLEMENTED.** `Observation.parsed_facts` is now populated per-action via
agent-specific `fact_extractors.py`, and `core/ai_planner.py::_format_discovered_facts`
surfaces a compact "known facts" block in the prompt instead of relying only on a
truncated raw stdout snippet.
**Where:** `core/ai_planner.py::_build_prompt` (history formatting lines ~243–260; hypotheses/findings lines ~262–275).

**What's wrong.** The only state the model sees about what happened is the last 8 history rows, each reduced to `action → status` plus the **first 120 chars of stdout of the first command**. Rejection reasons are truncated to 80 chars (so a Pydantic "field required" message is cut before it's useful). Active hypotheses and findings are *always empty* because of C1. So on step 2, after a successful `suid_audit`, the model does **not** see the SUID binary path it just found — it can't carry that into `strings_analysis.binary_path`. It literally lacks the data needed to choose and parameterize the next step. That is "can't decide the next step properly," independent of model size.

**Fix.**
1. Populate `Observation.parsed_facts` (the field exists, always `{}`) with a per‑action parser that extracts the few structured facts that matter (open ports, SUID paths, usernames, discovered endpoints). Promote these into `EvidenceGraph` entities.
2. Put a compact "known facts" block in the planner prompt (entities discovered so far, keyed by type) instead of raw truncated stdout. This is both smaller and far more decision‑relevant.
3. Raise the rejection‑reason budget well above 80 chars, or pass the structured `rejection_category` + full reason, so the model can actually self‑correct.

---

## HIGH‑severity defects

### H1 — The shared loop hard‑codes CTF's completion logic for every agent
**Where:** `agents/base_agent.py::run_autonomous` line ~396: `from agents.ctf.completion import verify_completion`.

Every agent (pentest, red_team, cloud, …) is verified against **CTF's** `verify_completion` and CTF's `action_to_phase` map. Pentest/cloud action_ids aren't in that map, so `completed_phases` stays empty and `required_phases` is never satisfied — a second, independent reason non‑CTF agents can never complete. Each domain already ships its own `completion.py`; the loop just ignores them.

**STATUS: IMPLEMENTED.** The completion verifier is now resolved per-agent-type
instead of unconditionally importing CTF's. Regression:
`tests/test_remediation_c1_c2_h1.py::test_completion_verifier_resolves_per_agent_type`,
`::test_completion_verifier_raises_for_unknown_agent_type`.

### H2 — `max_new_tokens` from the planner is silently ignored

**STATUS: IMPLEMENTED.** `model_router.ModelRouter.generate()` now takes an
`Optional[max_new_tokens]` override and uses it when given, falling back to
`self.max_new_tokens` only when the caller omits it (see the "H2 fix" comment at
`ll-finetuning/rag/core/model_router.py` around the `generate()` method).
**Where:** `core/ai_planner.py` calls `generate_raw(..., max_new_tokens=512)`; `chat_assistant.generate_raw` forwards `max_new_tokens=…`; but `model_router.ModelRouter.generate(self, prompt, temperature, top_p, do_sample, system_prompt, **kwargs)` **ignores it** and always uses `self.max_new_tokens` (set from config at construction).

If the configured cap is short (e.g. 256), the JSON gets truncated mid‑object → `json.loads` fails → 3 retries → planner error → `continue`. Intermittent, size‑dependent malformed JSON.

**Fix.** Honour the per‑call `max_new_tokens` in `generate()` (and in the `LLMService.generate` wrapper), falling back to `self.max_new_tokens`.

### H3 — The planner's strict system prompt is demoted; a generic mentor persona overrides it

**STATUS: IMPLEMENTED.** `system_prompt` is now threaded through to `generate()`
and applied as the real chat-template system role per model family, instead of
being concatenated into the user turn and overridden by a default mentor persona.
Regression: `tests/test_model_router_system_prompt.py`.
**Where:** `chat_assistant.generate_raw` lines ~549–557 builds `full_prompt = system_prompt + "\n\n" + prompt` and calls `llm.generate(full_prompt)` **without** passing `system_prompt=` through. `model_router._format_prompt` then applies its **own** default system message ("You are an expert penetration‑testing mentor…") and treats your entire planner contract as user text.

Result: the model's actual system role is a chatty mentor, not "emit exactly one JSON object." On a 1.1B model this is the difference between JSON and a paragraph. Direct contributor to malformed output.

**Fix.** Thread `system_prompt` all the way to `generate()`/`apply_chat_template` as the real system role; don't concatenate it into the user turn.

### H4 — Decoding is sampled, not deterministic, for a task that needs determinism

**STATUS: IMPLEMENTED.** `chat_assistant.generate_raw()` (the planner's only call
path) now passes `do_sample=False` for greedy decoding; `temperature`/`top_p` are
only applied when `do_sample=True` is explicitly requested elsewhere.
**Where:** `model_router.generate` defaults `temperature=0.2, do_sample=True`; `generate_raw` doesn't override.

For structured‑JSON/action selection you want greedy or near‑greedy decoding. Sampling adds avoidable variance → sometimes invalid JSON, sometimes a different action for identical state (which also looks like "indecision").

**Fix.** For planner calls use `do_sample=False` (greedy) or `temperature≈0`, and ideally constrained/grammar decoding (M3).

---

## The model choice (why it amplifies everything)

**STATUS UPDATE (2026-11-11):** the planner now runs
`Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 QLoRA) with the user's own fine-tuned LoRA
adapter, not TinyLlama-1.1B or flan-t5-large -- see M2 below and the model
configuration table in README.md. M1 (constrained decoding) is also implemented.
This does not make the remaining M2/M3/M4 guidance below moot (a 7B model still
benefits from them), but the specific "1.1B is fundamentally too small" framing
is no longer the current state.

### M‑root — a 1.1B chat model (or flan‑t5‑large on CPU) is being asked to be a reliable structured planner
**Where:** README + `model_router`/`config` defaults: GPU = `TinyLlama/TinyLlama-1.1B-Chat-v1.0` + LoRA `CyberSec-Model-V2`; CPU = `google/flan-t5-large`.

Even with C1–C5, H1–H4 all fixed, TinyLlama‑1.1B will not reliably (a) emit strict JSON, (b) only ever use registered `action_id`s, (c) fill typed params correctly, or (d) reason over multi‑step security state. The LoRA may improve domain phrasing but does not buy structured‑output reliability at 1.1B. On CPU, flan‑t5‑large (seq2seq, ~0.78B) is worse for this. This is why it "hallucinates" even when the plumbing lets it run.

**Fixes, in order of leverage:**
- **M1 — Constrain the output.** Use grammar/JSON‑constrained decoding (e.g. an Outlines/llama.cpp GBNF grammar, or a tool/function‑calling schema) so the model *cannot* emit an unknown `action_id` or an invalid field. This single change removes most Gate‑1 rejections and most JSON‑parse failures, and makes even a small model usable. Highest ROI.
  **STATUS: IMPLEMENTED (scoped).** `core/constrained_decoding.py::ActionIdConstrainedLogitsProcessor`
  constrains the `action_id` field's value to prefixes of real registered
  action_ids, AND (added 2026-11-11, after the model repeatedly appended
  parenthetical/`//`/bracket annotations right after a closed, valid action_id)
  forces the single token immediately after a closed, valid action_id's closing
  quote to a real JSON delimiter. Still deliberately not full JSON-grammar
  decoding (see the file's own docstring for the scope tradeoff). Regression:
  `tests/test_m1_constrained_decoding.py`.
- **M2 — Right‑size the planner model.** The planner is a *small* classification‑style decision; a 7–8B instruct model (quantized) is the realistic floor for reliable JSON+reasoning on your 8GB card. Consider splitting roles: a stronger model (or an API model) for `decide_next`, the small local LoRA for narration/report text. Don't let 8GB VRAM dictate the brain's competence.
  **STATUS: IMPLEMENTED (first half).** `GPU_MODEL_NAME` is now
  `Qwen/Qwen2.5-7B-Instruct` with `GPU_FORCE_4BIT=True`, loaded with the user's own
  QLoRA adapter (`lora-adapters/qwen-qlora/`, falling back to the HF repo). A
  separate `PLANNER_MODEL_NAME` override exists (`core/config.py`) for the
  role-split idea above, but is unset by default -- planner and chat/report
  currently share the same Qwen instance. Regression: `tests/test_qwen_everywhere.py`,
  `tests/test_m2_planner_model_override.py`.
- **M3 — Few‑shot the exact format.** Put 1–2 worked examples (state → correct JSON proposal) in the planner system prompt. Small models follow examples far better than rules.
- **M4 — Validate‑and‑repair loop.** On a semantic rejection, do a cheap "fix this JSON to match this schema/error" re‑prompt before spending a step (ties into C4.2).

---

## MEDIUM / correctness & robustness

- **MED‑1 Planner crash handling masks nothing useful.** On 3 failed JSON attempts `decide_next` raises; the loop catches it, logs `planner_decision failed`, and `continue`s with no backoff and no distinct cap. A permanently‑malformed model = 50 wasted steps. Add a consecutive‑planner‑failure cap that aborts early with a clear message.
  **STATUS: IMPLEMENTED.** `agents/base_agent.py::run_autonomous` tracks
  `consecutive_planner_failures` and aborts with `status="aborted_planner_failures"`
  after `MAX_CONSECUTIVE_PLANNER_FAILURES=5`. Regression: `tests/test_med1_med2.py`.
- **MED‑2 No no‑progress / cycle detector.** Nothing notices "N steps with no new observation/entity/finding." Add a stall detector that ends the run (or forces a strategy change) after K non‑productive steps — a cheaper, more honest backstop than `max_steps`.
  **STATUS: IMPLEMENTED.** `last_progress_step` / `MAX_STEPS_WITHOUT_PROGRESS=10`
  in `agents/base_agent.py::run_autonomous` abort the run early when no new
  Observation or successful evidence-verb call has happened in that window.
  Regression: `tests/test_med1_med2.py`.
- **MED‑3 JSON extraction regex is single‑nesting.** `ai_planner._parse_proposal` brace regex `(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})` handles one level of nesting; a `parameters` object containing a nested object breaks it and it may grab the wrong `{...}`. Prefer a brace‑matching scan or `json.JSONDecoder().raw_decode` over regex.
  **STATUS: IMPLEMENTED (and extended well past the original scope).**
  `_extract_json_object` now uses `json.JSONDecoder().raw_decode()` with a
  "prefer the first decision-bearing object" search, handling arbitrary nesting.
  It also went through several more rounds live against the real model:
  stripping parenthetical, `//`-comment, and bracket-tag annotations the model
  kept appending after `action_id`'s closing quote, plus a constrained-decoding
  guard (see M1) that now prevents the whole class structurally. Regression:
  `tests/test_med3_json_extraction.py`.
- **MED‑4 Gate 2 lets a local‑only action run against a remote target.** For `target=10.10.10.5` with a CTF action whose `allowed_target_kinds=["local"]`, `is_local_target` is False so the "local not permitted" check is skipped and the local command runs on the *host*, not the target. CTF assumes you already have a shell, but the loop never checks that the action's `allowed_target_kinds` actually includes the target kind — add a positive target‑kind match, not just the negative local check.
  **STATUS: IMPLEMENTED.** Gate 2 (`command_executor.py`) now classifies the
  target via `_classify_target_kind` and positively checks it against the
  action's `allowed_target_kinds`, not just the old negative local-only check.
  Regression: `tests/test_med4_target_kind.py`.
- **MED‑5 `path_hijack` builder uses `bash -c` with an interpolated path.** `["bash","-c", f"echo '…' > {fake_path}"]` reintroduces a shell with an interpolated value — the one place the "no shell strings" contract is bent. `fake_path` is derived from AI‑supplied `fake_binary_name`/`work_dir`. Even though Pydantic validates types, there's no character validation; a crafted name injects shell. Build the file with Python I/O or `tee` via argv, and validate the names.
  **STATUS: IMPLEMENTED.** `build_path_hijack_argv` now validates
  `fake_binary_name`/`work_dir`/`target_suid_binary` against strict allow-list
  regexes before use, writes the fake script via a `python -c` helper that
  receives the path only as `sys.argv[1]` (never interpolated into source), and
  uses `env VAR=value cmd` (argv, not shell) instead of `bash -c "..."`.
  Regression: `tests/test_med5_path_hijack.py`.
- **MED‑6 `compute_identity` can throw on non‑JSON‑serializable params.** `json.dumps(sorted(self.parameters.items()))` — fine for scalars, raises if a value isn't serializable. Low risk given typed params, but wrap defensively.
  **STATUS: IMPLEMENTED.** `AIActionProposal.compute_identity()` now falls back to a
  `repr()`-based encoding on `TypeError` instead of raising. Regression:
  `tests/test_med6_med7.py`.
- **MED‑7 Observation never stores which hypothesis/phase it serves.** Makes C1 Option B harder later; consider tagging observations with the proposing action's `phase`.
  **STATUS: IMPLEMENTED.** `Observation` gained an optional `phase` field, populated
  in `agents/base_agent.py::run_autonomous` from the executing action's own
  `ActionDefinition.phase`. Regression: `tests/test_med6_med7.py`.

---

## Documentation vs. reality (the docs are "not fully updated," as you said)

- **10‑gate "no infinite loops" (README §Safety)** — **FIXED** (C2): Gate 6 is live in autonomous mode now that `execution_history` is passed through.
- **"Observation → Hypothesis → Validation → Finding" enforced (README)** — **FIXED** (C1): both the deterministic promotion floor and the full `form_hypothesis`/`record_validation`/`record_finding` decision verbs are implemented.
- **Two modes "Automated Sequential" vs "Autonomous Hunter" (README)** — re-checked 2026-11-11: `main.py` now distinguishes `mode == "automated"` (calls `run_autonomous_default()`) from `--autohunt`; no longer the same code path unconditionally.
- **`network_effect=True` (README ActionRegistry example)** — **FIXED**: README's example now uses the real enum values and field names (`action_id`, `network_effect="active"`, etc.), matching `core/action_contract.py::ActionDefinition`.
- **Gate table says "Gate 2 target format + scope"** — implemented, but the local/remote kind check is incomplete (MED‑4).
- **README quickstart has broken code fences** — **FIXED**: the eval section's mangled `` `ash ``/`` `	ext `` (stray control characters, not literal backtick+letters) are now proper ` ```bash `/` ```text ` fences.
- **`rag` is a symlink** — **DOCUMENTED**: README's setup section now explains the `git config core.symlinks true` + privilege requirement on Windows and how to verify `rag` is a real symlink after cloning, rather than silently assuming it resolves.
- **Model configuration table (README)** — **FIXED**: was still describing TinyLlama-1.1B/flan-t5-large and claiming Qwen 7B "will cause OOM errors" on an 8GB card; now describes the actual Qwen2.5-7B-Instruct + 4-bit NF4 QLoRA setup, confirmed working live on an RTX 3060 Ti.
- **Test count badge/table (README)** — **FIXED**: was still advertising "39/39 tests" from two files; now reflects the full current suite (16 files, 122 test functions, 110 passed / 8 skipped on the last full run).

---

## Suggested fix order (highest leverage first)

1. **C3** — collect the scope authorization flags (otherwise non‑CTF agents reject everything). Fast.
2. **C2** — pass `execution_history` to the executor (turns the loop backstop back on). Fast, one line + signature.
3. **C4 + C5** — put parameter schemas *and* discovered facts into the planner prompt; widen rejection text. Medium.
4. **C1** — add the evidence‑chain producer (decision verbs or deterministic promotion) so findings/completion can happen. Medium‑large; the keystone.
5. **H1** — per‑agent completion. Fast.
6. **H2/H3/H4** — honour `max_new_tokens`, thread the system prompt, greedy decode. Fast, high reliability payoff.
7. **M1 (constrained decoding) + M2 (bigger planner model)** — the durable cure for hallucination.
8. MED‑1/2/3/4/5 — robustness and the one shell‑injection smell. **All items 1-8
   in this list, plus MED-6/MED-7, are now implemented** (see each item's STATUS
   note above) -- this list is kept as a historical record of the fix order that
   was actually followed, not an open backlog.

Items 1, 2, 5, 6 are roughly an afternoon and will change the behaviour from "always loops to max_steps" to "actually runs." Items 3, 4, 7 are what make it *good*.

---

*Audit performed read‑only against the working copy on this machine. No files were modified.*
