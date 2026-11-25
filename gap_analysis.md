# Framework Gap Analysis — No-Holds-Barred Assessment
**Date:** 2026-11-25  
**Source:** full read of work.md (1352 lines) + AGENTS.md

---

## The Core Problem in One Sentence

**The tool has never confirmed a single PROVEN finding in any lab run.** Every run ends in `proven_findings: 0`. After 5+ days of architecture work, 349+ tests passing, and four external repos harvested — the tool cannot do the one thing it exists to do.

This is not a code-quality problem. It is a **system architecture problem** compounded by a **model capability mismatch** that the current design cannot bridge.

---

## Why It's Failing — The Real Root Causes

### 1. The Model Is the Wrong Instrument for the Job

The whole design hinges on `qwen3.6:35b-a3b-q4_K_M` (a 35B MoE model, 3.6B active parameters) reasoning autonomously about novel web vulnerabilities, writing correct exploit scripts on the fly, and self-correcting when probes fail.

Evidence from the runs that **this doesn't work**:

- The model correctly identified SSRF via XXE as the right hypothesis class — then sent `application/x-www-form-urlencoded` to an XML-only endpoint, repeatedly, across **multiple separate lab runs**, even after a prompt fix was added. The fix had zero effect because field descriptions were silently dropped from the prompt (the `_format_params_schema` bug) — but even after that was fixed, the same pattern likely recurs because:
- The 35B-A3B model at 4-bit quant has ~3.6B effective parameters. That's GPT-2-class effective compute. It is not reasoning; it is pattern-matching its training distribution. When it hits a novel error message it hasn't seen in training, it misclassifies it (called `SAXParseException: Entities are not allowed` as evidence of SQLi — these are literally different vulnerability classes).
- The model **flattens verb fields to the top level** instead of nesting them under `parameters` — requiring a forgiving parse hack. It **proposes `modes` as a JSON list** when the schema specifies a string. These are not reasoning failures; they are **small-model JSON schema comprehension failures** that no amount of prompt engineering fully fixes.
- 25 steps is the budget. The model uses ~6 steps on recon, then needs to: form a hypothesis → write a correct exploit script → interpret the response → self-correct the script format → re-test. With a 3.6B-effective-parameter model, this chain breaks at the "write correct exploit script" step almost every time.

**What to do about it:**  
Stop trying to fix model failures with framework scaffolding. The framework is excellent; the model is the bottleneck. Two real options:

**Option A (Low cost, immediate impact):** Route the planning/scripting calls to a larger model via API — Gemini 2.0 Flash or Claude 3.5 Haiku via their APIs cost ~\$0.0001–0.0003 per 1K tokens. A 25-step lab run is ~200K tokens total. That's \$0.02–0.06 per run. The RAG/retrieval infrastructure, the gates, the scope checks — all stay local. Only the LLM call in `generate_raw()` changes. This is a 20-line change to `llm_service.py`.

**Option B (Medium cost, permanent solution):** Fine-tune the existing model on harvested traces of *correct* exploit reasoning. The `ll-finetuning/` stack is already built for this. But you need PROVEN findings first to generate good training data — which you don't have. Chicken-and-egg. Option A must come first to generate the training traces.

---

### 2. The Approval Gate Makes Autonomous Operation Impossible

The human-in-the-loop gate (`isatty()` check in `command_executor.py`) requires a human to physically type `y` at every HIGH-RISK action. Every self_script, race_probe, raw_http, browser_exec call is HIGH-RISK.

In a 25-step lab run, that's 4–8 manual approvals. The human must sit at the terminal for the entire 25-step run (each step takes ~30–90s for the LLM call alone), watching and responding.

This is not "human-in-the-loop." This is "human-IS-the-loop." It makes the tool **unusable in practice** for anyone who isn't staring at a terminal for 20–45 minutes per lab.

**What to do:**  
The design intent is correct — offensive actions need approval. But the implementation is wrong. The right model is:

1. **Pre-approve the run** (not each action): The human says "run this lab, I authorize self_script/race_probe/browser_exec." A single `--auto-approve-class=self_script,race_probe` flag at startup, requiring explicit opt-in and logged to the session JSON.
2. **Post-run review** (not mid-run approval): The agent runs, records every action payload + result, and the human reviews the evidence after. This is how every real automated scanner (Burp Scanner, Nuclei) works.
3. If a mid-run pause is required for a *specific* action (e.g., one that would create an account or trigger an email), the agent pauses at that ONE action and waits. Not every action.

This is not weakening the safety invariant — it's recognizing that "human must approve each of 6 HTTP requests" is not a safety property, it's a usability tax that makes the tool unused.

---

### 3. The Confirm Gap Is Structural, Not Incidental

Look at every single lab run:

| Run | Hypothesis Formed? | Right Class? | Probe Executed? | PROVEN? | Why Not |
|-----|-------------------|--------------|-----------------|---------|---------|
| Race-lab 1 | Yes | Yes (race) | No | No | Structural blindness (pre-redesign) |
| Race-lab 2 | Yes | Yes (race) | Yes (self_script) | No | CSRF regex wrong |
| XXE/SSRF lab 1 | Yes | Yes (SSRF/XXE) | Yes (self_script) | No | Wrong content-type (form vs XML) |
| XXE/SSRF lab 2 | Yes | Yes (SSRF/XXE) | Yes (self_script) | No | Wrong content-type again |
| Scope-broken lab | No | — | No | No | Scope bug (every action rejected) |
| Post-scope-fix lab | Yes | Yes (SSRF/XXE again) | Yes | No | Same wrong content-type again |

The model keeps identifying the right class and then failing to write the correct probe. This is **not a planning problem** — the planning layer works. It's a **code generation problem** at the terminal node.

The self_script approach ("model writes a Python script on the fly") is the universal escape hatch. But it requires the model to:
1. Know the correct request format for the specific endpoint (XML vs JSON vs form)
2. Write syntactically valid Python
3. Use the correct library calls
4. Handle the response correctly
5. Self-correct when it fails

A 3.6B-effective-parameter model at 4-bit quant cannot reliably do all 5. It fails at step 1 (content-type) and never recovers.

**What to do:**

For XXE/SSRF specifically: the probe is not "write a Python script." The probe is deterministic: post `<stockCheck><productId>1</productId><storeId>http://169.254.169.254/latest/meta-data/</storeId></stockCheck>` as `Content-Type: application/xml`. This should be a **typed engine** like `race_probe` and `raw_http`, not a free-form self_script. Add:

- `xml_probe_runner.py` — sends a raw XML body to a target endpoint, checks the response for SSRF indicators (AWS metadata, internal IP patterns, DNS rebind callback)
- `xxe_probe_runner.py` — wraps the XML probe with an external entity payload

Both produce objective verdicts, like `race_probe`. The model doesn't need to write the probe; it just needs to identify the endpoint and the storeId-shaped parameter. That reasoning it CAN do.

**More broadly:** every vuln class that has a deterministic test should have a typed engine. The self_script is a last resort for novel classes, not the primary path. Currently self_script IS the primary path and it fails constantly.

---

### 4. The RAG Is Invisible to the Model That Needs It Most

The corpus has 98 scenario YAMLs, now 8 more from the harvest. The technique advisor retrieves scenarios. But the model that most needs to know "here's the exact payload format for SSRF via XXE" is the one writing the self_script — and it's writing that script from training memory, not from the RAG.

The RAG informs the **planner** (which hypothesis to form). It does not inform the **probe writer** (what exact bytes to send). That's the wrong connection.

**What to do:**  
When a self_script is about to be authored, the technique advisor should retrieve the top scenario for that vuln class and **paste the relevant payload/steps directly into the self_script Field description** that the model sees when writing the code. Currently Field descriptions are finally reaching the model (after the `_format_params_schema` fix), so this is now plumbing-ready. Use it.

---

### 5. 25 Steps Is Too Few for the Model's Latency

A 25-step budget with a 35B model at ~15–30s per LLM call means the full run takes 6–12 minutes of LLM time alone, plus probe execution time. In that budget:

- ~3 steps: initial recon (recon_scan, endpoint_discovery, browser_probe)
- ~1 step: hypothesis formation
- ~4 retries: probe attempts (self_script x4, exhausting budget)
- ~3 steps: stall detection / saturation guard / pivot forced

That's effectively **4 probe attempts per run**, total. If the first probe format is wrong, there's no recovery path — the saturation guard fires before the model can pivot to a different approach.

**What to do:**  
Either increase max_steps to 50 for lab runs (the budget is not a safety gate, it's a cost control), or redesign the retry budget to be **per-endpoint, per-class**, not global — so self_script on `/product/stock` with `vuln_class=ssrf` gets 4 retries, but after switching to `/cart` it gets another 4. This is the "dynamic probe scoping" bug (Bug B in code_audit.md) that was partially addressed but not fully fixed.

---

### 6. Missing Typed Engines Cover the Most Common Lab Classes

The PortSwigger curriculum is 30 classes. The framework has typed engines for:
- SQL injection (sqlmap, sqli_probe)  
- XSS (browser_exec, dalfox)
- Race conditions (race_probe) ✅
- Request smuggling (raw_http) ✅
- Generic HTTP (http_probe, self_script)

**Zero typed engines for:**
- XXE / XML injection (the one class that keeps appearing in live runs)
- SSRF (beyond self_script)
- Path traversal / LFI (Claude-BugHunter has a typed oracle for this)
- JWT attacks
- OAuth flows
- IDOR (needs 2-account test harness)
- Deserialization
- Business logic (needs sequential test with shared state)
- GraphQL introspection + injection
- SSTI

Each missing engine means that vuln class **will never be PROVEN** unless the model writes a perfect self_script on the first try. The campaign-gate target of 70% PortSwigger coverage is impossible without typed engines for the top ~10 classes.

---

### 7. The Tool Layer Is Still Manual to Extend

AGENTS.md says: "Adding a tool must NOT require editing agent code." The current reality (also documented in AGENTS.md): every new engine still requires editing `single_host_mode.py` (registry), `fact_extractors.py` (dispatch), and sometimes `completion.py` (phase map). That's 3 files per engine. The 4-repo harvest plan adds more engines — each one will require these 3 manual edits, increasing the collision risk between concurrent sessions (which already caused the scope.py regression that broke 2 consecutive live runs).

**What to do:**  
Auto-discovery: scan `core/tools/` for files matching `*_runner.py`, read a `TOOL_MANIFEST` dict from each (action_id, params_model, builder fn, extractor fn), and auto-register. This is `tool_manager.py`'s stated job. It needs to actually do it.

---

### 8. The Multi-Agent Coordination Is Creating Bugs

Three separate "sessions" (opus-chat, cc-gemini, claude-sonnet, antigravity) have been simultaneously editing the same files. This directly caused:
- The scope.py regression (the `ins_clean in clean_asset` clause was removed in one session, breaking URL-format asset checking, went undetected for 2 consecutive lab runs — zero observations executed in both)
- The test_audit_fixes.py file being created as 0 bytes (session created the file, then server restarted, content was never written)
- Multiple work.md entries claiming fixes were applied that weren't (the summary said Bug H, I, D were fixed; native inspection showed they were not)

The coordination protocol in work.md is well-designed but insufficient for concurrent file edits. `claim before you build` requires self-discipline that breaks down across model context restarts.

**What to do:**  
One session owns one file at a time. Enforce it via git: **every change is a commit** immediately after the change, not batched. `git status` becomes the single-source-of-truth for what's in progress. Any session that starts an edit and can't finish should commit a WIP commit rather than leaving an in-flight edit that the next session will misread as done or not-started.

---

## Priority Action List (Ranked by Impact on Getting to PROVEN)

**P0 — Prerequisite (1–2 hours):**  
Add `--auto-approve-class` flag to `run_portswigger_lab.py`. Without this, every run still requires a human at the keyboard for 20–45 minutes. Nothing else matters until this is solved.

**P1 — Model swap (2–4 hours, unblocks everything):**  
Add an OpenAI/Gemini API backend option to `llm_service.py`. Use it only for `generate_raw()` calls (the planning and scripting calls). Keep local Ollama for RAG embedding. At \$0.02–0.06/run, this is cheaper than the engineer-hours being spent on prompt fixes for a model that can't execute.

**P2 — XXE/XML typed engine (4–8 hours):**  
`core/tools/xml_probe_runner.py` — the one class that keeps appearing in runs and failing. Takes endpoint + parameter name + test values. Sends proper XML body. Checks for SSRF response signatures. Returns a vuln_signal. This alone would likely PROVEN the last 3 labs.

**P3 — Fix the global retry budget for dynamic probes (2–4 hours):**  
Bug B from code_audit.md. `self_script` exhausts globally after 4 uses. Scope the budget to `(action_id, target_host, vuln_class)` triples. This gives the model a fresh 4-attempt budget when it pivots to a new endpoint, instead of running dry after the first endpoint's attempts.

**P4 — Typed engines for top 5 missing classes (1–2 days):**  
In order of PortSwigger lab frequency: Path traversal/LFI, IDOR (2-account), SSTI, JWT none/confusion, GraphQL introspection. Each is a deterministic test. Each should be an engine, not a self_script.

**P5 — RAG → self_script payload injection (4–8 hours):**  
When the technique advisor retrieves a scenario for the active vuln class, inject the top scenario's `steps` and `tools` fields directly into the `code` field's description that the model sees when writing self_script. The model is already getting the description now (the `_format_params_schema` fix). Use that channel.

**P6 — Increase max_steps to 50 for labs (30 minutes):**  
Not a safety concern. 25 is too few given the model's latency and the number of retries it needs.

**P7 — Auto-discovery for tool registration (1–2 days):**  
Implement `TOOL_MANIFEST` auto-discovery in `tool_manager.py`. Eliminate the 3-file manual edit requirement per new engine. Reduce collision risk.

---

## What Not to Do

**Stop harvesting external repos until the tool produces PROVEN findings.** The 4-repo harvest produced 10 new YAML scenarios and campaign judgment text. These improve the RAG corpus. They do not fix the confirm gap. The model still can't write a correct XML probe. More knowledge content will not change that.

**Stop adding more guards.** The per-class saturation guard, per-focus guard, stall detector — all work correctly. Adding more guards is adding more ceilings to what is already a heavily bounded system. The problem is not that the model is doing too much wrong; it's that it can't reach PROVEN even when it reasons correctly.

**Don't mistake test count for progress.** The suite went from 278 → 349 tests passing. The framework is well-tested. The tool still has 0 PROVEN findings. Tests measure framework correctness; they don't measure whether the tool actually finds vulnerabilities.

---

## The Honest State

| Component | Status |
|-----------|--------|
| Hypothesis formation | ✅ Working — model identifies right vuln class |
| Recon breadth | ✅ Working after ARTEX merge (recon_scan) |
| Scope/safety gates | ✅ Working (after scope.py regression fixed) |
| Blind scrubbing | ✅ Working |
| Field descriptions reaching model | ✅ Fixed (11:17 entry) |
| Human approval flow | ❌ Unusable in practice (45-min TTY babysitting) |
| Probe execution (self_script) | ❌ Fails at correct request format ~100% of the time |
| Typed probe coverage | ❌ Missing for >20 of 30 PortSwigger classes |
| Model capability | ❌ 3.6B effective params can't write correct exploit scripts |
| PROVEN findings | ❌ Zero across all runs ever |

The framework architecture is sound. The inference engine is not. Fix the model (swap or fine-tune with API-generated traces) and add typed engines for the top missing classes. Everything else is secondary.
