# AGENTS.md

Standing instructions for every session in this repo. Read before acting.

## Session start (do this first)
Read `work.md` at the repo root — it is the LIVE ledger: current state, what's
done, what's next, and MUST-PRESERVE invariants. Append a log entry to it after
each meaningful change (format is defined in that file).

## Repo facts (inspected, not guessed)
- **Stack:** Python 3.10+. CLI orchestrator (no web framework). AI/ML: PyTorch +
  PEFT (LoRA/QLoRA), FAISS RAG; Playwright (headless Chromium) for browser probes.
- **Architecture (4 layers):** Orchestrator `main.py` -> domain Agents (`agents/`)
  -> deterministic phase scripts (`agents/<agent>/phases/`) -> AI consultant
  (RAG/ChatAssistant, `ll-finetuning/rag/`). The AI never executes; every action
  passes a 10-gate safety pipeline.

### Directory layout
- `core/` — deterministic reasoning + safety: `action_contract.py`, `ai_planner.py`,
  `investigation.py` (Observation->Hypothesis->Validation->Finding), `scope.py` (scope
  gate), `tool_manager.py` (runtime tool discovery), `stealth.py`,
  `technique_advisor.py`, `agent_router.py`, `os_detector.py`.
  `core/tools/` — detection engines/runners: `browser_probe`, `browser_exec_runner`,
  `raw_http_runner`, `race_probe_runner`, `nuclei_template_runner`, `http_probe`,
  `script_runner`.
- `agents/` — `base_agent.py` (lifecycle, approval gates, autonomous loop) + 9 domain
  agents (bug_bounty, red_team, pentest, ctf, mobile, firewall_bypass,
  security_research, active_directory, cloud), each with its own `phases/`.
  `agents/common/` — shared primitives + model-authored action modules
  (`*_authoring.py`, `self_scripting.py`).
- `ll-finetuning/` — RAG corpus + LoRA/QLoRA training stack. `ll-finetuning/rag/core/`
  is merged into the `core.*` / `rag.*` import namespace; root `rag` is a symlink ->
  `ll-finetuning/rag`.
- `tests/` — pytest suite (40 files). Also `data/`, `eval/`, `scripts/`,
  `lora-adapters/`, `lora-output/`.

### Entry points
- `main.py` — interactive orchestrator:
  `python main.py --agent <ctf|pentest|bug_bounty|...> --autohunt --target <t>`,
  `--mode manual`, `--resume <session_id>`.
- `run_portswigger_lab.py <lab_url> [--blind] [--max-steps 25]` — single lab harness.
- `run_portswigger_batch.py` (+ `labs.txt`) — batch lab runner.
- `run_target.py` — authorized real-target harness (stealth on, scanners off, human
  approval enforced).
- `train_qlora.py` — LoRA/QLoRA training. Utilities: `add_scenarios.py`,
  `fix_jsonl*.py`, `verify_*_smoke_test.py`.

### Commands
- **Install:** `python -m venv .venv && . .venv/bin/activate`, then
  `pip install -r requirements.txt` (pulls `ll-finetuning/requirements.txt`).
  Dev tooling: `pip install -r requirements-dev.txt`.
  OS-specific setup under `ll-finetuning/setup/{ubuntu,windows11}/`.
  Browser probes need `playwright install chromium`.
  Autonomous model-authored probes need `export PENTEST_ENABLE_SELF_SCRIPTING=1`.
- **Run tests:** `python -m pytest` (pytest.ini sets `pythonpath=.`, `testpaths=tests`).
  Narrowest: `python -m pytest tests/<file>::<test>`.
- **Lint / format:** `ruff check .` (lint) and `ruff format .` (format). Config in
  `ruff.toml` (target py310, line-length 100, conservative E/F/W/I rules).

## What this is
- A multi-agent autonomous cybersecurity assessment / adversary-emulation
  framework. Authorized, in-scope targets ONLY.
- Has a fully autonomous mode and an AI-assisted manual mode.

## Hard rules
- Scope gate: every tool run must be checked against an allowed-targets list
  before executing. Refuse any out-of-scope target.
- Never expose secrets or commit them. Never add dependencies without reason.

## Model backend
- LLM backend is a local Ollama model reached over Tailscale.
- Keep the model name and base URL as config values, not hardcoded, so I can
  swap models (including an abliterated build) without code changes.
- Implemented in `ll-finetuning/rag/core/services/llm_service.py` via env vars:
  `OLLAMA_BASE_URL`, `OLLAMA_MODEL` (default `qwen3.6:35b-a3b`), plus
  `OLLAMA_NUM_CTX` / `OLLAMA_TIMEOUT` / `OLLAMA_THINK` / `OLLAMA_NUM_PREDICT`.
  Already config-driven — keep it that way.
- REQUIRED before any run/lab: set `OLLAMA_BASE_URL` and `OLLAMA_MODEL`. The URL is the
  Ollama host — from Windows ON the Ollama machine use `http://localhost:11434`; from WSL
  use the Windows-host gateway, e.g. `http://172.28.128.1:11434`. On Windows set env with
  `setx NAME "value"` then restart the shell (not `export`). If `OLLAMA_BASE_URL` is UNSET it
  silently falls back to the CPU `google/flan-t5-large` model (useless — any run is
  meaningless). Verify the startup line reads `(ollama @ ...)`, NOT `(cpu): google/flan-t5-large`.

## Tool layer
- Security tools (nmap, nuclei, etc.) are wrapped as hot-swappable tools the
  agent discovers at runtime. Adding a tool must NOT require editing agent code.
- Current reality (converge toward the rule): runtime discovery is
  `core/tool_manager.py` (`ToolManager`, `TOOL_PACKAGE_MAP`); engines live in
  `core/tools/`. A model-authored detection action today still also needs a
  `registry.register(...)` in the agent's registry builder (e.g.
  `agents/bug_bounty/single_host_mode.py`) and a fact-extractor dispatch entry.

## Working style
- Inspect before editing. Prefer targeted diffs over broad rewrites.
- Verify before claiming done: run the check and show output, never assume.
- Run the narrowest relevant test first; report modified files + results.

## Token discipline
- Delegate broad or multi-file searches to subagents; keep the conclusion,
  not the file dumps.
- Read only the needed slice of large files, not the whole thing.
- Answer first, no filler preamble, no restating the question.
