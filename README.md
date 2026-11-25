# 🔒 Multi-Agent Security Assessment & Adversary Emulation Framework (v5.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/Architecture-Autonomous%20AI%20Security%20Researcher-orange.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.1-red.svg)](https://pytorch.org/)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-green.svg)](https://github.com/huggingface/peft)
[![Tests](https://img.shields.io/badge/Tests-110%2F118%20Passing%20(8%20skipped)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

An enterprise-grade **Autonomous AI Security Assessment Framework**. Evolved from a workflow runner into a security reasoning system that discovers vulnerabilities through **observation → hypothesis → validation → finding** — with a 10-gate safety pipeline enforced on every action.

---

## 🏛️ The Four-Layer Architecture

The framework enforces strict separation between deterministic execution and AI reasoning:

```text
Layer 1: System Orchestrator (main.py)
   ▶ OS Detection (Windows, Linux Debian/Ubuntu, macOS)
   ▶ Pre-flight Tool Verification & Package Manager Installation (winget, apt, brew)
   ▶ Interactive Agent Selection (9 Specialized Agents)
   ▶ Scope & Safety Boundary Interview (in-scope, out-of-scope, prohibited tests)
   ▶ Mode Handoff (Manual Co-Pilot / Automated Sequential / Autonomous Hunter)
          ▼
Layer 2: Specialized Agent Classes (agents/)
   ▶ BaseAgent: lifecycle, approval gates, audit trails, Ctrl+C save/resume
   ▶ BaseAgent.run_autonomous(): shared autonomous hunter loop for all agents
   ▶ 9 Domain Agents: Bug Bounty, Red Team, Pentest, CTF, Mobile,
       Firewall Bypass, Security Research, Active Directory, Cloud Security
          ▼
Layer 3: Deterministic Phase Scripts (agents/<agent>/phases/)
   ▶ Deterministic modules: name, requires_tools, run(), parse_output(), next_hint()
   ▶ Safe by default: read_only=True unless active testing explicitly authorized
          ▼
Layer 4: AI Reasoning Consultant (ChatAssistant & RAG/YAML Stack)
   ▶ Consultant, NOT Driver: generates text, analysis, decisions — never executes
   ▶ AIPlanner wraps ChatAssistant to emit structured JSON action proposals
   ▶ All AI output validated by ActionRegistry before execution (Gate 1)
```

---

## 🤖 Autonomous Hunter Loop (`--autohunt`)

The autonomous mode upgrades agents from sequential phase runners into reasoning systems that **select, execute, and chain actions** based on live evidence — with every action passing through 10 mandatory safety gates.

### Quick Start

```powershell
# Windows — autonomous CTF hunt
python main.py --agent ctf --autohunt --target 10.10.10.5

# Resume an interrupted assessment
python main.py --resume ctf_1730500000

# Manual co-pilot mode (unchanged)
python main.py --agent pentest --mode manual --target 10.10.10.5
```

### The 10-Gate Safety Pipeline

Every AI-proposed action traverses these gates **in order** before execution:

| Gate | Check | Failure Action |
|------|-------|---------------|
| 1 | Registry validation — action exists, params match typed model | Reject |
| 2 | Target format + scope validation (structural, not substring) | Reject |
| 3 | Policy validation (network_effect, state_change, requires_active_testing) | Reject |
| 4 | Prohibited action gate (name blocklist) | Reject |
| 5 | Time window gate (within authorized window) | Defer |
| 6 | Retry budget gate (per action-identity, framework-enforced) | Reject |
| 7 | Command build (deterministic pure-Python builder, no AI, no shell strings) | Error |
| 8 | Final argv safety check (no shell metacharacters) | Reject |
| 9 | Approval gate (manual Y/N OR autonomous policy) | Hold/Reject |
| 10 | Execution (subprocess, shell=False, timeout, captured output) | — |

### Evidence Chain

The framework enforces an evidence chain. A **Finding** cannot be recorded without:

```text
OBSERVATION (tool produced output)
    └─► HYPOTHESIS (output may indicate vulnerability)
            └─► VALIDATION (targeted test to confirm/refute)
                    └─► FINDING (confirmed — evidence chain complete)
                            └─► IMPACT (consequence demonstrated)
```

`EvidenceStore.record_finding()` **refuses** findings that skip any step in this chain. This is framework-enforced, not AI-enforced.

### ActionRegistry Pattern

Each agent exposes a typed action registry. The AI proposes action names + params; the registry validates before any builder or executor is called:

```python
# agents/ctf/actions.py
from core.action_contract import ActionDefinition, ActionRegistry

SCAN_PORTS = ActionDefinition(
    action_id="scan_ports",
    phase="recon",
    description="TCP SYN scan target for open ports",
    params_model=ScanPortsParams,        # Pydantic model, extra="forbid"
    network_effect="active",             # "none" | "read_only" | "active"
    state_change="none",                 # "none" | "target_modify" | "credential_access"
    risk_level="medium",                 # "low" | "medium" | "high"
    requires_active_testing=True,
    command_builder="build_scan_ports_argv",
)

def build_ctf_registry() -> ActionRegistry:
    registry = ActionRegistry(agent_name="ctf")
    registry.register(SCAN_PORTS, builder=build_scan_ports_argv)
    ...
    return registry
```

```python
# agents/ctf/command_builders.py — pure Python, no AI, no shell strings
def build_scan_ports(params: ScanPortsParams) -> List[List[str]]:
    return [["nmap", "-sV", "-p", params.ports, params.target]]
```

---

## 🕵️ The 9 Specialized Assessment Agents

| Agent | Directory | Methodology Workflow | Primary Scenarios |
| :--- | :--- | :--- | :--- |
| **Bug Bounty Agent** | `agents/bug_bounty/` | Recon → Subdomain Enum → Endpoint Discovery → Param Fuzzing → Vuln Validation | Web apps, APIs, CORS |
| **Red Team Agent** | `agents/red_team/` | Recon → Initial Access → Privilege Escalation → Lateral Movement → Objective Impact | MITRE ATT&CK |
| **Pentest Agent** | `agents/pentest/` | Host Discovery → Service Enum → Safe Exploit Probes → Post-Exploitation → Audit Report | Formal PTES |
| **CTF / Wargame Agent** | `agents/ctf/` | Environment Recon → Local Enumeration → Privesc → Flag Capture | Linux shells, HTB, OverTheWire |
| **Mobile App Agent** | `agents/mobile/` | APK/IPA Static → Dynamic Runtime → Auth Tests → Local Storage → Network Traffic | Android/iOS, TLS pinning |
| **Firewall Bypass Agent** | `agents/firewall_bypass/` | WAF Fingerprint → Rule Mapping → Bypass & Mutation Techniques | Cloudflare, AWS WAF |
| **Security Research Agent** | `agents/security_research/` | CVE Analysis → PoC Dev → Root Cause Diffing → Writeup Synthesis | 0-day / N-day research |
| **Active Directory Agent** | `agents/active_directory/` | Domain Enum → Kerberoasting → AS-REP Roasting → DCSync Audit → Tier-0 Impact | Enterprise AD |
| **Cloud Security Agent** | `agents/cloud/` | IAM Policy Audit → Resource Inventory → Misconfiguration Validation | AWS, Azure, GCP |

---

## ⚡ Quickstart Guide

### 1. Setup & Environment

```powershell
# Windows — using uv (recommended)
uv venv .venv
uv pip install -r ll-finetuning/setup/windows11/requirements.txt
uv pip install pydantic>=2.0
.venv\Scripts\activate
```

```bash
# Linux (Debian / Ubuntu)
python3 -m venv .venv
source .venv/bin/activate
pip install -r ll-finetuning/setup/ubuntu/requirements.txt
pip install pydantic>=2.0
```

> **Windows symlink note:** `rag` at the repo root is a real directory symlink to
> `ll-finetuning/rag`, which several modules (`core/scope.py`, `core/os_detector.py`,
> `core/tool_manager.py`, `core/agent_router.py`, `core/constrained_decoding.py`, ...)
> import through. Git on Windows only creates a *real* symlink when
> `git config core.symlinks true` is set **before** cloning *and* the cloning user
> has the "Create symbolic links" privilege (Developer Mode, or an elevated/admin
> shell) -- otherwise `git clone` silently writes `rag` as a plain text file
> containing the link target path, and every `from rag.core... import ...` (and the
> `core/__init__.py::__path__.append()` merge it depends on) will fail immediately.
> If imports fail right after a fresh clone, check `git config core.symlinks` and
> whether `rag` is actually a directory (`Get-Item rag | Select LinkType`) rather
> than a text file, and re-clone with symlinks enabled if not.

### 2. Run the Full Test Suite

```bash
# Run the full suite (122 test functions across 16 files; a handful are skipped
# automatically where they need a loaded LLM)
pytest tests/ -v

# Just the two original acceptance-criteria files (24 + 15 = 39 tests)
pytest tests/test_autonomous_framework.py tests/test_multi_agent_framework.py -v

# Fast unit tests only (no GPU/LLM load)
pytest tests/test_autonomous_framework.py -v

# LLM-heavy regression suite (loads TinyLlama on GPU, ~3 min)
pytest tests/test_multi_agent_framework.py -v
```

### 3. Run the Evaluation Benchmark

```bash
# Run benchmark (13 scenarios across all 9 agents, no LLM load required)
python eval/run_benchmark.py

# Run and automatically compare against previous baseline
python eval/run_benchmark.py --compare

# Compare two specific benchmark runs across git commits
python eval/compare_commits.py --baseline eval/results/benchmark_A.json --current eval/results/benchmark_B.json

# Generate markdown report from latest results
python eval/report_benchmark.py
```

Expected output:
```text
Scenarios: 13/13   Discovery: 100%   Chain Integrity: 100%   Safety Violations: 0
```

### 4. Command-Line Options

```text
options:
  -h, --help            Show this help message and exit
  --resume RESUME       Resume an existing assessment session by name
  --agent AGENT         Select agent by number (1-9) or key (pentest, ctf, cloud, ...)
  --mode {manual,automated}
                        Execution mode
  --target TARGET       Target identifier (IP, domain, URL, CIDR)
  --autohunt            Enable autonomous hunter loop (requires --agent and --target)
```

### 5. Examples

```powershell
# Autonomous CTF hunt
python main.py --agent ctf --autohunt --target 10.10.10.5

# Manual penetration test co-pilot
python main.py --agent pentest --mode manual --target 10.10.10.5

# Automated bug bounty run
python main.py --agent bug_bounty --mode automated --target target.corp

# Resume an interrupted assessment
python main.py --resume pentest_1730500000
```

---

## 🛡️ Safety & Scope Gates

Before any active test executes:

1. **Out-of-Scope Precedence**: Any asset in `out_of_scope` immediately blocks execution, even if covered by an `in_scope` CIDR.
2. **Prohibited Tools & Tests**: Banned tools or techniques are blocked by deterministic gate checks before any execution.
3. **Read-Only Default**: All phases execute with `read_only=True` unless `active_network_allowed=True` in scope.
4. **Evidence Chain Integrity**: Findings require a complete `Observation → Hypothesis → Validation` chain — framework-verified.
5. **Retry Budget**: Each action-identity has a fixed retry budget. Exhausted budget blocks further attempts — no infinite loops.
6. **Audit Trail**: Every action, command, stdout, stderr, and timestamp is stored in `.assessments/<session>/artifacts/`.
7. **Ctrl+C Graceful Save**: Pressing `Ctrl+C` cleanly persists all session state and `EvidenceStore` to disk.

---

## 📁 Project Structure

```text
ll-fintuning/
├── main.py                          # System orchestrator (--autohunt flag)
├── agents/
│   ├── base_agent.py                # BaseAgent + run_autonomous() loop
│   ├── ctf/
│   │   ├── agent.py                 # CTFAgent + run_autonomous_default()
│   │   ├── actions.py               # CTF ActionRegistry (10 actions)
│   │   ├── command_builders.py      # Pure Python argv builders
│   │   ├── completion.py            # CTF_COMPLETION + verify_completion()
│   │   └── phases/                  # Deterministic phase scripts
│   ├── pentest/, red_team/, bug_bounty/, mobile/,
│   │   firewall_bypass/, security_research/,
│   │   active_directory/, cloud/    # Phase-based agents
├── core/
│   ├── action_contract.py           # ActionDefinition, AIActionProposal, ActionRegistry
│   ├── investigation.py             # EvidenceStore, Observation, Hypothesis, Validation, Finding
│   ├── ai_planner.py                # AIPlanner (structured JSON decisions)
│   ├── command_executor.py          # 10-gate executor (shell=False, argv vectors)
│   ├── session_manager.py           # State + EvidenceStore persistence
│   ├── model_router.py              # CPU/GPU model selection + LoRA adapter loading
│   └── ...
├── eval/
│   ├── scenarios/                   # 10 YAML benchmark scenarios
│   ├── run_benchmark.py             # Benchmark runner
│   └── report_benchmark.py          # Markdown report generator
├── tests/
│   ├── test_autonomous_framework.py # 12 autonomous-mode tests
│   └── test_multi_agent_framework.py# 15 regression tests
├── ll-finetuning/
│   ├── rag/core/                    # Original RAG + LLM core (ChatAssistant, RAG engine)
│   ├── scenarios/                   # 200+ YAML assessment playbooks
│   ├── corpus/                      # OWASP, PortSwigger, textbook chunks
│   └── SYSTEM_DOCUMENTATION.md     # Full architectural specification
└── scripts/                         # One-time patch scripts (idempotent, kept for reference)
```

---

## 🔬 Core Module Reference

| Module | Key Classes / Functions | Purpose |
|--------|------------------------|---------|
| `core/action_contract.py` | `ActionDefinition`, `AIActionProposal`, `ActionRegistry` | Typed action registry, proposal validation |
| `core/investigation.py` | `EvidenceStore`, `Observation`, `Hypothesis`, `Validation`, `Finding` | Evidence chain enforcement |
| `core/ai_planner.py` | `AIPlanner.decide_next()` | Wraps ChatAssistant → structured JSON decisions |
| `core/command_executor.py` | `execute_action()`, `execute_command()` | 10-gate executor, shell=False always |
| `core/session_manager.py` | `save_investigation_store()`, `load_investigation_store()`, `append_autonomous_history()` | Persistent state + evidence |
| `agents/base_agent.py` | `BaseAgent.run_autonomous()` | Shared autonomous hunter loop |
| `agents/ctf/actions.py` | `build_ctf_registry()` | 10 CTF ActionDefinitions |
| `agents/ctf/command_builders.py` | `build_*()` functions | Pure Python argv builders |
| `agents/ctf/completion.py` | `verify_completion()`, `CTF_COMPLETION` | Framework-verified completion criteria |

---

## 🧪 Test Coverage

| Test Suite | Count | Covers |
|------------|-------|--------|
| `test_autonomous_framework.py` | 24 | Scope gates, policy gates, retry budget, evidence chain, completion criteria, injection safety, mode isolation |
| `test_multi_agent_framework.py` | 15 | OS detection, agent instantiation, scope validation, command executor safety, AI reasoning points, audit trail (8 of these are skipped without a loaded LLM) |
| `test_med3_json_extraction.py` | 12 | Planner JSON extraction: nested objects, markdown fences, annotation-stripping repairs |
| `test_remediation_c1_decision_verbs.py` | 11 | `form_hypothesis` / `record_validation` / `record_finding` decision verbs, fabricated-id rejection |
| `test_med4_target_kind.py` | 8 | Target-kind classification and `allowed_target_kinds` enforcement (Gate 2) |
| `test_remediation_c1_c2_h1.py` | 7 | Retry-budget wiring, evidence promotion, per-agent completion verifier resolution |
| `test_m1_constrained_decoding.py` | 7 | action_id-constrained decoding, including the post-closing-quote delimiter guard |
| `test_resume_scope_persistence.py` | 5 | `--resume` round-trips the full `Scope`, not just the target |
| `test_qwen_everywhere.py` | 5 | Qwen2.5-7B-Instruct + local LoRA adapter auto-discovery, 4-bit NF4 path |
| `test_model_router_system_prompt.py` | 5 | System prompt threaded as the real chat-template system role, per model family |
| `test_med5_path_hijack.py` | 5 | `path_hijack_validation` argv safety (no shell strings, no path/command injection) |
| `test_min_new_tokens_floor.py` | 4 | `min_new_tokens` floor prevents premature EOS truncation of planner JSON |
| `test_m2_planner_model_override.py` | 4 | Planner-role model override / fallback-to-default caching |
| `test_exhausted_action_filtering.py` | 4 | Exhausted actions excluded from the planner's REGISTERED ACTIONS catalog |
| `test_dynamic_output_example.py` | 4 | OUTPUT FORMAT example rebuilt from a real, non-exhausted action each call |
| `test_med1_med2.py` | 2 | Consecutive-planner-failure abort, no-progress stall detector |
| **Total** | **122 test functions across 16 files** | **Run with `pytest tests/ -v`; a handful are skipped automatically where they need a loaded LLM** |

---

## ⚠️ Disclaimer

This framework and its underlying fine-tuned models are designed strictly for **authorized security auditing, educational research, and defensive threat modeling**. Testing targets without explicit written authorization is illegal. Always obtain proper authorization before running any assessment.

---

## 🤖 Model Configuration

| Setting | Value |
|---------|-------|
| GPU Model | `Qwen/Qwen2.5-7B-Instruct`, loaded in 4-bit NF4 (bitsandbytes QLoRA) |
| LoRA Adapter | Local fine-tune: `lora-adapters/qwen-qlora/` (falls back to the HF repo `DarkLight6996/CyberSec-Qwen-V1` if not present locally), auto-discovered and attached via PEFT |
| CPU Model | `google/flan-t5-large` |
| VRAM Required | ~5-6GB in 4-bit NF4 (fits an 8GB RTX 3060 Ti) |

> **Note:** Qwen2.5-7B-Instruct does **not** OOM on an 8GB card -- the earlier
> assumption in this section was wrong. Loading full-precision/fp16 would (~14GB),
> but `GPU_FORCE_4BIT=True` in `core/config.py` skips the fp16 attempt entirely and
> loads straight into 4-bit NF4, matching the quantization the LoRA adapter was
> trained with. Confirmed working end-to-end via live GPU smoke testing
> (`verify_autohunt_smoke_test.py`) on this exact hardware.
