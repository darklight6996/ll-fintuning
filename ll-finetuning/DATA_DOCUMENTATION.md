# 🛡️ Cybersecurity Instruction Dataset & RAG Data Documentation

## Overview

This document provides a comprehensive technical reference for all training datasets, format conversions, RAG knowledge sources, session state schemas, and fine-tuning pipelines within the cybersecurity assistant platform.

The system supports dual operation:
1. **Offline / Fast Baseline**: `DarkLight6996/CyberSec-Model-V2` (1.1B LoRA adapter on `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
2. **Next-Generation SFT / QLoRA**: `DarkLight6996/CyberSec-Qwen-V1` (7B parameter adapter on `Qwen/Qwen2.5-7B-Instruct` using native ChatML token framing).

---

## 📊 Dataset Inventory & File Registry

The `data/` directory houses specialized instruction-tuning corpora, multi-scenario datasets, and parsed web security writeups:

| File Name | Size | Primary Domain | Format | Description |
| :--- | :--- | :--- | :--- | :--- |
| `comprehensive_cybersec_finetune.jsonl` | ~195 KB | Multi-Domain Cyber & Compliance | Alpaca JSONL | Primary multi-turn dataset covering Linux privesc, AD, Cloud, PCI DSS, NIST risk assessments, and reporting. |
| `qwen_chatml.jsonl` *(generated)* | Dynamic | Instruction SFT | Qwen ChatML | SFT training dataset serialized with `<\|im_start\|>` and `<\|im_end\|>` tags for direct ingestion by `SFTTrainer`. |
| `redefining_hacking.jsonl` | ~676 KB | Offensive Operations & Methodology | Alpaca JSONL | In-depth technical exploitation workflows, custom payload generation, and Linux privilege escalation playbooks. |
| `structured_data.jsonl` | ~93 KB | Command Reference & Recon | Alpaca JSONL | Granular CLI command-and-flag pairs for Nmap, Burp Suite, SQLMap, FFUF, and NetExec. |
| `owasp_alpaca.jsonl` | ~252 KB | Web Application Security | Alpaca JSONL | OWASP Top 10 vulnerabilities, API security assessments, injection vectors, and mitigation rules. |
| `portswigger_alpaca.jsonl` | ~71 KB | Web Exploitation Labs | Alpaca JSONL | PortSwigger Web Security Academy scenarios, business logic flaws, SSRF, and authentication bypasses. |
| `data.jsonl` | ~69 KB | Baseline Instruction Pairs | Alpaca JSONL | Core baseline penetration testing query-response instructions. |
| `htb.jsonl` | ~27.8 MB | CTF & Machine Writeups | Alpaca / Text JSONL | Extensive HackTheBox machine writeups and walkthroughs detailing end-to-end user-to-root exploitation chains. |

---

## 🔄 Data Schemas & Format Evolution

### 1. Stanford Alpaca Format (Legacy / Input Corpus)
Each record in the raw training corpora is structured as an instruction-input-output triplet:

```json
{
  "instruction": "A question or scenario describing what needs to be accomplished.",
  "input": "Context, environmental details, or specific targets related to the instruction (optional).",
  "output": "The appropriate command(s) to execute, technical analysis, and expected outcome."
}
```

#### Example Entry (PCI DSS Compliance):
```json
{
  "instruction": "Explain how to conduct a compliance assessment for PCI DSS requirements in a payment card environment.",
  "input": "You are assessing an organization that processes credit card payments. Describe the systematic approach to validate PCI DSS v4.0 compliance across all 12 requirements.",
  "output": "PCI DSS v4.0 Compliance Assessment Methodology:\n\n**Phase 1: Scope Definition (Requirement 1)**\n- Identify all system components in Cardholder Data Environment (CDE)\n- Network segmentation validation: `nmap -sS -p- <CDE_RANGE>`\n..."
}
```

---

### 2. Qwen ChatML Format (Target SFT Training Format)
Modern instruction models such as `Qwen/Qwen2.5-7B-Instruct` require token-level role demarcations (`<|im_start|>` and `<|im_end|>`). 

The serialized ChatML format produced by `scripts/alpaca_to_chatml.py` wraps each dialogue into a single `"text"` string:

```json
{
  "text": "<|im_start|>system\nYou are a cybersecurity and penetration testing expert.<|im_end|>\n<|im_start|>user\nExplain how to conduct a compliance assessment for PCI DSS requirements...\n\nContext:\nYou are assessing an organization that processes credit card payments...<|im_end|>\n<|im_start|>assistant\nPCI DSS v4.0 Compliance Assessment Methodology:\n...<|im_end|>\n"
}
```

#### Conversion Script Pipeline (`scripts/alpaca_to_chatml.py`):
```python
import json
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def alpaca_to_chatml(record):
    user_content = record["instruction"].strip()
    if record.get("input") and record["input"].strip():
        user_content += f"\n\nContext:\n{record['input'].strip()}"
        
    messages = [
        {"role": "system", "content": "You are an expert penetration-testing mentor inside a live operator console."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": record["output"].strip()}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)
```

---

## 🎯 Content Categories & Coverage

The training and scenario datasets span the full offensive security, compliance, and systems auditing lifecycle:

### 1. Web Application & API Security
- **Reconnaissance & Enumeration**: Subdomain discovery (`subfinder`, `amass`), directory fuzzing (`ffuf`, `gobuster`), technology stack fingerprinting (`whatweb`, `httpx`).
- **Exploitation**: Parameterized SQLi (`sqlmap`), Cross-Site Scripting (XSS), Server-Side Request Forgery (SSRF), IDOR, and JWT token tampering.
- **WAF & Defensive Evasion**: Cloudflare, AWS WAF, ModSecurity fingerprinting, and rule bypass payloads (OWASP CRS).

### 2. Active Directory & Windows Domain Escalation
- **Kerberos Attacks**: AS-REP Roasting (`GetNPUsers.py`), Kerberoasting (`GetUserSPNs.py`), and Silver/Golden ticket forgery.
- **AD Infrastructure**: SMB null/guest share enumeration (`netexec`), LDAP object queries, ADCS certificate template abuse (`certipy` ESC1-ESC16).
- **Lateral Movement & Credential Dumping**: DCSync (`secretsdump.py`), Pass-the-Hash (`evil-winrm`, `psexec`), and LSASS memory extraction.

### 3. Linux Exploitation, CTF & Privilege Escalation
- **SUID / SGID Audit**: Unqualified relative path execution ($PATH variable hijacking), custom `/challenge/` binaries, POSIX capability inspection (`getcap`).
- **Access Control & Sudo**: Abusing `sudo -l` rules with GTFOBins, writable cron jobs, and wildcards in shell scripts.
- **Low-Level Exploit Development**: Stack-based buffer overflows, ROP chain construction (x64 ret2libc / execve syscalls), NX bypass, and bad character filtering.

### 4. Cloud Security & Container Auditing
- **Azure & AWS**: Azure RBAC role auditing via `az` CLI, Key Vault secret inspection, AWS IMDSv1/v2 SSRF abuse, IAM policy privilege escalation.
- **Containers**: Docker socket escape, privilege container breakouts (`--privileged`, `nsenter`), Kubernetes RBAC and service account secret auditing.

### 5. Enterprise Compliance, Risk Management & Reporting
- **PCI DSS v4.0**: Validation across all 12 operational requirements (CDE scope, network segmentation, encryption, access control, log retention).
- **Risk Assessment (NIST SP 800-30 / ISO 27005)**: Likelihood vs. impact evaluation, threat actor profiling, CVSS 4.0 scoring, and asset valuation.
- **Professional Reporting**: Executive summaries, technical finding templates (Root Cause, Evidence, Impact, Remediation Roadmap), and CVSS 4.0 vectors.

---

## ⚙️ Model Architecture Evolution: TinyLlama to Qwen2.5-7B

### 1. Rationale for Transitioning from TinyLlama-1.1B to Qwen2.5-7B-Instruct
The initial baseline relied on `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (`DarkLight6996/CyberSec-Model-V2`). While TinyLlama offered an ultra-fast proof-of-concept for verifying LoRA pipelines with minimal VRAM (~2.2 GB), it exhibited core architectural limitations when deployed in autonomous security operations:
* **Reasoning Depth & Tool Syntax**: Complex cybersecurity tasks (constructing ROP chains, auditing ADCS certificate templates, multi-stage Linux privilege escalation, parsing messy nmap/ffuf outputs) require deep multi-step causal reasoning. 1.1B models frequently hallucinated syntax or dropped necessary flags.
* **Context Retention & Instruction Following**: TinyLlama struggled to adhere simultaneously to strict system personas, live environment state (`whoami`, `pwd`, SUID binaries), untrusted evidence blocks, and FAISS RAG context.
* **Why Qwen2.5-7B-Instruct?**: Qwen2.5-7B is recognized as a premier open-weight model for coding, mathematical logic, structured JSON outputs, and tool-use reasoning. Through 4-bit NF4 QLoRA quantization, the entire 7B base model fits into ~4.5 GB of VRAM, allowing enterprise-grade reasoning directly on consumer GPUs (RTX 3060 Ti 8GB).

---

## 📜 Training & Utility Script Inventory

| Script Path | Purpose | Key Technologies |
| :--- | :--- | :--- |
| `train_qlora.py` *(Root)* | Primary QLoRA fine-tuning for `Qwen/Qwen2.5-7B-Instruct` on Windows with hardened memory caps and power fail-safes. | `transformers 5.17`, `trl 1.13`, `peft 0.21`, `bitsandbytes` 4-bit NF4, `bfloat16`. |
| `ll-finetuning/scripts/train_lora.py` | Legacy GPU training script for TinyLlama 1.1B baseline adapter. | `peft`, `transformers`, FP16 LoRA. |
| `ll-finetuning/scripts/train_lora_cpu.py` | Deterministic fallback CPU training pipeline (single precision FP32) for low-resource environments. | `torch.float32`, CPU execution, fixed padding. |
| `ll-finetuning/scripts/infer_lora.py` | Local CLI inference test script for validating LoRA adapters. | `peft`, `AutoModelForCausalLM`. |
| `ll-finetuning/scripts/upload_to_hf.py` | Automated deployment script to push trained adapters and tokenizer configs to Hugging Face Hub. | `huggingface_hub`, `HfApi`. |
| `scripts/alpaca_to_chatml.py` | Converts legacy 3-field Alpaca JSONL datasets into native Qwen ChatML format (`<\|im_start\|>`/`<\|im_end\|>`). | `transformers.AutoTokenizer.apply_chat_template`. |
| `scripts/verify_bnb.py` | Day 1 Go/No-Go smoke test validating `bitsandbytes` `Linear4bit` CUDA execution on Windows. | `bitsandbytes.nn.Linear4bit`, PyTorch CUDA. |
| `scripts/verify_tokenizer.py` | Verifies ChatML template serialization and special token formatting. | `transformers.AutoTokenizer`. |

---

## 🔄 Format Transition: Stanford Alpaca to Native Qwen ChatML

### 1. Why the Transition Was Necessary
* **Role Ambiguity in Alpaca**: Standard Alpaca format (`{"instruction", "input", "output"}`) flattens queries into monolithic text. The model cannot structurally distinguish between immutable system security instructions, operator prompts, untrusted scanner evidence, and assistant responses.
* **Special Token Demarcation**: ChatML uses hard boundary tokens (`<|im_start|>role\n...<|im_end|>`) natively trained into Qwen2.5. This strictly isolates system rules from user input, eliminating prompt injection risks from raw terminal evidence.
* **Multi-Turn Readiness**: ChatML natively supports arbitrary conversation turns, preserving conversation history cleanly for iterative CTF and assessment sessions.

### 2. How the Transformation Works (`scripts/alpaca_to_chatml.py`)
Each Alpaca record is extracted, combining `instruction` and optional `input` context into a structured user message, and mapped into standard Hugging Face conversation dictionaries:
```python
messages = [
    {"role": "system", "content": "You are a cybersecurity and penetration testing expert."},
    {"role": "user", "content": f"{instruction}\n\nContext:\n{input_text}" if input_text else instruction},
    {"role": "assistant", "content": output_text}
]
chatml_text = tokenizer.apply_chat_template(messages, tokenize=False)
```

---

## ⚡ Hardware Safety, Crash Root-Cause Analysis & Fail-Safe Engineering

### 1. Root Cause Analysis: Why Hardware Crashes Occur During 7B Training
During unhardened QLoRA training on 8GB GPUs (e.g. RTX 3060 Ti), systems can experience sudden hard shutdowns, power supply trips, or physical socket overheating due to three compounding factors:

1. **The Step-50 Evaluation Activation Surge**:
   - In standard Hugging Face `SFTTrainer` configurations with `eval_strategy="steps", eval_steps=50`, gradient checkpointing is automatically turned off during the evaluation loop over the test dataset (e.g. 41 samples).
   - Computing forward-pass activations across maximum sequence lengths without gradient checkpointing creates a sudden multi-gigabyte VRAM spike.
2. **LoRA Parameter Oversubscription**:
   - Targeting all 7 linear projection matrices (`q, k, v, o, gate, up, down`) inflates trainable gradients and optimizer states.
   - On Windows, where the Desktop Window Manager (WDDM) consumes ~1.4 GB of VRAM, total required memory exceeds physical 8 GB.
3. **Windows WDDM Paging Thrashing & Transient Power Spikes**:
   - When VRAM saturates, Windows WDDM forces memory paging between GPU VRAM and System RAM across the PCIe bus.
   - The concurrent load of 100% GPU Tensor Cores, PCIe saturation, CPU paging, and checkpoint disk I/O causes acute transient power spikes (jumping past 250W on Ampere cards).
   - This trips the PSU Over-Current Protection (OCP) or overheats electrical wall sockets/extension strips.

### 2. Implemented Fail-Safe Protections in `train_qlora.py`

To guarantee 100% stable, crash-proof fine-tuning on 8GB consumer hardware:

* **VRAM Memory Capping (`torch.cuda.set_per_process_memory_fraction(0.80, device=0)`)**:
  Hard-caps PyTorch to use at most 80% of total physical VRAM, guaranteeing a dedicated 20% buffer for Windows display/WDDM and completely preventing OS memory paging thrashing.
* **Bypassed In-Train Evaluation (`eval_strategy="no"`)**:
  Eliminates the uncheckpointed evaluation surge during the training run. Evaluation is conducted offline or on saved adapters.
* **Optimized Attention LoRA Targets**:
  Focused `LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]` to reduce trainable parameter overhead and activation tensors by >50%.
* **Sequence Length Budgeting (`MAX_SEQ_LEN = 512`)**:
  Empirical dataset auditing shows mean token length is 406 tokens (only 12.9% >512). Bounding sequence length to 512 slashes quadratic self-attention memory by $4\times$.
* **PyTorch Allocator Optimization**:
  `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` prevents CUDA memory fragmentation errors.
* **8-Bit Paged Optimizer**:
  `optim="paged_adamw_8bit"` automatically pages optimizer states to CPU RAM if allocations surge, preventing OOM crashes.
* **Hardware Power Limit Capping (Recommended)**:
  Run `nvidia-smi -pl 150` in an Administrator terminal to cap peak board power draw to 150W (75% TDP), preventing electrical socket overheating and transient PSU trips.

---

## 🧠 Live RAG & Dynamic Prompt Assembly Architecture

When an operator enters a query in `main.py`, `ChatAssistant` dynamically fuses five context streams into the final prompt:

```
+-------------------------------------------------------------+
|                     OPERATOR QUERY                          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 1. LIVE SESSION INTEL (extract_session_intelligence())     |
|    - Active user ID, groups (sudo/adm alerts)               |
|    - Discovered SUID / challenge binaries                   |
|    - Strings findings (e.g., 'fortune', '/bin/sh -p')       |
|    - Flag permissions and open network ports                |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 2. UNTRUSTED EVIDENCE BLOCK                                 |
|    <<<BEGIN_EVIDENCE>>>                                     |
|    Pasted scanner outputs, logs, terminal results           |
|    <<<END_EVIDENCE>>> (strictly treated as DATA ONLY)       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 3. SEMANTIC RAG KNOWLEDGE (rag_engine.py)                   |
|    - Top-K text chunks filtered with min_score = -2.5       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 4. SCENARIO TACTICS (yaml_search.py)                        |
|    - Step-by-step procedures from rag/scenarios/*.yaml      |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| 5. RECENT CHAT HISTORY (Last 3 conversational turns)        |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                  TOKENIZER CHAT TEMPLATE                    |
|       (Formats system, context, query into ChatML)          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    LLM GENERATION                           |
|       (Qwen2.5-7B-Instruct or TinyLlama-1.1B)               |
+-------------------------------------------------------------+
```

---

## 🛡️ Verification Gates & Acceptance Criteria

To ensure reliable, non-hallucinatory output during live assessments and government demonstrations, the data and model pipeline enforces strict verification gates:

### Gate 1: 4-Bit bitsandbytes CUDA Smoke Test (`scripts/verify_bnb.py`)
Confirms that 4-bit CUDA quantization layers execute properly on Windows before any training run:
```python
import torch
from bitsandbytes.nn import Linear4bit
layer = Linear4bit(64, 64).cuda()
x = torch.randn(1, 64).cuda()
y = layer(x)
assert y.shape == torch.Size([1, 64])
```

### Gate 2: ChatML Tokenizer Verification (`scripts/verify_tokenizer.py`)
Confirms that the active tokenizer correctly injects `<|im_start|>` and `<|im_end|>` markers without corrupting generation prompts.

### Gate 3: Core Query Acceptance Criteria

| Benchmark Query | Required Pass Behavior | Failure Condition |
| :--- | :--- | :--- |
| `"ubuntu ctf, terminal only, access control, lets start"` | Returns ≥ 3 actionable reconnaissance shell commands (`id`, `uname -a`, `sudo -l`, `find / -perm -4000`). | Returns the generic top-level CLI command menu. |
| `"no this is a different challenge"` | Acknowledges session reset request, wipes assumptions about prior SUID/PATH binaries, and asks for new initial output. | Repeats the prior exploit or assumes the same target binary exists. |
| `"where do i start, give me info gathering scripts"` | Emits structured 5-phase Linux reconnaissance commands. | Prints vague template placeholders without shell commands. |

### Gate 4: VRAM Safety Thresholds (RTX 3060 Ti 8GB)
- **Peak > 7.5 GB**: Lower `max_seq_length` to 384 and `lora_r` to 4. If still over, trigger fallback path (`DarkLight6996/CyberSec-Model-V2`).
- **Peak 6.5 – 7.5 GB**: Ideal operational stability. Maintain 512 sequence length.
- **Peak < 6.5 GB**: Headroom present; sequence length may optionally be evaluated at 768 tokens.

---

## 🏗️ Phase 1 — Services Layer Data Flow (September 2026)

### Updated Model Configuration

| Constant | Previous Value | Updated Value |
|---|---|---|
| `GPU_MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.2` | `DarkLight6996/CyberSec-Model-V2` |
| `ROOT_DIR` | *(not defined)* | `os.path.dirname(RAG_DIR)` — project root |
| `LORA_ADAPTER_PATH` | *(not defined)* | `ROOT_DIR/../lora-adapters` |

### Service Singleton Data Flow

```
[Process Start]
       │
       ▼
get_embedding_service()  ──► EmbeddingService(sentence-transformers/all-MiniLM-L6-v2)
get_rerank_service()     ──► RerankService(cross-encoder/ms-marco-MiniLM-L-6-v2)
       │
       ▼
get_rag_service()        ──► RAGService → RAGEngine (delegates to above singletons)
                                        → FAISS index + metadata
       │
       ▼
get_yaml_service()       ──► YAMLService → YAMLLoader → YAMLRegistry (214 templates)
                                        → YAMLSearch (embed index over templates)
       │
       ▼
get_llm_service()        ──► LLMService → ModelRouter(DarkLight6996/CyberSec-Model-V2, gpu)
                                        → 4-bit NF4 quantization on GPU
       │
       ▼
get_memory(session_id)   ──► ChatMemoryWrapper → ChatMemory(SQLite, sessions.db)
```

### YAML Corpus Health (Phase 1 Verification)

| Status | Count |
|---|---|
| Successfully loaded | 214 |
| Failed (syntax errors) | 19 |
| Total discovered | 233 |

**Known failing files** (pre-existing syntax errors — not introduced by Phase 1):
- `adcs-esc7/9/10/11/12-attack.yaml` — bad escape sequences in double-quoted strings
- `event-log-tampering.yaml`, `sysmon-evasion.yaml`, `malware-persistence-registry.yaml` — `\W`, `\S` used in YAML double-quote context
- `lfi-rfi.yaml`, `terraform-iam-misconfig.yaml` — backtick character used as YAML token
- `AZ-06/07`, `CM-01/04`, `HR-02`, `jenkins-misconfig`, `nosql-injection`, `PrintNightmare` — various block structure errors

These files require YAML repair (replace `\W` with `\\W`, use single-quote strings around regex patterns, remove raw backticks). Flagged for Phase 2 data maintenance pass.

---

## 🏛️ Multi-Agent Framework State & Scope Data Schemas (November 2026)

### 1. Scope Data Contract (`core/scope.py`)
```json
{
  "target": "10.10.10.5",
  "in_scope": ["10.10.10.0/24", "target.corp"],
  "out_of_scope": ["10.10.10.1", "prod-db.target.corp"],
  "prohibited_tests": ["no DoS", "no social engineering", "no credential dumping"],
  "prohibited_tools": ["metasploit", "sqlmap"],
  "time_window": "09:00 - 17:00 UTC",
  "authorization_ref": "SEC-AUDIT-2026-441",
  "active_testing_allowed": false
}
```

### 2. PhaseResult Data Contract (`agents/base_agent.py`)
```json
{
  "phase_name": "subdomain_enum",
  "status": "success",
  "commands_run": [
    {
      "command": "subfinder -d target.com -silent",
      "status": "success",
      "stdout": "api.target.com\nauth.target.com",
      "returncode": 0
    }
  ],
  "raw_output": "api.target.com\nauth.target.com",
  "findings": {
    "subdomains": ["api.target.com", "auth.target.com"]
  },
  "next_hint": "endpoint_discovery",
  "error": null
}
```
