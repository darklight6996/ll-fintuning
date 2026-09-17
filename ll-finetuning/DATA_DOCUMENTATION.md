# Penetration Testing Dataset Documentation (`data.jsonl`)

## Overview


## RAG & Retrieval Data Updates (September 2026)

- **Vector Distance Normalization**: FAISS index outputs are no longer reported as raw L2 squared euclidean distances. They are converted to a `0.0 - 1.0` semantic similarity score to filter out low-confidence matches. Any document scoring below `0.35` is dropped before entering the LLM context block.
- **Prompt Context Assembly**: LLM prompts dynamically fuse data from 3 streams:
  1. **Live Intel**: Extracted open ports, active users, groups, kernel versions, SUID binaries, and discovered banners.
  2. **Scenario Tactics**: YAML scenario steps ranked by vector similarity.
  3. **RAG Context**: Top matches from the static document corpus.
- **Evidence Formatting**: Pasted data is explicitly classified as *untrusted* data using `<<<BEGIN_EVIDENCE>>>` and `<<<END_EVIDENCE>>>` tags.

The `data.jsonl` file is an instruction-based dataset designed for fine-tuning Large Language Models (LLMs) on cybersecurity, penetration testing, and security auditing tasks. It contains high-quality, practical examples of command-line tools, methodologies, and security compliance checks.

## Data Structure
Each line in the JSONL file represents a single training example formatted as a JSON object with three key-value pairs:

```json
{
  "instruction": "A question or scenario describing what needs to be accomplished.",
  "input": "Context, environmental details, or specific targets related to the instruction.",
  "output": "The appropriate command(s) to execute and a brief explanation of the command's purpose or expected outcome."
}
```

### Example Entry
```json
{
  "instruction": "How do you test for SQL injection vulnerabilities in a user input field?",
  "input": "You suspect the username field on the login page might be vulnerable to SQL injection.",
  "output": "Command: `sqlmap -u \"http://example.com/login?username=admin&password=test\" --risk=3 --level=5 --dbs`\nExplanation: sqlmap automatically detects and exploits SQL injection vulnerabilities, retrieving database names if successful."
}
```

## Content Categories

The dataset comprehensively covers various domains within cybersecurity:

### 1. Web Application Penetration Testing
- **Reconnaissance & Enumeration**: Usage of tools like `nmap` (port scanning, service versioning), `nikto` (web vulnerability scanning), and `whatweb` (technology identification).
- **Vulnerability Exploitation**: Identifying and exploiting common vulnerabilities like SQL Injection (`sqlmap`), Server-Side Request Forgery (SSRF), and Directory Fuzzing (`ffuf`).
- **Brute Forcing**: Using `hydra` to brute-force login pages.

### 2. Active Directory (AD) & Windows Environments
- **Enumeration & Access**: Utilizing tools like `netexec` (SMB, LDAP), `rusthound-ce`, and `bloodhound-python` for Active Directory enumeration and mapping.
- **Credential Access & Lateral Movement**: Techniques such as Kerberoasting, password spraying, DCSync (`secretsdump.py`), shadow credential attacks (`certipy`), and manipulating ADCS (Active Directory Certificate Services) via ESC16.
- **Post-Exploitation**: Shell access using `evil-winrm` and resetting passwords via `bloodyAD`.

### 3. Linux Exploitation & Security Auditing
- **Privilege Escalation**: Identifying and exploiting SUID binaries, exploiting specific CVEs (e.g., NetData ndsudo, GitPython, Enlightenment CVE-2022-37706), and abusing cron jobs.
- **Baseline Security Auditing (SOC 2 / Compliance)**: Checking password expiration policies (`/etc/login.defs`), auditing root SSH access, checking firewall statuses (`ufw`), and verifying audit logs (`auditd`).

### 4. Cloud Security (Azure & AWS)
- **Azure Reconnaissance**: Enumerating subscriptions, resources, Active Directory users, role assignments, and Logic/App Function configurations using the `az` CLI.
- **Azure Exploitation**: Authenticating via Service Principals, exploring key vaults, enumerating shared storage, and identifying over-privileged role assignments.
- **AWS Auditing**: Identifying overly permissive IAM roles, checking CloudTrail logs, validating S3 bucket encryption, and auditing EC2 security groups using the `aws` CLI.

### 5. Containers & Orchestration
- **Docker**: Reviewing container image sources, verifying runtime configurations, and ensuring secure supply chains.
- **Kubernetes**: Checking Kubernetes RBAC configurations (`kubectl get rolebindings`) and auditing stored secrets (`kubectl get secrets`).

## Usage
This dataset can be directly ingested by supervised fine-tuning (SFT) pipelines (e.g., using frameworks like Hugging Face `transformers` or TRL) to teach an LLM how to assist a penetration tester or security auditor by providing context-aware commands and explanations.

---

## Data Engineering & Automation Scripts

The project includes a suite of scripts in the `Generic Scripts` directory designed to automate the creation of high-quality training and RAG data.

### 1. Document Extraction
- **`pdf_to_fintune.py`**: Extracts text from PDF documents and uses heuristic templates to generate Stanford Alpaca-style instruction pairs.
- **`pdf_to_rag_corpus.py`**: Chunks PDF content into a structured RAG (Retrieval-Augmented Generation) format with ID, Source, and Page metadata.

### 2. Web Scraping & Chunking
- **`scrape_writeups.py`**: Specialized scraper for gathering HackTheBox (HTB) writeups from established sources (e.g., 0xdf.gitlab.io), respecting rate limits.
- **`WebPage_chunker.py`**: A general-purpose tool to fetch web pages, strip noise (nav/footers), and create overlapping RAG chunks.

### 3. Dataset Conversion
- **`rag_to_alpaca.py`**: Converts structured RAG `.txt` files into `.jsonl` instruction datasets.
- **`rag_to_alpaca_multiple.py`**: Batch processes entire directories of RAG chunks into a single unified fine-tuning dataset.
- **`writeup_to_corpus.py`**: Converts raw text writeups into the project's standard RAG chunk format.

---

## Fine-Tuning Workflow

The training process is optimized for both development and production environments.

### Training Scripts (`ll-finetuning/scripts/`)
- **`train_lora.py`**: The primary training script for GPU-enabled systems. It uses `bitsandbytes` for 4/8-bit quantization and `peft` for Low-Rank Adaptation (LoRA).
- **`train_lora_cpu.py`**: A specialized version for CPU-only training, utilizing `torch.float32` and standard LoRA configurations.

### Model Configuration
- **Base Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Technique**: LoRA (Low-Rank Adaptation)
- **Parameters**: `r=8`, `alpha=16`, Target modules: `q_proj`, `v_proj`.

---

## Environment & Portability

To ensure the project can be deployed across different environments, specific requirements are managed in the `ll-finetuning/setup/` directory:

- **Windows 11**: Includes specific CUDA 12.1 index URLs and build-tool requirements.
- **Ubuntu**: Includes standard Linux build essentials and CUDA 12.1 configurations.

All core dependencies are also tracked in the root `requirements.txt`.

---

## Assessment Scenarios & Tactical Playbooks (`ll-finetuning/rag/scenarios/`)

In addition to Alpaca-style instruction pairs, the system uses structured YAML scenario specifications that encode end-to-end tactical procedures for live assessment execution.

### Scenario Schema (`.yaml`)
Each scenario contains operational steps, targeted tools, expected outputs, and risk assessments:
```yaml
id: linux_suid_privesc
name: Linux SUID Executable & Capability Audit
category: ctf_linux_privilege_escalation
target_environment: local_shell_or_container
steps:
  - sequence_number: 1
    action: Find SUID/SGID binaries on target filesystem
    commands: ["find / -perm -4000 -type f 2>/dev/null", "getcap -r / 2>/dev/null"]
    expected_output: List of binaries executing with effective root UID
    risk: Read-only filesystem search
```

### Supported Categories:
1. **CTF & Local Linux Shells:** Privilege escalation, flag discovery (`/flag`, `/challenge`), SUID/SGID, sudo rules, and path manipulation (e.g. `pwn.college` scenarios).
2. **Active Directory & Identity:** SMB anonymous shares, Kerberoasting (`GetNPUsers.py`), LDAP enumeration, and DCSync.
3. **Web Application & APIs:** SQL injection, authentication bypass, directory fuzzing, and CORS misconfiguration.
4. **Cloud Infrastructure:** Azure IAM role auditing, AWS EC2 metadata service (IMDSv1/v2) SSRF exploitation, and Key Vault exfiltration.

---

## Live Assessment Session State Schema (`.assessments/<session>/state.json`)

When an assessment or CTF is initiated, state is persisted in an atomic JSON structure:
- **`name`**: Unique session slug (e.g., `pwn_college_first_transient_path`).
- **`target`**: Target identifier (e.g., `localhost (pwn.college container)`, IP CIDR, domain).
- **`type`**: Assessment methodology (`ctf`, `linux`, `active_directory`, `web`, `cloud`).
- **`connectivity`**: Access mechanism (`VPN`, `Local`, `Direct`).
- **`scope`**: Authorized engagement boundaries.
- **`current_step`**: The active pending action item served by the planner.
- **`completed_steps`**: Chronological record of completed actions with execution outputs.
- **`evidence`**: Forensic execution proofs and raw outputs.
- **`notes`**: Operator observations and conversational AI Co-Pilot dialogue history.
- **`targets`**: Discovered target entities (`ips`, `domains`, `ports`, `services`, `cloud_resources`).

---

## Interactive CLI & AI Co-Pilot (`main.py`)

The `main.py` entry point is the primary operator interface, combining a structured assessment command shell with a free-form conversational AI Security Co-Pilot. It is launched via `python main.py` from the workspace root.

### Command Reference

| Command | Description |
| :--- | :--- |
| `new <name>` | Start a new assessment session with target, type, scope, and connectivity |
| `open <name>` | Resume an existing session. Supports **partial / prefix matching** (e.g., `open pwn_college` finds `pwn_college_first_transient_path`) |
| `next` | Fetch exactly one recommended methodology step from the planner |
| `run` | Execute the pending step: either run shell commands or mark as manually done |
| `paste` | Paste raw terminal output — automatically advances the active step and analyzes for new findings |
| `upload <path>` | Ingest an Nmap XML, Burp Suite, or cloud JSON scanner file |
| `status` | Show full session overview: targets, ports, services, vulnerabilities, progress |
| `scope [new]` | View or update engagement scope boundaries |
| `type [new]` | Switch assessment methodology (`ctf`, `linux`, `web`, `active_directory`, `cloud`) |
| `target set\|add\|rm\|clear` | Fine-tune target IPs, domains, ports, and services |
| `back` / `undo` | Roll back the last completed step to re-evaluate |
| `reset steps\|targets` | Clear completed steps or discovered targets |
| `report` | Compile notes and evidence into a Markdown report |
| `list` | List all saved assessment sessions |
| `quit` / `exit` | Save state and exit cleanly |

### AI Co-Pilot Chat

Any input that is **not** a reserved command is automatically routed to the `ChatAssistant` as a free-form question. The `chat <message>` command also works explicitly. The Co-Pilot:
1. Reads live session context (target, type, scope, steps, evidence).
2. Performs semantic RAG retrieval over the textbook and OWASP corpus.
3. Queries YAML scenario playbooks for matching tactics.
4. Returns tactical, copy-pasteable guidance with commands.

```
assess [pwn_college_first_transient_path] > how do I read the flag if I'm not root?
[AI Co-Pilot | Context: localhost (pwn.college container) (CTF)]
-----------------------------------------------------------------
🔍 Flag Access & Permission Breakdown:
If /flag is owned by root:root with mode 0400, check for SUID on /challenge/* or sudo -l
```

---

## RAG Core Module Architecture (`ll-finetuning/rag/core/`)

The refactored `rag/core/` package contains all subsystem modules:

| Module | Responsibility |
| :--- | :--- |
| `rag_engine.py` | FAISS vector search, embedding, cross-encoder reranking, context building |
| `memory.py` | Conversation history loading, saving, and retrieval (JSON-backed) |
| `prompts.py` | Prompt template formatting per model family (Mistral, FLAN-T5, default) |
| `models.py` | Model factory wrappers and lazy loader utilities |
| `config.py` | Centralized path constants (`INDEX_PATH`, `META_PATH`, `SCENARIOS_DIR`, etc.) |
| `tools.py` | Reusable utility helpers (text normalization, token counting, safe file ops) |
| `planner.py` | Phase-ordered methodology planner; serves exactly one step per `next` call |
| `session_manager.py` | Session lifecycle, state persistence, evidence collection, target tracking |
| `chat_assistant.py` | Conversational AI Co-Pilot: context fusion, RAG retrieval, reasoning |
| `ingestion.py` | Multi-format file parser (Nmap XML, Burp, Cloud JSON, plain text) |
| `report_generator.py` | Compiles session state into structured Markdown assessment reports |
| `command_executor.py` | Safe command preview, timeout enforcement, and output capture |
| `variable_engine.py` | Live target parameter substitution into command templates |
| `yaml_loader.py` | Loads and validates YAML scenario files from `rag/scenarios/` |
| `yaml_registry.py` | Indexes and catalogues all loaded scenarios for fast lookup |
| `yaml_search.py` | Keyword and semantic search over the YAML scenario registry |
| `model_router.py` | Device-aware inference routing (CPU → FLAN-T5, GPU → Mistral/LLaMA) |
| `assessment_router.py` | Maps operator queries to relevant YAML scenarios and RAG context |
| `evidence_manager.py` | Structured evidence ledger management and forensic record keeping |
| `build_index.py` | Rebuilds the FAISS vector index from corpus text files |
| `yaml_indexer.py` | Builds YAML scenario vector index for semantic scenario retrieval |

---

## Runtime Module Status (Post Code Audit - September 2026)

All modules have been audited and hardened. Key changes that affect data flow:

### chat_assistant.py
- **f-string fix:** `_explain_pasted_output_and_binaries()` now correctly substitutes the `target_bin` variable into response text (was silently printing `{target_bin}` as a literal string).
- **State mutation fix:** `respond()` no longer mutates the caller-provided empty dict default when no session is active; `chat_history` tracking only persists when a real session state is passed.
- **Evidence-Based Dynamic Reasoning:** `extract_session_intelligence()` parses actual session artifacts (`notes`, `evidence`, `completed_steps`). When no custom SUID binary is present in the pasted output, the assistant explicitly states that no binary was found yet, preventing hallucinated references to previous labs.
- **Session Reset Data Flush:** Acknowledging a different challenge purges prior `artifacts`, `notes`, and `completed_steps` from the session state to guarantee strict session isolation.

### rag_engine.py
- **Singleton Model Caching:** `get_global_rag_engine()` ensures `SentenceTransformer` and `CrossEncoder` weights are cached in process memory across CLI commands, eliminating duplicate model initializations.
- **Negative Logit Filtering:** Added `min_score = -2.5` threshold to `retrieve()`, eliminating noisy, irrelevant RAG chunks from contaminating Linux reconnaissance conversations.

### ingestion.py
- **Burp XML IPs:** BurpParser now correctly reads IP addresses from the `<host ip="..." />` XML attribute (was using invalid XPath `host[@ip]` which silently returned empty strings, losing all IP data from Burp exports).
- **Cloud JSON safety:** CloudCLIParser gracefully handles malformed JSON files without crashing the ingestion pipeline — returns a structured error result instead.
- **Copy diagnostics:** Artifact copy failures now emit a `warnings.warn()` so operators can see when evidence files could not be saved to the session directory.

### session_manager.py
- **Backward compatibility:** `add_ingested_findings()` now uses `.setdefault()` for both the top-level `ingested_findings` key and all category sub-keys, making the system compatible with older session state files on disk.

### main.py
- **Encoding guard:** Consolidated to a single clean `sys.stdout.reconfigure(encoding="utf-8")` guard.
- **Note truncation:** AI response notes now only append `...` when the response was actually truncated beyond 400 characters.
- **Bilingual Chat / CLI Routing:** The `default()` handler cleanly routes natural queries directly to `ChatAssistant.respond()`, while preserving rapid execution commands (`new`, `open`, `paste`, `next`, `run`).
