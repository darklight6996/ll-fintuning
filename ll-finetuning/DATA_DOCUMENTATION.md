# Penetration Testing Dataset Documentation (`data.jsonl`)

## Overview
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

