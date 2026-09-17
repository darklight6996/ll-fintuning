import sys
from pathlib import Path

ROOT = Path(r"c:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning")

FILES = {
    "SYSTEM_DOCUMENTATION.md": """
## Architectural Updates (September 2026)

### 1. Hybrid LLM Reasoning & Fallback Engine (`chat_assistant.py`)
- **Transition**: Migrated from a static N-branch decision tree to a **Hybrid LLM + Deterministic Fallback** architecture.
- **LLM First**: User queries are now passed to an LLM (`ModelRouter`) powered by a highly structured prompt containing live session intelligence, RAG documents, and YAML scenario tactics.
- **Rule Fallback**: The deterministic 8-branch rule engine only executes if the LLM is unavailable or fails to generate a valid response.
- **Safety**: Context resets are now confirmation-gated (`reset confirm`) to prevent accidental session wipes.
- **Prompt Injection Defense**: All untrusted terminal output ingested into the LLM prompt is heavily sandboxed within `<<<BEGIN_EVIDENCE>>>` delimiters and accompanied by strict data-only instructions.

### 2. Command Execution Security (`command_executor.py`, `variable_engine.py`)
- **Shell Escaping**: Completely removed `shell=True` from `CommandExecutor`. All commands are executed via list formats (`shell=False`) after passing through `shlex.split()`.
- **Injection Mitigation**: The `VariableEngine` now routes all substituted variables through `shlex.quote()` prior to command generation.
- **Network Validation**: Strict allow-list regex validation (`^[a-zA-Z0-9\.\-\_\:\/]+$`) enforced on all network parameters (IP, PORT, HOST, DOMAIN) to block unexpected shell characters.

### 3. Engine Configuration (`config.py`)
- **CPU Model**: Default CPU inference engine upgraded from `google/flan-t5-large` to `Qwen/Qwen2.5-1.5B-Instruct` to leverage a true chat-tuned causal LM.
- **Context Budgets**: CPU prompt context strictly capped at 1024 input tokens to prevent silent truncation.

### 4. Vector Search Tuning (`rag_engine.py`)
- **Similarity Normalization**: FAISS L2 distances are mathematically inverted and normalized using `1.0 / (1.0 + distance)`, translating lower distances into intuitive similarity scores `[0, 1]`.
- **Noise Thresholds**: Implemented hard similarity minimums (e.g., `> 0.35`) during context building to prevent irrelevant low-quality corpus matches from polluting the prompt.
""",

    "DATA_DOCUMENTATION.md": """
## RAG & Retrieval Data Updates (September 2026)

- **Vector Distance Normalization**: FAISS index outputs are no longer reported as raw L2 squared euclidean distances. They are converted to a `0.0 - 1.0` semantic similarity score to filter out low-confidence matches. Any document scoring below `0.35` is dropped before entering the LLM context block.
- **Prompt Context Assembly**: LLM prompts dynamically fuse data from 3 streams:
  1. **Live Intel**: Extracted open ports, active users, groups, kernel versions, SUID binaries, and discovered banners.
  2. **Scenario Tactics**: YAML scenario steps ranked by vector similarity.
  3. **RAG Context**: Top matches from the static document corpus.
- **Evidence Formatting**: Pasted data is explicitly classified as *untrusted* data using `<<<BEGIN_EVIDENCE>>>` and `<<<END_EVIDENCE>>>` tags.
""",

    "TEXTBOOK.md": """
## Engine Evolution: The Hybrid LLM Architecture

As of September 2026, the underlying Chat Assistant engine transitioned from a static decision-tree into a **Hybrid LLM + Deterministic Fallback** architecture. 

### Why the Shift?
In traditional architectures, the assistant could only answer questions that were explicitly accounted for in hardcoded regex rules. If a user asked "Explain Kerberoasting" or queried a non-standard port (e.g., port `31337`), the system hallucinated generic responses.

By implementing a local, offline LLM (such as `Qwen2.5-1.5B-Instruct` on CPU or `Mistral-7B` on GPU), the system now parses the full context of the session and generates dynamic, grounded responses.

### Security Defenses Implemented
- **Command Injection Prevention**: Variables substituted into command templates are strictly validated against allow-lists and encapsulated via `shlex.quote()`. All executions use `shell=False`.
- **Prompt Injection Containment**: Because operators paste potentially hostile terminal banners into the assistant, all pasted evidence is heavily sandboxed inside the AI prompt with strict instructions not to execute the enclosed text.
"""
}

for filename, content in FILES.items():
    filepath = ROOT / filename
    if filepath.exists():
        original = filepath.read_text(encoding="utf-8")
        
        # Avoid duplicate appends
        if "Architectural Updates (September 2026)" not in original and "RAG & Retrieval Data Updates (September 2026)" not in original:
            # We'll prepend to make it highly visible, right after the main title headers
            lines = original.split("\n")
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("---") or line.startswith("## "):
                    insert_idx = i + 1
                    break
            
            new_content = "\n".join(lines[:insert_idx]) + "\n\n" + content + "\n" + "\n".join(lines[insert_idx:])
            filepath.write_text(new_content, encoding="utf-8")
            print(f"[+] Updated {filename}")
        else:
            print(f"[-] {filename} already updated.")
    else:
        print(f"[!] {filename} not found.")

