"""
test_prompt_subsystem.py

Unit test suite verifying Phase 6 Prompt Builder & Dynamic Token Budgeting:
1. PromptBudget dynamic context partitioning and surplus reallocation.
2. HistoryBuilder rolling summaries and semantic deduplication.
3. IntelBuilder evidence sandboxing and list deduplication.
4. KnowledgeBuilder RAG context deduplication and compression.
5. ScenarioBuilder tactical playbook formatting.
6. Multi-model templates: Qwen, Llama 3, Mistral, Gemma, DeepSeek.
7. Full PromptBuilder orchestration.
8. Backward compatibility of PromptManager.
"""

import sys
from pathlib import Path
import pytest

CORE_DIR = Path(__file__).resolve().parent
RAG_DIR = CORE_DIR.parent
ROOT_DIR = RAG_DIR.parent

for p in (ROOT_DIR, RAG_DIR, CORE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.prompts import (
    PromptBudget,
    HistoryBuilder,
    IntelBuilder,
    KnowledgeBuilder,
    ScenarioBuilder,
    PromptTemplate,
    ModelFamily,
    detect_model_family,
    PromptBuilder,
    PromptManager,
    DEFAULT_SYSTEM_PERSONA
)
from core.builders.context_builder import ContextPackage
from core.intelligence.models import (
    SessionIntelligence,
    UserIntel,
    PortIntel,
    SUIDIntel
)


# ==========================================================
# 1. PROMPT BUDGET TESTS
# ==========================================================

def test_prompt_budget_allocation_and_surplus():
    budget = PromptBudget(total_tokens=2048, reserve_generation=250, reserve_system=200, reserve_query=150)
    # Available context: 2048 - 600 = 1448 tokens ≈ 5792 chars
    avail_tokens = budget.available_context_tokens
    assert avail_tokens == 1448
    avail_chars = budget.tokens_to_chars(avail_tokens)
    assert avail_chars == 5792

    # Case A: Minimal intel and history (surplus should flow to RAG and tactics)
    alloc = budget.allocate(intel_chars=50, history_chars=50, rag_chars=4000, tactics_chars=2000)
    assert alloc["intel"] == 100  # min floor
    assert alloc["history"] == 100  # min floor
    # Surplus went to needy components (rag and tactics)
    assert alloc["rag"] > int(avail_chars * 0.35)
    assert alloc["tactics"] > int(avail_chars * 0.20)


# ==========================================================
# 2. HISTORY BUILDER TESTS
# ==========================================================

def test_history_builder_dedup_and_rolling_summary():
    builder = HistoryBuilder(recent_turns_limit=2)

    # Contains consecutive duplicate turn
    raw_history = [
        {"role": "user", "message": "How do I scan open ports?"},
        {"role": "user", "message": "How do I scan open ports?"},  # duplicate
        {"role": "assistant", "message": "Use nmap -sC -sV <target>"},
        {"role": "user", "message": "Found port 80 open"},
        {"role": "assistant", "message": "Check web server banners with curl -I"},
        {"role": "user", "message": "Banner says Apache 2.4.41"},
        {"role": "assistant", "message": "Inspect available CGI scripts or SUID binaries"}
    ]

    formatted = builder.build(raw_history, max_chars=1000)

    # Check deduplication worked: "How do I scan open ports?" summarized once
    assert "Prior Context Summary:" in formatted
    assert "User: Banner says Apache 2.4.41" in formatted
    assert "Assistant: Inspect available CGI scripts or SUID binaries" in formatted


# ==========================================================
# 3. INTEL BUILDER TESTS
# ==========================================================

def test_intel_builder_sandboxing_and_deduplication():
    builder = IntelBuilder()

    # Intel with duplicates in ports and binaries
    intel_data = {
        "user": "hacker",
        "has_sudo": True,
        "open_ports": ["80/tcp open http", "80/tcp open http", "22/tcp open ssh"],
        "suid_binaries": ["/challenge/run", "/challenge/run", "/usr/bin/find"],
        "kernel": "Linux 5.15.0"
    }

    evidence = builder.build(intel_data, max_chars=800)
    assert "<<<BEGIN_EVIDENCE>>>" in evidence
    assert "<<<END_EVIDENCE>>>" in evidence
    assert "- User: hacker (sudo enabled)" in evidence
    assert "- Open Ports: 80/tcp open http, 22/tcp open ssh" in evidence
    assert "- SUID Targets: /challenge/run, /usr/bin/find" in evidence


def test_intel_builder_with_session_intelligence_model():
    builder = IntelBuilder()
    model = SessionIntelligence(
        user_intel=UserIntel(user="ctf_operator", has_sudo=False),
        port_intel=PortIntel(open_ports=["445/tcp"]),
        suid_intel=SUIDIntel(suid_binaries=["/opt/custom/exec"])
    )

    evidence = builder.build(model)
    assert "- User: ctf_operator" in evidence
    assert "- Open Ports: 445/tcp" in evidence
    assert "- SUID Targets: /opt/custom/exec" in evidence


# ==========================================================
# 4. KNOWLEDGE & SCENARIO BUILDER TESTS
# ==========================================================

def test_knowledge_builder_dedup():
    builder = KnowledgeBuilder()
    rag_snippet = (
        "[SOURCE: doc1.md]\n[CHUNK: 1]\nCommand injection can be tested via ; or |\n\n"
        "[SOURCE: doc1.md]\n[CHUNK: 1]\nCommand injection can be tested via ; or |\n\n"
        "[SOURCE: doc2.md]\n[CHUNK: 2]\nSQL injection can be tested with '"
    )
    res = builder.build(rag_snippet, max_chars=600)
    # The duplicate chunk 1 should only appear once
    assert res.count("Command injection can be tested via") == 1
    assert "SQL injection can be tested with" in res


def test_scenario_builder():
    builder = ScenarioBuilder()
    tactics = [
        {"phase": "recon", "name": "Port Scan", "command": "nmap -p- [TARGET]"},
        {"phase": "recon", "name": "Port Scan", "command": "nmap -p- [TARGET]"},  # duplicate
        {"phase": "privesc", "name": "SUID Scan", "command": "find / -perm -4000"}
    ]
    res = builder.build(tactics, max_chars=500)
    assert res.count("[RECON] Port Scan: nmap -p- [TARGET]") == 1
    assert "[PRIVESC] SUID Scan: find / -perm -4000" in res


# ==========================================================
# 5. MULTI-MODEL PROMPT TEMPLATE TESTS
# ==========================================================

def test_multi_model_templates():
    sys_prompt = "You are a cyber security mentor."
    user_content = "How do I exploit SUID binaries?"

    # Qwen (ChatML)
    qwen_prompt = PromptTemplate.format(sys_prompt, user_content, ModelFamily.QWEN)
    assert "<|im_start|>system\nYou are a cyber security mentor.<|im_end|>" in qwen_prompt
    assert "<|im_start|>user\nHow do I exploit SUID binaries?<|im_end|>" in qwen_prompt
    assert qwen_prompt.endswith("<|im_start|>assistant\n")

    # Llama 3
    llama_prompt = PromptTemplate.format(sys_prompt, user_content, ModelFamily.LLAMA)
    assert "<|start_header_id|>system<|end_header_id|>" in llama_prompt
    assert "<|start_header_id|>user<|end_header_id|>" in llama_prompt
    assert "<|start_header_id|>assistant<|end_header_id|>" in llama_prompt

    # Mistral
    mistral_prompt = PromptTemplate.format(sys_prompt, user_content, ModelFamily.MISTRAL)
    assert mistral_prompt.startswith("<s>[INST]")
    assert mistral_prompt.endswith("[/INST]")

    # Gemma
    gemma_prompt = PromptTemplate.format(sys_prompt, user_content, ModelFamily.GEMMA)
    assert "<start_of_turn>user" in gemma_prompt
    assert "<start_of_turn>model\n" in gemma_prompt

    # DeepSeek
    deepseek_prompt = PromptTemplate.format(sys_prompt, user_content, ModelFamily.DEEPSEEK)
    assert "<｜begin of sentence｜>" in deepseek_prompt
    assert "Assistant: <｜thought｜>\n" in deepseek_prompt


def test_detect_model_family():
    assert detect_model_family("Qwen/Qwen2.5-7B-Instruct") == ModelFamily.QWEN
    assert detect_model_family("DarkLight6996/CyberSec-Model-V2") == ModelFamily.LLAMA
    assert detect_model_family("mistralai/Mistral-7B-Instruct-v0.3") == ModelFamily.MISTRAL
    assert detect_model_family("google/gemma-2-9b-it") == ModelFamily.GEMMA
    assert detect_model_family("deepseek-ai/DeepSeek-Coder-V2") == ModelFamily.DEEPSEEK


# ==========================================================
# 6. UNIFIED PROMPT BUILDER TESTS
# ==========================================================

def test_prompt_builder_full_pipeline():
    builder = PromptBuilder(max_input_tokens=2048, default_model_family=ModelFamily.QWEN)

    ctx = ContextPackage(
        intel={"user": "root", "open_ports": ["22/tcp open ssh"]},
        rag_context="[SOURCE: suid.md]\nGTFOBins provides privilege escalation commands.",
        yaml_tactics=["[PRIVESC] Run SUID: /challenge/run"]
    )

    state = {
        "target": "10.10.10.5",
        "type": "Linux CTF",
        "chat_history": [{"role": "user", "message": "Ready to begin assessment."}]
    }

    prompt = builder.build(
        query="What is the next tactical step?",
        state=state,
        context=ctx,
        model_family=ModelFamily.QWEN
    )

    # Prompt adheres to ChatML framing
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "<|im_start|>assistant\n" in prompt
    assert "=== UNTRUSTED EVIDENCE" in prompt
    assert "<<<BEGIN_EVIDENCE>>>" in prompt
    assert "Target: 10.10.10.5" in prompt
    assert "GTFOBins provides privilege escalation commands." in prompt


# ==========================================================
# 7. PROMPT MANAGER BACKWARD COMPATIBILITY
# ==========================================================

def test_prompt_manager_backward_compatibility():
    pm = PromptManager()
    prompt = pm.build_prompt(
        mode="rag",
        query="What is a buffer overflow?",
        retrieved_docs=[({"filename": "pwn.md", "chunk_id": "c1", "text": "Buffer overflow occurs..."}, 0.9)],
        memory_rows=[("User", "Hello"), ("Assistant", "Welcome")]
    )

    assert "You are a cybersecurity mentor and assistant." in prompt
    assert "CONVERSATION HISTORY:" in prompt
    assert "[SOURCE 1] pwn.md" in prompt
    assert "USER QUESTION" in prompt
