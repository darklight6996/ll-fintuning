"""
test_memory_enrichment.py

Phase 10 — Enriched Session Memory verification tests:
1. Exact count tracking across message additions
2. Fast keyword text search (search_text)
3. Semantic vector memory retrieval (search_memory)
4. Empty session resilience
5. Conversation summary statistics (get_summary)
6. Structured JSON export (export_json)
7. Service wrapper delegation via MemoryService
8. ToolRegistry integration with session_memory_search
"""

import sys
import os
import tempfile
import pytest
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent
RAG_DIR = CORE_DIR.parent
ROOT_DIR = RAG_DIR.parent.parent

for p in (ROOT_DIR, RAG_DIR, CORE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.memory import ChatMemory
from core.services.memory_service import ChatMemoryWrapper
from core.tool_registry import get_tool_registry


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Clean up any open SQLite connections in MemoryService registry
    try:
        from core.services.memory_service import _REGISTRY
        for wrapper in list(_REGISTRY.values()):
            try:
                wrapper.close()
            except Exception:
                pass
        _REGISTRY.clear()
    except Exception:
        pass

    import gc
    gc.collect()

    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def test_chat_memory_add_and_count(temp_db):
    mem = ChatMemory(session_id="test_count", db_path=temp_db)
    assert mem.count() == 0

    mem.add_turn("What is SUID?", "SUID allows execution with file owner privileges.")
    assert mem.count() == 2

    mem.add_message("user", "How to find them?")
    assert mem.count() == 3
    mem.close()


def test_chat_memory_search_text(temp_db):
    mem = ChatMemory(session_id="test_search_text", db_path=temp_db)
    mem.add_turn("Scan the target IP 10.10.10.5", "Running nmap -sC -sV 10.10.10.5 on port 80 and 445.")
    mem.add_turn("Any web services?", "Apache 2.4 on port 80.")

    # Search for "nmap"
    results = mem.search_text("nmap")
    assert len(results) == 1
    assert "nmap -sC -sV" in results[0]["content"]

    # Search for "port 80"
    results_80 = mem.search_text("port 80")
    assert len(results_80) == 2

    # Search for non-existent keyword
    assert len(mem.search_text("kerberos")) == 0
    mem.close()


def test_chat_memory_search_memory_semantic(temp_db):
    mem = ChatMemory(session_id="test_semantic", db_path=temp_db)
    mem.add_turn("Found an interesting executable in /opt/secret", "Inspect file permissions and run strings on it.")
    mem.add_turn("Port 22 is open with OpenSSH", "Check for banner versions and weak credentials.")
    mem.add_turn("Kernel version is Linux 5.4.0", "Check for dirty pipe or overlayfs CVEs.")

    # Query related to binary investigation
    results = mem.search_memory("binary analysis and permissions", top_k=1)
    assert len(results) == 1
    assert "permissions" in results[0]["content"] or "executable" in results[0]["content"]
    assert "score" in results[0]
    mem.close()


def test_chat_memory_empty_history(temp_db):
    mem = ChatMemory(session_id="test_empty", db_path=temp_db)
    assert mem.count() == 0
    assert mem.search_memory("anything") == []
    assert mem.search_text("anything") == []
    assert mem.export_json() == []

    summary = mem.get_summary()
    assert summary["total_messages"] == 0
    assert summary["user_turns"] == 0
    assert summary["first_timestamp"] is None
    mem.close()


def test_chat_memory_get_summary(temp_db):
    mem = ChatMemory(session_id="test_summary", db_path=temp_db)
    mem.add_turn("Query 1", "Answer 1")
    mem.add_turn("Query 2", "Answer 2")

    summary = mem.get_summary()
    assert summary["session_id"] == "test_summary"
    assert summary["total_messages"] == 4
    assert summary["user_turns"] == 2
    assert summary["assistant_turns"] == 2
    assert summary["first_timestamp"] is not None
    assert summary["last_timestamp"] is not None
    mem.close()


def test_chat_memory_export_json(temp_db):
    mem = ChatMemory(session_id="test_export", db_path=temp_db)
    mem.add_turn("Who am I?", "You are user hacker.")

    exported = mem.export_json()
    assert len(exported) == 2
    assert exported[0]["role"] == "user"
    assert exported[0]["content"] == "Who am I?"
    assert exported[1]["role"] == "assistant"
    assert exported[1]["content"] == "You are user hacker."
    mem.close()


def test_memory_service_wrapper_delegation(temp_db):
    wrapper = ChatMemoryWrapper(session_id="test_wrapper", db_path=temp_db)
    assert wrapper.is_available is True
    assert wrapper.count() == 0

    wrapper.add_turn("What is Kerberoasting?", "Requesting SPN tickets to crack TGS offline.")
    assert wrapper.count() == 2

    # Test search_text delegation
    text_matches = wrapper.search_text("Kerberoasting")
    assert len(text_matches) == 1

    # Test get_summary delegation
    summary = wrapper.get_summary()
    assert summary["total_messages"] == 2

    # Test export_json delegation
    exported = wrapper.export_json()
    assert len(exported) == 2

    wrapper.close()


def test_session_memory_search_builtin_tool(temp_db):
    # Setup test memory
    mem = ChatMemory(session_id="tool_session_test", db_path=temp_db)
    mem.add_turn("Run linpeas on the target", "linpeas discovered SUID binary /usr/bin/find.")
    mem.close()

    # Use tool registry
    reg = get_tool_registry()
    assert "session_memory_search" in reg

    result = reg.dispatch(
        "session_memory_search",
        {"session_id": "tool_session_test", "query": "linpeas SUID", "top_k": 2, "db_path": temp_db}
    )
    assert result.success is True
    assert "linpeas" in result.output.lower() or "suid" in result.output.lower()
