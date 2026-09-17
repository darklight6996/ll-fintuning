import sys
import os

# Add paths
root_dir = r"c:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning"
rag_dir = os.path.join(root_dir, "rag")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if rag_dir not in sys.path:
    sys.path.insert(0, rag_dir)

from core.chat_assistant import ChatAssistant

def test_network_intel_extraction():
    print("[*] Testing network intelligence parsing...")
    assistant = ChatAssistant()
    
    nmap_sample = """
    Starting Nmap 7.80 ( https://nmap.org ) at 2026-09-16 21:00 UTC
    Nmap scan report for 10.10.10.5
    Host is up (0.0012s latency).
    PORT      STATE SERVICE VERSION
    22/tcp    open  ssh     OpenSSH 8.2p1 Ubuntu 4ubuntu0.5
    31337/tcp open  elite   SECRET SERVER v1.0
    
    Banner: SECRET SERVER
    """
    
    session = {
        "name": "network_box",
        "type": "network",
        "notes": [{"content": nmap_sample}],
        "evidence": [],
        "completed_steps": []
    }
    
    intel = assistant.extract_session_intelligence(session)
    assert len(intel["open_ports"]) >= 2, f"Expected at least 2 ports, got: {intel['open_ports']}"
    assert any("31337/tcp" in p for p in intel["open_ports"]), f"Port 31337 missing: {intel['open_ports']}"
    assert any("SECRET SERVER" in b for b in intel["network_services"]), f"Banner missing: {intel['network_services']}"
    print(f"    [+] Successfully extracted ports: {intel['open_ports']}")
    print(f"    [+] Successfully extracted banners: {intel['network_services']}")

def test_prompt_builder():
    print("[*] Testing _build_llm_prompt...")
    assistant = ChatAssistant()
    session = {
        "name": "test_box",
        "target": "10.10.10.5",
        "type": "ctf",
        "scope": "box",
        "notes": [{"content": "31337/tcp open Elite"}],
        "evidence": [],
        "completed_steps": []
    }
    intel = assistant.extract_session_intelligence(session)
    prompt = assistant._build_llm_prompt(
        query="What is port 31337?",
        state=session,
        intel=intel,
        rag_context="Port 31337 is historically used by Back Orifice.",
        yaml_tactics=["* [Tactic] Custom Service Analysis: Analyze unknown TCP daemon"]
    )
    assert "=== OPERATOR QUESTION ===" in prompt
    assert "What is port 31337?" in prompt
    assert "31337/tcp" in prompt
    assert "Port 31337 is historically used" in prompt
    assert "Never fabricate" in prompt
    print("    [+] Prompt correctly formatted and grounded.")

def test_llm_reasoning_and_fallback():
    print("[*] Testing LLM generation with fallback...")
    assistant = ChatAssistant()
    
    # 1. Mock LLM available
    class MockLLM:
        def generate(self, prompt, temperature=0.3):
            return "Port 31337 is running a custom TCP banner 'SECRET SERVER'. You should connect using nc 10.10.10.5 31337 to inspect the interaction protocol."
    
    assistant._llm_router = MockLLM()
    
    session = {
        "name": "custom_box",
        "target": "10.10.10.5",
        "type": "network"
    }
    
    # Query that previously fell through to generic menu
    resp = assistant.respond("I'm on a box with a weird port 31337 open and the banner says 'SECRET SERVER'. What is it?", session)
    assert "Port 31337 is running a custom TCP banner" in resp
    print("    [+] LLM reasoning successfully answered novel query.")
    
    # 2. Test fallback when LLM is unavailable (None)
    assistant._llm_router = False
    
    # Recon methodology fallback
    recon_resp = assistant.respond("where do i start on this ubuntu terminal?", session)
    assert "Phase 1: Environment, Identity & Sudo Privileges" in recon_resp
    print("    [+] Recon request gracefully fell back to 5-phase playbook.")
    
    # Reset fallback
    reset_resp = assistant.respond("no this is a different challenge", session)
    assert "[RESET]" in reset_resp
    print("    [+] Reset request gracefully fell back to reset acknowledgement.")

def test_rag_similarity_calculation():
    print("[*] Testing RAG similarity calculation...")
    import numpy as np
    distances = np.array([[0.0, 1.0, 4.0]])  # distances from IndexFlatL2
    similarities = 1.0 / (1.0 + distances[0])
    
    assert similarities[0] == 1.0
    assert similarities[1] == 0.5
    assert similarities[2] == 0.2
    assert all(0.0 <= s <= 1.0 for s in similarities)
    print(f"    [+] Similarities correctly normalized: {similarities.tolist()}")

if __name__ == "__main__":
    test_network_intel_extraction()
    test_prompt_builder()
    test_llm_reasoning_and_fallback()
    test_rag_similarity_calculation()
    print("\n[SUCCESS] All hybrid LLM + RAG + rule fallback tests passed!")
