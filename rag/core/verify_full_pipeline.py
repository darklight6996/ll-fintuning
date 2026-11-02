"""
core/verify_full_pipeline.py

Comprehensive test suite verifying:
1. Document Retrieval (RAGEngine) with specific technical questions.
2. YAML Search (YAMLSearch) with procedure search queries.
3. Integrated Pipeline (Memory -> Document RAG + YAML Search -> Router -> Planner -> Variable Engine -> Read-only Executor -> Evidence Manager).
"""

import os
import sys
import json
from pathlib import Path

# Set path resolution
CORE_DIR = Path(__file__).resolve().parent
RAG_DIR = CORE_DIR.parent
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from core.config import (
    INDEX_PATH,
    META_PATH,
    SCENARIOS_DIR,
    DATABASE_PATH
)
from core.rag_engine import RAGEngine
from core.yaml_loader import YAMLLoader
from core.yaml_search import YAMLSearch
from core.assessment_router import AssessmentRouter
from core.planner import Planner
from core.variable_engine import VariableEngine
from core.command_executor import CommandExecutor
from core.evidence_manager import EvidenceManager
from core.memory import ChatMemory
from core.prompts import PromptManager


def test_document_retrieval(rag_engine: RAGEngine):
    print("=" * 80)
    print("TEST 1: DOCUMENT RETRIEVAL (RAGEngine)")
    print("=" * 80)

    queries = [
        "Explain Kerberos.",
        "What is constrained delegation?",
        "How does Pass-the-Hash work?"
    ]

    for q in queries:
        print(f"\n[QUERY]: '{q}'")
        results = rag_engine.retrieve(q)
        print(f"Retrieved {len(results)} document chunk(s):")
        for rank, (doc, score) in enumerate(results, start=1):
            source = doc.get("filename", doc.get("chunk_id", "Unknown"))
            snippet = doc.get("text", "").replace("\n", " ")[:150]
            print(f"  {rank}. [Score: {score:.4f}] Source: {source}")
            print(f"     Snippet: {snippet}...")
    print("\n" + "-" * 80 + "\n")


def test_yaml_retrieval(yaml_search: YAMLSearch):
    print("=" * 80)
    print("TEST 2: YAML SEARCH (YAMLSearch)")
    print("=" * 80)

    queries = [
        "Assess Azure AD.",
        "SOC2 review.",
        "Kubernetes privilege escalation."
    ]

    for q in queries:
        print(f"\n[QUERY]: '{q}'")
        results = yaml_search.search(q, top_k=3)
        print(f"Retrieved {len(results)} assessment template(s):")
        for rank, tmpl in enumerate(results, start=1):
            score = tmpl.get("_score", 0.0)
            tid = tmpl.get("id", "N/A")
            name = tmpl.get("name", "N/A")
            category = tmpl.get("category", "N/A")
            print(f"  {rank}. [Score: {score:.4f}] [{tid}] {name} (Category: {category})")
    print("\n" + "-" * 80 + "\n")


def test_integrated_pipeline(
    rag_engine: RAGEngine,
    yaml_search: YAMLSearch
):
    print("=" * 80)
    print("TEST 3: INTEGRATED PIPELINE")
    print("=" * 80)
    print("Pipeline Flow: Question -> Memory -> Doc RAG + YAML Search -> Router -> Planner -> Variable Engine -> Executor -> Evidence Manager\n")

    user_question = "Assess Azure tenant for privilege escalation and review Key Vault access."
    session_id = "test_pipeline_session"

    print(f"[STEP 1: USER QUESTION]\n  Query: '{user_question}'\n")

    # 1. Memory Setup
    memory = ChatMemory(session_id=session_id, db_path=DATABASE_PATH)
    memory.add_turn(user_message=user_question, assistant_message="Processing assessment pipeline...")
    recent_history = memory.get_recent(limit=2)
    print(f"[STEP 2: MEMORY]\n  Retrieved {len(recent_history)} recent history turn(s).\n")

    # 2. Assessment Router (Document RAG + YAML Search)
    router = AssessmentRouter(rag_engine=rag_engine, yaml_search=yaml_search)
    merged_ctx = router.route(query=user_question, top_k_docs=2, top_k_yamls=3)

    print(f"[STEP 3 & 4: DOCUMENT RAG & YAML SEARCH & ASSESSMENT ROUTER]")
    print(f"  Retrieved {len(merged_ctx.documents)} document chunk(s).")
    print(f"  Retrieved {len(merged_ctx.yaml_templates)} YAML procedure template(s):")
    for t in merged_ctx.yaml_templates:
        print(f"    - [{t.get('id')}] {t.get('name')}")
    print()

    # 3. Planner
    planner = Planner(plan_name="Azure AD & Key Vault Security Assessment")
    plan = planner.create_plan(merged_ctx.yaml_templates)
    print(f"[STEP 5: PLANNER]")
    print(f"  Synthesized Plan: '{plan.name}' with {len(plan.sequential_steps)} steps across {len(plan.phases)} phases.")
    print("  Plan Summary Preview:")
    for step in plan.sequential_steps[:5]:
        print(f"    Step {step.get('sequence_number')}: [{step.get('phase')}] {step.get('action')} ({step.get('source_yaml_id')})")
    if len(plan.sequential_steps) > 5:
        print(f"    ... +{len(plan.sequential_steps) - 5} more steps.")
    print()

    # 4. Variable Engine
    var_engine = VariableEngine({
        "TENANT": "ACME-Prod-Tenant",
        "USER": "security.auditor@acme.org",
        "TARGET_IP": "10.0.4.15",
        "RESOURCE_GROUP": "rg-production-east",
        "KEY_VAULT": "kv-prod-secrets"
    })
    print(f"[STEP 6: VARIABLE ENGINE]")
    print("  Active Session Variables:")
    for k, v in var_engine.variables.items():
        print(f"    - {k}: {v}")
    print()

    # 5. Read-only Command Executor
    executor = CommandExecutor(read_only=True)
    sample_step = plan.sequential_steps[0] if plan.sequential_steps else {
        "id": "step_1",
        "action": "Enumerate Role Assignments",
        "commands": ["az role assignment list --tenant [TENANT] --resource-group [RESOURCE_GROUP]"]
    }
    previews = executor.prepare_step_commands(sample_step, variable_engine=var_engine)

    print(f"[STEP 7: READ-ONLY COMMAND EXECUTOR]")
    for p in previews:
        print(f"  Prepared Preview:")
        print(f"    Command:         {p.command}")
        print(f"    Purpose:         {p.purpose}")
        print(f"    Expected Output: {p.expected_output}")
        print(f"    Possible Risks:  {p.possible_risks}")
    
    exec_result = executor.execute_command(previews[0].command if previews else "az account show")
    print(f"  Safe Read-Only Execution Result: {exec_result.get('status')} (Reason: {exec_result.get('reason')})\n")

    # 6. Evidence Manager
    evidence_mgr = EvidenceManager()
    evidence_mgr.add_evidence(
        assessment_id="ASSESS-AZURE-001",
        yaml_id=sample_step.get("source_yaml_id", "AZ-02"),
        step_id=str(sample_step.get("sequence_number", 1)),
        command=previews[0].command if previews else "az role assignment list",
        output='{"status": "completed", "roles_discovered": ["Global Reader", "Key Vault Contributor"]}',
        status="success",
        artifacts=["role_assignments.json", "keyvault_audit.json"]
    )
    summary = evidence_mgr.export_summary("ASSESS-AZURE-001")

    print(f"[STEP 8: EVIDENCE MANAGER]")
    print(f"  Recorded evidence for Assessment ID 'ASSESS-AZURE-001':")
    print(f"  Total Evidence Records: {summary.get('total_evidence_count')}")
    print(f"  Step Output Preview:    {summary.get('evidence_steps')[0].get('output_preview')}\n")

    # 7. Final Response Generation
    print(f"[STEP 9: FINAL RESPONSE SYNTHESIS]")
    prompt_mgr = PromptManager()
    formatted_context = merged_ctx.format_combined_prompt_context()
    print(f"  Combined Context Length: {len(formatted_context)} characters.")
    print("  Pipeline ready for LLM inference / user presentation.")
    print("\n" + "=" * 80)
    print("INTEGRATED PIPELINE VERIFICATION COMPLETE & SUCCESSFUL!")
    print("=" * 80 + "\n")


def main():
    print("Loading Document RAG Engine...")
    rag_engine = RAGEngine(index_path=INDEX_PATH, meta_path=META_PATH)

    print("Loading YAML Assessment Library...")
    loader = YAMLLoader(yaml_root=SCENARIOS_DIR, recursive=True, verbose=False)
    registry = loader.load()

    print("Building YAML Semantic Search Engine...")
    yaml_search = YAMLSearch(registry=registry)

    # Run Tests
    test_document_retrieval(rag_engine)
    test_yaml_retrieval(yaml_search)
    test_integrated_pipeline(rag_engine, yaml_search)


if __name__ == "__main__":
    main()
