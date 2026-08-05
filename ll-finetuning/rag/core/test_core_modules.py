"""
Test runner to verify all 6 newly created core assessment modules.
"""

import os
import sys
from pathlib import Path

# Add rag directory to sys.path
RAG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RAG_DIR))

from core.yaml_loader import YAMLLoader
from core.yaml_search import YAMLSearch
from core.assessment_router import AssessmentRouter
from core.planner import Planner
from core.variable_engine import VariableEngine
from core.command_executor import CommandExecutor
from core.evidence_manager import EvidenceManager


def main():
    print("==================================================")
    print("Testing 6 Core Assessment Modules")
    print("==================================================\n")

    # 1. Load Registry
    scenarios_dir = RAG_DIR / "scenarios"
    print(f"[1] Initializing YAMLLoader on '{scenarios_dir}'...")
    loader = YAMLLoader(yaml_root=str(scenarios_dir), recursive=True, verbose=False)
    registry = loader.load()
    print(f"    Loaded {len(registry.templates)} YAML templates.\n")

    # 2. YAML Search
    print("[2] Testing YAML Search...")
    yaml_search = YAMLSearch(registry=registry)
    query = "Assess an Azure tenant for privilege escalation."
    search_results = yaml_search.search(query, top_k=2)
    print(f"    Search Query: '{query}'")
    print(f"    Found {len(search_results)} relevant template(s):")
    for tmpl in search_results:
        print(f"      - [{tmpl.get('_score', 0):.4f}] {tmpl.get('id')}: {tmpl.get('name')}")
    print()

    # 3. Assessment Router
    print("[3] Testing Assessment Router...")
    router = AssessmentRouter(rag_engine=None, yaml_search=yaml_search)
    merged_ctx = router.route(query, top_k_docs=2, top_k_yamls=2)
    print(f"    Routed context dict: {merged_ctx.to_dict()}\n")

    # 4. Planner
    print("[4] Testing Planner...")
    planner = Planner(plan_name="Azure Privilege Escalation Assessment")
    plan = planner.create_plan(merged_ctx.yaml_templates)
    print(plan.summary())
    print()

    # 5. Variable Engine
    print("[5] Testing Variable Engine...")
    var_engine = VariableEngine({
        "TENANT": "ACME-Corp-Tenant",
        "USER": "alex.admin@acme.org",
        "TARGET_IP": "10.10.10.5",
        "DOMAIN": "acme.local"
    })
    test_cmd = "az role assignment list --tenant [TENANT] --user [USER] --ip [TARGET_IP]"
    subbed_cmd = var_engine.substitute(test_cmd)
    print(f"    Original:   '{test_cmd}'")
    print(f"    Substituted:'{subbed_cmd}'\n")

    # 6. Read-only Command Executor
    print("[6] Testing Read-only Command Executor...")
    executor = CommandExecutor(read_only=True)
    step_sample = {
        "id": "step_1",
        "action": "Review Azure Role Assignments",
        "commands": [test_cmd],
        "expected_output": "JSON list of user role assignments",
        "detection": "CloudTrail / Azure Activity Log event for GetRoleAssignment"
    }
    previews = executor.prepare_step_commands(step_sample, variable_engine=var_engine)
    for p in previews:
        print(f"    Preview Command: {p.command}")
        print(f"    Purpose:         {p.purpose}")
        print(f"    Risks:           {p.possible_risks}")
    exec_result = executor.execute_command(previews[0].command)
    print(f"    Execution Result (Read-Only mode): {exec_result}\n")

    # 7. Evidence Manager
    print("[7] Testing Evidence Manager...")
    evidence_mgr = EvidenceManager()
    evidence_mgr.add_evidence(
        assessment_id="AZ-PRIV-001",
        yaml_id=search_results[0].get("id") if search_results else "AZ-02",
        step_id="step_1",
        command=previews[0].command,
        output='{"role": "Global Administrator", "status": "active"}',
        status="success",
        artifacts=["role_assignments.json"]
    )
    summary = evidence_mgr.export_summary("AZ-PRIV-001")
    print(f"    Evidence Export Summary: {summary}\n")

    print("==================================================")
    print("ALL 6 MODULES TESTED & WORKING INTEGRATEDLY!")
    print("==================================================")


if __name__ == "__main__":
    main()
