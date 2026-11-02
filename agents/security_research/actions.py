"""
agents/security_research/actions.py

Action definitions and registry factory for SecurityResearchAgent.
Covers 0-day / N-day vulnerability research: CVE advisory analysis, patch diffing,
root cause analysis, benign PoC harness development, mitigation assessment, and write-ups.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition, ActionRegistry


# ---------------------------------------------------------------------------
# Typed Action Parameter Models (Pydantic v2, extra="forbid")
# ---------------------------------------------------------------------------

class CveLookupParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target CVE ID, product name, or advisory reference (e.g. CVE-2023-38606)")


class SourceDiffParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Repository URL or local patch diff file")
    commit_hash: Optional[str] = Field(None, description="Optional commit SHA for patch diffing")


class RootCauseAnalysisParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target component or vulnerability identifier")
    flaw_type: str = Field("memory_corruption", description="Flaw type (memory_corruption, logic_bug, auth_bypass, injection)")


class HarnessGenerationParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target application or library name")
    harness_language: str = Field("python", description="Test harness language (python, c)")


class BenignPocDevParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target service URL or binary path")
    trigger_type: str = Field("benign_crash_probe", description="PoC probe mode (benign_crash_probe, memory_leak_probe)")


class ExploitabilityAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target binary or execution environment")


class VariationFuzzingParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target endpoint or parser harness")
    mutations: int = Field(50, description="Number of benign input mutations to evaluate")


class RemediationGuideParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Vulnerability or CVE identifier")


class ResearchWriteupParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Research project or CVE identifier")
    output_format: str = Field("markdown", description="Advisory output format")


class CvssCalculatorParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target vulnerability title")
    attack_vector: str = Field("network", description="AV (network, adjacent, local, physical)")
    privileges_required: str = Field("none", description="PR (none, low, high)")


# ---------------------------------------------------------------------------
# Action Definitions
# ---------------------------------------------------------------------------

CVE_LOOKUP = ActionDefinition(
    action_id="cve_lookup",
    phase="cve_research",
    description="Retrieve CVE details, CVSS metrics, vendor advisories, and public references",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=CveLookupParams,
    allowed_target_kinds=["domain", "url", "local"],
    command_builder="build_cve_lookup",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=True,
)

SOURCE_DIFF = ActionDefinition(
    action_id="source_diff",
    phase="cve_research",
    description="Analyze source code diffs and patch commits to isolate security fixes and bug introduction",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=SourceDiffParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_source_diff",
    timeout_seconds=45,
    max_retries=1,
    satisfies_phase=True,
)

ROOT_CAUSE_ANALYSIS = ActionDefinition(
    action_id="root_cause_analysis",
    phase="analysis",
    description="Model vulnerability mechanism, memory state, and input parsing flow causing the flaw",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=RootCauseAnalysisParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_root_cause_analysis",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

HARNESS_GENERATION = ActionDefinition(
    action_id="harness_generation",
    phase="poc_dev",
    description="Generate a localized, sandboxed test harness to reproduce target behavior safely",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=HarnessGenerationParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_harness_generation",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

BENIGN_POC_DEV = ActionDefinition(
    action_id="benign_poc_dev",
    phase="poc_dev",
    description="Construct non-destructive proof-of-concept probe to verify flaw existence",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=BenignPocDevParams,
    allowed_target_kinds=["local", "domain", "url"],
    command_builder="build_benign_poc_dev",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

EXPLOITABILITY_AUDIT = ActionDefinition(
    action_id="exploitability_audit",
    phase="analysis",
    description="Evaluate platform mitigations (ASLR, DEP, SafeSEH, CFI, seccomp) impacting exploitability",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=ExploitabilityAuditParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_exploitability_audit",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

VARIATION_FUZZING = ActionDefinition(
    action_id="variation_fuzzing",
    phase="poc_dev",
    description="Execute benign input mutation fuzzing to identify adjacent boundary conditions",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=VariationFuzzingParams,
    allowed_target_kinds=["local", "domain", "url"],
    command_builder="build_variation_fuzzing",
    timeout_seconds=60,
    max_retries=1,
    satisfies_phase=True,
)

REMEDIATION_GUIDE = ActionDefinition(
    action_id="remediation_guide",
    phase="writeup",
    description="Generate defensive mitigation guidelines, patch recommendations, and detection rules",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=RemediationGuideParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_remediation_guide",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

RESEARCH_WRITEUP = ActionDefinition(
    action_id="research_writeup",
    phase="writeup",
    description="Synthesize comprehensive vulnerability research advisory and root cause analysis report",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=ResearchWriteupParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_research_writeup",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

CVSS_CALCULATOR = ActionDefinition(
    action_id="cvss_calculator",
    phase="analysis",
    description="Calculate CVSS v3.1 vector string and qualitative severity score",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=CvssCalculatorParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_cvss_calculator",
    timeout_seconds=20,
    max_retries=1,
    satisfies_phase=True,
)


def build_research_registry() -> ActionRegistry:
    """Build and return the ActionRegistry for SecurityResearchAgent."""
    registry = ActionRegistry()
    registry.register(CVE_LOOKUP)
    registry.register(SOURCE_DIFF)
    registry.register(ROOT_CAUSE_ANALYSIS)
    registry.register(HARNESS_GENERATION)
    registry.register(BENIGN_POC_DEV)
    registry.register(EXPLOITABILITY_AUDIT)
    registry.register(VARIATION_FUZZING)
    registry.register(REMEDIATION_GUIDE)
    registry.register(RESEARCH_WRITEUP)
    registry.register(CVSS_CALCULATOR)
    return registry

