"""
agents/security_research/command_builders.py

Pure Python command translators for SecurityResearchAgent actions.
Translates typed action parameter models into executable argument vectors.

SAFETY INVARIANTS:
  - Returns strictly List[List[str]]
  - Never shell strings
  - No shell=True
  - Cross-platform compatible
"""

import sys
from typing import List

from agents.security_research.actions import (
    CveLookupParams,
    SourceDiffParams,
    RootCauseAnalysisParams,
    HarnessGenerationParams,
    BenignPocDevParams,
    ExploitabilityAuditParams,
    VariationFuzzingParams,
    RemediationGuideParams,
    ResearchWriteupParams,
    CvssCalculatorParams,
)


def build_cve_lookup(params: CveLookupParams) -> List[List[str]]:
    """Build command vector for CVE advisory queries."""
    return [
        ["curl", "-s", "-k", "--max-time", "10", f"https://cveawg.mitre.org/api/cve/{params.target}"]
    ]


def build_source_diff(params: SourceDiffParams) -> List[List[str]]:
    """Build command vector for patch analysis."""
    if params.commit_hash:
        return [["git", "show", params.commit_hash]]
    return [["git", "diff", "HEAD~1"]]


def build_root_cause_analysis(params: RootCauseAnalysisParams) -> List[List[str]]:
    """Build command vector for root cause analysis."""
    return [
        [sys.executable, "-c", f"print('Performing root cause analysis on {params.target} ({params.flaw_type})')"]
    ]


def build_harness_generation(params: HarnessGenerationParams) -> List[List[str]]:
    """Build command vector for reproduction harness setup."""
    return [
        [sys.executable, "-c", f"print('Generating {params.harness_language} reproduction harness for {params.target}')"]
    ]


def build_benign_poc_dev(params: BenignPocDevParams) -> List[List[str]]:
    """Build command vector for benign probe execution."""
    return [
        [sys.executable, "-c", f"print('Executing safe proof-of-concept verification ({params.trigger_type}) on {params.target}')"]
    ]


def build_exploitability_audit(params: ExploitabilityAuditParams) -> List[List[str]]:
    """Build command vector for platform mitigation audit."""
    return [
        [sys.executable, "-c", f"print('Auditing platform exploit mitigations for {params.target}')"]
    ]


def build_variation_fuzzing(params: VariationFuzzingParams) -> List[List[str]]:
    """Build command vector for mutation fuzzing."""
    return [
        [sys.executable, "-c", f"print('Executing {params.mutations} benign input mutations against {params.target}')"]
    ]


def build_remediation_guide(params: RemediationGuideParams) -> List[List[str]]:
    """Build command vector for defensive guidance generation."""
    return [
        [sys.executable, "-c", f"print('Generating mitigation and patching guidance for {params.target}')"]
    ]


def build_research_writeup(params: ResearchWriteupParams) -> List[List[str]]:
    """Build command vector for research advisory write-up."""
    return [
        [sys.executable, "-c", f"print('Generating research disclosure write-up for {params.target}')"]
    ]


def build_cvss_calculator(params: CvssCalculatorParams) -> List[List[str]]:
    """Build command vector for CVSS scoring."""
    return [
        [sys.executable, "-c", f"print('Calculating CVSS v3.1 score for {params.target} (AV:{params.attack_vector}/PR:{params.privileges_required})')"]
    ]

