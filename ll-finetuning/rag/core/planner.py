"""
core/planner.py

Purpose
-------
Synthesize one or more YAML assessment templates into a logical, ordered assessment plan.

Responsibilities
----------------
1. Merge multiple assessment checklists into unified tactical phases.
2. Maintain standard methodology ordering (Recon -> Enum -> PrivEsc -> Persistence -> Evidence).
3. Assign sequential step numbers and track source YAML origins.
"""

from typing import List, Dict, Any, Optional


PHASE_ORDER = [
    "reconnaissance",
    "recon",
    "enumeration",
    "initial_access",
    "privilege_escalation",
    "privesc",
    "lateral_movement",
    "persistence",
    "exfiltration",
    "evidence_collection",
    "cleanup"
]


def get_phase_rank(phase: str) -> int:
    """Return standard ordering rank for a phase name."""
    clean_phase = phase.strip().lower().replace(" ", "_").replace("-", "_")
    for rank, known in enumerate(PHASE_ORDER):
        if known in clean_phase or clean_phase in known:
            return rank
    return 99  # Default fallback for unknown phases


class AssessmentPlan:
    """
    Structured assessment plan synthesized from template procedures.
    """

    def __init__(
        self,
        name: str,
        sources: List[str],
        phases: Dict[str, List[Dict[str, Any]]],
        sequential_steps: List[Dict[str, Any]]
    ):
        self.name = name
        self.sources = sources
        self.phases = phases
        self.sequential_steps = sequential_steps

    def summary(self) -> str:
        """Return formatted summary of the assessment plan."""
        lines = [
            f"=== ASSESSMENT PLAN: {self.name} ===",
            f"Source Templates: {', '.join(self.sources)}",
            f"Total Steps: {len(self.sequential_steps)}",
            ""
        ]

        current_phase = None
        for step in self.sequential_steps:
            phase = step.get("phase", "General").title()
            if phase != current_phase:
                current_phase = phase
                lines.append(f"\n[{current_phase}]")

            seq_num = step.get("sequence_number")
            action = step.get("action", "")
            tmpl_id = step.get("source_yaml_id", "")
            lines.append(f"  {seq_num}. {action} ({tmpl_id})")

        return "\n".join(lines)


class Planner:
    """
    Synthesizes YAML templates into unified assessment execution plans.
    """

    def __init__(self, plan_name: str = "Custom Assessment Plan"):
        self.default_name = plan_name

    def create_plan(
        self,
        templates: List[Dict[str, Any]],
        custom_name: Optional[str] = None
    ) -> AssessmentPlan:
        """
        Merge templates into a structured assessment plan sorted by methodology phases.

        Args:
            templates: List of YAML template dicts retrieved by YAML search or router
            custom_name: Optional override name for the assessment plan

        Returns:
            AssessmentPlan instance
        """
        plan_name = custom_name or self.default_name
        source_ids = []
        raw_steps = []

        for tmpl in templates:
            tmpl_id = tmpl.get("id", "unknown")
            source_ids.append(tmpl_id)
            tmpl_name = tmpl.get("name", tmpl_id)

            steps = tmpl.get("steps", [])
            for step in steps:
                if isinstance(step, dict):
                    step_copy = dict(step)
                    step_copy["source_yaml_id"] = tmpl_id
                    step_copy["source_yaml_name"] = tmpl_name
                    raw_steps.append(step_copy)

        # Sort steps primary by phase methodology rank, secondary by original ID/index
        raw_steps.sort(key=lambda s: get_phase_rank(s.get("phase", "")))

        # Assign global sequential step numbers and group into phases
        phases: Dict[str, List[Dict[str, Any]]] = {}
        sequential_steps: List[Dict[str, Any]] = []

        for idx, step in enumerate(raw_steps, start=1):
            step["sequence_number"] = idx
            phase_name = step.get("phase", "general").lower()
            
            if phase_name not in phases:
                phases[phase_name] = []
            phases[phase_name].append(step)
            sequential_steps.append(step)

        return AssessmentPlan(
            name=plan_name,
            sources=source_ids,
            phases=phases,
            sequential_steps=sequential_steps
        )
