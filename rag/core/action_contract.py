"""
core/action_contract.py

Core architectural contract for structured actions, typed parameter validation,
and per-agent action registries.

Enforces:
1. Pure structured action representation (no shell strings emitted by AI).
2. Pydantic typed parameter validation with extra="forbid".
3. Framework-governed policy attributes (network_effect, state_change, risk_level).
4. Deterministic action identity hashing for per-target retry tracking.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Literal, Optional, Type
from pydantic import BaseModel, ConfigDict, Field


class ActionDefinition(BaseModel):
    """
    Formal definition of an approved, safety-governed action.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    action_id: str
    phase: str
    description: str
    requires_active_testing: bool = False
    network_effect: Literal["none", "read_only", "active"] = "none"
    state_change: Literal["none", "target_modify", "credential_access"] = "none"
    risk_level: Literal["low", "medium", "high"] = "low"
    params_model: Type[BaseModel]
    allowed_target_kinds: List[Literal["ip", "cidr", "domain", "url", "local"]] = Field(
        default_factory=lambda: ["local"]
    )
    command_builder: str  # Function name in command_builders.py or callable
    timeout_seconds: int = 60
    max_retries: int = 2
    satisfies_phase: bool = True


class AIActionProposal(BaseModel):
    """
    Contract for AI planner decisions.
    The AI proposes structured parameters, NEVER shell commands or raw argv.
    """
    decision: Literal["propose_action", "propose_completion"] = "propose_action"
    action_id: Optional[str] = None
    target: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    confidence: float = 1.0

    def compute_identity(self) -> str:
        """
        Compute a stable deterministic hash for retry budget and deduplication.
        Tracks (action_id, target, normalized_params).
        """
        raw_key = {
            "action_id": self.action_id or "",
            "target": str(self.target or "").strip().lower(),
            "params": sorted(self.parameters.items()) if self.parameters else [],
        }
        encoded = json.dumps(raw_key, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


class ActionRegistry:
    """
    Per-agent registry holding all authorized actions, their schemas,
    and command builders.
    """

    def __init__(self, agent_name: str = "generic"):
        self.agent_name = agent_name
        self.actions: Dict[str, ActionDefinition] = {}
        self.builders: Dict[str, Callable[[BaseModel], List[List[str]]]] = {}

    def register(
        self,
        definition: ActionDefinition,
        builder: Optional[Callable[[BaseModel], List[List[str]]]] = None,
    ) -> None:
        """Register an action definition and optional builder function."""
        self.actions[definition.action_id] = definition
        if builder:
            self.builders[definition.command_builder] = builder

    def register_builder(
        self,
        name: str,
        builder: Callable[[BaseModel], List[List[str]]],
    ) -> None:
        """Explicitly register a command builder by name."""
        self.builders[name] = builder

    def get(self, action_id: str) -> Optional[ActionDefinition]:
        return self.actions.get(action_id)

    def list_actions(self) -> List[ActionDefinition]:
        return list(self.actions.values())

    def get_builder(self, builder_name: str) -> Optional[Callable[[BaseModel], List[List[str]]]]:
        return self.builders.get(builder_name)

    def validate_proposal(self, proposal: AIActionProposal) -> tuple[bool, Optional[BaseModel], Optional[str]]:
        """
        Validate that proposal targets a registered action and parameters match
        its typed Pydantic model (with extra='forbid').
        """
        if not proposal.action_id:
            return False, None, "No action_id specified in proposal."

        defn = self.get(proposal.action_id)
        if not defn:
            return False, None, f"Action '{proposal.action_id}' is not registered for agent '{self.agent_name}'."

        try:
            typed_params = defn.params_model(**proposal.parameters)
            return True, typed_params, None
        except Exception as exc:
            return False, None, f"Parameter validation failed for '{proposal.action_id}': {exc}"

    def to_prompt_schema(self) -> List[Dict[str, Any]]:
        """
        Export registered actions as schema documentation for the AI planner prompt.
        """
        catalog = []
        for defn in self.actions.values():
            schema = {}
            try:
                schema = defn.params_model.model_json_schema().get("properties", {})
            except Exception:
                pass
            catalog.append({
                "action_id": defn.action_id,
                "phase": defn.phase,
                "description": defn.description,
                "risk_level": defn.risk_level,
                "network_effect": defn.network_effect,
                "state_change": defn.state_change,
                "requires_active_testing": defn.requires_active_testing,
                "allowed_target_kinds": defn.allowed_target_kinds,
                "parameters_schema": schema,
            })
        return catalog

