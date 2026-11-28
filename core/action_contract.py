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
    # Request-volume hint for the stealth layer (core/stealth.py). "high" marks
    # loud scanners (sqlmap --crawl, full nuclei, ffuf wordlists) that send many
    # requests; on a real target under a stealth profile these are gated off in
    # favor of surgical actions. Default "low" = unaffected, so existing actions
    # and lab runs behave exactly as before.
    request_volume: Literal["low", "high"] = "low"
    # Suspicion gate (2026-11-16): the vuln class this action tests, and whether
    # the planner may offer it ONLY after a matching suspicion (an open or
    # confirmed Hypothesis) already exists. This enforces "form a suspicion
    # BEFORE firing the tool" deterministically -- an offensive/loud action is
    # withheld from the REGISTERED ACTIONS catalog until the evidence chain
    # warrants it, instead of being offered every turn for the model to fire
    # blindly (the list-walking failure mode). Recon/coverage actions leave both
    # at their defaults and are therefore always offered. See action_is_unlocked().
    vuln_class: Optional[str] = None
    requires_suspicion: bool = False


class AIActionProposal(BaseModel):
    """
    Contract for AI planner decisions.
    The AI proposes structured parameters, NEVER shell commands or raw argv.
    """
    # Full C1 extension: three evidence-chain decision verbs alongside the original
    # two. form_hypothesis/record_validation/record_finding let the AI build the
    # OBSERVATION -> HYPOTHESIS -> VALIDATION -> FINDING chain explicitly instead of
    # relying solely on BaseAgent's deterministic satisfies_phase promotion as the
    # only path to a Finding. Framework-side id verification (not this Pydantic model)
    # is what stops a fabricated obs_id/hyp_id from being accepted -- see
    # base_agent.run_autonomous() and AUDIT.md "Option A".
    decision: Literal[
        "propose_action", "propose_completion",
        "form_hypothesis", "record_validation", "record_finding",
        "update_app_model",
    ] = "propose_action"
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
        # MED-6 fix: json.dumps can raise TypeError for a parameter value that isn't
        # JSON-serializable (bytes, datetime, a set, a custom object, ...). Typed
        # Pydantic params make this unlikely in practice, but compute_identity() backs
        # the retry-budget gate (Gate 6) -- an uncaught exception here would crash the
        # whole action-identity computation (and the executor with it), which is a far
        # worse failure than a slightly less precise identity hash. repr() can encode
        # any Python object, so it's used as a fallback rather than letting this raise.
        try:
            encoded = json.dumps(raw_key, sort_keys=True).encode("utf-8")
        except TypeError:
            encoded = repr(raw_key).encode("utf-8", errors="backslashreplace")
        return hashlib.sha256(encoded).hexdigest()[:16]



def action_is_unlocked(defn, open_vuln_classes, any_suspicion_exists: bool) -> bool:
    """
    Suspicion gate. Decide whether an action may be OFFERED to the planner given
    the current evidence chain.

    SUSPICION-BEFORE-FIRING is preserved: an offensive/gated action is still
    withheld until the model has formed at least one open hypothesis. What
    changed (BUILD SPEC v1 workstream B, Part 3) is that the vuln_class STRING
    match is no longer a HARD LOCK -- it is a soft prioritization hint handled by
    the planner's ranking, not an access control. The SVG-lab failure showed a
    mis-inferred class locking the model out of the correct specialist engine and
    forcing a buggy self_script; this removes that failure mode. Which engine
    actually RUNS is governed by the risk-tiered approval + blast-radius gate in
    command_executor.py (Parts 1-2), not by the class label, so loosening the
    OFFER is safe.

    - requires_suspicion=False  -> ALWAYS unlocked (recon/coverage).
    - requires_suspicion=True   -> unlocked once ANY open/confirmed hypothesis
      exists (named class OR vuln_class=None). The class no longer has to match.

    `open_vuln_classes` is retained in the signature (still passed by callers and
    used elsewhere for ranking) but no longer gates the unlock. This is a pure
    function so the planner and the tests share one definition.
    """
    if not getattr(defn, "requires_suspicion", False):
        return True
    # Soft prioritization, not a hard lock: a formed suspicion of ANY class is
    # sufficient basis to OFFER the engine; the approval gate decides execution.
    return bool(any_suspicion_exists)


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

        # BUG FIX (found via live GPU smoke test, Qwen2.5-7B-Instruct + qwen-qlora,
        # pentest agent, 2026-11-14): AIPlanner._parse_proposal() already resolves a
        # safe top-level proposal.target (falling back to scope.target when the
        # model's JSON omits it) -- but that fallback never reached the NESTED
        # proposal.parameters dict that params_model actually validates against. A
        # weak local model that supplies action_id and some params but forgets to
        # repeat "target" inside "parameters" (observed on 10/10 steps of one run --
        # host_discovery, service_scan, vuln_scan, web_enum, smb_enum all rejected on
        # the exact same missing-field error) burned its entire retry budget on a
        # value the framework had already safely resolved, never reaching Gate 2's
        # real scope check at all.
        #
        # This fills proposal.parameters["target"] from the already-resolved
        # proposal.target ONLY when: the action's params_model actually declares a
        # "target" field, AND the model didn't already supply one itself (never
        # overrides an explicit value), AND a fallback value exists. The injected
        # value is proposal.target -- the SAME value Gate 2 (out-of-scope rejection)
        # independently re-checks against scope.is_asset_in_scope() right after this
        # -- so this does not bypass or weaken scope enforcement; it just stops an
        # omittable, already-known value from failing Gate 1 before Gate 2 gets a
        # chance to do the real authorization check.
        params = dict(proposal.parameters or {})
        if "target" in defn.params_model.model_fields and not params.get("target"):
            fallback_target = getattr(proposal, "target", None)
            if fallback_target:
                params["target"] = fallback_target

        try:
            typed_params = defn.params_model(**params)
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
            required_params: List[str] = []
            try:
                full_schema = defn.params_model.model_json_schema()
                schema = full_schema.get("properties", {})
                required_params = full_schema.get("required", [])
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
                "request_volume": defn.request_volume,
                "vuln_class": defn.vuln_class,
                "requires_suspicion": defn.requires_suspicion,
                "allowed_target_kinds": defn.allowed_target_kinds,
                "parameters_schema": schema,
                "required_params": required_params,
            })
        return catalog

