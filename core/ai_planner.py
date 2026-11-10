"""
core/ai_planner.py

AI Planner — the reasoning engine that proposes structured actions.

ARCHITECTURE CONTRACT:
- The AI is a consultant, NEVER a driver.
- The AI proposes structured AIActionProposal objects, NEVER shell strings.
- The framework enforces retry budgets, scope checks, and approval gates
  BEFORE any action is executed.
- The planner's output is ALWAYS validated by ActionRegistry before use.

PROMPT CONTRACT (enforced by the planner):
The AI must return EXACTLY one JSON object per call:
{
  "decision": "propose_action" | "propose_completion"
            | "form_hypothesis" | "record_validation" | "record_finding",
  "action_id": "<registered action_id or null>",
  "target": "<target string or null>",
  "parameters": { ... },
  "rationale": "<brief reasoning>",
  "confidence": 0.0..1.0
}

DECISION VERBS (full C1 extension -- see AUDIT.md "Option A"):
Running an action only ever produces an Observation. To build a real evidence
chain (OBSERVATION -> HYPOTHESIS -> VALIDATION -> FINDING) the AI must also use
the three evidence verbs below. The framework independently checks every
obs_id/hyp_id it is given against what actually exists in the EvidenceStore
and rejects fabricated ones -- this is a planner capability, not a trust
upgrade. A deterministic, framework-only promotion
(BaseAgent._promote_evidence_if_warranted) still runs underneath this as a
floor so a run is never *solely* dependent on the AI using these verbs
correctly.
  - "form_hypothesis":  parameters={observation_ids, description, validation_test, confidence}
  - "record_validation": parameters={hypothesis_id, observation_id, verdict, reasoning}
  - "record_finding":    parameters={hypothesis_id, title, description, severity,
                                      affected_target, evidence_summary,
                                      reproducible_steps, impact, confidence}
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from core.action_contract import ActionRegistry, AIActionProposal


# ---------------------------------------------------------------------------
# System prompt template for AI planner
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """\
You are a structured security assessment AI planner operating inside a safety-gated framework.

YOUR ROLE:
- Analyze the current assessment state, completed actions, and the evidence chain so far.
- Propose the SINGLE most valuable next step from the five decisions below.
- Build the evidence chain explicitly: OBSERVATION -> HYPOTHESIS -> VALIDATION -> FINDING.
  Running an action only ever produces an Observation. A Finding requires you to also
  form a Hypothesis and record a Validation that supports it first -- the framework
  refuses to create a Finding without a confirmed Hypothesis backed by a supporting
  Validation already on record.

DECISIONS (choose exactly one per turn):

1. "propose_action" -- run a registered tool action.
   parameters: that action's own typed parameters (see REGISTERED ACTIONS below).

2. "form_hypothesis" -- propose a candidate explanation from observations you have
   ALREADY seen below (never ones you expect, assume, or recall from memory).
   parameters: {
     "observation_ids": ["<obs_id copied verbatim from OBSERVATIONS below>", ...],
     "description": "<what you think this evidence shows>",
     "validation_test": "<the action_id that would prove/disprove it, if relevant>",
     "confidence": "speculative" | "plausible" | "likely"
   }

3. "record_validation" -- judge whether one specific observation supports or refutes
   an open hypothesis.
   parameters: {
     "hypothesis_id": "<hyp_id copied verbatim from ACTIVE HYPOTHESES below>",
     "observation_id": "<obs_id copied verbatim from OBSERVATIONS below>",
     "verdict": "supports" | "refutes" | "inconclusive",
     "reasoning": "<why this observation's actual output supports/refutes it>"
   }

4. "record_finding" -- promote a hypothesis you have already validated with
   verdict="supports" into a Finding. The framework independently re-checks that the
   hypothesis is confirmed and will refuse this decision (telling you why) if it is not.
   parameters: {
     "hypothesis_id": "<hyp_id of a hypothesis you have already validated>",
     "title": "<short finding title>",
     "description": "<what was found>",
     "severity": "info" | "low" | "medium" | "high" | "critical",
     "affected_target": "<target>",
     "evidence_summary": "<concrete evidence, quote real output>",
     "reproducible_steps": ["<step>", ...],
     "impact": "<practical impact>",
     "confidence": 0.0-1.0
   }

5. "propose_completion" -- declare the assessment complete. The framework
   independently verifies this against required phases and minimum findings; it will
   reject premature completion and tell you exactly what is still missing.

RULES (NON-NEGOTIABLE):
1. You MUST return exactly one JSON object — no markdown, no explanation outside JSON.
2. You MUST only use action_id values from the "REGISTERED ACTIONS" list for propose_action.
3. You MUST NOT emit shell commands, shell strings, or argv vectors.
4. You MUST NOT propose an action that has already hit its retry budget (see RETRY STATUS).
5. obs_id / hyp_id values MUST be copied verbatim from the sections below. An id you
   invent, paraphrase, or recall from memory instead of copying will be rejected — the
   framework checks every id against what actually exists in the evidence store.
6. A hypothesis cannot become a Finding until it has been explicitly validated with
   "record_validation" and verdict="supports". Running an action is not enough by itself.

OUTPUT FORMAT (propose_action example):
{
  "decision": "propose_action",
  "action_id": "<action_id from catalog>",
  "target": "<target or null>",
  "parameters": {},
  "rationale": "<one sentence>",
  "confidence": 0.85
}
"""

_PLANNER_USER_TEMPLATE = """\
=== ASSESSMENT STATE ===
Target: {target}
Mode: {mode}
Active Testing Authorized: {active_testing}
Active Network Authorized: {active_network}
State Changes Authorized: {state_changes}

=== REGISTERED ACTIONS ===
{action_catalog}

=== COMPLETION CRITERIA ===
{completion_status}

=== RETRY BUDGET STATUS ===
{retry_status}

=== RECENT EXECUTION HISTORY (last {history_count} steps) ===
{execution_history}

=== DISCOVERED FACTS & ENTITIES ===
{discovered_facts}

=== RECENT OBSERVATIONS (cite obs_id verbatim in form_hypothesis / record_validation) ===
{observations}

=== ACTIVE HYPOTHESES (cite hyp_id verbatim in record_validation / record_finding) ===
{active_hypotheses}

=== CONFIRMED FINDINGS ({finding_count}) ===
{findings_summary}

Propose the next action, evidence-chain step, or completion. Return ONLY the JSON object.
"""


class AIPlanner:
    """
    AI Planner — calls ChatAssistant.generate_raw() to obtain structured
    AIActionProposal decisions.

    The planner NEVER executes anything. It returns a validated AIActionProposal
    that the autonomous loop then passes through the 10-gate safety pipeline.
    """

    MAX_JSON_RETRIES = 3
    MAX_HISTORY_STEPS = 8

    def __init__(
        self,
        registry: ActionRegistry,
        chat_assistant,  # ChatAssistant instance
        scope,          # Scope instance
    ):
        self.registry = registry
        self.chat_assistant = chat_assistant
        self.scope = scope

    def decide_next(
        self,
        session_state: Dict[str, Any],
        evidence_store,           # EvidenceStore
        completion_criteria,      # CompletionCriteria or None
        execution_history: List[Dict[str, Any]],
        retry_budget_status: Dict[str, int],
    ) -> AIActionProposal:
        """
        Ask the AI to propose the next action or declare completion.

        Returns a validated AIActionProposal (may have decision='propose_completion').
        Raises ValueError if JSON parsing fails after MAX_JSON_RETRIES attempts.
        """
        prompt = self._build_prompt(
            session_state=session_state,
            evidence_store=evidence_store,
            completion_criteria=completion_criteria,
            execution_history=execution_history,
            retry_budget_status=retry_budget_status,
        )

        raw_output: Optional[str] = None
        last_error: Optional[str] = None

        for attempt in range(self.MAX_JSON_RETRIES):
            try:
                raw_output = self.chat_assistant.generate_raw(
                    prompt=prompt,
                    max_new_tokens=512,
                    system_prompt=_PLANNER_SYSTEM_PROMPT,
                    # M1: nudge decoding toward a real, registered action_id in the
                    # "action_id" JSON field (advisory only -- see
                    # core/constrained_decoding.py). Gate 1 registry validation in
                    # command_executor.py is unchanged and remains authoritative.
                    action_id_choices=[a.action_id for a in self.registry.list_actions()],
                )
                proposal = self._parse_proposal(raw_output)
                return proposal
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = str(exc)
                if attempt < self.MAX_JSON_RETRIES - 1:
                    # Add error context to prompt for retry
                    prompt += (
                        f"\n\n[RETRY {attempt + 1}] Your previous response could not be "
                        f"parsed as valid JSON: {last_error}. "
                        f"Return ONLY the JSON object."
                    )

        raise ValueError(
            f"AIPlanner.decide_next() failed after {self.MAX_JSON_RETRIES} attempts. "
            f"Last error: {last_error}. Last raw output: {str(raw_output)[:300]}"
        )

    def _parse_proposal(self, raw: str) -> AIActionProposal:
        """
        Extract and validate the JSON proposal from the AI's raw output.
        Handles markdown code fences and leading/trailing text.
        """
        if not raw or not raw.strip():
            raise ValueError("AI returned empty response.")

        data = self._extract_json_object(raw)

        # Validate required fields
        if "decision" not in data:
            raise ValueError("Missing 'decision' field in AI proposal.")

        decision = data["decision"]
        valid_decisions = (
            "propose_action", "propose_completion",
            "form_hypothesis", "record_validation", "record_finding",
        )
        if decision not in valid_decisions:
            raise ValueError(f"Invalid decision value: '{decision}'.")

        if decision == "propose_action" and not data.get("action_id"):
            raise ValueError("decision='propose_action' requires a non-null 'action_id'.")

        # Full C1 extension: lightweight structural validation of each evidence-verb's
        # parameters at parse time. This is NOT the authoritative check (base_agent /
        # EvidenceStore re-verify every id against what actually exists before recording
        # anything) -- it exists so a malformed verb call feeds back into the same
        # MAX_JSON_RETRIES retry loop as a JSON parse failure, with a specific error
        # message, instead of silently reaching base_agent with missing fields.
        verb_required_params = {
            "form_hypothesis": ("observation_ids", "description", "validation_test"),
            "record_validation": ("hypothesis_id", "observation_id", "verdict", "reasoning"),
            "record_finding": (
                "hypothesis_id", "title", "description", "severity",
                "affected_target", "evidence_summary",
            ),
        }
        if decision in verb_required_params:
            params = data.get("parameters") or {}
            if not isinstance(params, dict):
                raise ValueError(f"decision='{decision}' requires 'parameters' to be a JSON object.")
            missing = [k for k in verb_required_params[decision] if not params.get(k)]
            if missing:
                raise ValueError(f"decision='{decision}' is missing required parameters: {missing}")

        proposal = AIActionProposal(
            decision=decision,
            action_id=data.get("action_id"),
            target=data.get("target") or str(getattr(self.scope, "target", "local")),
            parameters=data.get("parameters", {}),
            rationale=str(data.get("rationale", ""))[:500],
            confidence=float(data.get("confidence", 0.5)),
        )
        return proposal

    def _extract_json_object(self, raw: str) -> Dict[str, Any]:
        """
        Extract the first complete, balanced JSON object from the AI's raw output.

        MED-3 fix: the previous implementation matched JSON with a single-level-nesting
        regex (`\\{[^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\}`). That silently mis-extracts (or
        fails outright on) anything with two or more levels of nested braces -- exactly
        what a `parameters` dict containing a nested object, or `reproducible_steps`
        holding dict-shaped items, produces. It also didn't account for braces that
        legitimately appear inside a JSON string value. json.JSONDecoder.raw_decode()
        does real, depth-unlimited, string-literal-aware parsing instead of
        approximating it with a regex: this scans for the first '{' from which a
        complete JSON object actually parses, regardless of markdown fences,
        leading/trailing commentary, or nesting depth.
        """
        decoder = json.JSONDecoder()
        search_from = 0
        while True:
            start = raw.find("{", search_from)
            if start == -1:
                raise ValueError(
                    f"No JSON object found in AI response. Raw output: {raw[:300]!r}"
                )
            try:
                obj, _end = decoder.raw_decode(raw, start)
                if isinstance(obj, dict):
                    return obj
                search_from = start + 1  # a '{'-start decode is always a dict; defensive only
            except json.JSONDecodeError:
                search_from = start + 1

    def _build_prompt(
        self,
        session_state: Dict[str, Any],
        evidence_store,
        completion_criteria,
        execution_history: List[Dict[str, Any]],
        retry_budget_status: Dict[str, int],
    ) -> str:
        """Build the full planner prompt from current state."""
        # Action catalog — from registry schema
        # C4 fix: render each action's parameter schema (field names, types, which are
        # required, and defaults). Previously only action_id + description were shown,
        # so the model had to guess parameter names — guesses that fail Pydantic's
        # extra="forbid" validation at Gate 1 and burn a step on every wrong guess.
        catalog_items = []
        for item in self.registry.to_prompt_schema():
            params_line = self._format_params_schema(
                item.get("parameters_schema", {}),
                item.get("required_params", []),
            )
            catalog_items.append(
                f"  - {item['action_id']} [{item['risk_level']} risk, "
                f"network={item['network_effect']}, state={item['state_change']}]: "
                f"{item['description']}\n"
                f"      parameters: {params_line}"
            )
        action_catalog = "\n".join(catalog_items) if catalog_items else "  (none registered)"

        # Completion status
        completion_status = self._format_completion_status(
            criteria=completion_criteria,
            evidence_store=evidence_store,
            execution_history=execution_history,
        )

        # Retry budget
        retry_lines = []
        for action_id, attempts in retry_budget_status.items():
            defn = self.registry.get(action_id)
            max_r = defn.max_retries if defn else "?"
            retry_lines.append(f"  - {action_id}: {attempts}/{max_r} attempts used")
        retry_status = "\n".join(retry_lines) if retry_lines else "  (no retries recorded)"

        # Recent execution history (last N steps)
        recent_history = execution_history[-self.MAX_HISTORY_STEPS:]
        hist_lines = []
        for i, entry in enumerate(recent_history):
            status = entry.get("status", "?")
            action = entry.get("action_id", "?")
            rejection = entry.get("rejection_reason", "")
            line = f"  [{i+1}] {action} → {status}"
            if rejection:
                line += f" (REJECTED: {rejection[:300]})"
            elif entry.get("results"):
                results = entry["results"]
                if isinstance(results, list) and results:
                    first = results[0]
                    stdout = str(first.get("stdout", ""))[:120]
                    if stdout:
                        line += f" | output: {stdout!r}"
            hist_lines.append(line)
        execution_hist_text = "\n".join(hist_lines) if hist_lines else "  (no history)"

        # Active hypotheses
        hyp_lines = []
        for hyp in evidence_store.hypotheses.values():
            if hyp.status in ("open", "validating"):
                hyp_lines.append(f"  [{hyp.hyp_id}] {hyp.status}: {hyp.description[:100]}")
        active_hypotheses = "\n".join(hyp_lines) if hyp_lines else "  (none)"

        # Confirmed findings
        findings_lines = []
        for fnd in evidence_store.findings.values():
            findings_lines.append(
                f"  [{fnd.finding_id}] {fnd.severity.upper()}: {fnd.title}"
            )
        findings_summary = "\n".join(findings_lines) if findings_lines else "  (none yet)"

        # C5: aggregate structured facts (open ports, subdomains, headers, ...)
        # extracted from every observation so far, instead of relying solely on
        # truncated raw stdout snippets in the execution history.
        discovered_facts = self._format_discovered_facts(evidence_store)

        # Full C1 extension: list real obs_ids the planner can cite verbatim when
        # calling form_hypothesis / record_validation. Without this the model has no
        # way to reference a real observation except by copying truncated stdout text
        # from the execution-history section, which is not an id the framework can verify.
        observations_text = self._format_observations(evidence_store)

        return _PLANNER_USER_TEMPLATE.format(
            target=getattr(self.scope, "target", "local"),
            mode="automated",
            # C3 alignment fix: show all three authorization flags the executor's
            # Gate 3 actually enforces (active_testing_allowed, active_network_allowed,
            # state_changes_allowed) distinctly -- previously this line was labeled
            # "Active Testing Authorized" but read active_network_allowed's value,
            # and active_testing_allowed was never shown to the model at all.
            active_testing=getattr(self.scope, "active_testing_allowed", False),
            active_network=getattr(self.scope, "active_network_allowed", False),
            state_changes=getattr(self.scope, "state_changes_allowed", False),
            action_catalog=action_catalog,
            completion_status=completion_status,
            retry_status=retry_status,
            history_count=len(recent_history),
            execution_history=execution_hist_text,
            discovered_facts=discovered_facts,
            observations=observations_text,
            active_hypotheses=active_hypotheses,
            finding_count=len(evidence_store.findings),
            findings_summary=findings_summary,
        )

    def _format_params_schema(self, schema: Dict[str, Any], required: List[str]) -> str:
        """
        Render a Pydantic-derived JSON schema's properties as a compact,
        model-readable parameter list: 'name:type (required)' or 'name:type=default'.
        """
        if not schema:
            return "(no parameters)"
        parts = []
        for name, spec in schema.items():
            ptype = spec.get("type", "any")
            if name in required:
                parts.append(f"{name}:{ptype} (required)")
            else:
                default = spec.get("default")
                if default is not None:
                    parts.append(f"{name}:{ptype}={default!r}")
                else:
                    parts.append(f"{name}:{ptype} (optional)")
        return ", ".join(parts)

    def _format_observations(self, evidence_store, limit: int = 10) -> str:
        """
        List the most recent observations with their real obs_id, so the planner has
        something concrete to cite in form_hypothesis / record_validation. Capped to
        the most recent `limit` to keep prompt size bounded on long runs -- older
        observations are still fully available via discovered_facts (C5) and findings.
        """
        obs_list = list(evidence_store.observations.values())
        if not obs_list:
            return "  (no observations recorded yet)"
        recent = obs_list[-limit:]
        lines = []
        for obs in recent:
            excerpt = (obs.raw_stdout or "").strip().replace("\n", " ")[:150]
            lines.append(
                f"  [{obs.obs_id}] action={obs.action_id} exit_code={obs.exit_code} "
                f"stdout: {excerpt!r}"
            )
        return "\n".join(lines)

    def _format_discovered_facts(self, evidence_store) -> str:
        """
        Aggregate Observation.parsed_facts (C5) across the whole run into a compact,
        model-readable summary: list-valued facts are unioned and deduplicated
        (capped so one noisy observation can't blow out the prompt), dict-valued
        facts are merged, everything else keeps the most recent value.
        """
        aggregated: Dict[str, Any] = {}
        for obs in evidence_store.observations.values():
            facts = getattr(obs, "parsed_facts", None) or {}
            for key, value in facts.items():
                if isinstance(value, list):
                    existing = aggregated.setdefault(key, [])
                    if isinstance(existing, list):
                        for item in value:
                            if item not in existing:
                                existing.append(item)
                elif isinstance(value, dict):
                    existing = aggregated.setdefault(key, {})
                    if isinstance(existing, dict):
                        existing.update(value)
                    else:
                        aggregated[key] = value
                else:
                    aggregated[key] = value

        if not aggregated:
            return "  (no structured facts extracted yet)"

        lines = []
        for key, value in aggregated.items():
            if isinstance(value, list):
                shown = value[:15]
                suffix = f" (+{len(value) - 15} more)" if len(value) > 15 else ""
                lines.append(f"  - {key}: {shown}{suffix}")
            else:
                lines.append(f"  - {key}: {value}")
        return "\n".join(lines)

    def _format_completion_status(
        self,
        criteria,
        evidence_store,
        execution_history: List[Dict[str, Any]],
    ) -> str:
        """Summarize completion criteria against current progress."""
        if criteria is None:
            return "  No completion criteria defined."

        lines = []
        completed_action_ids = {
            e.get("action_id") for e in execution_history if e.get("status") == "success"
        }
        required = getattr(criteria, "required_phases", [])
        lines.append(f"  Required phases: {required}")
        lines.append(f"  Successful actions so far: {sorted(completed_action_ids)}")

        min_findings = getattr(criteria, "minimum_findings", 0)
        current_findings = len(evidence_store.findings)
        lines.append(
            f"  Findings: {current_findings}/{min_findings} required"
        )

        obs_required = getattr(criteria, "evidence_required_for_phases", [])
        obs_count = len(evidence_store.observations)
        lines.append(f"  Observations recorded: {obs_count} (required for phases: {obs_required})")

        return "\n".join(lines)
