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
  "decision": "propose_action" | "propose_completion",
  "action_id": "<registered action_id or null>",
  "target": "<target string or null>",
  "parameters": { ... },
  "rationale": "<brief reasoning>",
  "confidence": 0.0..1.0
}
"""

from __future__ import annotations

import json
import time
import re
from typing import Any, Dict, List, Optional

from core.action_contract import ActionRegistry, AIActionProposal


# ---------------------------------------------------------------------------
# System prompt template for AI planner
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """\
You are a structured security assessment AI planner operating inside a safety-gated framework.

YOUR ROLE:
- Analyze the current assessment state, completed actions, and active hypotheses.
- Propose the SINGLE most valuable next action from the registered action catalog.
- If all completion criteria are verifiably met, propose completion instead.

RULES (NON-NEGOTIABLE):
1. You MUST return exactly one JSON object — no markdown, no explanation outside JSON.
2. You MUST only use action_id values from the "REGISTERED ACTIONS" list.
3. You MUST NOT emit shell commands, shell strings, or argv vectors.
4. You MUST NOT propose an action that has already hit its retry budget (see RETRY STATUS).
5. If decision is "propose_action", action_id and parameters are required.
6. If decision is "propose_completion", action_id must be null.

OUTPUT FORMAT:
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
State Changes Authorized: {state_changes}

=== REGISTERED ACTIONS ===
{action_catalog}

=== COMPLETION CRITERIA ===
{completion_status}

=== RETRY BUDGET STATUS ===
{retry_status}

=== RECENT EXECUTION HISTORY (last {history_count} steps) ===
{execution_history}

=== ACTIVE HYPOTHESES ===
{active_hypotheses}

=== CONFIRMED FINDINGS ({finding_count}) ===
{findings_summary}

Propose the next action or completion. Return ONLY the JSON object.
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

        # Strip markdown code fences if present
        text = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            # Try to extract the first JSON object from the text
            brace_match = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", text, re.DOTALL)
            if brace_match:
                text = brace_match.group(1)

        data = json.loads(text)

        # Validate required fields
        if "decision" not in data:
            raise ValueError("Missing 'decision' field in AI proposal.")

        decision = data["decision"]
        if decision not in ("propose_action", "propose_completion"):
            raise ValueError(f"Invalid decision value: '{decision}'.")

        if decision == "propose_action" and not data.get("action_id"):
            raise ValueError("decision='propose_action' requires a non-null 'action_id'.")

        proposal = AIActionProposal(
            decision=decision,
            action_id=data.get("action_id"),
            target=data.get("target") or str(getattr(self.scope, "target", "local")),
            parameters=data.get("parameters", {}),
            rationale=str(data.get("rationale", ""))[:500],
            confidence=float(data.get("confidence", 0.5)),
        )
        return proposal

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
        catalog_items = []
        for item in self.registry.to_prompt_schema():
            catalog_items.append(
                f"  - {item['action_id']} [{item['risk_level']} risk, "
                f"network={item['network_effect']}, state={item['state_change']}]: "
                f"{item['description']}"
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
                line += f" (REJECTED: {rejection[:80]})"
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

        return _PLANNER_USER_TEMPLATE.format(
            target=getattr(self.scope, "target", "local"),
            mode="automated",
            active_testing=getattr(self.scope, "active_network_allowed", False),
            state_changes=getattr(self.scope, "state_changes_allowed", False),
            action_catalog=action_catalog,
            completion_status=completion_status,
            retry_status=retry_status,
            history_count=len(recent_history),
            execution_history=execution_hist_text,
            active_hypotheses=active_hypotheses,
            finding_count=len(evidence_store.findings),
            findings_summary=findings_summary,
        )

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
