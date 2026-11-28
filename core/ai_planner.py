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
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from core.action_contract import ActionRegistry, AIActionProposal, action_is_unlocked
from core.campaign_judgment import render_campaign_judgment

# REASON-THEN-FORMAT planner path (2026-11-25). Hypothesis under test: the
# in-tool model underperforms partly because its ONLY call is constrained to
# "return exactly one compact JSON object, no text around it" -- exactly the
# output shape a reasoning model is weakest at, and exactly where raw Ollama
# chat (free prose, thinking allowed) is strongest. This splits decide_next()
# into two LLM calls when enabled:
#   Call 1 (reasoning): the SAME state/context, but asked to think freely in
#     plain prose -- no JSON constraint at all.
#   Call 2 (format): the STRICT prompt/schema as before, with Call 1's own
#     reasoning text inserted as additional context, asked to return ONLY the
#     JSON action implied by that reasoning.
# All gates and confirmation rigor are UNCHANGED -- this only affects how a
# proposal is PRODUCED, never how it is validated or executed. Off by default
# (PLANNER_REASON_THEN_FORMAT unset): decide_next() is byte-identical to the
# single-call path that existed before this toggle.
_REASON_THEN_FORMAT_ENV = "PLANNER_REASON_THEN_FORMAT"


def reason_then_format_enabled() -> bool:
    """True only when the operator has explicitly opted in via env var."""
    return os.environ.get(_REASON_THEN_FORMAT_ENV) == "1"


# ---------------------------------------------------------------------------
# System prompt template for AI planner
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert security investigator operating inside a safety-gated assessment framework.

YOUR MISSION:
Reason like a human security researcher — do NOT pattern-match to a fixed catalog of vulnerability classes.
You discover vulnerabilities (injection, business logic, race conditions, auth flaws, access control, state bugs, request smuggling, and novel logic flaws) by modeling the target, finding fragile developer assumptions, forming specific falsifiable hypotheses, designing discriminating tests, and confirming through concrete proof.

THE 6-STEP INVESTIGATOR SCAFFOLD (Reason, do not blindly scan):
1. MODEL IT. From observations, state what this app/endpoint is FOR, its moving parts, the actors/roles, what state or limited resources exist (e.g. coupons, discounts, inventory, balances, tokens, permissions), and for each — what must the developer be ASSUMING?
2. FIND THE FRAGILE ASSUMPTION. Which assumption, if false, causes harm? Where does attacker-controlled data reach something powerful (fetcher, query, parser, auth decision, shared/limited resource)? What invariant must hold, and what enforces it (e.g. is coupon redemption atomic, or can parallel requests redeem twice)?
3. HYPOTHESIZE SPECIFICALLY. Form a clear hypothesis: "I suspect ___ because ___. The minimal test that would PROVE OR KILL it is ___. Only this flaw would explain ___." Hypotheses are free-form text — no fixed class name required!
4. DESIGN A DISCRIMINATING TEST — control vs attack, so the result cannot be coincidence. Author your own test via `self_script` (which supports requests, threading, asyncio, and raw sockets for parallel bursts and timing tests) if no specialized tool fits; a scanner is optional.
5. JUDGE THE EVIDENCE — does ONLY the vulnerability/violation explain the result? (e.g. single-use resource consumed twice, limit exceeded, negative total vs control). If not, refute and update your model.
6. CHAIN & FOLLOW SMELLS — a confirmed primitive opens new assumptions; an anomaly that fits no class is still a lead.

DECISIONS (choose exactly one per turn):

1. "update_app_model" -- record or update your working memory about the target application model:
   purpose, components, roles, trust boundaries, state/limited resources (e.g. single-use coupon, stock),
   and fragile developer assumptions.
   parameters: {
     "purpose": "<what this app/endpoint does>",
     "state_and_resources": ["<limited or shared resource, e.g. single-use discount coupon, stock>"],
     "assumptions": [
       {"target_or_component": "<endpoint/param>", "description": "<developer assumption, e.g. coupon redemption is atomic>", "fragile": true}
     ]
   }

2. "propose_action" -- run a registered tool action (or "self_script" to test a fragile assumption).
   parameters: that action's own typed parameters (see REGISTERED ACTIONS below).

3. "form_hypothesis" -- propose a candidate explanation from observations you have
   ALREADY seen below (never ones you expect, assume, or recall from memory).
   parameters: {
     "observation_ids": ["<obs_id copied verbatim from OBSERVATIONS below>", ...],
     "description": "<falsifiable hypothesis: what assumption is fragile and why>",
     "validation_test": "<action_id, e.g. self_script or tool>",
     "vuln_class": "<optional label for knowledge lookup, e.g. race-condition, logic, sqli, ssrf, or null>",
     "confidence": "speculative" | "plausible" | "likely"
   }
   Forming a hypothesis is how you UNLOCK offensive tools like self_script.

4. "record_validation" -- judge whether one specific observation supports or refutes
   an open hypothesis.
   parameters: {
     "hypothesis_id": "<hyp_id copied verbatim from ACTIVE HYPOTHESES below>",
     "observation_id": "<obs_id copied verbatim from OBSERVATIONS below>",
     "verdict": "supports" | "refutes" | "inconclusive",
     "reasoning": "<why this observation's actual output supports/refutes it>"
   }

5. "record_finding" -- promote a hypothesis you have already validated with
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

6. "propose_completion" -- declare the assessment complete. The framework
   independently verifies this against required phases and minimum findings; it will
   reject premature completion and tell you exactly what is still missing.

RULES (NON-NEGOTIABLE):
1. REASON FREELY in prose first (think out loud -- this is encouraged, not penalized),
   then end with EXACTLY ONE fenced ```action block containing your chosen decision as
   compact JSON OR simple `key: value` lines. Everything OUTSIDE the block is reasoning
   and is ignored by the parser. If you emit more than one block, only the LAST is used.
2. You MUST only use action_id values from the "REGISTERED ACTIONS" list for propose_action.
3. You MUST NOT emit shell commands, shell strings, or argv vectors.
4. REGISTERED ACTIONS below already excludes any action whose retry budget is exhausted -- it is simply not listed as an option any more (see RETRY STATUS for which ones). If REGISTERED ACTIONS is empty, you MUST choose "propose_completion" or an evidence-chain decision instead of re-proposing something already exhausted.
5. obs_id / hyp_id values MUST be copied verbatim from the sections below. An id you
   invent, paraphrase, or recall from memory instead of copying will be rejected — the
   framework checks every id against what actually exists in the evidence store.
6. A hypothesis cannot become a Finding until it has been explicitly validated with
   "record_validation" and verdict="supports". Running an action is not enough by itself.
7. INSIDE the ```action block put ONLY the action itself (JSON or key: value) -- no
   comments, parentheticals, // line comments, or /* */ blocks inside the block. All your
   commentary and reasoning belong OUTSIDE the block, where they are read but not parsed.
8. The OUTPUT FORMAT example below is rebuilt fresh every turn from the CURRENT
   REGISTERED ACTIONS and RETRY STATUS below -- its action_id and rationale are a
   SHAPE to copy, not a recommendation. Always re-derive
   action_id, parameters, and rationale from what REGISTERED ACTIONS, RETRY STATUS,
   OBSERVATIONS, and ACTIVE HYPOTHESES actually say on THIS turn.

CONFIRMATION STANDARD (every vulnerability & invariant violation):
A vulnerability is PROVEN only by a DISTINCTIVE POSITIVE or an OBSERVABLE INVARIANT VIOLATION versus a control:
- Invariant / Logic / Race condition: an invariant is observably violated vs control (e.g. single-use resource consumed twice, limit/quota exceeded, balance became negative, state reached unauthorized value in parallel requests).
- SSRF: the fetched internal page body quoted as evidence.
- SQLi: flipped result set or DB error quoted.
- XSS: unencoded reflected bytes in execution context.
- IDOR: another principal's private data quoted.
- LFI: file contents (/etc/passwd) quoted.
- RCE: command execution output quoted.
Always send a CONTROL that SHOULD fail, and treat it as proof only if the control fails AND the attack succeeds. NEVER call something confirmed because "responses differ", a status code changed, or generic text like "login"/"admin" appears -- those are LEADS, not proof. In a self_script, put the proof in an `evidence` field inside the vuln_signal; a confirmed signal with no evidence is recorded only as a LEAD for human review, never as a solved finding.

OUTPUT FORMAT -- reason in prose first if useful, then END with ONE fenced action block
(__EXAMPLE_DECISION__ example -- copy this SHAPE, but replace every value with one that fits
the CURRENT state above, not these example values). Minimal JSON or `key: value` lines both work:
```action
__EXAMPLE_JSON__
```
"""

def _render_system_prompt(example_action_id, example_rationale: str, example_proposal=None) -> str:
    """
    Fill in the OUTPUT FORMAT example in _PLANNER_SYSTEM_PROMPT_TEMPLATE with a
    *real, currently-available* action_id (and a rationale that names it) instead
    of a single frozen literal.

    BUG FIX (live GPU smoke test, 2026-11-11): the template used to hard-code a
    literal example, e.g. "action_id": "environment_recon" with a fixed rationale
    sentence. This 7B model reliably just copied that literal example verbatim as
    its *answer* on step after step -- same action_id, same word-for-word
    rationale -- regardless of current state, instead of treating it as a shape to
    fill in. Once that action's retry budget was exhausted, every copy got
    rejected, and the model never varied its output enough to try anything else
    (confirmed: 5 consecutive REJECTED steps with byte-identical rationale text).

    Rebuilding the example from the *current* first-available REGISTERED ACTION
    each call fixes this without needing the model to reason correctly: even a
    model that just copies the example now copies something valid and different
    once the previous one is exhausted and filtered out of the catalog (see the
    exhausted_action_ids filtering in _build_prompt), so copying becomes
    self-correcting instead of a stuck loop.
    """
    if example_proposal is not None:
        # State-aware example built by _build_prompt (e.g. a form_hypothesis shape
        # when coverage surfaced a suspicion candidate but no hypothesis is open yet).
        # This replaces the frozen literal so the model is shown the RIGHT NEXT MOVE
        # as a shape, not an action to blindly copy.
        example_decision = example_proposal.get("decision", "propose_action")
        example_json = json.dumps(example_proposal, indent=2)
        return (
            _PLANNER_SYSTEM_PROMPT_TEMPLATE
            .replace("__EXAMPLE_DECISION__", example_decision)
            .replace("__EXAMPLE_JSON__", example_json)
        )
    if example_action_id is None:
        example_decision = "propose_completion"
        example_dict = {
            "decision": "propose_completion",
            "action_id": None,
            "target": None,
            "parameters": {},
            "rationale": (
                "Every registered action has exhausted its retry budget and no "
                "further evidence-chain step is available; declaring the "
                "assessment complete."
            ),
            "confidence": 0.6,
        }
    else:
        example_decision = "propose_action"
        example_dict = {
            "decision": "propose_action",
            "action_id": example_action_id,
            "target": None,
            "parameters": {},
            "rationale": example_rationale,
            "confidence": 0.85,
        }
    # json.dumps handles all quoting/escaping -- action descriptions are
    # code-defined, but this removes any risk of a stray quote in a rationale
    # string breaking the example's own JSON syntax.
    example_json = json.dumps(example_dict, indent=2)
    return (
        _PLANNER_SYSTEM_PROMPT_TEMPLATE
        .replace("__EXAMPLE_DECISION__", example_decision)
        .replace("__EXAMPLE_JSON__", example_json)
    )


# REASON-THEN-FORMAT's Call 1 system prompt: the SAME mission/scaffold/
# confirmation-standard guidance as the strict system prompt above, but with
# the DECISIONS JSON-shape listing, the RULES (NON-NEGOTIABLE) JSON-format
# rules, and the OUTPUT FORMAT example all dropped -- those are exactly the
# "return exactly one JSON object" constraints this pass deliberately lifts.
# Written standalone rather than derived from _PLANNER_SYSTEM_PROMPT_TEMPLATE
# via string surgery, so it stays correct even if that template's structure
# changes later.
_REASONING_SYSTEM_PROMPT = """\
You are an expert security investigator operating inside a safety-gated assessment framework.

YOUR MISSION:
Reason like a human security researcher — do NOT pattern-match to a fixed catalog of vulnerability classes.
You discover vulnerabilities (injection, business logic, race conditions, auth flaws, access control, state bugs, request smuggling, and novel logic flaws) by modeling the target, finding fragile developer assumptions, forming specific falsifiable hypotheses, designing discriminating tests, and confirming through concrete proof.

THE 6-STEP INVESTIGATOR SCAFFOLD (Reason, do not blindly scan):
1. MODEL IT. From observations, state what this app/endpoint is FOR, its moving parts, the actors/roles, what state or limited resources exist (e.g. coupons, discounts, inventory, balances, tokens, permissions), and for each — what must the developer be ASSUMING?
2. FIND THE FRAGILE ASSUMPTION. Which assumption, if false, causes harm? Where does attacker-controlled data reach something powerful (fetcher, query, parser, auth decision, shared/limited resource)? What invariant must hold, and what enforces it (e.g. is coupon redemption atomic, or can parallel requests redeem twice)?
3. HYPOTHESIZE SPECIFICALLY. Form a clear hypothesis: "I suspect ___ because ___. The minimal test that would PROVE OR KILL it is ___. Only this flaw would explain ___." Hypotheses are free-form text — no fixed class name required!
4. DESIGN A DISCRIMINATING TEST — control vs attack, so the result cannot be coincidence. Author your own test via `self_script` (which supports requests, threading, asyncio, and raw sockets for parallel bursts and timing tests) if no specialized tool fits; a scanner is optional.
5. JUDGE THE EVIDENCE — does ONLY the vulnerability/violation explain the result? (e.g. single-use resource consumed twice, limit exceeded, negative total vs control). If not, refute and update your model.
6. CHAIN & FOLLOW SMELLS — a confirmed primitive opens new assumptions; an anomaly that fits no class is still a lead.

The decision you will eventually act on is one of: update_app_model, propose_action (run a registered
tool, or self_script to test a fragile assumption), form_hypothesis (a falsifiable candidate explanation
citing real obs_ids), record_validation (judge a real observation against an open hypothesis),
record_finding (promote an already-validated hypothesis), or propose_completion.

CONFIRMATION STANDARD (every vulnerability & invariant violation):
A vulnerability is PROVEN only by a DISTINCTIVE POSITIVE or an OBSERVABLE INVARIANT VIOLATION versus a control:
- Invariant / Logic / Race condition: an invariant is observably violated vs control (e.g. single-use resource consumed twice, limit/quota exceeded, balance became negative, state reached unauthorized value in parallel requests).
- SSRF: the fetched internal page body quoted as evidence.
- SQLi: flipped result set or DB error quoted.
- XSS: unencoded reflected bytes in execution context.
- IDOR: another principal's private data quoted.
- LFI: file contents (/etc/passwd) quoted.
- RCE: command execution output quoted.
Always send a CONTROL that SHOULD fail, and treat it as proof only if the control fails AND the attack succeeds. NEVER call something confirmed because "responses differ", a status code changed, or generic text like "login"/"admin" appears -- those are LEADS, not proof.

This is a FREE-FORM REASONING pass -- think out loud in plain prose. Do NOT output JSON, markdown, or
any structured format of any kind. A second call will convert your conclusion into the required strict
JSON format, so end with one concrete, specific paragraph naming your intended decision and exactly
what you would do (the action/parameters, or the hypothesis/obs_id/hyp_id involved) -- specific enough
that converting it to JSON is mechanical, not another judgment call.
"""


_PLANNER_USER_TEMPLATE = """\
=== ASSESSMENT STATE ===
Target: {target}
Mode: {mode}
Active Testing Authorized: {active_testing}
Active Network Authorized: {active_network}
State Changes Authorized: {state_changes}

=== REGISTERED ACTIONS (available now) ===
{action_catalog}

=== LOCKED TOOLS (exist but NOT offered yet -- each needs a matching suspicion first) ===
These offensive tools are deliberately withheld until the evidence actually warrants
them. Do NOT ask for them directly; instead form the suspicion each one names (via
form_hypothesis) and it becomes available next turn. This is by design: never fire an
attack tool on a target you have no specific reason to suspect.
{locked_tools}

=== COMPLETION CRITERIA ===
{completion_status}

=== RETRY BUDGET STATUS ===
{retry_status}

=== RECENT EXECUTION HISTORY (last {history_count} steps) ===
{execution_history}

=== DISCOVERED FACTS & ENTITIES ===
{discovered_facts}

=== APPLICATION MODEL & WORKING MEMORY ===
{app_model_summary}

=== RECENT OBSERVATIONS (cite obs_id verbatim in form_hypothesis / record_validation) ===
{observations}

=== SUSPICION CANDIDATES (concrete features worth a specific test -- act on these) ===
Each line is a feature coverage actually observed, the bug class it suggests, and the
test that would confirm it. This is your primary to-do list: pick one, form_hypothesis
for it (citing the obs_id that revealed it), and that unlocks the tool to test it.
{suspicion_candidates}

=== ACTIVE HYPOTHESES (cite hyp_id verbatim in record_validation / record_finding) ===
{active_hypotheses}

=== CONFIRMED FINDINGS ({finding_count}) ===
{findings_summary}

=== RELEVANT TECHNIQUE KNOWLEDGE (retrieved from the knowledge base for THIS target/state) ===
These are proven methods, methodology templates, and payloads from the knowledge
base. You do NOT need to invent techniques -- prefer applying the ones below and
choose the action + parameters that carry them out. When they do not apply, fall
back to your own judgement.
{technique_knowledge}

=== CAMPAIGN JUDGMENT (standing discipline -- applies every turn) ===
{campaign_judgment}

=== CLASS-SWEEP COVERAGE LEDGER (--sweep mode; empty outside it) ===
Each line is a (vuln class, endpoint) pair the context gate already decided is
worth testing, and whether it's been tested yet. Your job here is to ROUTE --
pick the next UNTESTED/LEAD pair and confirm/pivot via its named engine, not
to invent a payload yourself. Once a pair is PROVEN, EXHAUSTED, or NO_ENGINE,
move to the next one instead of re-testing it.
{sweep_coverage}
{prior_reasoning}
{final_instruction}
"""

# The strict-format final instruction, factored into its own constant so the
# REASON-THEN-FORMAT path (below) can swap it out for a free-prose variant on
# Call 1 while reusing the identical rendered prompt otherwise.
_FINAL_DECISION_INSTRUCTION = (
    'Decide the next action, evidence-chain step, or completion. Reason in prose first if it helps, '
    'then END with EXACTLY ONE fenced ```action block containing the decision as compact JSON OR simple '
    '`key: value` lines (decision, action_id, target, parameters, rationale, confidence as needed). '
    'Everything outside the block is ignored; if multiple blocks appear, the LAST is used. Keep the block '
    'itself minimal -- one short sentence for "rationale", no essays or comments INSIDE the block.'
)

# Call 1's instruction (REASON-THEN-FORMAT, free-prose pass): deliberately the
# OPPOSITE of the strict instruction above -- no JSON, no length cap, think
# out loud. Call 2 then gets this reasoning back as {prior_reasoning} context
# alongside the unchanged strict instruction.
_REASONING_ONLY_INSTRUCTION = (
    "This is a free-form REASONING pass, not your final answer. Do NOT output JSON, markdown, or any "
    "structured format -- think out loud in plain prose. Walk through the 6-step investigator scaffold "
    "above against the CURRENT state shown: what do you conclude about the target, which ONE decision "
    "(update_app_model / propose_action / form_hypothesis / record_validation / record_finding / "
    "propose_completion) is the right next move, and exactly why. End with one concrete paragraph "
    "naming your intended action/parameters/obs_id/hyp_id in plain English (e.g. \"I will propose "
    "self_script against the shopId parameter, injecting an internal IP, because...\") -- a second pass "
    "will convert this into the required strict JSON format, so be specific enough for that conversion "
    "to be mechanical, not another judgment call."
)


# Free-form-then-convert (#13): the fenced block the model ends with. Tagged
# ```action specifically, so the parser can tell the decision block apart from
# any other code fences the model's prose might contain.
_ACTION_BLOCK_RE = re.compile(r"```[ \t]*action[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


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
        # Cheap-model NORMALIZER for the free-form-then-convert fallback (#13).
        # None = not yet resolved; False = tried and unavailable; otherwise a
        # client with a generate_raw() interface. NEVER the primary reasoning
        # model. Tests inject a fake here directly.
        self._normalizer = None

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

        # REASON-THEN-FORMAT (2026-11-25, PLANNER_REASON_THEN_FORMAT=1): an
        # EXTRA free-prose LLM call before the strict JSON call, testing
        # whether the model reasons better unconstrained than when forced
        # into "return exactly one JSON object, nothing else" from the first
        # token. Call 1's conclusion becomes Call 2's {prior_reasoning}
        # context (inserted via _build_prompt's own parameter, not string
        # surgery, so it goes through the same formatting as every other
        # section). Any failure here (timeout, empty response, exception)
        # degrades silently to the ORIGINAL single-call prompt -- this is an
        # experiment layered on top of the existing path, never a new way for
        # decide_next() itself to fail.
        if reason_then_format_enabled():
            try:
                reasoning_prompt = prompt.replace(_FINAL_DECISION_INSTRUCTION, _REASONING_ONLY_INSTRUCTION)
                reasoning_text = self.chat_assistant.generate_raw(
                    prompt=reasoning_prompt,
                    max_new_tokens=2048,
                    system_prompt=_REASONING_SYSTEM_PROMPT,
                )
                if reasoning_text and reasoning_text.strip():
                    prior_reasoning_block = (
                        "=== YOUR OWN REASONING (free-form first pass over this exact state) ===\n"
                        f"{reasoning_text.strip()[:4000]}\n\n"
                        "Base your JSON decision below on the reasoning above -- convert its "
                        "conclusion into the matching structured fields; do not re-derive it from "
                        "scratch or contradict it without a clearly stated reason."
                    )
                    prompt = self._build_prompt(
                        session_state=session_state,
                        evidence_store=evidence_store,
                        completion_criteria=completion_criteria,
                        execution_history=execution_history,
                        retry_budget_status=retry_budget_status,
                        prior_reasoning=prior_reasoning_block,
                    )
            except Exception as exc:
                print(f"    [!] REASON-THEN-FORMAT Call 1 failed ({type(exc).__name__}: {exc}); "
                      f"falling back to the single-call prompt for this step.")

        # _build_prompt() sets these as a side effect (first currently-available
        # action, if any) so the OUTPUT FORMAT example below is rebuilt from real,
        # non-exhausted state every call instead of a frozen literal -- see
        # _render_system_prompt()'s docstring for the bug this fixes.
        system_prompt = _render_system_prompt(
            getattr(self, "_example_action_id", None),
            getattr(self, "_example_rationale", ""),
            getattr(self, "_example_proposal", None),
        )

        raw_output: Optional[str] = None
        last_error: Optional[str] = None

        for attempt in range(self.MAX_JSON_RETRIES):
            raw_output = self.chat_assistant.generate_raw(
                prompt=prompt,
                max_new_tokens=3072,  # generous; reasoning prose + the action block both fit
                system_prompt=system_prompt,
                # M1: nudge decoding toward a real, registered action_id (advisory
                # only -- see core/constrained_decoding.py). Gate 1 registry
                # validation in command_executor.py remains authoritative.
                action_id_choices=[a.action_id for a in self.registry.list_actions()],
            )

            # EXTRACTION PIPELINE (#13), fail-safe, in order:
            # 1. Deterministic parse of the ```action block (prose ignored).
            try:
                proposal = self._parse_proposal(raw_output)
                self._reject_if_unregistered(proposal)
                return proposal
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = f"deterministic: {exc}"

            # 2. ONE normalization call to a CHEAP model (never the primary brain)
            #    to convert the free-form action into schema JSON.
            try:
                normalized = self._normalize_with_cheap_model(raw_output)
                if normalized:
                    proposal = self._build_proposal_from_data(self._extract_json_object(normalized))
                    self._reject_if_unregistered(proposal)
                    return proposal
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = f"{last_error} | normalizer: {exc}"

            # 3. Corrective retry: regenerate with an error hint (and, if enabled,
            #    the reason-then-format pre-pass already ran above). Fall through.
            if attempt < self.MAX_JSON_RETRIES - 1:
                prompt += (
                    f"\n\n[RETRY {attempt + 1}] Your previous response could not be parsed into a "
                    f"valid action: {last_error}. End your reply with EXACTLY ONE fenced ```action "
                    f"block containing a compact JSON object (or key: value lines) with a valid "
                    f"'decision' and, for propose_action, a registered 'action_id'. Reasoning prose "
                    f"outside the block is fine and ignored."
                )

        # 4. ALL paths failed -- reject with a clear error. NEVER fabricate an
        #    action or silently drop the model's decision.
        raise ValueError(
            f"AIPlanner.decide_next() failed to extract a valid action after "
            f"{self.MAX_JSON_RETRIES} attempts (deterministic parse + cheap-model normalizer + "
            f"corrective retries all failed). Last error: {last_error}. "
            f"Last raw output: {str(raw_output)[:300]}"
        )

    def _reject_if_unregistered(self, proposal: AIActionProposal) -> None:
        """Fail-safe (consistent with the risk-tier gate): a propose_action whose
        action_id is not a registered action is REJECTED here, so the free-form
        path can never smuggle an unknown/unclassified action past the gate. The
        executor's Gate 1 re-validates too; this just rejects earlier and feeds
        the retry loop a clear reason."""
        if proposal.decision != "propose_action":
            return
        registered = {a.action_id for a in self.registry.list_actions()}
        if proposal.action_id not in registered:
            raise ValueError(
                f"unknown/unregistered action_id {proposal.action_id!r} -- rejected "
                f"(not in REGISTERED ACTIONS; the free-form path is not a way around "
                f"classification or the gate)."
            )

    def _get_normalizer(self):
        """Lazily resolve a CHEAP model client (generate_raw interface) for action
        normalization, or None. NEVER the primary reasoning model. Configured via
        PLANNER_NORMALIZER_MODEL (+ optional backend env read by LLMService);
        unset or unavailable -> None (pipeline simply skips to the next fallback).
        Tests inject a fake into self._normalizer directly."""
        if self._normalizer is False:
            return None
        if self._normalizer is not None:
            return self._normalizer
        model = os.environ.get("PLANNER_NORMALIZER_MODEL")
        if not model:
            self._normalizer = False
            return None
        try:
            from core.services.llm_service import LLMService
            svc = LLMService(verbose=False, model_name_override=model)
            if getattr(svc, "is_available", False):
                self._normalizer = svc
                return svc
        except Exception:
            pass
        self._normalizer = False
        return None

    def _normalize_with_cheap_model(self, raw: str) -> Optional[str]:
        """ONE cheap-model call to convert a free-form action decision into schema
        JSON. Returns the model's raw text, or None if no normalizer is available
        or the call fails (fail-safe: the pipeline then moves on)."""
        client = self._get_normalizer()
        if client is None:
            return None
        convert_prompt = (
            "Convert the action the assistant chose in the text below into ONE compact JSON object "
            "with keys: decision, action_id, target, parameters, rationale, confidence. Output ONLY "
            "the JSON object -- no prose, no code fence.\n\n--- assistant response ---\n"
            + str(raw)[:4000]
        )
        try:
            return client.generate_raw(
                prompt=convert_prompt,
                max_new_tokens=512,
                system_prompt="You convert a free-form action decision into one strict JSON object.",
            )
        except Exception:
            return None

    def _parse_proposal(self, raw: str) -> AIActionProposal:
        """
        Free-form-then-convert (#13): the model reasons in free prose and ends with
        ONE fenced ```action block (compact JSON OR simple key: value lines).
        Everything outside the block is reasoning and is ignored. If multiple
        blocks appear, the LAST is used. Falls back to scanning the whole response
        for a decision-bearing JSON object, so strict-JSON callers still work.
        """
        if not raw or not raw.strip():
            raise ValueError("AI returned empty response.")
        data = self._extract_action_data(raw)
        return self._build_proposal_from_data(data)

    def _extract_action_data(self, raw: str) -> Dict[str, Any]:
        """Prefer the LAST fenced ```action block (JSON or key: value); else fall
        back to the whole-response JSON scan."""
        block = self._extract_action_block(raw)
        if block is not None:
            data = self._parse_action_block_text(block)
            if data is not None:
                return data
        # Fallback: scan the whole response (handles ```json fences + bare JSON).
        return self._extract_json_object(raw)

    @staticmethod
    def _extract_action_block(raw: str) -> Optional[str]:
        """Return the inner text of the LAST ```action ... ``` fenced block, or
        None if there is no action-tagged block."""
        blocks = _ACTION_BLOCK_RE.findall(raw or "")
        if not blocks:
            return None
        return blocks[-1].strip()

    def _parse_action_block_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse an ```action block's inner text, tolerating both minimal-JSON and
        simple `key: value` forms. Returns a dict, or None if nothing parseable."""
        text = (text or "").strip()
        if not text:
            return None
        # 1. The block IS a JSON object.
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        # 2. A JSON object embedded inside the block.
        try:
            return self._extract_json_object(text)
        except ValueError:
            pass
        # 3. Simple `key: value` lines (one per line).
        data: Dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip().strip("\"'")
            v = v.strip().rstrip(",").strip().strip("\"'")
            if k:
                data[k] = v
        if not data:
            return None
        # A key:value block naming an action but no explicit decision -> propose_action.
        if "decision" not in data and data.get("action_id"):
            data["decision"] = "propose_action"
        return data

    def _build_proposal_from_data(self, data: Dict[str, Any]) -> AIActionProposal:
        """Validate a parsed action dict (from any source) and build the typed
        AIActionProposal. NEVER fabricates: a dict missing a decision, or with an
        invalid one, raises -- it is not silently coerced into some default."""
        # Validate required fields
        if "decision" not in data:
            raise ValueError("Missing 'decision' field in AI proposal.")

        decision = data["decision"]
        valid_decisions = (
            "propose_action", "propose_completion",
            "form_hypothesis", "record_validation", "record_finding",
            "update_app_model",
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
        # Forgiving param extraction: small models frequently FLATTEN a verb's
        # fields to the top level of the JSON instead of nesting them under
        # "parameters" (observed abort loop: form_hypothesis with observation_ids/
        # description/validation_test at top level -> "missing required parameters").
        # Lift any non-proposal-level top-level key into parameters so a correctly
        # reasoned proposal is never rejected over JSON SHAPE alone. base_agent /
        # EvidenceStore still re-verify every id before anything is recorded.
        _PROPOSAL_LEVEL = {"decision", "action_id", "target", "rationale", "confidence", "parameters"}
        params = data.get("parameters")
        params = dict(params) if isinstance(params, dict) else {}
        for _k, _v in data.items():
            if _k not in _PROPOSAL_LEVEL and _k not in params:
                params[_k] = _v

        if decision in verb_required_params:
            missing = [k for k in verb_required_params[decision] if not params.get(k)]
            if missing:
                raise ValueError(f"decision='{decision}' is missing required parameters: {missing}")

        try:
            _conf = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            _conf = 0.5

        proposal = AIActionProposal(
            decision=decision,
            action_id=data.get("action_id"),
            target=data.get("target") or str(getattr(self.scope, "target", "local")),
            parameters=params,
            rationale=str(data.get("rationale", ""))[:500],
            confidence=_conf,
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

        BUG FIX (2026-11-25): the three annotation-stripping regexes below used to
        run UNCONDITIONALLY, before any decode attempt -- including on otherwise
        VALID JSON. Each one's "a string's own closing quote is never immediately
        followed by X" reasoning is false when a string VALUE itself legitimately
        starts with that pattern (the live case: `"target": "//evil.com/ssrf"` --
        a protocol-relative URL, exactly the shape an SSRF/open-redirect proposal
        needs to carry). The `//`-comment regex has no `\r\n` to stop at in a
        single-line payload, so `[^\r\n]*` ate everything from that opening quote
        to the end of the string, corrupting `raw` down to `'{"target": "'` and
        destroying the whole proposal. Fix: try the UNMODIFIED text first; only
        fall back to the regex repairs (and re-scan) when the first pass found no
        dict carrying "decision" at all -- so valid JSON is never touched, and the
        repairs still catch the actual annotation-habit artifacts they were
        written for.
        """
        decoder = json.JSONDecoder()

        def _scan(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
            """Return (decision_dict_or_None, first_parseable_dict_or_None)."""
            search_from = 0
            first_valid_dict: Optional[Dict[str, Any]] = None
            while True:
                start = text.find("{", search_from)
                if start == -1:
                    break
                try:
                    obj, _end = decoder.raw_decode(text, start)
                    if isinstance(obj, dict):
                        if "decision" in obj:
                            return obj, first_valid_dict
                        # Defensive robustness fix (same incident as above): don't settle
                        # for the first dict that merely parses -- a small, valid-but-wrong
                        # nested object (like {}) can parse successfully while the real,
                        # intended top-level object sits elsewhere (or was unparseable and
                        # already skipped above). Keep scanning for one that actually has
                        # "decision"; only fall back to the first parseable dict, preserving
                        # the original behavior, if nothing better is ever found.
                        if first_valid_dict is None:
                            first_valid_dict = obj
                    search_from = start + 1
                except json.JSONDecodeError:
                    search_from = start + 1
            return None, first_valid_dict

        decision_obj, first_valid_dict = _scan(raw)
        if decision_obj is not None:
            return decision_obj

        # First pass found nothing decision-bearing -- NOW try the annotation
        # repairs, on a copy, as a fallback (never mutating what already parsed).
        repaired = raw

        # BUG FIX (found via live GPU smoke test with Qwen2.5-7B-Instruct + qwen-qlora,
        # 2026-11-11): the model sometimes appends a human-readable annotation directly
        # after a string value, e.g. "action_id": "environment_recon" (from catalog), --
        # syntactically invalid JSON that raw_decode correctly refuses to parse at that
        # position (traced to the OUTPUT FORMAT example above using the placeholder
        # "<action_id from catalog>", which the model partially echoed -- now fixed to
        # show a concrete, literal example instead; the rule above and this repair are
        # defense in depth for whenever a model still does this anyway).
        repaired = re.sub(r'"\s*\([^)]{0,100}\)(\s*[,}\]])', r'"\1', repaired)

        # BUG FIX (found via live GPU smoke test, 2026-11-11, same incident as
        # above): the model's next attempt at the same annotation habit used a
        # JS/JSON5-style line comment instead of a parenthetical --
        # `"action_id": "environment_recon" // from catalog` -- equally invalid in
        # standard JSON.
        repaired = re.sub(r'"\s*//[^\r\n]*', '"', repaired)

        # BUG FIX (found via live GPU smoke test, 2026-11-11, third variant of the
        # same incident): once the OUTPUT FORMAT example started showing real
        # per-action metadata in REGISTERED ACTIONS (e.g.
        # "  - environment_recon [low risk, network=none, state=none]: ..."), the
        # model started copying that bracketed tag directly after the action_id
        # value too -- "action_id": "environment_recon" [low risk, network=none,
        # state=none], -- same invalid-JSON-annotation habit, now with square
        # brackets instead of parens or "//".
        repaired = re.sub(r'"\s*\[[^\]]{0,200}\](\s*[,}\]])', r'"\1', repaired)

        decision_obj, repaired_first_valid_dict = _scan(repaired)
        if decision_obj is not None:
            return decision_obj

        fallback = repaired_first_valid_dict if repaired_first_valid_dict is not None else first_valid_dict
        if fallback is not None:
            return fallback
        raise ValueError(
            f"No JSON object found in AI response. Raw output: {raw[:300]!r}"
        )

    def _build_prompt(
        self,
        session_state: Dict[str, Any],
        evidence_store,
        completion_criteria,
        execution_history: List[Dict[str, Any]],
        retry_budget_status: Dict[str, int],
        prior_reasoning: str = "",
    ) -> str:
        """
        Build the full planner prompt from current state.

        prior_reasoning: REASON-THEN-FORMAT's Call 2 passes Call 1's own
        free-prose reasoning text here so it's inserted as extra context
        right before the final decision instruction. Empty by default --
        every existing caller (and Call 1's own prompt, before its final
        instruction is swapped for the reasoning-only variant) gets a
        byte-identical prompt to before this parameter existed.
        """
        # BUG FIX (found via live GPU smoke test, 2026-11-11): compute exhausted
        # actions up front so the catalog below can exclude them entirely, instead of
        # relying on the model to cross-reference two separate lists (the full
        # catalog + RETRY STATUS below) and compute "attempts >= max_retries" itself.
        # RULE 4 already told it not to propose an exhausted action -- in practice,
        # with this model, it kept re-proposing "environment_recon" every single step
        # after its retry budget was exhausted on step 2, getting rejected 8/8 times
        # and never reaching flag_capture, record_finding, or anything else. The
        # framework's retry-budget rejection was never bypassed (nothing unsafe
        # happened), it just made zero progress. Not offering an exhausted action as
        # a choice at all is a stronger, more reliable nudge than a rule buried in a
        # long system prompt.
        exhausted_action_ids = set()
        for action_id, attempts in retry_budget_status.items():
            defn = self.registry.get(action_id)
            max_r = defn.max_retries if defn else None
            if max_r is not None and attempts >= max_r:
                exhausted_action_ids.add(action_id)

        # Action catalog — from registry schema
        # C4 fix: render each action's parameter schema (field names, types, which are
        # required, and defaults). Previously only action_id + description were shown,
        # so the model had to guess parameter names — guesses that fail Pydantic's
        # extra="forbid" validation at Gate 1 and burn a step on every wrong guess.
        # SUSPICION GATE (2026-11-16). Split the catalog into actions AVAILABLE now
        # vs offensive tools LOCKED until a matching suspicion exists. "form a
        # suspicion before firing the tool" is thereby enforced by the catalog, not
        # left to the model. See core.action_contract.action_is_unlocked().
        open_classes = self._open_vuln_classes(evidence_store)
        any_suspicion = self._any_suspicion(evidence_store)

        catalog_items = []
        locked_items = []
        self._example_action_id = None
        self._example_proposal = None  # may be set to a state-aware example below
        self._hyp_steer_note = ""
        self._example_rationale = (
            "No registered actions remain available; declare completion or advance "
            "the evidence chain instead."
        )
        for item in self.registry.to_prompt_schema():
            aid = item["action_id"]
            if aid in exhausted_action_ids:
                continue  # retry budget exhausted -- don't even offer it as an option
            defn = self.registry.get(aid)
            params_line = self._format_params_schema(
                item.get("parameters_schema", {}),
                item.get("required_params", []),
            )
            unlocked = action_is_unlocked(defn, open_classes, any_suspicion) if defn is not None else True
            if not unlocked:
                vc = item.get("vuln_class")
                if vc:
                    why = (f'needs an open/confirmed hypothesis with vuln_class="{vc}" '
                           f'-- form that suspicion first')
                else:
                    why = "needs at least one open hypothesis -- form any suspicion first"
                locked_items.append(
                    f"  - {aid} [{item['risk_level']} risk]: {item['description']}\n"
                    f"      LOCKED: {why}"
                )
                continue
            catalog_items.append(
                f"  - {aid} [{item['risk_level']} risk, "
                f"network={item['network_effect']}, state={item['state_change']}]: "
                f"{item['description']}\n"
                f"      parameters: {params_line}"
            )
            if self._example_action_id is None:
                self._example_action_id = aid
                self._example_rationale = (
                    f"{aid} advances the assessment given the CURRENT observations and "
                    f"hypotheses; re-derive the real action and parameters from state."
                )
        action_catalog = (
            "\n".join(catalog_items) if catalog_items
            else "  (none available right now -- advance the evidence chain with "
                 "form_hypothesis / record_validation / record_finding, or propose_completion.)"
        )
        locked_tools = ("\n".join(locked_items) if locked_items
                        else "  (none -- no gated tools remain, or all are already unlocked.)")

        # Feature -> vuln-class bridge: turn coverage into concrete, testable leads.
        suspicion_candidates, candidate_classes = self._suspicion_candidates(
            evidence_store, open_classes, exhausted_action_ids
        )

        # If coverage surfaced a candidate but NO suspicion is open yet, show the
        # model a form_hypothesis example (the right next move) instead of an action
        # example -- this is what breaks the "walk the action list" failure mode and
        # makes the loop reason feature -> hypothesis -> test.
        # Steer the example to the RIGHT next move for the CURRENT state:
        #   - a CONFIRMED hypothesis has no Finding yet -> record_finding NOW
        #     (top priority -- see _confirmed_unpromoted_hypotheses's docstring:
        #     this is cheap, already-earned progress; nothing else should
        #     outrank cashing it in before the model wanders off);
        #   - an open real suspicion exists  -> ACT on the strongest (do not form
        #     yet another hypothesis: that was the observed form_hypothesis spam loop);
        #   - candidates exist, no suspicion  -> form_hypothesis on the top candidate;
        #   - otherwise                       -> the default action example set above.
        confirmed_unpromoted = self._confirmed_unpromoted_hypotheses(evidence_store)
        open_hyps = [
            h for h in evidence_store.hypotheses.values()
            if not getattr(h, "milestone", False) and h.status in ("open", "validating")
        ]
        if confirmed_unpromoted:
            conf_rank = {"likely": 3, "plausible": 2, "speculative": 1}
            strongest = sorted(
                confirmed_unpromoted,
                key=lambda h: (conf_rank.get(h.confidence, 0), h.created_at),
                reverse=True,
            )[0]
            supporting = [
                v for v in evidence_store.validations.values()
                if v.hypothesis_id == strongest.hyp_id and v.verdict == "supports"
            ]
            evidence_hint = supporting[-1].reasoning[:150] if supporting else "<the supporting validation's reasoning>"
            self._example_proposal = {
                "decision": "record_finding",
                "action_id": None,
                "target": None,
                "parameters": {
                    "hypothesis_id": strongest.hyp_id,
                    "title": "<short finding title>",
                    "description": strongest.description[:150],
                    "severity": "<info|low|medium|high|critical, matching the ACTUAL impact>",
                    "affected_target": "<the exact endpoint/parameter this hypothesis names>",
                    "evidence_summary": evidence_hint,
                    "reproducible_steps": ["<step>"],
                    "impact": "<practical impact>",
                    "confidence": 0.8,
                },
                "rationale": f"Hypothesis {strongest.hyp_id} is already CONFIRMED by a supporting validation "
                             f"-- promoting it to a Finding now instead of forming a new hypothesis.",
                "confidence": 0.8,
            }
            self._hyp_steer_note = (
                f"  >> Hypothesis {strongest.hyp_id} is CONFIRMED (a supporting validation was already "
                f"recorded) but has NO Finding yet. Call record_finding on {strongest.hyp_id} NOW -- "
                f"do NOT form a new hypothesis or re-test anything until this confirmed result is either "
                f"recorded as a Finding or the framework tells you why it can't be."
            )
        elif open_hyps:
            conf_rank = {"likely": 3, "plausible": 2, "speculative": 1}
            strongest = sorted(
                open_hyps,
                key=lambda h: (conf_rank.get(h.confidence, 0), h.created_at),
                reverse=True,
            )[0]
            tool = strongest.validation_test if strongest.validation_test in self.registry.actions else "self_script"
            if tool in exhausted_action_ids:
                tool = "self_script"
            tool_dead = tool in exhausted_action_ids  # self_script itself is spent

            if tool_dead:
                # The hypothesis has already been probed and the probe budget is spent
                # -> stop re-running the tool (the observed 10x reject->stall loop) and
                # JUDGE the result instead: record_validation, then record_finding.
                tested_obs = [o for o in evidence_store.observations.values()
                              if o.action_id == strongest.validation_test]
                obs_ref = (tested_obs[-1].obs_id if tested_obs else
                           (list(evidence_store.observations.values())[-1].obs_id
                            if evidence_store.observations else "<obs_id from OBSERVATIONS>"))
                self._example_proposal = {
                    "decision": "record_validation",
                    "action_id": None,
                    "target": None,
                    "parameters": {
                        "hypothesis_id": strongest.hyp_id,
                        "observation_id": obs_ref,
                        # BUG FIX (2026-11-26): this literally hardcoded "supports" here --
                        # a live run followed this worked example verbatim once its probe
                        # budget was exhausted and confirmed a hypothesis its own probe
                        # produced no evidence for (see work.md). The verdict must be
                        # JUDGED from the real observation output, never defaulted.
                        "verdict": "<supports|refutes|inconclusive -- judge HONESTLY from the "
                                   "observation's ACTUAL output; do NOT default to supports>",
                        "reasoning": "<what the probe's ACTUAL output shows -- quote it>",
                    },
                    "rationale": f"Probe budget for {strongest.hyp_id} is spent; judging the result it produced.",
                    "confidence": 0.7,
                }
                self._hyp_steer_note = (
                    f"  >> You have ALREADY tested hypothesis {strongest.hyp_id} and its probe budget is "
                    f"spent -- do NOT propose {strongest.validation_test}/self_script for it again. "
                    f"record_validation on {strongest.hyp_id} (verdict supports/refutes/inconclusive) "
                    f"citing the probe's observation, then record_finding if it is supported."
                )
            else:
                self._example_proposal = {
                    "decision": "propose_action",
                    "action_id": tool,
                    "target": None,
                    "parameters": ({
                        "target": "<the EXACT in-scope endpoint your hypothesis names>",
                        "hypothesis": "<one line: what this probe confirms or refutes>",
                        "code": ("import requests\n"
                                 "# 1) use the form's METHOD (POST form-encoded if POST; a GET query may be ignored).\n"
                                 "# 2) send a CONTROL that SHOULD fail plus the real payload; confirm ONLY if the\n"
                                 "#    control fails AND the payload returns a DISTINCTIVE positive you can quote.\n"
                                 "# 3) print EXACTLY one JSON line: a vuln_signal object with keys confirmed (true/false),\n"
                                 "#    vuln_class, severity, summary, and evidence (the quoted proof). No evidence = LEAD."),
                    } if tool == "self_script" else {}),
                    "rationale": f"Testing open hypothesis {strongest.hyp_id}: {strongest.description[:70]}",
                    "confidence": 0.8,
                }
                self._hyp_steer_note = (
                    f"  >> You have {len(open_hyps)} OPEN hypothesis(es). TEST the strongest "
                    f"({strongest.hyp_id}) NOW with {tool}, against the EXACT parameter and endpoint it "
                    f"names -- do NOT form another hypothesis until you have tested one."
                )
        elif candidate_classes and not any_suspicion:
            self._example_proposal = {
                "decision": "form_hypothesis",
                "action_id": None,
                "target": None,
                "parameters": {
                    "observation_ids": ["<obs_id copied from OBSERVATIONS that revealed the feature>"],
                    "description": ("<the suspicion, naming the EXACT parameter + endpoint, e.g. "
                                    "'stockApi on /product/stock takes a URL -> possible SSRF'>"),
                    "validation_test": "self_script",
                    "confidence": "plausible",
                },
                "rationale": "<which SUSPICION CANDIDATE you are acting on and why it is plausible>",
                "confidence": 0.7,
            }

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
            tag = " [EXHAUSTED -- no longer offered in REGISTERED ACTIONS above]" if action_id in exhausted_action_ids else ""
            retry_lines.append(f"  - {action_id}: {attempts}/{max_r} attempts used{tag}")
        retry_status = "\n".join(retry_lines) if retry_lines else "  (no retries recorded)"

        # Recent execution history (last N steps).
        # BUILD SPEC v1 (2026-11-28): the old flat stdout[:120] window frequently
        # cut off the meaningful part of tool output before the model could use
        # it. The window is now budget-aware (an equal share of the history render
        # budget per shown step, derived from the real context window), and when
        # output is abbreviated that is stated explicitly with a reference to the
        # full stored step results.
        recent_history = execution_history[-self.MAX_HISTORY_STEPS:]
        per_step = max(200, self._render_budgets()["history"] // max(1, len(recent_history)))
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
                    raw = str(first.get("stdout", ""))
                    if len(raw) > per_step:
                        stdout = raw[:per_step]
                        trunc = (f" ...[abbreviated; {len(raw) - per_step} more chars in "
                                 f"step {entry.get('step', i + 1)} results]")
                    else:
                        stdout = raw
                        trunc = ""
                    if stdout:
                        line += f" | output: {stdout!r}{trunc}"
            hist_lines.append(line)
        execution_hist_text = "\n".join(hist_lines) if hist_lines else "  (no history)"

        # Active hypotheses. A CONFIRMED hypothesis with no Finding yet stays
        # listed here too (tagged, not silently dropped) -- otherwise its
        # hyp_id vanishes from the model's view the instant it's confirmed,
        # and record_finding's "copied verbatim from ACTIVE HYPOTHESES below"
        # requirement becomes impossible to satisfy (diagnosed 2026-11-25 from
        # a live run that confirmed 2 hypotheses and recorded 0 findings).
        confirmed_unpromoted_ids = {h.hyp_id for h in confirmed_unpromoted}
        hyp_lines = []
        for hyp in evidence_store.hypotheses.values():
            if hyp.status in ("open", "validating"):
                hyp_lines.append(f"  [{hyp.hyp_id}] {hyp.status}: {hyp.description[:100]}")
            elif hyp.hyp_id in confirmed_unpromoted_ids:
                hyp_lines.append(
                    f"  [{hyp.hyp_id}] CONFIRMED (no Finding yet -- call record_finding on this "
                    f"hyp_id): {hyp.description[:100]}"
                )
        active_hypotheses = "\n".join(hyp_lines) if hyp_lines else "  (none)"
        if getattr(self, "_hyp_steer_note", ""):
            active_hypotheses = active_hypotheses + "\n" + self._hyp_steer_note

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

        # RAG/YAML technique knowledge for THIS state. Cross-agent: this runs for
        # every agent via the shared planner. Fully fail-silent -- an empty string
        # when the knowledge base isn't built/available, so the prompt and the
        # planner's behavior are byte-identical to before when RAG is off.
        technique_knowledge = self._retrieve_technique_knowledge(
            evidence_store=evidence_store,
            available_action_ids=[
                item["action_id"] for item in self.registry.to_prompt_schema()
                if item["action_id"] not in exhausted_action_ids
            ],
            candidate_classes=candidate_classes,
        )

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
            app_model_summary=getattr(evidence_store, "get_app_model_summary", lambda: "  (no app model)")(),
            observations=observations_text,
            active_hypotheses=active_hypotheses,
            finding_count=len(evidence_store.findings),
            findings_summary=findings_summary,
            technique_knowledge=technique_knowledge,
            locked_tools=locked_tools,
            suspicion_candidates=suspicion_candidates,
            campaign_judgment=render_campaign_judgment(),
            sweep_coverage=self._format_sweep_coverage(evidence_store),
            prior_reasoning=prior_reasoning,
            final_instruction=_FINAL_DECISION_INSTRUCTION,
        )

    def _retrieve_technique_knowledge(
        self,
        evidence_store,
        available_action_ids: List[str],
        candidate_classes: Optional[list] = None,  # relevance-ordered (not a set); see #5
    ) -> str:
        """
        Build a retrieval query from the current assessment state and ask the
        framework-level TechniqueAdvisor (RAG corpus + YAML scenario templates)
        for relevant methods/payloads.

        Cross-agent and fail-silent: the query is composed generically from
        state (agent domain, target, discovered facts, recent findings,
        available actions), so it works for any agent; and any failure or an
        unbuilt index yields "(no matching technique knowledge retrieved ...)",
        never an exception. See core/technique_advisor.py.
        """
        try:
            from core.technique_advisor import get_technique_advisor

            domain = getattr(self.registry, "agent_name", "") or ""
            target = getattr(self.scope, "target", "") or ""

            parts: List[str] = []
            if domain:
                parts.append(str(domain))
            if target:
                parts.append(f"target {target}")

            # Hypothesis free text: semantic retrieval query from researcher's active reasoning
            for hyp in getattr(evidence_store, "hypotheses", {}).values():
                if getattr(hyp, "milestone", False):
                    continue
                if getattr(hyp, "status", "") in ("open", "validating"):
                    desc = getattr(hyp, "description", "")
                    if desc:
                        parts.append(desc[:150])

            # Candidate vuln classes observed in the current surface are the
            # strongest retrieval signal -- pull the playbook for what we actually
            # see (e.g. "ssrf", "idor"), not just the generic action list.
            if candidate_classes:
                # Preserve the relevance ORDER from _suspicion_candidates (strongest
                # first) -- do NOT re-sort alphabetically (#5).
                parts.append("suspected: " + ", ".join(candidate_classes))

            # Most salient state: recent confirmed findings + discovered-fact keys.
            recent_findings = list(evidence_store.findings.values())[-3:]
            for fnd in recent_findings:
                title = getattr(fnd, "title", "")
                if title:
                    parts.append(str(title))

            # The available actions encode which vuln classes/techniques are in
            # play right now -- a strong retrieval signal.
            if available_action_ids:
                parts.append("techniques: " + ", ".join(available_action_ids[:10]))

            query = " ; ".join(p for p in parts if p).strip()
            # Keep this tight: a concise hint, not a wall of prose. A long block
            # here both eats the prompt budget and gives a small model text to
            # get lost echoing (observed live 2026-11-16). 700 chars keeps the
            # top methodology hit without crowding out the decision instruction.
            advice = get_technique_advisor().advise(query, rag_k=2, yaml_k=1, max_chars=700)

            # Deterministic payload pack (2026-11-25 4-repo harvest, asset 8) for
            # the single strongest candidate class -- same "suspected:" signal
            # already used above, so this never does its own retrieval/scoring,
            # just a fixed-cost lookup keyed on the class the model already
            # suspects. Advisory text only, same posture as campaign_judgment.py.
            pack = ""
            if candidate_classes:
                try:
                    from core.payload_pack import build_payload_pack
                    # list(...)[0] is the STRONGEST lead by evidence (the internal
                    # caller passes a relevance-ordered list; list() also tolerates a
                    # set from external callers). NOT sorted(...)[0] -- that showed the
                    # payload pack for whichever class sorted first by NAME (#5 fix:
                    # e.g. idor before ssrf).
                    pack = build_payload_pack(list(candidate_classes)[0], max_chars=900)
                except Exception:
                    pack = ""

            combined = "\n".join(p for p in (advice, pack) if p)
            return combined or "(no matching technique knowledge retrieved for this state)"
        except Exception:
            # Absolutely never let retrieval break planning.
            return "(no matching technique knowledge retrieved for this state)"

    def _format_params_schema(self, schema: Dict[str, Any], required: List[str]) -> str:
        """
        Render a Pydantic-derived JSON schema's properties as a compact,
        model-readable parameter list: 'name:type (required)' or 'name:type=default'.

        BUG FIX (2026-11-25, found live: a carefully-written field-level
        Field(description=...) -- e.g. self_script's "code" field, which
        teaches the model to recognize an XML-parser-error signature
        (SAXParseException / "Entities are not allowed") and rewrite its
        probe as raw XML instead of mislabeling the result as SQLi -- had NO
        EFFECT on a live run, because this function only ever read `type`
        and `default` off each property spec and silently dropped
        `description` entirely. The guidance was written, present in
        `defn.params_model.model_json_schema()`, and simply never reached the
        model's prompt. This renders it now. This model's context window is
        262144 tokens (see work.md), so the real constraint historically was
        OUTPUT token budget, not input size -- field descriptions are shown
        close to verbatim (capped generously, not aggressively truncated).
        """
        if not schema:
            return "(no parameters)"
        parts = []
        for name, spec in schema.items():
            ptype = spec.get("type", "any")
            if name in required:
                header = f"{name}:{ptype} (required)"
            else:
                default = spec.get("default")
                header = f"{name}:{ptype}={default!r}" if default is not None else f"{name}:{ptype} (optional)"
            desc = (spec.get("description") or "").strip()
            if desc:
                if len(desc) > 2200:
                    desc = desc[:2197] + "..."
                parts.append(f"{header} -- {desc}")
            else:
                parts.append(header)
        return "\n          ".join(parts)

    # ------------------------------------------------------------------
    # Shared prompt render-budget policy
    # (2026-11-28, BUILD SPEC v1 / workstream A pt1 -- audit defects #1/#2/#3-#21)
    # ------------------------------------------------------------------
    # ONE place that decides "how much evidence may each prompt section render",
    # derived from the ACTUAL configured model context (OLLAMA_NUM_CTX -- the same
    # env the live Ollama backend reads, NOT the 262144 figure some configs cite),
    # reserving room for the system prompt, objective, hypotheses, findings,
    # retrieval knowledge, and the model's own output. This replaces the ad-hoc
    # [:15]/[:150]/[:120] constants that each silently discarded evidence with no
    # disclosure (the [:15] slice directly caused the false-IDOR incident: the
    # model's "IDs 1-15" hypothesis was the literal boundary of value[:15], with
    # IDs 16-20 present in the SAME list one layer below the cut -- audit S1).

    def _context_tokens(self) -> int:
        """Actual configured model context window, in tokens. Reads the SAME env
        the live Ollama backend uses (OLLAMA_NUM_CTX, default 16384); never
        assumes a documented-but-unconfigured larger window."""
        try:
            return max(2048, int(os.environ.get("OLLAMA_NUM_CTX", "16384")))
        except (TypeError, ValueError):
            return 16384

    def _render_budgets(self) -> Dict[str, int]:
        """Per-section CHARACTER budgets for the evidence-bearing prompt sections,
        derived from the real context window. ~4 chars/token; ~50% of the window
        is reserved (un-budgeted here) for the system prompt, objective,
        hypotheses, findings, retrieval knowledge, and the model's own decision
        output. The remaining half is split across the three evidence sections so
        one large (possibly irrelevant) response can never consume the whole
        prompt at the expense of the others."""
        evidence_chars = int(self._context_tokens() * 0.50) * 4
        return {
            "discovered_facts": int(evidence_chars * 0.50),
            "observations": int(evidence_chars * 0.30),
            "history": int(evidence_chars * 0.20),
        }

    @staticmethod
    def _integer_series(values):
        """If EVERY item is '<common-prefix><integer>' with an IDENTICAL prefix,
        return (prefix, [ints in original order]); else None. Lets a list of
        sequential-ID endpoints be rendered as a truthful range instead of a
        silently-truncated partial list. The identical-prefix requirement makes
        this safe: unrelated items never get collapsed into a fake range."""
        values = list(values)
        if not values:
            return None
        m0 = re.match(r"^(.*?)(\d+)$", str(values[0]))
        if not m0:
            return None
        prefix = m0.group(1)
        ints = []
        for v in values:
            m = re.match(r"^(.*?)(\d+)$", str(v))
            if not m or m.group(1) != prefix:
                return None
            ints.append(int(m.group(2)))
        return prefix, ints

    def _render_list_faithfully(self, values, budget_chars: int) -> str:
        """Render a list within budget_chars WITHOUT ever silently implying a
        partial list is complete. When the full list fits, it is rendered
        verbatim. When it does not, the representation ALWAYS discloses the true
        total, what is shown vs omitted, and that the full set is retained in the
        EvidenceStore (so omitted detail is retrievable via existing facts). A
        bare `(+N more)` -- which conceals which items were cut -- is never
        produced."""
        values = list(values)
        full = repr(values)
        if len(full) <= budget_chars:
            return full
        total = len(values)
        # 1) Sequential-ID compaction: a truthful range beats a truncated list.
        series = self._integer_series(values)
        if series is not None:
            prefix, ints = series
            lo, hi = min(ints), max(ints)
            if sorted(ints) == list(range(lo, hi + 1)):
                return (f"{total} values of form {prefix!r}+N, CONTIGUOUS range "
                        f"{lo}-{hi} (complete set retained in EvidenceStore)")
            missing = sorted(set(range(lo, hi + 1)) - set(ints))
            return (f"{total} values of form {prefix!r}+N spanning {lo}-{hi} with "
                    f"gaps; values ABSENT in this range: {missing} "
                    f"(complete set retained in EvidenceStore)")
        # 2) Generic: show as many WHOLE items as fit, then state the rest plainly.
        shown = []
        used = 0
        for v in values:
            s = repr(v)
            if used + len(s) + 2 > budget_chars and shown:
                break
            shown.append(v)
            used += len(s) + 2
        remaining = total - len(shown)
        if remaining <= 0:
            return repr(shown)
        return (f"{shown} -- PARTIAL LIST: {len(shown)} of {total} shown here, "
                f"{remaining} more NOT rendered (complete set of {total} retained "
                f"in EvidenceStore; do NOT treat this as the full set)")

    def _format_observations(self, evidence_store, limit: int = 10) -> str:
        """
        List the most recent observations with their real obs_id, so the planner has
        something concrete to cite in form_hypothesis / record_validation. Capped to
        the most recent `limit` to keep prompt size bounded on long runs -- older
        observations are still fully available via discovered_facts (C5) and findings.

        Per-observation stdout render length is now BUDGET-AWARE (BUILD SPEC v1):
        each recent observation gets an equal share of the observations render
        budget (derived from the real context window) instead of a flat 150-char
        cut that routinely sliced off meaningful evidence before the model saw it.
        When an excerpt is abbreviated, that is stated explicitly and the obs_id
        (already shown on the line) is the reference to the full stored stdout.
        """
        obs_list = list(evidence_store.observations.values())
        if not obs_list:
            return "  (no observations recorded yet)"
        recent = obs_list[-limit:]
        per_obs = max(200, self._render_budgets()["observations"] // max(1, len(recent)))
        lines = []
        for obs in recent:
            raw = (obs.raw_stdout or "").strip().replace("\n", " ")
            if len(raw) > per_obs:
                excerpt = raw[:per_obs]
                tail = f" ...[abbreviated; full stdout in observation {obs.obs_id}]"
            else:
                excerpt = raw
                tail = ""
            lines.append(
                f"  [{obs.obs_id}] action={obs.action_id} exit_code={obs.exit_code} "
                f"stdout: {excerpt!r}{tail}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Suspicion gate + feature->vuln-class bridge (2026-11-16)
    # ------------------------------------------------------------------
    def _open_vuln_classes(self, evidence_store) -> set:
        """vuln_class strings carried by open/validating/confirmed hypotheses."""
        classes = set()
        for hyp in evidence_store.hypotheses.values():
            if getattr(hyp, "milestone", False):
                continue  # benign phase marker, not a real suspicion
            if hyp.status in ("open", "validating", "confirmed"):
                vc = getattr(hyp, "vuln_class", None)
                if vc:
                    classes.add(vc)
        return classes

    def _any_suspicion(self, evidence_store) -> bool:
        """True once at least one non-refuted hypothesis exists."""
        for hyp in evidence_store.hypotheses.values():
            if getattr(hyp, "milestone", False):
                continue  # benign phase marker, not a real suspicion
            if hyp.status in ("open", "validating", "confirmed"):
                return True
        return False

    def _confirmed_unpromoted_hypotheses(self, evidence_store) -> List[Any]:
        """
        Hypotheses the model (or the framework floor) already marked
        status="confirmed" via record_validation, but that have NOT yet been
        turned into a Finding via record_finding.

        FIX (2026-11-25, diagnosed from a live lab run: 2 hypotheses reached
        'confirmed' status via record_validation, findings_count stayed 0,
        and the run burned its remaining steps on form_hypothesis instead).
        ROOT CAUSE: a hypothesis disappears from BOTH the "ACTIVE HYPOTHESES"
        prompt section (which only lists open/validating) and the example-
        steering ladder (which only looks at open_hyps) the instant it becomes
        "confirmed" -- so the model loses all visibility into it and has no
        nudge to call record_finding, even though that is the only decision
        left that actually counts toward completion. This method restores
        that visibility; see its two call sites below.
        """
        promoted_hyp_ids = {f.hypothesis_id for f in evidence_store.findings.values()}
        return [
            hyp for hyp in evidence_store.hypotheses.values()
            if not getattr(hyp, "milestone", False)
            and hyp.status == "confirmed"
            and hyp.hyp_id not in promoted_hyp_ids
        ]

    def _aggregate_facts(self, evidence_store) -> Dict[str, Any]:
        """Union/merge Observation.parsed_facts across the whole run (shared by
        _format_discovered_facts and _suspicion_candidates)."""
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
        return aggregated

    def _suspicion_candidates(self, evidence_store, open_classes, exhausted_action_ids=None):
        """
        THE FEATURE -> SUSPICION SURFACE. Scan every parameter / form input /
        endpoint coverage has surfaced and flag the ones carrying EVIDENCE of being
        interesting (an attacker-controllable value reaching the server, a reflected
        input, a single-use resource on a state-changing endpoint). These surface
        NEUTRALLY, ranked by how much evidence they carry -- NOT by vuln-class
        identity. The reasoning step decides what each one means; a class label,
        when attached, is only a retrieval hint, never a verdict.

        Returns (formatted_text, candidate_classes:set).
        """
        from urllib.parse import urlsplit, parse_qsl

        facts = self._aggregate_facts(evidence_store)

        # --- gather (name, value, endpoint) across all surfaced inputs ---
        pairs = []
        param_method = {}   # param name -> HTTP method of the form it belongs to (e.g. "POST")
        def add_pair(name, value, ep):
            if name and isinstance(name, str):
                pairs.append((name, value, ep))

        for key in ("discovered_endpoints", "network_endpoints", "js_endpoints",
                    "urls", "endpoints", "links"):
            v = facts.get(key)
            if isinstance(v, list):
                for u in v:
                    if not isinstance(u, str):
                        continue
                    try:
                        parts = urlsplit(u)
                        ep = parts.path or u
                        for nm, val in parse_qsl(parts.query, keep_blank_values=True):
                            add_pair(nm, val, ep)
                    except Exception:
                        continue
        for key in ("parameters", "params", "discovered_params"):
            v = facts.get(key)
            if isinstance(v, list):
                for nm in v:
                    add_pair(nm, None, None)
        forms = facts.get("forms")
        if isinstance(forms, list):
            for f in forms:
                if isinstance(f, dict):
                    ep = f.get("action")
                    meth = str(f.get("method") or "get").upper()
                    for nm in (f.get("inputs") or []):
                        add_pair(nm, None, ep)
                        param_method.setdefault(nm, meth)
        reflected = set()
        for rk in ("reflected_params", "reflections", "reflected"):
            rv = facts.get(rk)
            if isinstance(rv, list):
                reflected.update(x for x in rv if isinstance(x, str))

        if not pairs and not reflected:
            return ("  (none yet -- run coverage first: endpoint_discovery, param_fuzz, "
                    "browser_probe to surface parameters, forms, and endpoints.)", [])

        # --- signature table (general, class-keyed) ---
        SSRF = {"url","uri","link","src","source","dest","destination","redirect","redir",
                "return","returnurl","next","continue","callback","cb","feed","host","hostname",
                "site","domain","page","path","out","view","load","target","image","img","imageurl",
                "avatar","data","ref","reference","api","stock","stockapi","fetch","proxy","remote",
                "resource","u","to","go","document","server","endpoint","webhook","upstream","forward"}
        REDIRECT = {"redirect","redir","return","returnurl","returnuri","next","continue","url",
                    "dest","destination","goto","go","out","target","forward","rurl","r"}
        IDOR = {"id","uid","userid","user","user_id","account","acct","accountid","order","orderid",
                "order_id","invoice","invoiceid","doc","docid","document_id","pid","productid",
                "product_id","itemid","number","no","num","key","record","recordid","row","customerid",
                "customer","profile","profileid","groupid","group","fileid","objectid","oid","ticket",
                "ticketid","msgid","messageid"}
        SQLI = {"id","category","cat","search","q","query","filter","sort","order","orderby","name",
                "user","username","product","productid","page","limit","offset","from","year","month",
                "day","code","ref","group","select","where"}
        LFI = {"file","filename","path","filepath","template","tpl","include","inc","page","doc",
               "document","folder","dir","directory","download","load","lang","language","view",
               "style","theme","pdf","report","log","conf","config","module","layout","content","locale"}
        CMD = {"cmd","exec","command","run","ping","host","ip","query","q","search","jump","code",
               "do","func","function","action","call","shell","system","process"}
        RACE_LOGIC = {"coupon","code","promo","discount","voucher","gift","giftcard","cart",
                      "checkout","order","quantity","qty","item","stock","balance","transfer",
                      "amount","credit","points","vote","like","redeem","apply","claim","token",
                      "ticket","withdraw","deposit","pay","payment"}

        import re as _re
        def urlish(v):
            if not v: return False
            vv = v.strip().lower()
            return (vv.startswith(("http://","https://","//")) or "://" in vv
                    or vv.startswith("/") or bool(_re.match(r"^[a-z0-9.-]+\.[a-z]{2,}(/|$|:)", vv)))
        def numish(v):
            if v is None: return False
            vv = v.strip()
            return bool(_re.fullmatch(r"\d+", vv)) or bool(_re.fullmatch(r"[0-9a-fA-F-]{8,}", vv))
        def pathish(v):
            if not v: return False
            vv = v.strip()
            return ("/" in vv) or ("\\" in vv) or (".." in vv) or bool(
                _re.search(r"\.(php|txt|log|conf|ini|xml|json|pdf|html?|bak|yml|yaml|env)$", vv.lower()))
        def hit(low, names): return any(k == low or k in low for k in names)

        # Per-class playbooks: how to CONFIRM (with a control + a distinctive positive)
        # and the common VARIANTS to try. The confirmation bar is the same for every
        # class -- a distinctive positive you can quote as evidence, never "responses
        # differ", a bare status code, or generic page text.
        HINTS = {
            "ssrf": ("PROVE the server fetched an internal target: use an unroutable CONTROL "
                     "(http://127.0.0.1:1/) vs a live internal URL, and capture a DISTINCTIVE "
                     "internal body (e.g. an admin page) as evidence. Variants: localhost/127.x; "
                     "internal ranges 192.168.0.0/24, 10.0.0.0/8, 172.16/12 (often :8080); cloud "
                     "metadata 169.254.169.254; filter bypasses (127.1, [::1], decimal IP, @ / # tricks). "
                     "'another back-end system' => SCAN the internal subnet for the admin host."),
            "open-redirect": ("PROVE a 3xx with a Location header to YOUR external host. Variants: "
                              "//evil, https:evil, /\\evil, @evil, whitelisted-host.evil."),
            "idor": ("PROVE you read ANOTHER principal's data: force an adjacent/other id and quote "
                     "distinctive content that isn't yours. Same data for your own id is NOT proof. "
                     "Variants: numeric increment, GUID swap, param pollution, method change."),
            "sqli": ("PROVE it: a boolean pair must FLIP the result set (1=1 vs 1=2), OR a time "
                     "payload adds a measurable delay, OR an error names the DB -- quote it. "
                     "'Responses differ' is NOT proof. Variants: error / boolean-blind / time-blind / "
                     "UNION; confirm depth with sqli_deep_scan."),
            "lfi": ("PROVE file disclosure: ../../../../etc/passwd must return 'root:x:0:0' content "
                    "(quote it). Variants: depth, URL/double-encoding, null byte, php://filter, absolute paths."),
            "cmd-injection": ("PROVE execution: ;id / |id / $(id) returns 'uid=' output (quote it), OR "
                              ";sleep 5 adds a delay. Variants: ; | && $() backticks newline."),
            "xss": ("PROVE the payload reflects UNENCODED in its context (quote the reflected bytes); "
                    "a canary returned HTML-encoded is NOT XSS. Variants: HTML body, attribute, JS "
                    "string, href/URL, template context."),
            "race-condition": ("PROVE timing window / invariant violation: send a sequential CONTROL "
                               "(which properly enforces single-use or limit) vs a BURST of parallel concurrent "
                               "requests using self_script (threading / concurrent.futures). Quote the multi-redemption "
                               "or limit overrun as proof. Variants: simultaneous coupon redemption, balance overdraft, "
                               "cart checkout race, double-spend."),
            "business-logic": ("PROVE state invariant failure: submit unexpected sequencing, negative quantities, "
                               "or bypass step validation. Compare against control."),
        }
        TOOL = {"ssrf":"self_script","open-redirect":"self_script","lfi":"self_script",
                "cmd-injection":"self_script","idor":"idor_probe","sqli":"sqli_probe","xss":"xss_probe",
                "race-condition":"self_script","business-logic":"self_script"}

        # --- score ---
        best = {}  # (cls,name) -> dict(weight, ep, cls, name)
        # Strong URL/fetch-shaped names: a param whose NAME implies the server will
        # fetch a URL is a high-confidence SSRF/redirect lead -- more specific than
        # "a number might be IDOR". These outrank a bare numeric-id IDOR guess.
        STRONG_SSRF = {"url", "uri", "link", "src", "dest", "destination", "redirect",
                       "redir", "returnurl", "return", "next", "continue", "callback",
                       "feed", "host", "site", "domain", "fetch", "proxy", "remote",
                       "webhook", "upstream", "forward", "api", "stock", "stockapi",
                       "imageurl", "resource", "endpoint", "load", "image", "img"}

        def consider(name, val, ep):
            low = name.lower()
            out = []
            # Weight by the EVIDENCE an observation carries, never by vuln-class
            # identity: a shape we can SEE in the value (confirmed) outranks a bare
            # name match (speculative). No class is privileged -- the reasoning step,
            # not this table, decides what is worth pursuing.
            CONFIRMED, NAMEONLY = 3, 2
            if urlish(val):
                out.append(("ssrf", CONFIRMED))          # value IS a URL -> server may fetch it
            elif hit(low, SSRF) or any(tok == low or tok in low for tok in STRONG_SSRF):
                out.append(("ssrf", NAMEONLY))           # name only -- speculative, not privileged
            if hit(low, REDIRECT):
                out.append(("open-redirect", CONFIRMED if urlish(val) else NAMEONLY))
            if hit(low, RACE_LOGIC) or (ep and any(tok in ep.lower() for tok in ("coupon", "cart", "checkout", "discount", "redeem", "gift", "balance", "voucher"))):
                _state_ep = param_method.get(name) in ("POST", "PUT", "PATCH", "DELETE")
                out.append(("race-condition", CONFIRMED if _state_ep else NAMEONLY))
            if hit(low, IDOR):
                out.append(("idor", CONFIRMED if numish(val) else NAMEONLY))
            if hit(low, LFI):
                out.append(("lfi", CONFIRMED if pathish(val) else NAMEONLY))
            if hit(low, SQLI) and val is not None:
                out.append(("sqli", NAMEONLY))
            if hit(low, CMD):
                out.append(("cmd-injection", NAMEONLY))
            if name in reflected:
                out.append(("xss", CONFIRMED))           # reflection is observed evidence
            # keep this param's top 2 classes only, to avoid spamming every class
            out.sort(key=lambda t: t[1], reverse=True)
            for cls, w in out[:2]:
                k = (cls, name)
                if k not in best:
                    # "order" = discovery index -> a deterministic, NON-alphabetical
                    # tiebreak for equal-evidence candidates (#5).
                    best[k] = {"weight": w, "ep": ep, "cls": cls, "name": name, "order": len(best)}
                elif w > best[k]["weight"]:
                    best[k]["weight"] = w
                    best[k]["ep"] = ep

        for name, val, ep in pairs:
            consider(name, val, ep)
        for name in reflected:
            consider(name, None, None)

        if not best:
            return ("  (parameters/forms were seen but none match a known vuln signature; "
                    "use param_fuzz / browser_probe for deeper coverage, or self_script for a "
                    "bespoke test if you suspect something specific.)", [])

        # Rank by EVIDENCE weight (desc), then discovery order -- NEVER alphabetical
        # (#5): alphabetical order must not decide what the model sees first.
        ranked = sorted(best.values(), key=lambda d: (-d["weight"], d.get("order", 0)))[:6]
        lines = []
        classes = []  # relevance-ORDERED list (strongest first), not an unordered set
        exhausted = exhausted_action_ids or set()
        for d in ranked:
            cls = d["cls"]
            if cls not in classes:
                classes.append(cls)
            on = f' on {d["ep"]}' if d.get("ep") else ""
            method = param_method.get(d["name"])
            method_hint = (f" Submit it with the {method} method (the form's method), "
                           f"form-encoded if POST -- a GET query is ignored.") if method and method != "GET" else ""
            tool = TOOL.get(cls, "self_script")
            if tool in exhausted:
                tool = "self_script"
            mtag = method or "GET"
            if cls in open_classes:
                lines.append(
                    f'  - param "{d["name"]}"{on} [{mtag}] -> {cls.upper()} (hypothesis already OPEN). '
                    f'Test: {HINTS.get(cls, "")}{method_hint} '
                    f'Act: run {tool} against this exact parameter+endpoint, or record_validation on the '
                    f'open hypothesis. Do NOT form_hypothesis for this again.'
                )
            else:
                _spec = "" if tool == "self_script" else f", or {tool} if that specialist clearly fits"
                lines.append(
                    f'  - param "{d["name"]}"{on} [{mtag}] -- attacker-controllable input reaching the server. '
                    f'REASON about what the app DOES with it and which assumption it trusts; '
                    f'a known angle worth checking ({cls}): {HINTS.get(cls, "")}{method_hint} '
                    f'Act: form_hypothesis citing the obs_id that revealed it, describing the fragile '
                    f'assumption in your OWN words (name a class only if you recognize one); '
                    f'then test it (use self_script{_spec}).'
                )
        advisory = (
            "  (ADVISORY, ranked by evidence -- NOT a menu or a filter: these are leads, "
            "strongest first. You MAY pursue a class not listed here, or form a hypothesis "
            "with no class at all (vuln_class=None). The approval gate, not this list, "
            "governs what actually runs.)"
        )
        return (advisory + "\n" + "\n".join(lines), classes)


    def _format_sweep_coverage(self, evidence_store) -> str:
        """
        Render --sweep mode's coverage ledger (populated deterministically by
        BaseAgent._update_sweep_coverage, never by the planner itself). Empty
        EvidenceStore / non-sweep runs render the same fixed placeholder text
        every time, so this is a byte-identical no-op for every existing
        caller that never touches surface_map/coverage_ledger.
        """
        ledger = getattr(evidence_store, "coverage_ledger", None)
        if not ledger:
            return "  (no sweep coverage -- not in --sweep mode, or surface not characterized yet)"
        lines = evidence_store.coverage_summary_lines()
        return "\n".join(lines) if lines else "  (coverage ledger empty)"

    def _format_discovered_facts(self, evidence_store) -> str:
        """
        Aggregate Observation.parsed_facts (C5) across the whole run into a
        model-readable summary: list-valued facts are unioned and deduplicated,
        dict-valued facts are merged, everything else keeps the most recent value.

        BUILD SPEC v1 (2026-11-28): the old hard `value[:15]` + bare `(+N more)`
        slice silently hid facts and -- because it landed exactly at the "IDs
        1-15" boundary on a sequential-ID list -- directly produced the false
        IDOR hypothesis (audit S1). List rendering is now budget-aware and
        FAITHFUL: the complete set is rendered when it fits the (context-derived)
        budget; when it doesn't, a truthful representation is emitted that
        discloses the true total, the value range / which items are omitted, and
        that the complete set remains retrievable from the EvidenceStore. A
        partial list is never presented as if it were complete.
        """
        aggregated = self._aggregate_facts(evidence_store)

        if not aggregated:
            return "  (no structured facts extracted yet)"

        budget = self._render_budgets()["discovered_facts"]
        list_keys = [k for k, v in aggregated.items() if isinstance(v, list)]
        # Give each list-valued fact an equal share of the section budget so one
        # huge list can't starve the others (floor keeps every key readable).
        per_key = max(400, budget // max(1, len(list_keys))) if list_keys else budget

        lines = []
        for key, value in aggregated.items():
            if isinstance(value, list):
                lines.append(f"  - {key}: {self._render_list_faithfully(value, per_key)}")
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
            e.get("action_id") for e in execution_history
            if e.get("status") == "success" and e.get("action_id")
        }  # action_id is None for evidence-verb steps (form_hypothesis/...); drop it so sorted() can't hit str<None
        required = getattr(criteria, "required_phases", [])
        lines.append(f"  Required phases: {required}")
        lines.append(f"  Successful actions so far: {sorted(completed_action_ids)}")

        # FIX (live WSL smoke test, 2026-11-13): previously this only listed the
        # required phase NAMES with no indication of which action_id would
        # satisfy an unmet one. On the live run the model kept re-proposing
        # suid_audit/find_writable (already satisfied) for all 8 steps and
        # never once proposed flag_search, because nothing told it "flag_capture"
        # specifically needed that action. phase_actions is an OPTIONAL mapping
        # (phase name -> action_ids) a completion module can attach to its
        # CompletionCriteria; when present, call out exactly which unmet phases
        # still need action and which action_id(s) would satisfy each one. This
        # stays fully agent-agnostic: if phase_actions isn't set, this section is
        # skipped and behavior is unchanged from before.
        phase_actions = getattr(criteria, "phase_actions", None)
        if phase_actions:
            unmet_phases = []
            for phase in required:
                candidates = phase_actions.get(phase, [])
                phase_satisfied = any(a in completed_action_ids for a in candidates)
                if not phase_satisfied:
                    unmet_phases.append((phase, candidates))
            if unmet_phases:
                lines.append("  Unmet required phases -- propose one of the listed actions for each:")
                for phase, candidates in unmet_phases:
                    if candidates:
                        lines.append(f"    - '{phase}': try {candidates}")
                    else:
                        lines.append(f"    - '{phase}': (no action in the catalog maps to this phase)")
            else:
                lines.append("  All required phases are satisfied by a successful action.")

        min_findings = getattr(criteria, "minimum_findings", 0)
        current_findings = len(evidence_store.findings)
        lines.append(
            f"  Findings: {current_findings}/{min_findings} required"
        )

        obs_required = getattr(criteria, "evidence_required_for_phases", [])
        obs_count = len(evidence_store.observations)
        lines.append(f"  Observations recorded: {obs_count} (required for phases: {obs_required})")

        return "\n".join(lines)
