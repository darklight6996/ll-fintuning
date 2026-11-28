
def datetime_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
"""
agents/base_agent.py

BaseAgent class that all specialized assessment agents inherit from.
Coordinates:
- Deterministic phase sequence execution
- AI reasoning integration (AI as consultant, not driver)
- Safety gating via CommandExecutor (read_only unless scope authorizes active testing)
- Scope compliance checking before active testing
- Operator approval gates (Manual: prompt per phase; Automated: sequential with safety checks)
- Audit trail persistence via SessionManager and EvidenceManager
- Ctrl+C state saving and resuming
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path

from core.scope import Scope
from core.session_manager import SessionManager
from core.command_executor import CommandExecutor
from core.chat_assistant import ChatAssistant
from core.tool_manager import ToolManager


@dataclass
class PhaseResult:
    """Standardized result returned by every deterministic phase script."""
    phase_name: str
    status: str                              # 'success', 'failed', 'skipped', 'blocked'
    commands_run: List[Dict[str, Any]] = field(default_factory=list)
    raw_output: str = ""
    findings: Dict[str, Any] = field(default_factory=dict)
    next_hint: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "status": self.status,
            "commands_run": self.commands_run,
            "raw_output": self.raw_output[:500] if self.raw_output else "",
            "findings": self.findings,
            "next_hint": self.next_hint,
            "error": self.error
        }


class BaseAgent:
    """
    Abstract BaseAgent providing the unified four-layer assessment lifecycle.
    """

    agent_type: str = "base"
    agent_name: str = "Base Security Agent"
    description: str = "Base security assessment agent."

    def __init__(
        self,
        scope: Scope,
        mode: str = "manual",
        session_manager: Optional[SessionManager] = None,
        chat_assistant: Optional[ChatAssistant] = None,
        command_executor: Optional[CommandExecutor] = None,
    ):
        self.scope = scope
        self.mode = mode.lower()  # "manual" or "automated"
        # BLIND LAB mode (PENTEST_BLIND_LAB=1): strip PortSwigger lab-identifying
        # markers (title, lab banner, the portswigger.net/web-security/<class> solution
        # link) out of observations before the planner sees them, so the agent must
        # find the bug from the real app surface -- a fair proxy for a live target,
        # which has no such answer-key header. Off by default (zero effect).
        self._blind = os.environ.get("PENTEST_BLIND_LAB") == "1"
        self.session_mgr = session_manager or SessionManager()
        self.chat_assistant = chat_assistant or ChatAssistant()
        
        # Read-only executor by default unless active testing is explicitly authorized
        read_only = not self.scope.active_testing_allowed
        self.executor = command_executor or CommandExecutor(read_only=read_only)
        self.tool_manager = ToolManager()
        self.phase_modules: List[Any] = []
        self._load_phases()

    def _load_phases(self) -> None:
        """Must be implemented by subclasses to register their deterministic phase modules."""
        raise NotImplementedError("Subclasses must implement _load_phases()")

    def get_phase_names(self) -> List[str]:
        return [getattr(p, "name", str(p)) for p in self.phase_modules]

    def _get_completion_verifier(self) -> Callable[..., Tuple[bool, str]]:
        """
        Resolve this agent's own verify_completion() function.

        Every agents/<type>/completion.py module exposes a verify_completion(criteria,
        session_state, evidence_store) function with an identical signature. Importing
        it by self.agent_type (rather than hardcoding one agent's module) is what lets
        run_autonomous() be shared across all 9 agents while each one is checked against
        its own required_phases / action_to_phase mapping.
        """
        import importlib

        module_name = f"agents.{self.agent_type}.completion"
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Could not resolve completion module '{module_name}' for agent_type "
                f"'{self.agent_type}'. Every agent package must define completion.py "
                f"with a verify_completion() function. Original error: {exc}"
            ) from exc

        verify_fn = getattr(module, "verify_completion", None)
        if verify_fn is None:
            raise AttributeError(
                f"Module '{module_name}' does not define verify_completion()."
            )
        return verify_fn

    def _get_fact_extractor(self) -> Optional[Callable[[str, str, str], Dict[str, Any]]]:
        """
        Resolve this agent's optional agents/<type>/fact_extractors.py::extract_facts()
        function (C5). Unlike _get_completion_verifier(), this is NOT safety-critical --
        an agent without a fact_extractors.py module simply gets no structured fact
        extraction (Observation.parsed_facts stays {}), so this never raises.
        """
        import importlib

        try:
            module = importlib.import_module(f"agents.{self.agent_type}.fact_extractors")
        except ImportError:
            return None
        return getattr(module, "extract_facts", None)

    def _extract_parsed_facts(self, action_id: str, stdout: str, stderr: str) -> Dict[str, Any]:
        """
        Best-effort structured fact extraction (C5) for a single command's output.
        Always returns a dict, even if the agent has no extractor or the extractor
        itself raises -- parsed_facts is a planner-prompt convenience, never a gate.
        """
        extractor = self._get_fact_extractor()
        if extractor is None:
            return {}
        try:
            facts = extractor(action_id, stdout, stderr)
            return facts if isinstance(facts, dict) else {}
        except Exception:
            return {}

    _MODEL_AUTHORED_ACTIONS = {"self_script", "nuclei_template"}

    # Shared rigor vocabulary (factored out 2026-11-26 so _confirmation_strength
    # and _validate_supports_claim can't drift apart): phrases that mean "I am
    # guessing from a diff/heuristic", and the XML-parser-error signatures that
    # mean neither side of a control/attack pair ever reached real app logic.
    _WEAK_EVIDENCE_PHRASES = (
        "differ", "divergence", "different response", "!= 200", "status code",
        "suggest", "might", "may indicate", "may be", "heuristic", "probably",
        "appears", "could be", "seems", "status !=", "changed",
    )
    _XML_PARSER_ERROR_SIGNATURES = (
        "saxparseexception", "content is not allowed in prolog",
        "entities are not allowed", "doctype is disallowed",
    )

    def _validate_supports_claim(self, observation, reasoning: str) -> Tuple[bool, str]:
        """
        Structural rigor gate for record_validation's verdict='supports'.

        BUG FIX (2026-11-26, found via a live run report): EvidenceStore.add_validation()
        flips a Hypothesis to 'confirmed' on the AI's bare say-so -- there was NO check
        that the cited observation's actual content supports the claim. In the reported
        run, step 4's self_script was dead code (a `pass`-only loop that never sent a
        request) yet printed a verdict and was marked SUCCESS; step 6 then cited a "504
        timeout" that no recorded observation produced; and the planner's own tool_dead
        worked example hardcoded `"verdict": "supports"` (see core/ai_planner.py), which
        the model followed verbatim once its probe budget was spent. The combination
        produced a false 'proven' SSRF finding that reached propose_completion and was
        written into the training-trajectory JSONL files -- i.e. training-data
        contamination, not just a bad report.

        This runs BEFORE add_validation() for every verdict='supports' claim, regardless
        of which action produced the cited observation -- the same discipline
        _confirmation_strength already applies to engine-authored vuln_signals, applied
        here to the AI's own evidence citation at the point a Hypothesis is allowed to
        flip to 'confirmed' at all (the earliest point in the chain, per the fix priority
        in that report). A failing check REJECTS the record_validation call outright
        (never silently downgrades the verdict) so the planner must either cite real
        evidence or honestly choose refutes/inconclusive on its next turn.

        Returns (ok, reason).
        """
        facts = getattr(observation, "parsed_facts", None) or {}
        vuln_signal = facts.get("vuln_signal") if isinstance(facts, dict) else None
        if isinstance(vuln_signal, dict):
            # The cited observation came from an engine/probe that already emits its
            # own objective verdict -- the AI's "supports" judgement must agree with it,
            # not override it.
            if not vuln_signal.get("confirmed"):
                return False, (
                    "the cited observation's own vuln_signal reports confirmed=false (or no "
                    "confirmation) -- verdict='supports' contradicts the probe's own verdict. "
                    "Use 'refutes' or 'inconclusive', or cite an observation that actually "
                    "confirms this."
                )
            return True, "engine-confirmed vuln_signal backs this claim"

        raw = (getattr(observation, "raw_stdout", "") or "").strip()
        if not raw:
            return False, (
                "the cited observation produced NO output at all (empty stdout) -- there is "
                "nothing in it to support this hypothesis. Cite an observation that actually "
                "ran and produced real output, or use 'inconclusive'."
            )

        reasoning_clean = (reasoning or "").strip()
        if len(reasoning_clean) < 15:
            return False, (
                "reasoning is too short/generic to be a real evidence citation -- quote the "
                "ACTUAL distinguishing output from the observation."
            )

        low = reasoning_clean.lower()
        if any(sig in low for sig in self._XML_PARSER_ERROR_SIGNATURES):
            return False, (
                "reasoning cites an XML-parser error as support -- that means the request "
                "never reached real application logic, which is not confirming evidence. Use "
                "'refutes' or 'inconclusive'."
            )
        for w in self._WEAK_EVIDENCE_PHRASES:
            if w in low:
                return False, (
                    f"reasoning rests on a weak/heuristic basis ({w!r}) instead of quoting "
                    f"concrete evidence from the observation's actual output -- quote the real "
                    f"distinguishing content, or use 'inconclusive'."
                )
        return True, "concrete evidence cited"

    def _confirmation_strength(self, action_id, vuln_signal):
        """
        General confirmation-rigor gate (2026-11-20). A CONFIRMED vuln_signal is
        'proven' only when the evidence is trustworthy; otherwise it is a 'lead'
        (surfaced for human verification, never counted toward completion). This is
        vuln-class-agnostic -- it applies to SSRF, SQLi, XSS, IDOR, LFI, race conditions,
        business logic, anything.

          * Framework detection engines (sqlmap / nuclei / dalfox / the typed probes)
            are trusted oracles: a confirmed verdict is proven.
          * MODEL-AUTHORED probes (self_script, nuclei_template) are scrutinised --
            they must carry concrete `evidence` and must NOT rest on a weak/heuristic
            basis ("responses differ", a bare status code, "suggests/likely/might").
            However, concrete evidence of an INVARIANT or ASSUMPTION violation
            (e.g. single-use resource consumed twice, limit/quota exceeded, negative
            balance, race window overrun vs control) IS proven evidence.

        Returns (strength, reason).
        """
        if action_id not in self._MODEL_AUTHORED_ACTIONS:
            return "proven", "framework detection engine"
        sig = vuln_signal or {}
        if str(sig.get("strength") or "").lower() == "lead":
            return "lead", "probe self-reported a lead"
        evidence = str(sig.get("evidence") or "").strip()
        basis = (str(sig.get("summary") or "") + " " + evidence).lower()
        if not evidence or len(evidence) < 8:
            return "lead", "no concrete evidence captured (verdict only)"

        # BUG FIX (2026-11-25, found via a live blind A/B run): a self_script sent a
        # form-encoded POST to an endpoint that expects a raw XML body (confirmed by
        # self_scripting.py's OWN documented signatures below). BOTH the control and
        # the attack request got XML-PARSER errors -- "Content is not allowed in
        # prolog" for the control, "Entities are not allowed" for the attack -- i.e.
        # NEITHER request ever reached real application logic; the two different
        # error MESSAGES are an artifact of which malformed-request text happened to
        # confuse the parser differently, not evidence of anything the hypothesis
        # claims. The model's own evidence string said "Response body changed with
        # storeId=...", which slipped past the WEAK list below (that list has
        # "differ"/"divergence"/"different response", not "changed") and got
        # recorded as a PROVEN high-severity SSRF finding -- a real false positive.
        # This is exactly the "request FORMAT is wrong, not the hypothesis" pattern
        # self_scripting.py's own prompt guidance already teaches the model to
        # recognize and fix (rewrite as XML) -- but nothing stopped the model from
        # ignoring that guidance and reporting confirmed=True anyway. Structural fix:
        # when the evidence itself contains one of these XML-parser-error
        # signatures, NO diff between control and attack can be confirming evidence
        # of anything (both sides never got past the parser), so this is checked
        # BEFORE the invariant-violation branch and forces "lead" unconditionally.
        _XML_PARSER_ERROR_SIGNATURES = (
            "saxparseexception", "content is not allowed in prolog",
            "entities are not allowed", "doctype is disallowed",
        )
        if any(sig_text in basis for sig_text in _XML_PARSER_ERROR_SIGNATURES):
            return "lead", (
                "confirmation rests on an XML-parser format error (the request body format was "
                "wrong for this endpoint, not evidence of the suspected vulnerability -- rewrite "
                "the control AND attack requests as well-formed XML before re-testing)"
            )

        # Invariant / state violation checks:
        # A concrete violation of an invariant or assumption (e.g. single-use resource
        # consumed twice, quota/limit exceeded, illegal state reached) is a PROVEN
        # vulnerability, not a weak response-diff heuristic.
        invariant_violation = bool(
            sig.get("invariant_violated")
            or sig.get("violation")
            or any(kw in basis for kw in (
                "consumed twice", "redeemed twice", "applied multiple times",
                "overrun", "limit exceeded", "quota exceeded", "double-spend",
                "negative balance", "negative total", "invariant violated",
                "race window", "concurrent redemption",
            ))
        )

        WEAK = ("differ", "divergence", "different response", "!= 200", "status code",
                "suggest", "might", "may indicate", "may be", "heuristic", "probably",
                "appears", "could be", "seems", "status !=",
                # "changed" (2026-11-25): the exact synonym for "differ"/"divergence"
                # that slipped a body-diff-only false positive past this filter --
                # see the XML-parser-error check above for the live incident.
                "changed")

        if invariant_violation:
            SPECULATIVE = ("suggest", "might", "may indicate", "may be", "heuristic", "probably", "appears", "could be", "seems")
            if any(s in basis for s in SPECULATIVE) and not any(c in basis for c in ("applied", "redeemed", "consumed", "exceeded", "balance", "count", "total")):
                return "lead", "confirmation rests on speculative reasoning"
            return "proven", "concrete invariant violation captured"

        for w in WEAK:
            if w in basis:
                return "lead", "confirmation rests on a weak signal (%r)" % w.strip()
        return "proven", "concrete evidence captured"

    def _promote_evidence_if_warranted(
        self,
        defn: Optional[Any],
        proposal: Any,
        observations: List[Any],
        evidence_store: Any,
    ) -> None:
        """
        Minimal deterministic evidence-chain producer (C1).

        BACKGROUND: the AI planner's only decision verbs are propose_action and
        propose_completion. Nothing in the autonomous loop ever created a Hypothesis,
        Validation, or Finding, so EvidenceStore.record_finding() (which requires a
        confirmed Hypothesis with a supporting Validation) was unreachable — and every
        agent's completion criteria require minimum_findings >= 1. The practical effect
        was that no autonomous run could ever pass verify_completion(); it always ran
        to max_steps.

        This is intentionally a FRAMEWORK decision, not an AI one: it only promotes an
        action that the agent's own actions.py already curated as a real methodology
        milestone (satisfies_phase=True — authored per-action in every agents/*/actions.py
        but previously never read anywhere in the codebase) and that actually executed
        successfully with real output. It does not attempt to judge severity or exploit
        impact; that stays severity="info" / moderate confidence, clearly labeled as a
        deterministic framework promotion. This is a floor, not a substitute for the
        richer AI-driven hypothesis/validation reasoning described in core/investigation.py
        — see AUDIT.md "Option A" for the fuller decision-verb design.
        """
        if defn is None or not getattr(defn, "satisfies_phase", False):
            return
        if not observations:
            return

        successful_obs = [obs for obs in observations if obs.exit_code == 0 and obs.raw_stdout.strip()]
        if not successful_obs:
            return

        from core.investigation import Hypothesis, Validation

        target = str(getattr(proposal, "target", None) or getattr(self.scope, "target", "local"))
        stdout_excerpt = successful_obs[0].raw_stdout.strip()[:400]
        action_id = getattr(defn, "action_id", "unknown")
        phase = getattr(defn, "phase", "unknown")

        # VERDICT-AWARE PROMOTION (2026-11-14, first used by the sqlmap
        # detection-engine action). A specialist engine's fact extractor emits a
        # structured 'vuln_signal' in the observation's parsed_facts; when
        # present it is the ORACLE, replacing the blunt exit-0 floor below:
        #   - confirmed=True  -> a real Finding at the engine's reported severity
        #                        (a confirmed SQLi is HIGH, not "info").
        #   - confirmed=False -> NO finding. The scan ran and came back clean;
        #                        that is a valid, already-recorded observation
        #                        (so the step still counts as progress), not a
        #                        false positive. This is exactly the "solid
        #                        pass/fail reasoning" the engine integration is
        #                        for: a clean scan must not manufacture a finding.
        # Actions with no vuln_signal (every generic probe) fall through to the
        # unchanged deterministic exit-0 milestone promotion below.
        vuln_signal = None
        signal_obs = None
        for _obs in successful_obs:
            _sig = (getattr(_obs, "parsed_facts", None) or {}).get("vuln_signal")
            if isinstance(_sig, dict):
                vuln_signal, signal_obs = _sig, _obs
                break

        if vuln_signal is not None:
            if not vuln_signal.get("confirmed"):
                # Clean scan: a genuine negative result, deliberately NOT a finding.
                return
            v_severity = vuln_signal.get("severity", "medium")
            if v_severity not in ("info", "low", "medium", "high", "critical"):
                v_severity = "medium"
            v_summary = vuln_signal.get("summary", f"{action_id} confirmed a vulnerability.")
            v_excerpt = signal_obs.raw_stdout.strip()[:400] if signal_obs else stdout_excerpt
            v_hyp = Hypothesis(
                observation_ids=[o.obs_id for o in successful_obs],
                description=f"{action_id} ({phase}) against '{target}': {v_summary}",
                validation_test=action_id,
                confidence="plausible",
                vuln_class=(vuln_signal.get("vuln_class") or None),
            )
            evidence_store.add_hypothesis(v_hyp)
            evidence_store.add_validation(Validation(
                hypothesis_id=v_hyp.hyp_id,
                action_id=action_id,
                observation=signal_obs or successful_obs[0],
                verdict="supports",
                reasoning=f"Detection engine returned a confirmed verdict: {v_summary}",
            ))
            _strength, _why = self._confirmation_strength(action_id, vuln_signal)
            if _strength == "lead":
                # Unverified/weak confirmation -> a LEAD, not a proven finding. Cap the
                # severity and mark it clearly so a human verifies before any report,
                # and so it never satisfies completion as a solved vulnerability.
                if v_severity in ("medium", "high", "critical"):
                    v_severity = "low"
                _title = f"[LEAD - verify] {v_summary} - {target}"
                _impact = (f"UNVERIFIED LEAD ({_why}). Manual verification required before "
                           f"reporting. {v_summary}")
            else:
                _title = f"{v_summary} - {target}"
                _impact = v_summary
            v_finding, v_msg = evidence_store.record_finding(
                hypothesis_id=v_hyp.hyp_id,
                title=_title,
                description=(
                    f"Detection via '{action_id}' against '{target}' [{_strength}]. "
                    f"{v_summary} Evidence excerpt: {v_excerpt!r}"
                ),
                severity=v_severity,
                affected_target=target,
                evidence_summary=v_excerpt,
                reproducible_steps=[" ".join(o.argv) for o in successful_obs],
                impact=_impact,
                confidence=0.9 if _strength == "proven" else 0.5,
                strength=_strength,
            )
            if v_finding is None:
                print(f"    [!] Verdict-aware evidence promotion failed unexpectedly: {v_msg}")
            return

        hyp = Hypothesis(
            observation_ids=[obs.obs_id for obs in successful_obs],
            description=(
                f"Action '{action_id}' ({phase} phase) executed successfully against "
                f"'{target}' and produced output consistent with its intended check."
            ),
            validation_test=action_id,
            confidence="plausible",
            milestone=True,  # benign phase marker -- must NOT unlock offensive tools
        )
        evidence_store.add_hypothesis(hyp)

        validation = Validation(
            hypothesis_id=hyp.hyp_id,
            action_id=action_id,
            observation=successful_obs[0],
            verdict="supports",
            reasoning=f"Exit code 0 with non-empty output from '{action_id}': {stdout_excerpt!r}",
        )
        evidence_store.add_validation(validation)  # flips hyp.status -> "confirmed"

        finding, msg = evidence_store.record_finding(
            hypothesis_id=hyp.hyp_id,
            title=f"{getattr(defn, 'description', action_id)} — verified against {target}",
            description=(
                f"Deterministic framework promotion: action '{action_id}' in phase "
                f"'{phase}' completed successfully with evidence: {stdout_excerpt!r}"
            ),
            severity="info",
            affected_target=target,
            evidence_summary=stdout_excerpt,
            reproducible_steps=[" ".join(obs.argv) for obs in successful_obs],
            impact="Phase milestone reached; see linked observation(s) for full detail.",
            confidence=0.6,
            strength="milestone",
        )
        if finding is None:
            # can_create_finding() was just satisfied by the validation we added above,
            # so this should not happen — but never silently swallow a framework rejection.
            print(f"    [!] Evidence promotion failed unexpectedly: {msg}")

    def _promote_lead_findings_from_recon(
        self,
        observations: List[Any],
        evidence_store: Any,
    ) -> None:
        """
        ARTEX merge round 2 (2026-11-24): recon_scan's secrets/exposed_files/
        takeovers modes are regex/pattern/content-signature matches, not actual
        confirming probes -- per the GRADING RULE this always lands as
        strength="lead", NEVER "proven". This function has NO branch that can
        emit "proven": that is enforced here structurally (hard-coded), not by
        trusting a prompt, a convention, or anything the engine itself reports.
        A framework decision, same posture as _promote_evidence_if_warranted.

        dns/ports/tls/tech/real_ip are deliberately NOT handled here -- they
        stay pure facts with no Finding at all (same as before this round).
        """
        from core.investigation import Hypothesis, Validation

        for obs in observations:
            if getattr(obs, "action_id", None) != "recon_scan":
                continue
            recon = (getattr(obs, "parsed_facts", None) or {}).get("recon")
            if not isinstance(recon, dict):
                continue

            target = str(getattr(self.scope, "target", "local"))

            secrets = recon.get("secrets") or []
            for hit in secrets:
                if not isinstance(hit, dict):
                    continue
                name = hit.get("name", "secret")
                masked = hit.get("masked_value", "")
                severity = hit.get("severity") if hit.get("severity") in (
                    "info", "low", "medium", "high", "critical") else "medium"
                hyp = Hypothesis(
                    observation_ids=[obs.obs_id],
                    description=f"recon_scan pattern-matched a possible {name} on '{target}'.",
                    validation_test="recon_scan:secrets",
                    confidence="speculative",
                )
                evidence_store.add_hypothesis(hyp)
                evidence_store.add_validation(Validation(
                    hypothesis_id=hyp.hyp_id, action_id="recon_scan", observation=obs,
                    verdict="supports",
                    reasoning=f"Regex pattern match for {name} (masked: {masked}); unverified.",
                ))
                evidence_store.record_finding(
                    hypothesis_id=hyp.hyp_id,
                    title=f"[LEAD - verify] Possible {name} exposed - {target}",
                    description=(
                        f"recon_scan's secrets mode pattern-matched a possible {name} on "
                        f"'{target}' (masked value: {masked}). This is a REGEX MATCH, not a "
                        f"confirmed live credential -- verify manually before treating as a "
                        f"real exposure."
                    ),
                    severity=severity if severity == "info" else "low",
                    affected_target=target,
                    evidence_summary=f"Pattern '{name}' matched (masked: {masked}).",
                    reproducible_steps=[" ".join(obs.argv)],
                    impact="UNVERIFIED LEAD (pattern match only). Manual verification required.",
                    confidence=0.3,
                    strength="lead",
                )

            exposed_files = recon.get("exposed_files") or []
            for hit in exposed_files:
                if not isinstance(hit, dict) or not hit.get("signature_matched"):
                    # A bare 200 with no content-signature match is not even
                    # lead-worthy noise; only a signature match gets surfaced.
                    continue
                path = hit.get("path", "?")
                url = hit.get("url", target)
                severity = hit.get("severity") if hit.get("severity") in (
                    "info", "low", "medium", "high", "critical") else "medium"
                hyp = Hypothesis(
                    observation_ids=[obs.obs_id],
                    description=f"recon_scan found an exposed sensitive file at '{url}'.",
                    validation_test="recon_scan:exposed_files",
                    confidence="plausible",
                )
                evidence_store.add_hypothesis(hyp)
                evidence_store.add_validation(Validation(
                    hypothesis_id=hyp.hyp_id, action_id="recon_scan", observation=obs,
                    verdict="supports",
                    reasoning=f"HTTP 200 + content-signature match for {path}.",
                ))
                evidence_store.record_finding(
                    hypothesis_id=hyp.hyp_id,
                    title=f"[LEAD - verify] Exposed file {path} - {target}",
                    description=(
                        f"recon_scan's exposed_files mode got HTTP 200 and a matching "
                        f"content-signature at '{url}' ({hit.get('description', path)}). "
                        f"Verify manually before treating as a real exposure."
                    ),
                    severity=severity if severity in ("info", "low") else "low",
                    affected_target=url,
                    evidence_summary=f"200 + content-signature matched at {path}.",
                    reproducible_steps=[" ".join(obs.argv)],
                    impact="UNVERIFIED LEAD (content-signature match only). Manual verification required.",
                    confidence=0.4,
                    strength="lead",
                )

            takeovers = recon.get("takeovers") or []
            for hit in takeovers:
                if not isinstance(hit, dict):
                    continue
                subdomain = hit.get("subdomain", "?")
                service = hit.get("service", "unknown")
                hyp = Hypothesis(
                    observation_ids=[obs.obs_id],
                    description=f"recon_scan found a possible dangling CNAME takeover on '{subdomain}'.",
                    validation_test="recon_scan:takeovers",
                    confidence="plausible",
                )
                evidence_store.add_hypothesis(hyp)
                evidence_store.add_validation(Validation(
                    hypothesis_id=hyp.hyp_id, action_id="recon_scan", observation=obs,
                    verdict="supports",
                    reasoning=f"CNAME points at {service} with an 'unclaimed' content signature.",
                ))
                evidence_store.record_finding(
                    hypothesis_id=hyp.hyp_id,
                    title=f"[LEAD - verify] Possible subdomain takeover ({service}) - {subdomain}",
                    description=(
                        f"recon_scan's subdomains mode found '{subdomain}' with a CNAME to "
                        f"{hit.get('cname', service)} and the service's distinctive "
                        f"'unclaimed' content signature. Claiming the resource to confirm "
                        f"is a state-changing action and requires explicit human approval "
                        f"-- this is surfaced as a lead only."
                    ),
                    severity="low",
                    affected_target=subdomain,
                    evidence_summary=f"CNAME to {service}, unclaimed-content signature matched.",
                    reproducible_steps=[" ".join(obs.argv)],
                    impact="UNVERIFIED LEAD (CNAME + content signature only). Manual verification required.",
                    confidence=0.4,
                    strength="lead",
                )

            auth_sweep = recon.get("auth_sweep") or []
            for hit in auth_sweep:
                if not isinstance(hit, dict) or "url" not in hit:
                    # Either an {"error": ...} entry from a failed sweep, or
                    # malformed -- never finding-shaped.
                    continue
                url = hit.get("url", target)
                status = hit.get("status", "?")
                hyp = Hypothesis(
                    observation_ids=[obs.obs_id],
                    description=f"recon_scan's auth_sweep found a declared-secured GET "
                                f"endpoint answering unauthenticated at '{url}'.",
                    validation_test="recon_scan:auth_sweep",
                    confidence="plausible",
                )
                evidence_store.add_hypothesis(hyp)
                evidence_store.add_validation(Validation(
                    hypothesis_id=hyp.hyp_id, action_id="recon_scan", observation=obs,
                    verdict="supports",
                    reasoning=f"Unauthenticated GET to a declared-secured operation returned "
                              f"HTTP {status} (reached the handler/validation layer).",
                ))
                evidence_store.record_finding(
                    hypothesis_id=hyp.hyp_id,
                    title=f"[LEAD - verify] Possible broken function-level auth - {url}",
                    description=(
                        f"recon_scan's auth_sweep mode discovered an OpenAPI/Swagger spec "
                        f"declaring '{url}' as requiring auth, then got HTTP {status} on an "
                        f"UNAUTHENTICATED GET request. This means the request reached the "
                        f"handler/validation layer rather than being rejected at the auth "
                        f"layer, which is a genuine signal -- but it is still only a single "
                        f"unauthenticated probe against a declared contract, not a confirmed "
                        f"read of real protected data. Verify the response body actually "
                        f"contains protected data before treating this as a real exposure."
                    ),
                    severity="medium",
                    affected_target=url,
                    evidence_summary=f"Declared-secured GET answered HTTP {status} unauthenticated.",
                    reproducible_steps=[" ".join(obs.argv)],
                    impact="UNVERIFIED LEAD (declared-secured endpoint reached unauthenticated). "
                           "Manual verification of response content required.",
                    confidence=0.4,
                    strength="lead",
                )

    def _update_sweep_coverage(self, observations: List[Any], evidence_store: Any) -> None:
        """
        --sweep mode's deterministic ENUMERATE+CHARACTERIZE+gate bookkeeping.
        Pure bookkeeping on EvidenceStore.surface_map/coverage_ledger -- never
        creates a Hypothesis/Finding, never calls the AI. For every new
        Observation, turns its parsed_facts into surface entries (name-based
        characterization, see agents/common/surface_characterize.py), runs
        the context gate to decide which vuln classes are even plausible for
        each (endpoint, param), and upserts a coverage-ledger row for every
        gated class not already tracked, so the planner's prompt (via
        AIPlanner._format_sweep_coverage) shows a stable "what's covered,
        what's next" ledger instead of nothing.

        Only called when BaseAgent.run_autonomous(sweep=True).
        """
        from agents.common.surface_characterize import characterize_observation, CLASS_TO_ENGINE
        from core.tools.class_gate import gate_classes_for_entry

        for obs in observations:
            action_id = getattr(obs, "action_id", None)
            facts = getattr(obs, "parsed_facts", None) or {}
            for entry in characterize_observation(action_id, facts):
                endpoint_key = entry["endpoint_key"]
                evidence_store.add_surface_entry(endpoint_key, **{k: v for k, v in entry.items() if k != "endpoint_key"})
                for vuln_class in gate_classes_for_entry(entry):
                    ledger_key = f"{vuln_class}::{endpoint_key}"
                    if ledger_key in evidence_store.coverage_ledger:
                        continue  # already tracked -- CONFIRM OR MOVE ON owns status updates from here.
                    engine = CLASS_TO_ENGINE.get(vuln_class)
                    status = "untested" if engine else "no_engine"
                    note = None if engine else f"no typed engine for {vuln_class} -- low confidence (self_script fallback)"
                    evidence_store.record_coverage(vuln_class, endpoint_key, status, engine=engine, note=note)

            # CONFIRM OR MOVE ON: when an observation came from one of the
            # typed engines and carries an objective vuln_signal, advance
            # every ledger row of that vuln_class whose endpoint_key is a
            # substring/prefix match of the engine's own target argv (a
            # best-effort match -- this only drives DISPLAY/steering, never
            # confirmation rigor, which stays entirely on
            # _confirmation_strength/record_finding regardless of this
            # ledger's contents).
            vuln_signal = facts.get("vuln_signal") if isinstance(facts, dict) else None
            if isinstance(vuln_signal, dict) and action_id in CLASS_TO_ENGINE.values():
                sig_class = str(vuln_signal.get("vuln_class") or "").lower()
                target_hint = " ".join(str(a) for a in (getattr(obs, "argv", None) or []))
                new_status = "proven" if vuln_signal.get("confirmed") else "tested"
                for key, row in evidence_store.coverage_ledger.items():
                    if row.get("vuln_class") != sig_class and sig_class not in (row.get("vuln_class") or ""):
                        continue
                    ep = row.get("endpoint_key", "")
                    endpoint_part = ep.split("::", 1)[0].split(" ", 1)[-1] if "::" in ep else ep
                    if endpoint_part and endpoint_part in target_hint:
                        if row.get("status") != "proven":
                            evidence_store.record_coverage(
                                row["vuln_class"], row["endpoint_key"], new_status,
                                engine=row.get("engine"), note=row.get("note"),
                            )

    def _handle_form_hypothesis(self, proposal, evidence_store) -> Dict[str, Any]:
        """
        AI decision verb (full C1 extension): create a Hypothesis from observations the
        AI claims to have already seen. Every observation_id is independently checked
        against evidence_store.observations before anything is recorded -- an AI that
        fabricates, paraphrases, or misremembers an obs_id gets a rejection that names
        exactly which id was unknown, instead of a hypothesis silently built on evidence
        that does not exist. See AUDIT.md "Option A" for why this exists alongside the
        deterministic satisfies_phase promotion in _promote_evidence_if_warranted.
        """
        from core.investigation import Hypothesis

        params = proposal.parameters or {}
        observation_ids = params.get("observation_ids") or []
        description = str(params.get("description", ""))[:1000]
        validation_test = str(params.get("validation_test", ""))[:200]
        confidence = params.get("confidence", "plausible")
        if confidence not in ("speculative", "plausible", "likely"):
            confidence = "plausible"

        if not isinstance(observation_ids, list) or not observation_ids:
            return {
                "status": "rejected",
                "rejection_reason": "form_hypothesis requires a non-empty 'observation_ids' list.",
            }

        unknown_ids = [oid for oid in observation_ids if oid not in evidence_store.observations]
        if unknown_ids:
            return {
                "status": "rejected",
                "rejection_reason": (
                    f"form_hypothesis cited observation_id(s) that do not exist in the "
                    f"evidence store: {unknown_ids}. Only cite obs_id values copied "
                    f"verbatim from the OBSERVATIONS section of the prompt."
                ),
            }

        vuln_class = self._infer_vuln_class(
            params.get("vuln_class"), validation_test, description
        )

        # Anti-spam (observed live 2026-11-16: 8 near-identical hypotheses in a row
        # instead of testing one). Refuse a hypothesis that duplicates an OPEN one
        # -- same description, or same evidence + same class -- and tell the planner
        # to TEST the existing hypothesis instead of re-forming it.
        norm_desc = " ".join((description or "").lower().split())
        obs_key = frozenset(observation_ids)
        for h in evidence_store.hypotheses.values():
            if getattr(h, "milestone", False) or h.status not in ("open", "validating"):
                continue
            dup_desc = norm_desc and " ".join((h.description or "").lower().split()) == norm_desc
            dup_focus = (frozenset(h.observation_ids) == obs_key and vuln_class
                         and getattr(h, "vuln_class", None) == vuln_class)
            if dup_desc or dup_focus:
                return {
                    "status": "rejected",
                    "rejection_reason": (
                        f"A matching OPEN hypothesis already exists ({h.hyp_id}). Do NOT re-form it -- "
                        f"TEST it instead: propose self_script (or its validation_test) against the exact "
                        f"parameter and endpoint, or record_validation against it."
                    ),
                }

        # Per-focus saturation guard (observed 2026-11-21: the agent spent all 25 steps
        # on /cart, evading the per-class guard by RELABELING the class while re-citing the
        # same evidence). Count hypotheses sharing this one's observation focus, any class.
        focus_key = frozenset(observation_ids)
        same_focus = [h for h in evidence_store.hypotheses.values()
                      if not getattr(h, "milestone", False)
                      and frozenset(getattr(h, "observation_ids", [])) == focus_key]
        if len(same_focus) >= 3 and getattr(evidence_store, "proven_finding_count", 0) == 0:
            return {
                "status": "rejected",
                "rejection_reason": (
                    f"You have formed {len(same_focus)} hypotheses on the SAME evidence/endpoint and "
                    f"none is PROVEN -- this is tunnel vision. STOP testing this focus. Run coverage "
                    f"(endpoint_discovery / browser_probe / param_fuzz) on a DIFFERENT part of the app, "
                    f"form a hypothesis citing a DIFFERENT observation/endpoint, or propose_completion."
                ),
            }

        # Per-class saturation guard (observed 2026-11-21: after the probe tool was
        # spent, 7 near-identical race hypotheses were formed+refuted in a row, burning
        # 14 steps). Once a class has several hypotheses and none is PROVEN, re-forming
        # it is a sterile loop -- force a different test, a pivot, or completion.
        if vuln_class:
            same_class = [h for h in evidence_store.hypotheses.values()
                          if getattr(h, "vuln_class", None) == vuln_class
                          and not getattr(h, "milestone", False)]
            if len(same_class) >= 3 and getattr(evidence_store, "proven_finding_count", 0) == 0:
                return {
                    "status": "rejected",
                    "rejection_reason": (
                        f"'{vuln_class}' already has {len(same_class)} hypotheses and none is PROVEN -- "
                        f"re-forming it again is a sterile loop. Do ONE of: (a) run a test you have NOT "
                        f"tried against the exact endpoint (a reliable engine, or a self_script that FIXES "
                        f"what failed before -- e.g. robust CSRF/session handling), (b) PIVOT to a different "
                        f"endpoint or vuln class, or (c) propose_completion if nothing else is testable."
                    ),
                }

        hyp = Hypothesis(
            observation_ids=observation_ids,
            description=description or "(no description provided)",
            validation_test=validation_test or "(unspecified)",
            confidence=confidence,
            vuln_class=vuln_class,
        )
        evidence_store.add_hypothesis(hyp)
        return {
            "status": "success",
            "hypothesis_id": hyp.hyp_id,
            "result_summary": f"Hypothesis {hyp.hyp_id} recorded ({confidence}): {description[:150]}",
        }

    def _scrub_lab_markers(self, text):
        """Remove PortSwigger lab identifiers that would tell the agent the answer
        (blind-lab mode only). Real targets have none of these."""
        if not text:
            return text
        import re as _re
        t = text
        t = _re.sub("<!--\\s*LAB_HEAD_START\\s*-->.*?<!--\\s*LAB_HEAD_END\\s*-->",
                    "<head><title>Shop</title></head>", t, flags=_re.S | _re.I)
        t = _re.sub("<!--\\s*LAB_HEADER_START\\s*-->.*?<!--\\s*LAB_HEADER_END\\s*-->",
                    "", t, flags=_re.S | _re.I)
        t = _re.sub("<title>.*?</title>", "<title>Shop</title>", t, flags=_re.S | _re.I)
        t = _re.sub("<div id=.?academyLabHeader.*?</div>", "", t, flags=_re.S | _re.I)
        t = _re.sub("https?://(?:www\\.)?portswigger\\.net/web-security/[^\\s<>)\"']*",
                    "https://example.invalid/", t, flags=_re.I)
        t = _re.sub("(?i)web security academy", "application", t)
        t = _re.sub("(?i)portswigger", "the-vendor", t)
        return t

    def _scrub_lab_facts(self, facts):
        """Drop solution-leaking portswigger.net links and lab markers out of
        parsed_facts (blind-lab mode). Keeps the lab's own host -- that's the
        target.

        BUG FIX (2026-11-28, found via a live --sweep + OpenRouter run): this
        was SHALLOW -- it only scrubbed top-level string/list-of-string
        values, so a self_script's own vuln_signal JSON (which commonly
        embeds a raw HTML snippet as *evidence*, nested as
        vuln_signal -> evidence -> hidden_products[i] -> snippet) passed
        straight through the `else: out[k] = v` branch untouched, leaking the
        literal `<!--LAB_HEAD_START-->` marker into what the planner's prompt
        shows the model -- a real blind-constraint violation (MUST-PRESERVE
        #3), not just a cosmetic issue. Now recurses into dicts and lists at
        any depth so no nested string value can carry a marker through
        unscrubbed.
        """
        return self._scrub_lab_value(facts)

    def _scrub_lab_value(self, value):
        """Recursive helper for _scrub_lab_facts -- scrubs strings at any
        depth inside dicts/lists/tuples; non-container, non-string values
        (int/float/bool/None) pass through unchanged. Matches the original
        shallow function's exact semantics, just applied recursively: a list
        entry that IS a solution-leaking portswigger.net link is DROPPED
        (unchanged behavior); every other string anywhere just gets
        _scrub_lab_markers's text substitution (never dropped)."""
        if isinstance(value, str):
            return self._scrub_lab_markers(value)
        if isinstance(value, dict):
            return {k: self._scrub_lab_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [
                self._scrub_lab_value(x) for x in value
                if not (isinstance(x, str) and "portswigger.net" in x.lower())
            ]
        if isinstance(value, tuple):
            return tuple(self._scrub_lab_value(x) for x in value)
        return value

    def _infer_vuln_class(self, explicit, validation_test, description):
        """
        Resolve the vuln_class a hypothesis is about, so the suspicion gate can
        unlock the matching specialist tool. Priority:
          1. an explicit vuln_class the planner supplied,
          2. the vuln_class of the action it named as validation_test,
          3. a keyword scan of the hypothesis description.
        Returns a lowercase class string or None (a generic suspicion -- still
        unlocks the generic weapons like self_script).
        """
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip().lower()
        reg = getattr(self, "_registry", None)
        if reg is not None and validation_test:
            try:
                d = reg.get(validation_test)
                if d is not None and getattr(d, "vuln_class", None):
                    return d.vuln_class
            except Exception:
                pass
        low = (description or "").lower()
        KW = [
            ("ssrf", ("ssrf", "server-side request", "localhost", "169.254", "internal url", "metadata")),
            ("sqli", ("sql injection", "sqli", "sql error", "union select", "boolean-based", "' or ")),
            ("xss", ("xss", "cross-site scripting", "reflect", "<script", "javascript:")),
            ("idor", ("idor", "insecure direct object", "object reference", "another user",
                      "cross-tenant", "increment")),
            ("open-redirect", ("open redirect", "open-redirect", "location header")),
            ("lfi", ("lfi", "local file inclusion", "path traversal", "../", "etc/passwd",
                     "directory traversal")),
            ("cmd-injection", ("command injection", "cmd injection", "os command", "remote code", " rce")),
            ("race-condition", ("race condition", "race-condition", "concurrency", "concurrent",
                                "timing window", "time-of-check", "toctou", "double-spend",
                                "overrun", "redeem twice", "single-use")),
            ("business-logic", ("business logic", "logic flaw", "limit bypass", "coupon",
                                "discount", "negative balance", "workflow bypass", "state machine")),
            ("auth-bypass", ("auth bypass", "authentication bypass", "privilege escalation", "role bypass")),
            ("request-smuggling", ("request smuggling", "http request smuggling", "cl.te", "te.cl")),
        ]
        for cls, needles in KW:
            if any(n in low for n in needles):
                return cls
        return None

    def _handle_record_validation(self, proposal, evidence_store) -> Dict[str, Any]:
        """
        AI decision verb (full C1 extension): record a Validation linking one real
        Observation to one real Hypothesis. Both ids are independently checked to exist
        before the Validation is created -- this is what lets
        EvidenceStore.add_validation() legitimately flip a Hypothesis to 'confirmed'
        (verdict='supports'), which is the only thing that later lets record_finding's
        can_create_finding() gate pass. The AI's own judgement of the verdict is not
        second-guessed here (that is its job), only the ids it claims the judgement rests on.
        """
        from core.investigation import Validation

        params = proposal.parameters or {}
        hypothesis_id = params.get("hypothesis_id")
        observation_id = params.get("observation_id")
        verdict = params.get("verdict")
        reasoning = str(params.get("reasoning", ""))[:1000]

        if hypothesis_id not in evidence_store.hypotheses:
            return {
                "status": "rejected",
                "rejection_reason": (
                    f"record_validation cited hypothesis_id '{hypothesis_id}' which does "
                    f"not exist. Only cite hyp_id values copied verbatim from ACTIVE HYPOTHESES."
                ),
            }
        if observation_id not in evidence_store.observations:
            return {
                "status": "rejected",
                "rejection_reason": (
                    f"record_validation cited observation_id '{observation_id}' which "
                    f"does not exist. Only cite obs_id values copied verbatim from OBSERVATIONS."
                ),
            }
        if verdict not in ("supports", "refutes", "inconclusive"):
            return {
                "status": "rejected",
                "rejection_reason": (
                    f"record_validation requires verdict in supports|refutes|inconclusive, "
                    f"got '{verdict}'."
                ),
            }

        observation = evidence_store.observations[observation_id]

        # Structural rigor gate (2026-11-26): a verdict='supports' claim must be
        # backed by the cited observation's ACTUAL content -- see
        # _validate_supports_claim's docstring for the live false positive this
        # closes. 'refutes'/'inconclusive' are never gated here (they do not
        # advance a hypothesis toward being confirmed, so there is nothing to
        # fabricate).
        if verdict == "supports":
            ok, why = self._validate_supports_claim(observation, reasoning)
            if not ok:
                return {
                    "status": "rejected",
                    "rejection_reason": (
                        f"record_validation verdict='supports' REJECTED: {why} "
                        f"(hypothesis {hypothesis_id} remains unconfirmed)."
                    ),
                }

        validation = Validation(
            hypothesis_id=hypothesis_id,
            action_id=observation.action_id,
            observation=observation,
            verdict=verdict,
            reasoning=reasoning or "(no reasoning provided)",
        )
        evidence_store.add_validation(validation)  # may flip hypothesis status
        new_status = evidence_store.hypotheses[hypothesis_id].status
        return {
            "status": "success",
            "validation_id": validation.val_id,
            "result_summary": (
                f"Validation {validation.val_id} recorded (verdict={verdict}) against "
                f"hypothesis {hypothesis_id}; hypothesis status is now '{new_status}'."
            ),
        }

    def _compute_finding_strength(self, hypothesis_id: str, evidence_store) -> Tuple[str, str]:
        """
        CRITICAL FIX (2026-11-28, found via a live OpenRouter-brain run,
        bug_bounty_autohunt_1795852409): _handle_record_finding() below NEVER
        passed a `strength` to EvidenceStore.record_finding(), so every single
        AI-driven record_finding call defaulted to strength="proven" -- with
        ZERO rigor check -- regardless of how weak the evidence actually was.
        Confirmed live: a self_script's own naive "response length differs"
        logic set vuln_signal.confirmed=True; record_validation's verdict=
        'supports' was accepted (correctly, per _validate_supports_claim's
        contract: it trusts a vuln_signal's own confirmed flag rather than
        re-deriving it); but nothing downstream ever re-applied
        _confirmation_strength's WEAK-keyword/XML-parser-error scrutiny to
        that SAME self_script-authored signal before calling it "proven" --
        so a medium-severity "proven" IDOR finding was recorded, completion
        would have accepted it, and the trajectory was already written to the
        training JSONL files before this run was interrupted for diagnosis.

        This is the SAME discipline _promote_evidence_if_warranted already
        applies to the FRAMEWORK-automatic path, applied here to the AI's own
        manual form_hypothesis->record_validation->record_finding chain: walk
        every supporting Validation's cited Observation; a framework ENGINE's
        confirmed vuln_signal (action_id not in _MODEL_AUTHORED_ACTIONS) is
        trusted fully; a MODEL-AUTHORED action's vuln_signal (self_script/
        nuclei_template) is re-scored through the REAL _confirmation_strength
        gate, not just its own self-reported `confirmed` flag; and lacking any
        structured vuln_signal at all (pure narrative reasoning, no engine/
        probe oracle) is treated as unproven -- a human-verifiable lead, never
        a framework-certified proven finding. The AI's own `severity`/
        `confidence` params can never override this; the computed strength here
        is authoritative, same as the framework-automatic path.

        Returns (strength, reason).
        """
        supporting = [
            v for v in evidence_store.validations.values()
            if v.hypothesis_id == hypothesis_id and v.verdict == "supports"
        ]
        if not supporting:
            return "lead", "no supporting validation on record"

        best_strength = "lead"
        best_reason = "no structured vuln_signal in any supporting observation -- unverified narrative only"
        for val in supporting:
            obs = getattr(val, "observation", None)
            facts = getattr(obs, "parsed_facts", None) if obs else None
            vuln_signal = facts.get("vuln_signal") if isinstance(facts, dict) else None
            if not isinstance(vuln_signal, dict) or not vuln_signal.get("confirmed"):
                continue
            action_id = getattr(obs, "action_id", None) or getattr(val, "action_id", "unknown")
            strength, reason = self._confirmation_strength(action_id, vuln_signal)
            if strength == "proven":
                return "proven", reason  # a single genuinely proven citation is enough
            if best_strength != "proven":
                best_strength, best_reason = "lead", reason
        return best_strength, best_reason

    def _handle_record_finding(self, proposal, evidence_store) -> Dict[str, Any]:
        """
        AI decision verb (full C1 extension): attempt to promote a hypothesis into a
        Finding. EvidenceStore.record_finding() is the actual gate here -- it
        independently re-checks can_create_finding() (confirmed hypothesis + at least
        one supporting validation) and refuses with a specific reason if that is not
        satisfied, regardless of what the AI claims. This method only shapes the AI's
        parameters into that call and relays the framework's verdict; it adds no
        additional trust of its own.

        `strength` is NEVER taken from the AI -- see _compute_finding_strength()'s
        docstring for the live incident that made this a hard requirement, not a
        style preference.
        """
        params = proposal.parameters or {}
        hypothesis_id = params.get("hypothesis_id")
        title = str(params.get("title", ""))[:200]
        description = str(params.get("description", ""))[:2000]
        severity = params.get("severity", "medium")
        if severity not in ("info", "low", "medium", "high", "critical"):
            severity = "medium"
        affected_target = str(params.get("affected_target") or getattr(self.scope, "target", "local"))
        evidence_summary = str(params.get("evidence_summary", ""))[:1000]
        reproducible_steps = params.get("reproducible_steps") or []
        if not isinstance(reproducible_steps, list):
            reproducible_steps = [str(reproducible_steps)]
        impact = str(params.get("impact", "pending further validation"))[:1000]
        try:
            confidence = float(params.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        if not hypothesis_id or not title or not description or not evidence_summary:
            return {
                "status": "rejected",
                "rejection_reason": (
                    "record_finding requires hypothesis_id, title, description, and "
                    "evidence_summary."
                ),
            }

        strength, strength_reason = self._compute_finding_strength(hypothesis_id, evidence_store)

        finding, msg = evidence_store.record_finding(
            hypothesis_id=hypothesis_id,
            title=title,
            description=description,
            severity=severity,
            affected_target=affected_target,
            evidence_summary=evidence_summary,
            reproducible_steps=reproducible_steps,
            impact=impact,
            confidence=confidence,
            strength=strength,
        )
        if finding is None:
            # can_create_finding() refused -- e.g. hypothesis not yet confirmed, or no
            # supporting validation on record. Relay the framework's exact reason so the
            # planner's next turn can see what is still missing (e.g. "call
            # record_validation with verdict=supports first").
            return {"status": "rejected", "rejection_reason": msg}
        return {
            "status": "success",
            "finding_id": finding.finding_id,
            "result_summary": (
                f"Finding {finding.finding_id} ({severity}) recorded as strength={strength} "
                f"({strength_reason}): {title}"
            ),
        }

    def _handle_update_app_model(self, proposal, evidence_store) -> Dict[str, Any]:
        """
        AI decision verb: record or update working memory about the target application
        (purpose, components, roles, trust boundaries, state/resources, fragile assumptions).
        """
        params = proposal.parameters or {}
        evidence_store.update_app_model(
            purpose=params.get("purpose"),
            components=params.get("components"),
            roles=params.get("roles"),
            trust_boundaries=params.get("trust_boundaries"),
            state_and_resources=params.get("state_and_resources"),
            data_flows=params.get("data_flows"),
            assumptions=params.get("assumptions"),
        )
        summary_excerpt = evidence_store.get_app_model_summary()[:150].replace("\n", " ")
        return {
            "status": "success",
            "result_summary": f"AppModel updated: {summary_excerpt}",
        }

    def run(self) -> Dict[str, Any]:
        """
        Execute the assessment workflow across all registered phases.
        Handles:
        - Manual vs Automated execution flow
        - Pre-flight tool verification
        - AI scope compliance verification
        - Adaptive phase recommendations from YAML + RAG
        - Real-time audit recording in SessionManager & EvidenceManager
        - Graceful Ctrl+C interruption and state persistence
        """
        print("\n" + "=" * 70)
        print(f"  LAUNCHING: {self.agent_name.upper()}")
        print(f"  Target:    {self.scope.target}")
        print(f"  Mode:      {self.mode.upper()}")
        print(f"  Active:    {'AUTHORIZED' if self.scope.active_testing_allowed else 'READ-ONLY (Passive)'}")
        print("=" * 70 + "\n")

        # Ensure session exists
        if not self.session_mgr.active_session_name:
            session_name = f"{self.agent_type}_{int(time.time())}"
            self.session_mgr.create_session(
                name=session_name,
                scope=str(self.scope.in_scope),
                target=self.scope.target,
                target_type=self.agent_type,
                mode=self.mode,
            )
            # --resume fix: persist the full Scope (every authorization flag, not just
            # the target) so --resume can reconstruct it exactly instead of falling back
            # to a bare, fully-unauthorized Scope(target, in_scope).
            self.session_mgr.save_scope(self.scope)

        completed_phases: List[PhaseResult] = []
        phase_queue = list(self.phase_modules)
        current_idx = 0

        try:
            while current_idx < len(phase_queue):
                phase = phase_queue[current_idx]
                phase_name = getattr(phase, "name", f"phase_{current_idx}")
                req_tools = getattr(phase, "requires_tools", [])

                print(f"\n[*] Entering Phase [{current_idx + 1}/{len(phase_queue)}]: {phase_name.upper()}")

                # 1. Check required tools
                missing_tools = [t for t in req_tools if not self.tool_manager.is_tool_installed(t)]
                if missing_tools:
                    print(f"    [!] Missing required tool(s): {', '.join(missing_tools)}")
                    if self.mode == "manual":
                        install_choice = input(f"    Attempt installation via {self.tool_manager.os_info.package_manager}? [Y/n]: ").strip().lower()
                        if install_choice != "n":
                            for t in missing_tools:
                                print(f"    [*] Installing {t}...")
                                ok, msg = self.tool_manager.install_tool(t)
                                print(f"        {msg}")
                    else:
                        print("    [*] Automated mode: proceeding with fallback mechanisms if tool missing.")

                # 2. Gather AI unified context (Session + YAML + RAG + History + Tools)
                tools_info = [
                    {"name": t, "installed": self.tool_manager.is_tool_installed(t)}
                    for t in req_tools
                ]
                ai_context = self.chat_assistant.gather_ai_context(
                    agent_type=self.agent_type,
                    current_phase=phase_name,
                    session_state=self.session_mgr.state,
                    phase_results={p.phase_name: p.to_dict() for p in completed_phases},
                    available_tools=tools_info
                )

                # 3. AI Scope Compliance Gate (Part 2, Item 6)
                if self.scope.active_testing_allowed:
                    comp_check = self.chat_assistant.check_scope_compliance(
                        intended_command=f"Phase {phase_name} on {self.scope.target}",
                        scope=self.scope,
                        prohibited_items=self.scope.prohibited_tests + self.scope.prohibited_tools
                    )
                    if not comp_check.get("allowed", True):
                        print(f"    [!] SCOPE VIOLATION DETECTED BY AI: {comp_check.get('reason')}")
                        print(f"    [!] Halting phase '{phase_name}'.")
                        res = PhaseResult(phase_name=phase_name, status="blocked", error=comp_check.get("reason"))
                        completed_phases.append(res)
                        current_idx += 1
                        continue

                # 4. Manual Approval Gate
                if self.mode == "manual":
                    print(f"    Ready to execute phase: {phase_name}")
                    print(f"    Required Tools: {', '.join(req_tools) or 'None (Built-in)'}")
                    user_action = input("    Proceed with phase? [(Y)es / (N)o / (S)kip / (P)aste external]: ").strip().lower()
                    
                    if user_action in ("s", "skip"):
                        print(f"    [*] Skipped phase {phase_name}.")
                        completed_phases.append(PhaseResult(phase_name=phase_name, status="skipped"))
                        current_idx += 1
                        continue
                    elif user_action in ("p", "paste"):
                        print("    Enter/paste external terminal output below (finish with EOF or empty line):")
                        pasted_lines = []
                        while True:
                            try:
                                line = input()
                                if line.strip().upper() in ("EOF", "END") or not line.strip():
                                    break
                                pasted_lines.append(line)
                            except (KeyboardInterrupt, EOFError):
                                break
                        raw_paste = "\n".join(pasted_lines).strip()
                        interp = self.chat_assistant.interpret_pasted_output(raw_paste, ai_context)
                        print(f"\n    [+] AI Interpretation of pasted output:\n        {interp.get('summary')}")
                        self.session_mgr.add_note(f"Pasted in phase {phase_name}:\n{raw_paste}")
                        res = PhaseResult(
                            phase_name=phase_name,
                            status="success",
                            raw_output=raw_paste,
                            findings=interp
                        )
                        completed_phases.append(res)
                        current_idx += 1
                        continue
                    elif user_action == "n":
                        print("    [*] Phase execution declined by operator.")
                        completed_phases.append(PhaseResult(phase_name=phase_name, status="declined"))
                        current_idx += 1
                        continue

                # 5. Execute Phase Script
                try:
                    result: PhaseResult = phase.run(
                        session=self.session_mgr.state,
                        tools=self.tool_manager,
                        executor=self.executor,
                        scope=self.scope
                    )
                except Exception as exc:
                    print(f"    [!] Error during {phase_name}: {exc}")
                    # Novel situation handling via AI (Part 2, Item 3)
                    recovery = self.chat_assistant.suggest_recovery(str(exc), ai_context)
                    print(f"    [*] AI Diagnostic: {recovery.get('diagnostic')}")
                    print(f"    [*] AI Suggested Recovery: {recovery.get('ai_reasoning')}")
                    result = PhaseResult(phase_name=phase_name, status="failed", error=str(exc))

                completed_phases.append(result)

                # Record evidence for each command executed in this phase
                for cmd_entry in result.commands_run:
                    self.session_mgr.add_evidence(
                        command=cmd_entry.get("command", phase_name),
                        output=cmd_entry.get("stdout", "") or cmd_entry.get("output", ""),
                        step_id=phase_name,
                        status=cmd_entry.get("status", "completed")
                    )

                # Update session findings if phase produced any
                if result.findings:
                    self.session_mgr.add_ingested_findings(
                        finding_type=f"{self.agent_type}_{phase_name}",
                        findings=result.findings,
                        raw_source=phase_name
                    )

                # 6. Adaptive Planning via AI (Part 2, Item 1)
                remaining_candidates = [
                    getattr(p, "name", str(p))
                    for p in phase_queue[current_idx + 1:]
                ]
                if remaining_candidates:
                    ai_recommendation = self.chat_assistant.recommend_next_phase(
                        context=ai_context,
                        candidate_phases=remaining_candidates
                    )
                    recommended_next = ai_recommendation.get("next")
                    print(f"    [AI Consultant] Recommended next phase: {recommended_next}")
                    print(f"    [AI Consultant] Rationale: {ai_recommendation.get('reason')}")
                    if ai_recommendation.get("scenario_id"):
                        print(f"    [AI Consultant] Grounded in YAML scenario: {ai_recommendation.get('scenario_id')}")

                current_idx += 1

            # 7. End of assessment: AI Report Synthesis (Part 2, Item 7)
            print("\n" + "=" * 70)
            print("  ALL METHODOLOGY PHASES COMPLETED. SYNTHESIZING REPORT...")
            print("=" * 70)
            report_text = self.chat_assistant.synthesize_report(self.session_mgr.state)
            report_file = self.session_mgr.session_dir / "report.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"[+] Final Audit Report successfully compiled to: {report_file}\n")

            return {
                "status": "completed",
                "completed_phases": [p.to_dict() for p in completed_phases],
                "report_path": str(report_file)
            }

        except KeyboardInterrupt:
            print("\n\n[!] Execution paused by operator (Ctrl+C).")
            print("[*] Persisting session state to disk...")
            self.session_mgr.save_current_state()
            print(f"[+] State saved to session '{self.session_mgr.active_session_name}'.")
            print(f"    Resume anytime with: python main.py --resume {self.session_mgr.active_session_name}\n")
            return {
                "status": "interrupted",
                "session_name": self.session_mgr.active_session_name,
                "completed_phases": [p.to_dict() for p in completed_phases]
            }


    def _find_hypothesis_for_manual_payload(self, evidence_store) -> Optional[Any]:
        """
        The hypothesis the self_script proposal we're intercepting is almost
        certainly testing: the strongest OPEN/VALIDATING non-milestone
        hypothesis, same "which hypothesis is the model actually acting on"
        logic core/ai_planner.py's own steering ladder already uses (there is
        normally exactly one active line of investigation at a time). Used to
        surface vuln_class in the manual-payload brief -- the self_script
        proposal's own parameters carry only a one-line hypothesis string,
        not the structured vuln_class the Hypothesis object itself has.
        """
        conf_rank = {"likely": 3, "plausible": 2, "speculative": 1}
        open_hyps = [
            h for h in evidence_store.hypotheses.values()
            if not getattr(h, "milestone", False) and h.status in ("open", "validating")
        ]
        if not open_hyps:
            return None
        return sorted(open_hyps, key=lambda h: (conf_rank.get(h.confidence, 0), h.created_at), reverse=True)[0]

    def _handle_manual_payload(self, proposal, evidence_store) -> Optional[Any]:
        """
        MANUAL PAYLOAD MODE (--manual-payload): pause right before the
        planner's own self_script would run, print a payload-request brief,
        and read a human-supplied PAYLOAD:/CURL:/SCRIPT: from stdin instead.

        Returns a REPLACEMENT AIActionProposal (http_probe for PAYLOAD:,
        self_script for CURL:/SCRIPT:) that run_autonomous()'s caller then
        runs through the UNCHANGED execute_action() gated path -- same scope
        check, same approval gate, same confirmation-rigor grading a
        model-authored proposal would get. Returns None if the operator
        enters nothing (this step is then recorded as skipped).
        """
        from core.action_contract import AIActionProposal
        from agents.common.manual_payload import (
            infer_param_name, parse_curl, build_curl_replay_script, build_payload_brief,
        )

        params = proposal.parameters or {}
        target = str(params.get("target") or proposal.target or self.scope.target)
        hypothesis_text = str(params.get("hypothesis", ""))

        recent_obs = list(evidence_store.observations.values())[-1] if evidence_store.observations else None
        recent_excerpt = (recent_obs.raw_stdout or "")[:400] if recent_obs else "(no prior observation)"
        inferred_param = infer_param_name(target, hypothesis_text)
        matching_hyp = self._find_hypothesis_for_manual_payload(evidence_store)
        vuln_class = getattr(matching_hyp, "vuln_class", None)
        # The matching hypothesis's own description is usually more specific
        # than the self_script proposal's one-line "hypothesis" param (which
        # the model sometimes leaves thin) -- prefer it when available.
        if matching_hyp is not None and getattr(matching_hyp, "description", None):
            hypothesis_text = matching_hyp.description

        print("\n" + "-" * 70)
        print("  [MANUAL PAYLOAD] The planner wants to test this itself:")
        print()
        print(build_payload_brief(target, inferred_param, vuln_class, hypothesis_text, recent_excerpt))
        print()
        print("  Paste exactly ONE of the following, then finish with a blank line or EOF:")
        print("    PAYLOAD:<value>            -- your crafted payload for the parameter above")
        print("    PAYLOAD:<param>=<value>    -- or name a different parameter explicitly")
        print("    CURL:<curl command>        -- replay this exact request (single line)")
        print("    SCRIPT:                    -- a Python self_script probe follows (same contract the AI uses)")
        print("  Leave blank to skip this step.")
        print("-" * 70)

        lines: List[str] = []
        while True:
            try:
                line = input()
            except (KeyboardInterrupt, EOFError):
                break
            if not line.strip():
                break
            lines.append(line)
        raw = "\n".join(lines).strip()
        if not raw:
            print("  [MANUAL PAYLOAD] Nothing entered -- skipping this step.")
            return None

        upper = raw.upper()

        if upper.startswith("PAYLOAD:"):
            value = raw[len("PAYLOAD:"):].strip()
            param = inferred_param
            if "=" in value and (param is None or value.split("=", 1)[0].strip() in (param, "")):
                maybe_param, _, maybe_value = value.partition("=")
                if maybe_param.strip():
                    param, value = maybe_param.strip(), maybe_value
            if not param:
                print("  [MANUAL PAYLOAD] Could not infer a parameter name. "
                      "Re-paste as PAYLOAD:<param>=<value>.")
                return None
            self._manual_payload_context = {"kind": "payload", "target": target, "param": param, "payload": value}
            return AIActionProposal(
                decision="propose_action", action_id="http_probe", target=target,
                parameters={"target": target, "focus_param": param, "payload": value},
                rationale=f"[manual-payload] human-supplied payload for parameter '{param}'.",
                confidence=0.5,
            )

        if upper.startswith("CURL:"):
            curl_line = raw[len("CURL:"):].strip()
            try:
                method, url, headers, body = parse_curl(curl_line)
            except Exception as exc:
                print(f"  [MANUAL PAYLOAD] Could not parse curl command: {exc}")
                return None
            self._manual_payload_context = {"kind": "curl", "method": method, "url": url,
                                             "headers": headers, "body": body}
            code = build_curl_replay_script(method, url, headers, body)
            return AIActionProposal(
                decision="propose_action", action_id="self_script", target=target,
                parameters={"target": target,
                            "hypothesis": f"[manual-payload curl replay] {hypothesis_text}"[:1000],
                            "code": code},
                rationale="[manual-payload] human-supplied curl command replayed through the gated sandbox.",
                confidence=0.5,
            )

        if upper.startswith("SCRIPT:"):
            code = raw[len("SCRIPT:"):].lstrip("\n")
            if not code.strip():
                print("  [MANUAL PAYLOAD] No script body found after SCRIPT:.")
                return None
            self._manual_payload_context = {"kind": "script"}
            return AIActionProposal(
                decision="propose_action", action_id="self_script", target=target,
                parameters={"target": target,
                            "hypothesis": f"[manual-payload script] {hypothesis_text}"[:1000],
                            "code": code},
                rationale="[manual-payload] human-authored probe.",
                confidence=0.5,
            )

        print("  [MANUAL PAYLOAD] Input must start with PAYLOAD:, CURL:, or SCRIPT: -- skipping this step.")
        return None

    def _emit_manual_payload_solving_curl(self, ctx: Dict[str, Any]) -> None:
        """
        Called right after a step that ran a manual-payload proposal and
        produced a PROVEN finding (checked by the caller via proven_finding_count
        before/after this step). Prints the exact confirming request as a
        copy-pasteable curl command -- straightforward for PAYLOAD: (we know
        the exact url/param/value) and CURL: (the human's own command, echoed
        back since it IS the confirming request); SCRIPT: has no single
        reconstructable request (arbitrary code), so no curl is fabricated for it.
        """
        from agents.common.manual_payload import build_solving_curl

        if ctx["kind"] == "payload":
            from core.tools.http_probe import _set_param
            confirming_url = _set_param(ctx["target"], ctx["param"], ctx["payload"])
            print(f"\n  [MANUAL PAYLOAD] CONFIRMED. Solving request:\n"
                  f"    {build_solving_curl('GET', confirming_url)}\n")
        elif ctx["kind"] == "curl":
            print(f"\n  [MANUAL PAYLOAD] CONFIRMED. Solving request (your own curl, replayed):\n"
                  f"    {build_solving_curl(ctx['method'], ctx['url'], ctx['headers'], ctx['body'])}\n")
        else:
            print("\n  [MANUAL PAYLOAD] CONFIRMED via a human-authored script -- "
                  "see the finding's evidence_summary for the exact proof captured "
                  "(no single request to reconstruct from arbitrary code).\n")

    def run_autonomous(
        self,
        registry,
        completion_criteria=None,
        evidence_store=None,
        max_steps: int = 50,
        manual_payload: bool = False,
        sweep: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute an autonomous, hypothesis-driven security assessment.

        Enforces:
        - AI proposes structured actions via ActionRegistry (never raw shell).
        - 10-gate safety pipeline executed before every action.
        - Observations, hypotheses, validations, and findings recorded in EvidenceStore.
        - Framework-verified completion (AI cannot declare completion prematurely).
        - Atomic state persistence on each step and on Ctrl+C.

        manual_payload: when True (--manual-payload on run_portswigger_lab.py),
        every time the planner would propose self_script (the model writing its
        own probe), the loop instead pauses and asks a HUMAN to supply the test
        -- see _handle_manual_payload(). Everything downstream (scope/approval
        gates, Observation recording, confirmation-rigor grading) is UNCHANGED;
        only the authorship of the probe's payload/script/curl differs.
        """
        from core.action_contract import ActionRegistry, AIActionProposal
        from core.investigation import EvidenceStore, Observation, Hypothesis, Validation, Finding
        from core.ai_planner import AIPlanner

        self._manual_payload_mode = manual_payload
        self._manual_payload_context = None
        self._sweep_mode = sweep

        print("\n" + "=" * 70)
        print(f"  LAUNCHING AUTONOMOUS MODE: {self.agent_name.upper()}")
        print(f"  Target:    {self.scope.target}")
        print(f"  Max Steps: {max_steps}")
        print(f"  Active Net Allowed: {getattr(self.scope, 'active_network_allowed', False)}")
        print(f"  State Changes Allowed: {getattr(self.scope, 'state_changes_allowed', False)}")
        print(f"  Manual Payload Mode: {'ON (human supplies self_script probes)' if manual_payload else 'OFF'}")
        print(f"  Sweep Mode: {'ON (enumerate -> characterize -> context-gated class sweep)' if sweep else 'OFF'}")
        print("=" * 70 + "\n")

        # Ensure session exists
        if not self.session_mgr.active_session_name:
            session_name = f"{self.agent_type}_autohunt_{int(time.time())}"
            self.session_mgr.create_session(
                name=session_name,
                scope=str(self.scope.in_scope),
                target=self.scope.target,
                target_type=self.agent_type,
                mode=self.mode,
            )
            # --resume fix: persist the full Scope (every authorization flag, not just
            # the target) so --resume can reconstruct it exactly instead of falling back
            # to a bare, fully-unauthorized Scope(target, in_scope).
            self.session_mgr.save_scope(self.scope)

        # Initialize or load EvidenceStore
        if evidence_store is None:
            loaded_store = self.session_mgr.load_investigation_store()
            evidence_store = loaded_store if loaded_store is not None else EvidenceStore()

        # M2: the planner gets its own ChatAssistant(role="planner"). When
        # core.config.PLANNER_MODEL_NAME is unset (the default), this resolves to the
        # exact same shared LLMService as self.chat_assistant -- zero extra cost, zero
        # behavior change. Setting PLANNER_MODEL_NAME lets decide_next() use a
        # different (e.g. larger, more instruction-following) model than chat/report
        # synthesis without forcing that model on the rest of the framework. See
        # core/config.py "PLANNER MODEL OVERRIDE (M2)" for the tradeoffs.
        planner_chat_assistant = ChatAssistant(role="planner")
        planner = AIPlanner(
            registry=registry,
            chat_assistant=planner_chat_assistant,
            scope=self.scope,
        )

        step_count = 0
        status = "running"
        # MED-1: a planner that repeatedly returns unparseable JSON (common on a weak
        # or under-sized local model) previously just burned every remaining step via
        # `continue`, with the loop only ever ending at max_steps -- indistinguishable
        # from a healthy-but-slow run in the returned status. Tracking *consecutive*
        # failures (reset on any successful decide_next() call) lets a genuinely broken
        # planner abort early with a clear status instead of silently exhausting the budget.
        consecutive_planner_failures = 0
        MAX_CONSECUTIVE_PLANNER_FAILURES = 5
        # MED-2: a planner that keeps proposing actions which execute "successfully"
        # at the gate level but never advance the investigation (same read-only probe
        # over and over, or a string of rejected/no-op evidence-verb calls) previously
        # ran all the way to max_steps indistinguishable from real progress. "Progress"
        # here is deliberately narrow and framework-defined, not AI-claimed: a new
        # Observation recorded, or a successful form_hypothesis/record_validation/
        # record_finding call. last_progress_step starts at 0 so a planner that never
        # produces a single successful result is caught by the same mechanism.
        last_progress_step = 0
        MAX_STEPS_WITHOUT_PROGRESS = 10
        # BUG FIX (found via live GPU smoke test, pentest agent, 2026-11-14):
        # once every registered action had exhausted its retry budget, the
        # planner correctly switched to propose_completion -- but a genuinely
        # unsatisfiable required phase (e.g. "exploitation" against a target
        # with no exploitable service, a legitimate and common real-world
        # pentest outcome) made verify_completion() reject it every time, and
        # the model re-proposed completion identically for 8 steps straight
        # until max_steps. That is a provably terminal state: no action can
        # advance the unmet phase (all are exhausted) and completion cannot be
        # accepted, so additional steps cannot change the outcome. This guard
        # ends the run promptly with a status and reason naming the actual
        # blocker instead of burning the budget and reporting a generic
        # max_steps_reached. It NEVER lets the agent falsely claim success --
        # completion is still only ever accepted by verify_completion() -- it
        # only stops the futile spin once completion has been rejected for the
        # same unmet requirement several times with no intervening progress.
        consecutive_completion_rejections = 0
        MAX_CONSECUTIVE_COMPLETION_REJECTIONS = 3
        last_completion_block_reason = None

        try:
            while step_count < max_steps:
                step_count += 1
                print(f"\n[*] --- Step {step_count}/{max_steps} ---")

                # 1. Compute retry budget status from execution history
                execution_history = self.session_mgr.state.get("autonomous_execution_history", [])
                retry_counts: Dict[str, int] = {}
                for entry in execution_history:
                    act_id = entry.get("action_id")
                    if act_id:
                        retry_counts[act_id] = retry_counts.get(act_id, 0) + 1

                # 2. Ask AI planner for next action or completion
                try:
                    proposal: AIActionProposal = planner.decide_next(
                        session_state=self.session_mgr.state,
                        evidence_store=evidence_store,
                        completion_criteria=completion_criteria,
                        execution_history=execution_history,
                        retry_budget_status=retry_counts,
                    )
                except Exception as exc:
                    consecutive_planner_failures += 1
                    print(
                        f"    [!] Planner error ({consecutive_planner_failures}/"
                        f"{MAX_CONSECUTIVE_PLANNER_FAILURES} consecutive): {exc}"
                    )
                    # Record planner failure and try next step
                    self.session_mgr.append_autonomous_history({
                        "step": step_count,
                        "action_id": "planner_decision",
                        "status": "failed",
                        "rejection_reason": str(exc),
                        "timestamp": datetime_iso(),
                    })
                    if consecutive_planner_failures >= MAX_CONSECUTIVE_PLANNER_FAILURES:
                        print(
                            f"    [!] {MAX_CONSECUTIVE_PLANNER_FAILURES} consecutive planner "
                            f"failures -- aborting early instead of burning the rest of "
                            f"max_steps on a planner that cannot produce a valid proposal."
                        )
                        status = "aborted_planner_failures"
                        break
                    continue

                # MED-1: any successful decision resets the consecutive-failure counter --
                # only a *run* of failures aborts the loop, not a lifetime count of them.
                consecutive_planner_failures = 0

                print(f"    AI Decision: {proposal.decision.upper()}")
                print(f"    Action:      {proposal.action_id}")
                print(f"    Rationale:   {proposal.rationale}")
                print(f"    Confidence:  {proposal.confidence}")

                # The repeated-completion guard counts only a *run* of
                # back-to-back rejected completions -- any other decision resets
                # it, so a run still making real moves is never cut short by it.
                if proposal.decision != "propose_completion":
                    consecutive_completion_rejections = 0

                # 2b. MANUAL PAYLOAD MODE: the planner still decides WHAT to test
                # (it already formed the hypothesis and proposed self_script to
                # test it) -- in this mode a human supplies the actual probe
                # instead of the model. The replacement proposal returned here
                # flows through the EXACT SAME completion/evidence-verb/10-gate
                # path below as any planner-authored proposal; only its origin
                # differs. If the human enters nothing, this step is recorded as
                # skipped and the loop re-asks the planner on the next step.
                if (
                    getattr(self, "_manual_payload_mode", False)
                    and proposal.decision == "propose_action"
                    and proposal.action_id == "self_script"
                ):
                    manual_proposal = self._handle_manual_payload(proposal, evidence_store)
                    if manual_proposal is None:
                        self.session_mgr.append_autonomous_history({
                            "step": step_count,
                            "action_id": "self_script",
                            "status": "skipped",
                            "rejection_reason": "manual-payload: operator entered nothing",
                            "timestamp": datetime_iso(),
                        })
                        continue
                    proposal = manual_proposal

                # 3. Handle Completion Proposal
                if proposal.decision == "propose_completion":
                    if completion_criteria is None:
                        print("    [+] No completion criteria registered. Accepting completion.")
                        status = "completed"
                        break

                    # H1 fix: every agent package ships its own completion.py::verify_completion
                    # with an identical (criteria, session_state, evidence_store) signature.
                    # Resolve it dynamically by agent_type instead of hardcoding CTF's — otherwise
                    # every other agent is checked against CTF's action_to_phase map and can
                    # never satisfy its own required_phases.
                    verify_completion = self._get_completion_verifier()
                    can_complete, reason = verify_completion(
                        criteria=completion_criteria,
                        session_state=self.session_mgr.state,
                        evidence_store=evidence_store,
                    )

                    if can_complete:
                        print(f"    [+] Framework VERIFIED completion: {reason}")
                        self.session_mgr.append_autonomous_history({
                            "step": step_count,
                            "decision": "propose_completion",
                            "status": "success",
                            "reason": reason,
                            "timestamp": datetime_iso(),
                        })
                        status = "completed"
                        break
                    else:
                        print(f"    [!] Framework REJECTED completion: {reason}")
                        self.session_mgr.append_autonomous_history({
                            "step": step_count,
                            "decision": "propose_completion",
                            "status": "rejected",
                            "rejection_reason": reason,
                            "timestamp": datetime_iso(),
                        })
                        consecutive_completion_rejections += 1
                        last_completion_block_reason = reason
                        if consecutive_completion_rejections >= MAX_CONSECUTIVE_COMPLETION_REJECTIONS:
                            print(
                                f"    [!] Completion proposed and rejected "
                                f"{consecutive_completion_rejections} times in a row for the "
                                f"same unmet requirement, and every registered action is "
                                f"exhausted -- no remaining step can change this. Ending the "
                                f"run with a clear terminal status instead of burning the "
                                f"rest of max_steps."
                            )
                            status = "terminated_incomplete"
                            break
                        continue

                # 3b. Handle Evidence-Chain Decision Verbs (full C1 extension).
                # These never enter the 10-gate command pipeline below -- they only
                # read/write the EvidenceStore, which independently verifies every
                # obs_id/hyp_id the AI cites before recording anything (see the three
                # _handle_* methods above). This is additive on top of, not a
                # replacement for, _promote_evidence_if_warranted's deterministic floor.
                if proposal.decision in ("form_hypothesis", "record_validation", "record_finding", "update_app_model"):
                    self._registry = registry  # let _handle_form_hypothesis infer vuln_class from validation_test
                    verb_handlers = {
                        "form_hypothesis": self._handle_form_hypothesis,
                        "record_validation": self._handle_record_validation,
                        "record_finding": self._handle_record_finding,
                        "update_app_model": self._handle_update_app_model,
                    }
                    verb_result = verb_handlers[proposal.decision](proposal, evidence_store)
                    verb_status = verb_result.get("status", "unknown")
                    if verb_status == "success":
                        last_progress_step = step_count
                    print(
                        f"    Evidence-Verb Status: {verb_status.upper()} — "
                        f"{verb_result.get('result_summary') or verb_result.get('rejection_reason')}"
                    )
                    self.session_mgr.append_autonomous_history({
                        "step": step_count,
                        "decision": proposal.decision,
                        "action_id": None,
                        "status": verb_status,
                        "rejection_reason": verb_result.get("rejection_reason"),
                        "result_summary": verb_result.get("result_summary"),
                        "timestamp": datetime_iso(),
                    })
                    self.session_mgr.save_investigation_store(evidence_store)
                    # Stall check in the evidence-verb path too: a run that only emits
                    # (and re-emits) rejected verb decisions makes no framework progress,
                    # and previously bypassed the end-of-loop stall detector via `continue`,
                    # burning the rest of max_steps (observed live 2026-11-17).
                    if step_count - last_progress_step >= MAX_STEPS_WITHOUT_PROGRESS:
                        print(
                            f"\n[!] No progress (new observation or recorded evidence) in "
                            f"the last {MAX_STEPS_WITHOUT_PROGRESS} steps -- stopping early."
                        )
                        status = "stalled_no_progress"
                        break
                    continue

                # 4. Handle Action Proposal through 10-gate safety pipeline
                # NOTE: Gate 6 (retry budget) computes prior_attempts from execution_history.
                # It MUST be passed as execution_history=, not session_state= — the executor
                # has no 'session_state' parameter, so that kwarg silently fell into **kwargs
                # and Gate 6 always saw an empty list (retry budget never tripped).
                prev_proven_count = evidence_store.proven_finding_count
                # Consume the manual-payload context now (one-shot, whatever this
                # step's outcome): otherwise a context left set after a step that
                # did NOT confirm anything would wrongly attach to a LATER, unrelated
                # step's proven finding.
                manual_ctx_for_this_step = getattr(self, "_manual_payload_context", None)
                self._manual_payload_context = None
                exec_result = self.executor.execute_action(
                    proposal=proposal,
                    registry=registry,
                    scope=self.scope,
                    execution_history=execution_history,
                    mode=self.mode,
                )

                exec_status = exec_result.get("status", "unknown")
                print(f"    Execution Status: {exec_status.upper()}")

                history_entry = {
                    "step": step_count,
                    "action_id": proposal.action_id,
                    "action_identity": exec_result.get("action_identity"),
                    "status": exec_status,
                    "rejection_reason": exec_result.get("rejection_reason"),
                    "rejection_category": exec_result.get("rejection_category"),
                    "results": exec_result.get("results", []),
                    "timestamp": datetime_iso(),
                }
                self.session_mgr.append_autonomous_history(history_entry)

                # 5. On successful execution: Record Observation in EvidenceStore
                if exec_status == "success":
                    new_observations: List[Any] = []
                    for res in exec_result.get("results", []):
                        argv = res.get("argv", [])
                        stdout = res.get("stdout", "")
                        stderr = res.get("stderr", "")
                        returncode = res.get("returncode", 0)

                        # C5 fix: populate parsed_facts via the agent's own
                        # fact_extractors.py (if any) so the planner prompt can show
                        # structured discoveries (open ports, subdomains, headers, ...)
                        # instead of only a truncated stdout snippet.
                        parsed_facts = self._extract_parsed_facts(
                            proposal.action_id or "unknown", stdout, stderr
                        )
                        # Blind-lab mode: scrub the answer-key markers out of what the
                        # planner will see (raw_stdout + parsed_facts), so targeting has
                        # to come from the real app surface, not the lab title/link.
                        if self._blind:
                            stdout = self._scrub_lab_markers(stdout)
                            stderr = self._scrub_lab_markers(stderr)
                            parsed_facts = self._scrub_lab_facts(parsed_facts)
                        # MED-7 fix: tag this Observation with the phase of the action
                        # that produced it (recon/enum/privesc/...), read from the
                        # registry's own ActionDefinition -- previously left unset on
                        # every Observation.
                        action_defn = registry.get(proposal.action_id) if proposal.action_id else None
                        obs = Observation(
                            action_id=proposal.action_id or "unknown",
                            argv=argv,
                            raw_stdout=stdout,
                            raw_stderr=stderr,
                            exit_code=returncode,
                            parsed_facts=parsed_facts,
                            phase=action_defn.phase if action_defn else None,
                        )
                        evidence_store.add_observation(obs)
                        new_observations.append(obs)

                    if new_observations:
                        last_progress_step = step_count

                    # C1 fix (minimal deterministic producer): the AI planner can only
                    # ever propose_action / propose_completion — there was no decision
                    # verb that created a Hypothesis, Validation, or Finding, so
                    # EvidenceStore.record_finding() was never reachable and every
                    # agent's minimum_findings>=1 completion requirement could never be
                    # satisfied (every run burned through max_steps). This promotes
                    # deterministically, gated on the action's own satisfies_phase flag
                    # (already authored per-action in every agents/*/actions.py but
                    # previously never read anywhere) — the framework decides what
                    # counts as evidence, not the AI.
                    action_defn = registry.get(proposal.action_id) if proposal.action_id else None
                    self._promote_evidence_if_warranted(
                        defn=action_defn,
                        proposal=proposal,
                        observations=new_observations,
                        evidence_store=evidence_store,
                    )
                    # ARTEX merge round 2: recon_scan's secrets/exposed_files/
                    # takeovers hits get surfaced as lead-only findings (never
                    # proven) -- a separate, dedicated promotion path from the
                    # one above so the lead-only guarantee is structural.
                    self._promote_lead_findings_from_recon(
                        observations=new_observations,
                        evidence_store=evidence_store,
                    )
                    if sweep:
                        self._update_sweep_coverage(new_observations, evidence_store)

                    # MANUAL PAYLOAD MODE: a human-supplied probe just ran and this
                    # step's promotion pass yielded a NEW proven finding -- emit the
                    # exact confirming request as a copy-pasteable curl command.
                    if (
                        manual_ctx_for_this_step is not None
                        and evidence_store.proven_finding_count > prev_proven_count
                    ):
                        self._emit_manual_payload_solving_curl(manual_ctx_for_this_step)

                # 6. Save investigation store to disk atomically
                self.session_mgr.save_investigation_store(evidence_store)

                # MED-2: stall detector. Checked at the end of every step (after this
                # step's own progress, if any, has already been recorded above) so a
                # run that never makes framework-verified progress ends with a clear,
                # distinct status instead of silently consuming the rest of max_steps.
                if step_count - last_progress_step >= MAX_STEPS_WITHOUT_PROGRESS:
                    print(
                        f"\n[!] No progress (new observation or recorded evidence) in "
                        f"the last {MAX_STEPS_WITHOUT_PROGRESS} steps -- stopping early."
                    )
                    status = "stalled_no_progress"
                    break

            if step_count >= max_steps and status not in (
                "completed", "aborted_planner_failures", "stalled_no_progress",
                "terminated_incomplete",
            ):
                print(f"\n[!] Reached max_steps limit ({max_steps}). Assessment stopped.")
                status = "max_steps_reached"

            # Synthesize final report
            print("\n" + "=" * 70)
            print("  AUTONOMOUS ASSESSMENT FINISHED. GENERATING AUDIT REPORT...")
            print("=" * 70)
            report_text = self.chat_assistant.synthesize_report(
                self.session_mgr.state, context={"evidence_store": evidence_store}
            )
            report_file = self.session_mgr.session_dir / "report.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"[+] Final Audit Report compiled to: {report_file}\n")

            return {
                "status": status,
                "steps_run": step_count,
                "findings_count": len(evidence_store.findings),
                "proven_findings": evidence_store.proven_finding_count,
                "leads": sum(1 for _f in evidence_store.findings.values() if getattr(_f, "strength", "proven") == "lead"),
                "observations_count": len(evidence_store.observations),
                "session_name": self.session_mgr.active_session_name,
                "report_path": str(report_file),
                # Populated only when the run ended via the repeated-completion
                # guard: the exact verify_completion() reason that could not be
                # satisfied (e.g. an unmet required phase). None otherwise.
                "completion_block_reason": last_completion_block_reason,
            }

        except KeyboardInterrupt:
            print("\n\n[!] Autonomous execution interrupted by operator (Ctrl+C).")
            print("[*] Persisting session state and evidence store to disk...")
            self.session_mgr.save_current_state()
            self.session_mgr.save_investigation_store(evidence_store)
            print(f"[+] State saved to session '{self.session_mgr.active_session_name}'.")
            return {
                "status": "interrupted",
                "steps_run": step_count,
                "session_name": self.session_mgr.active_session_name,
                "findings_count": len(evidence_store.findings),
                "proven_findings": evidence_store.proven_finding_count,
                "leads": sum(1 for _f in evidence_store.findings.values() if getattr(_f, "strength", "proven") == "lead"),
            }
