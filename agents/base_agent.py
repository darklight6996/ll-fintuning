
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
                "appears", "could be", "seems", "status !=")

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
        """Drop solution-leaking portswigger.net links out of parsed_facts URL lists
        (blind-lab mode). Keeps the lab's own host -- that's the target."""
        if not isinstance(facts, dict):
            return facts
        out = {}
        for k, v in facts.items():
            if isinstance(v, list):
                out[k] = [x for x in v if not (isinstance(x, str) and "portswigger.net" in x.lower())]
            elif isinstance(v, str):
                out[k] = self._scrub_lab_markers(v)
            else:
                out[k] = v
        return out

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

    def _handle_record_finding(self, proposal, evidence_store) -> Dict[str, Any]:
        """
        AI decision verb (full C1 extension): attempt to promote a hypothesis into a
        Finding. EvidenceStore.record_finding() is the actual gate here -- it
        independently re-checks can_create_finding() (confirmed hypothesis + at least
        one supporting validation) and refuses with a specific reason if that is not
        satisfied, regardless of what the AI claims. This method only shapes the AI's
        parameters into that call and relays the framework's verdict; it adds no
        additional trust of its own.
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
            "result_summary": f"Finding {finding.finding_id} ({severity}) recorded: {title}",
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


    def run_autonomous(
        self,
        registry,
        completion_criteria=None,
        evidence_store=None,
        max_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Execute an autonomous, hypothesis-driven security assessment.

        Enforces:
        - AI proposes structured actions via ActionRegistry (never raw shell).
        - 10-gate safety pipeline executed before every action.
        - Observations, hypotheses, validations, and findings recorded in EvidenceStore.
        - Framework-verified completion (AI cannot declare completion prematurely).
        - Atomic state persistence on each step and on Ctrl+C.
        """
        from core.action_contract import ActionRegistry, AIActionProposal
        from core.investigation import EvidenceStore, Observation, Hypothesis, Validation, Finding
        from core.ai_planner import AIPlanner

        print("\n" + "=" * 70)
        print(f"  LAUNCHING AUTONOMOUS MODE: {self.agent_name.upper()}")
        print(f"  Target:    {self.scope.target}")
        print(f"  Max Steps: {max_steps}")
        print(f"  Active Net Allowed: {getattr(self.scope, 'active_network_allowed', False)}")
        print(f"  State Changes Allowed: {getattr(self.scope, 'state_changes_allowed', False)}")
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
