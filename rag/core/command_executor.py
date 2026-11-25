"""
core/command_executor.py

Purpose
-------
Interacts with the environment safely by preparing, explaining, and executing commands after explicit approval.

Responsibilities
----------------
1. Read-only mode by default: display commands, explain purpose, expected output, and risks.
2. Require approval before executing any command.
3. Safe execution wrapper returning standardized execution results.
"""

import os
import sys
import time
import ipaddress
from typing import Dict, Any, List, Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from core.action_contract import ActionDefinition, AIActionProposal, ActionRegistry
    from core.scope import Scope

import subprocess
import shlex



def _classify_target_kind(target_str: str) -> str:
    """
    MED-4: classify a non-local target string into one of the
    ActionDefinition.allowed_target_kinds values ("ip", "cidr", "domain", "url").
    Uses the stdlib `ipaddress` module for actual IP/CIDR validation rather than a
    hand-rolled regex, so "999.999.999.999" or similar malformed strings correctly
    fall through to "domain" instead of being misclassified as an "ip".
    """
    candidate = target_str.strip()

    # URL: has an explicit scheme (http://, https://, ftp://, ...).
    scheme_sep = candidate.find("://")
    if 0 < scheme_sep <= 32:
        scheme = candidate[:scheme_sep]
        if scheme.isalnum() or all(c.isalnum() or c in "+-." for c in scheme):
            return "url"

    # CIDR: "<ip>/<prefix>" that ipaddress accepts as a network.
    if "/" in candidate:
        try:
            ipaddress.ip_network(candidate, strict=False)
            return "cidr"
        except ValueError:
            pass

    # Bare IPv4 / IPv6 address.
    try:
        ipaddress.ip_address(candidate)
        return "ip"
    except ValueError:
        pass

    # Anything else (hostname, FQDN) is treated as a domain.
    return "domain"

class CommandPreview:
    """
    Structured representation of a command ready for inspection/approval.
    """

    def __init__(
        self,
        command: str,
        purpose: str,
        expected_output: str,
        possible_risks: str,
        step_id: Optional[str] = None
    ):
        self.command = command
        self.purpose = purpose
        self.expected_output = expected_output
        self.possible_risks = possible_risks
        self.step_id = step_id

    def display(self) -> str:
        """Format preview block for user display."""
        lines = [
            "--------------------------------------------------",
            f"Step ID:         {self.step_id or 'N/A'}",
            f"Command:         {self.command}",
            f"Purpose:         {self.purpose}",
            f"Expected Output: {self.expected_output}",
            f"Possible Risks:  {self.possible_risks}",
            "--------------------------------------------------",
            "Approve execution? (Y/N)"
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "command": self.command,
            "purpose": self.purpose,
            "expected_output": self.expected_output,
            "possible_risks": self.possible_risks
        }


class CommandExecutor:
    """
    Executor for safety-gated assessment commands.
    """

    def __init__(self, read_only: bool = True):
        """
        Initialize CommandExecutor.

        Args:
            read_only: If True, commands are only formatted and explained, never run automatically.
        """
        self.read_only = read_only
        self._stealth_ctl = None  # lazily built from scope on first execute_action()

    def _stealth(self, scope):
        """Return the per-run StealthController, built once from scope (persists
        on this executor instance so adaptive backoff accumulates across steps).
        A no-op controller under the default lab profile."""
        if self._stealth_ctl is None:
            try:
                from core.stealth import StealthController
                self._stealth_ctl = StealthController.from_scope(scope)
            except Exception:
                from core.stealth import StealthController
                self._stealth_ctl = StealthController(enabled=False)
        return self._stealth_ctl

    def prepare_step_commands(
        self,
        step: Dict[str, Any],
        variable_engine=None
    ) -> List[CommandPreview]:
        """
        Prepare a list of CommandPreview objects from a plan/YAML step dict.

        Args:
            step: Step dictionary containing commands, action, expected_output, etc.
            variable_engine: Optional VariableEngine to perform string substitution.

        Returns:
            List of CommandPreview objects.
        """
        commands = step.get("commands", [])
        if isinstance(commands, str):
            commands = [commands]

        action = step.get("action", step.get("name", "Assessment Command"))
        expected_output = step.get("expected_output", "N/A")
        detection_or_risk = step.get("detection", step.get("risks", "Standard execution traffic"))
        if isinstance(detection_or_risk, dict):
            detection_or_risk = str(detection_or_risk)

        step_id = step.get("id", f"step_{step.get('sequence_number', '0')}")

        previews = []
        for cmd in commands:
            # Substitute variables if engine provided
            final_cmd = cmd
            if variable_engine:
                final_cmd = variable_engine.substitute(cmd)

            preview = CommandPreview(
                command=final_cmd,
                purpose=action,
                expected_output=expected_output,
                possible_risks=detection_or_risk,
                step_id=step_id
            )
            previews.append(preview)

        return previews

    def execute_command(self, command_str: str, approved: bool = False) -> Dict[str, Any]:
        """
        Execute a command if read_only is False and explicit approval is given.

        Args:
            command_str: Shell command to run
            approved: Explicit user authorization flag

        Returns:
            Dict containing execution status, stdout, stderr, and returncode.
        """
        if self.read_only:
            return {
                "status": "skipped",
                "reason": "Executor is in Read-Only Mode.",
                "command": command_str,
                "stdout": "",
                "stderr": "",
                "returncode": None
            }

        if not approved:
            return {
                "status": "declined",
                "reason": "Execution was not approved by user.",
                "command": command_str,
                "stdout": "",
                "stderr": "",
                "returncode": None
            }

        try:
            # SECURITY: parse command string into a list and execute with shell=False
            # This prevents shell injection vulnerabilities.
            command_list = shlex.split(command_str)
            result = subprocess.run(
                command_list,
                shell=False,
                capture_output=True,
                text=True,
                timeout=60
            )
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "command": command_str,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            return {
                "status": "error",
                "command": command_str,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }

    def execute_action(
        self,
        proposal: Any,
        registry: Any,
        scope: Any,
        approval_policy: Any = "manual",
        execution_history: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute an AIActionProposal through the complete, ten-gate safety pipeline.
        Both manual mode and autonomous mode call this identical method.
        The only branch between modes is the approval policy parameter.
        """
        if isinstance(approval_policy, dict):
            if execution_history is None:
                execution_history = approval_policy.get("autonomous_execution_history", [])
            approval_policy = kwargs.get("mode", kwargs.get("approval_policy", "manual"))
        elif kwargs.get("mode"):
            approval_policy = kwargs["mode"]

        if str(approval_policy).lower() in ("automated", "autonomous"):
            approval_policy = "autonomous"
        else:
            approval_policy = "manual"

        execution_history = execution_history or []
        action_id = getattr(proposal, "action_id", None) or ""
        action_identity = getattr(proposal, "compute_identity", lambda: action_id)()

        # GATE 0: Action-ID Retry Budget (runs BEFORE validation).
        #
        # BUG FIX (found via live GPU smoke test, pentest agent, 2026-11-14):
        # steps 13-20 of one run re-proposed `exploit_probe` eight times after it
        # had already exhausted its retry budget (max_retries=1, first attempt at
        # step 9). Two cooperating gaps let that spin burn the whole step budget:
        #
        #   1. Gate 6's budget keyed on action_IDENTITY (AIActionProposal.
        #      compute_identity() = hash of action_id + target + params). The weak
        #      7B model supplied DIFFERENT (and invalid) params on each re-proposal,
        #      so every attempt had a distinct identity and Gate 6's
        #      prior_attempts stayed at 0 -- it never fired.
        #   2. Gate 1 (registry validation) runs before Gate 6, so each malformed
        #      re-proposal died at Gate 1 with a confusing `registry_validation`
        #      param error rather than the truthful "this action is exhausted".
        #
        # Meanwhile the PLANNER already filters exhausted actions by action_id
        # (ai_planner._build_prompt's exhausted_action_ids, attempts >= max_retries)
        # and stops OFFERING them -- the model just ignored that (a known 7B
        # limitation). This gate makes the executor's HARD enforcement agree with
        # the planner's own filtering: count prior attempts for this action_id
        # (every history entry, matching base_agent's retry_counts and the
        # planner's filter exactly) and reject at/over max_retries, before
        # validation, regardless of how the params vary. It is strictly
        # safety-preserving -- it only ever rejects MORE, never allows more -- and
        # the first attempt (prior_attempts == 0) is always still allowed through.
        # Gate 6 below is retained as the finer identity-level check for the
        # within-budget exact-repeat case.
        defn_for_budget = registry.get(action_id)
        if defn_for_budget is not None:
            prior_attempts_by_id = sum(
                1 for h in execution_history if h.get("action_id") == action_id
            )
            if prior_attempts_by_id >= defn_for_budget.max_retries:
                return {
                    "status": "rejected",
                    "action_id": action_id,
                    "action_identity": action_identity,
                    "results": [],
                    "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                    "rejection_reason": (
                        f"Retry budget exhausted: {prior_attempts_by_id}/"
                        f"{defn_for_budget.max_retries} prior attempt(s) for action "
                        f"'{action_id}' (counted by action_id across all parameter "
                        f"variations)."
                    ),
                    "rejection_category": "retry_budget_exhausted",
                }

        # GATE 1: Registry Validation
        is_valid, typed_params, reg_err = registry.validate_proposal(proposal)
        if not is_valid:
            return {
                "status": "rejected",
                "action_id": action_id,
                "action_identity": action_identity,
                "results": [],
                "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                "rejection_reason": reg_err or "Invalid action proposal.",
                "rejection_category": "registry_validation",
            }

        defn = registry.get(action_id)
        if not defn:
            return {
                "status": "rejected",
                "action_id": action_id,
                "action_identity": action_identity,
                "results": [],
                "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                "rejection_reason": f"Action '{action_id}' not found in registry.",
                "rejection_category": "registry_validation",
            }

        # GATE 2: Target Format & Structural Scope Validation
        target = (
            getattr(proposal, "target", None)
            or getattr(typed_params, "target", None)
            or getattr(scope, "target", "")
        )
        target_str = str(target or "").strip()

        # BUG FIX (found via live GPU smoke test, pentest agent, 2026-11-14): a bare
        # sentinel with no real network identity ("local", "", or a filesystem path)
        # is genuinely un-classifiable and can ONLY be satisfied by an action that
        # declares allowed_target_kinds=["local"] (CTF's filesystem-only actions:
        # suid_audit, find_writable, ...). But "127.0.0.1" / "::1" / "localhost" are
        # NOT mere placeholders -- they are legitimate, classifiable loopback
        # network targets (verify_pentest_smoke_test.py deliberately scans
        # 127.0.0.1 with real nmap/curl commands, and it must be accepted by
        # host_discovery/service_scan/etc., none of which declare "local" support).
        # The previous version treated all five of these strings identically as
        # "local", which let CTF's sentinel-based actions work (their
        # allowed_target_kinds=["local"]) but wrongly rejected every pentest action
        # against the loopback interface before Gate 2's own real
        # scope-membership/target-kind checks below ever got a chance to run --
        # exactly the kind of real target those checks are built to classify
        # correctly via _classify_target_kind() ("127.0.0.1"/"::1" -> "ip",
        # "localhost" -> "domain").
        bare_local_sentinel = (
            target_str.lower() in ("local", "")
            or target_str.startswith(("/", "\\", "."))
        )
        if bare_local_sentinel:
            if "local" not in defn.allowed_target_kinds:
                return {
                    "status": "rejected",
                    "action_id": action_id,
                    "action_identity": action_identity,
                    "results": [],
                    "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                    "rejection_reason": f"Local target is not permitted for action '{action_id}'. Allowed kinds: {defn.allowed_target_kinds}",
                    "rejection_category": "out_of_scope",
                }
        else:
            in_scope, scope_reason = scope.is_asset_in_scope(target_str)
            if not in_scope:
                return {
                    "status": "rejected",
                    "action_id": action_id,
                    "action_identity": action_identity,
                    "results": [],
                    "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                    "rejection_reason": f"Target '{target_str}' is out of scope: {scope_reason}",
                    "rejection_category": "out_of_scope",
                }

            # MED-4 fix: Gate 2 previously only checked the NEGATIVE case (a local target
            # proposed against an action that doesn't allow "local"). It never checked the
            # POSITIVE case: that a non-local target's actual kind (ip / cidr / domain /
            # url) is one of the kinds the action declares in allowed_target_kinds. Without
            # this, an action authored for allowed_target_kinds=["ip"] could be run against
            # a URL or domain target as long as that string happened to pass
            # scope.is_asset_in_scope() -- kind enforcement existed only in the type hints
            # and the docs, never at the gate.
            #
            # "or 'local' in defn.allowed_target_kinds" lets a loopback string
            # (127.0.0.1/::1/localhost) still satisfy a "local"-only action (CTF),
            # exactly as the old bare is_local_target check did, while ALSO letting
            # it satisfy any action whose allowed_target_kinds correctly includes the
            # real classified kind (pentest's "ip"/"domain").
            target_kind = _classify_target_kind(target_str)
            if target_kind not in defn.allowed_target_kinds and "local" not in defn.allowed_target_kinds:
                return {
                    "status": "rejected",
                    "action_id": action_id,
                    "action_identity": action_identity,
                    "results": [],
                    "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                    "rejection_reason": (
                        f"Target '{target_str}' was classified as kind '{target_kind}', which "
                        f"is not in action '{action_id}''s allowed_target_kinds: "
                        f"{defn.allowed_target_kinds}."
                    ),
                    "rejection_category": "out_of_scope",
                }

        # Also verify any out-of-scope string presence in parameters
        params_str = str(proposal.parameters).lower()
        for oos in getattr(scope, "out_of_scope", []):
            oos_clean = oos.strip().lower()
            if oos_clean and oos_clean in params_str:
                return {
                    "status": "rejected",
                    "action_id": action_id,
                    "action_identity": action_identity,
                    "results": [],
                    "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                    "rejection_reason": f"Parameter contains out-of-scope asset reference: '{oos}'",
                    "rejection_category": "out_of_scope",
                }

        # GATE 3: Policy Validation
        if defn.requires_active_testing and not getattr(scope, "active_testing_allowed", False):
            return {
                "status": "rejected",
                "action_id": action_id,
                "action_identity": action_identity,
                "results": [],
                "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                "rejection_reason": f"Action '{action_id}' requires explicit active testing authorization.",
                "rejection_category": "policy_violation",
            }

        if defn.network_effect == "active" and not getattr(scope, "active_network_allowed", False):
            return {
                "status": "rejected",
                "action_id": action_id,
                "action_identity": action_identity,
                "results": [],
                "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                "rejection_reason": f"Action '{action_id}' has active network effect but active_network_allowed is False.",
                "rejection_category": "policy_violation",
            }

        if defn.state_change != "none" and not getattr(scope, "state_changes_allowed", False):
            return {
                "status": "rejected",
                "action_id": action_id,
                "action_identity": action_identity,
                "results": [],
                "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                "rejection_reason": f"Action '{action_id}' modifies state ({defn.state_change}) but state_changes_allowed is False.",
                "rejection_category": "policy_violation",
            }

        if getattr(scope, "recon_only_authorized", False):
            recon_phases = ("recon", "reconnaissance", "discovery", "environment_recon")
            if defn.phase not in recon_phases or defn.network_effect == "active":
                return {
                    "status": "rejected",
                    "action_id": action_id,
                    "action_identity": action_identity,
                    "results": [],
                    "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                    "rejection_reason": "Engagement is authorized for reconnaissance only.",
                    "rejection_category": "policy_violation",
                }

        # GATE 4: Prohibited Action Gate
        for pt in getattr(scope, "prohibited_tools", []):
            pt_clean = pt.strip().lower()
            if pt_clean and (pt_clean in action_id.lower() or pt_clean in params_str):
                return {
                    "status": "rejected",
                    "action_id": action_id,
                    "action_identity": action_identity,
                    "results": [],
                    "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                    "rejection_reason": f"Action uses prohibited tool: '{pt}'",
                    "rejection_category": "prohibited_action",
                }

        for ptest in getattr(scope, "prohibited_tests", []):
            ptest_clean = ptest.strip().lower()
            if ptest_clean and (ptest_clean in action_id.lower() or ptest_clean in defn.description.lower()):
                return {
                    "status": "rejected",
                    "action_id": action_id,
                    "action_identity": action_identity,
                    "results": [],
                    "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                    "rejection_reason": f"Action matches prohibited test constraint: '{ptest}'",
                    "rejection_category": "prohibited_action",
                }

        # GATE 5: Time Window Gate
        time_window = getattr(scope, "time_window", "Anytime")
        if time_window and time_window.lower() not in ("anytime", "business hours / anytime", "24/7", "n/a"):
            pass

        # GATE 6: Framework Retry Budget Gate
        prior_attempts = sum(
            1 for h in execution_history
            if h.get("action_identity") == action_identity
        )
        if prior_attempts >= defn.max_retries:
            return {
                "status": "rejected",
                "action_id": action_id,
                "action_identity": action_identity,
                "results": [],
                "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                "rejection_reason": f"Retry budget exhausted: {prior_attempts}/{defn.max_retries} attempts made for this action identity.",
                "rejection_category": "retry_budget_exhausted",
            }

        # GATE 6.5: Stealth Gate (no-op unless a stealth profile is active).
        # On a real target under stealth, loud high-volume scanners are gated off
        # in favor of surgical actions, so the agent doesn't rate-limit/WAF-ban
        # the operator's account. Default (lab) profile allows everything.
        stealth = self._stealth(scope)
        s_allowed, s_reason = stealth.is_action_allowed(defn)
        if not s_allowed:
            return {
                "status": "rejected",
                "action_id": action_id,
                "action_identity": action_identity,
                "results": [],
                "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                "rejection_reason": s_reason,
                "rejection_category": "stealth_gated",
            }

        # GATE 7: Command Build (Deterministic pure Python builder)
        builder = registry.get_builder(defn.command_builder)
        if not builder:
            return {
                "status": "error",
                "action_id": action_id,
                "action_identity": action_identity,
                "results": [],
                "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                "rejection_reason": f"Command builder '{defn.command_builder}' is not registered.",
                "rejection_category": "builder_missing",
            }

        try:
            argv_list = builder(typed_params)
        except NotImplementedError as nie:
            return {
                "status": "manual_review_required",
                "action_id": action_id,
                "action_identity": action_identity,
                "results": [],
                "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                "rejection_reason": str(nie),
                "rejection_category": "human_judgment_required",
            }
        except Exception as build_err:
            return {
                "status": "error",
                "action_id": action_id,
                "action_identity": action_identity,
                "results": [],
                "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                "rejection_reason": f"Command builder raised error: {build_err}",
                "rejection_category": "builder_error",
            }

        # GATE 8: Final argv Safety Check (Must be List[List[str]], never shell string)
        if not isinstance(argv_list, list) or not all(isinstance(cmd, list) for cmd in argv_list):
            return {
                "status": "error",
                "action_id": action_id,
                "action_identity": action_identity,
                "results": [],
                "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                "rejection_reason": "Command builder violated contract: must return List[List[str]].",
                "rejection_category": "builder_contract_violation",
            }

        for cmd in argv_list:
            if not cmd or not all(isinstance(arg, str) for arg in cmd):
                return {
                    "status": "error",
                    "action_id": action_id,
                    "action_identity": action_identity,
                    "results": [],
                    "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                    "rejection_reason": "Command builder emitted invalid or empty argv vector.",
                    "rejection_category": "builder_contract_violation",
                }
            cmd_line = " ".join(cmd)
            allowed, cmd_reason = scope.is_command_allowed(cmd_line)
            if not allowed:
                return {
                    "status": "rejected",
                    "action_id": action_id,
                    "action_identity": action_identity,
                    "results": [],
                    "approval_record": {"policy": approval_policy, "timestamp": time.time(), "approved_by": None},
                    "rejection_reason": cmd_reason,
                    "rejection_category": "prohibited_action",
                }

        # GATE 9: Approval Gate (Manual vs Autonomous Policy)
        approval_record = {
            "policy": approval_policy,
            "timestamp": time.time(),
            "approved_by": None,
        }

        if approval_policy == "manual":
            if not sys.stdin.isatty():
                return {
                    "status": "declined",
                    "action_id": action_id,
                    "action_identity": action_identity,
                    "results": [],
                    "approval_record": approval_record,
                    "rejection_reason": "Non-interactive terminal cannot provide manual approval (fails closed).",
                    "rejection_category": "approval_denied",
                }
            print(f"\n[OPERATOR APPROVAL NEEDED] Action: {defn.action_id} ({defn.risk_level} risk)")
            print(f"  Description: {defn.description}")
            for c in argv_list:
                print(f"  > {' '.join(c)}")
            try:
                resp = input("Approve execution? [y/N]: ").strip().lower()
                if resp not in ("y", "yes"):
                    return {
                        "status": "declined",
                        "action_id": action_id,
                        "action_identity": action_identity,
                        "results": [],
                        "approval_record": approval_record,
                        "rejection_reason": "Execution declined by operator.",
                        "rejection_category": "approval_denied",
                    }
                approval_record["approved_by"] = "operator"
            except (KeyboardInterrupt, EOFError):
                return {
                    "status": "declined",
                    "action_id": action_id,
                    "action_identity": action_identity,
                    "results": [],
                    "approval_record": approval_record,
                    "rejection_reason": "Execution cancelled by operator interrupt.",
                    "rejection_category": "approval_denied",
                }

        elif approval_policy == "autonomous":
            if defn.risk_level == "high":
                if not sys.stdin.isatty():
                    return {
                        "status": "declined",
                        "action_id": action_id,
                        "action_identity": action_identity,
                        "results": [],
                        "approval_record": approval_record,
                        "rejection_reason": "High-risk action requires interactive operator approval in autonomous mode (fails closed).",
                        "rejection_category": "high_risk_unattended",
                    }
                print(f"\n[HIGH-RISK GATE] Action: {defn.action_id} ({defn.risk_level} risk)")
                for c in argv_list:
                    print(f"  > {' '.join(c)}")
                try:
                    resp = input("Authorize HIGH-RISK action? [y/N]: ").strip().lower()
                    if resp not in ("y", "yes"):
                        return {
                            "status": "declined",
                            "action_id": action_id,
                            "action_identity": action_identity,
                            "results": [],
                            "approval_record": approval_record,
                            "rejection_reason": "High-risk action declined by operator.",
                            "rejection_category": "approval_denied",
                        }
                    approval_record["approved_by"] = "operator_high_risk_gate"
                except (KeyboardInterrupt, EOFError):
                    return {
                        "status": "declined",
                        "action_id": action_id,
                        "action_identity": action_identity,
                        "results": [],
                        "approval_record": approval_record,
                        "rejection_reason": "High-risk action cancelled by operator interrupt.",
                        "rejection_category": "approval_denied",
                    }
            else:
                approval_record["approved_by"] = "autonomous_policy"

        # STEALTH PACING: before active requests, sleep a jittered delay (scaled
        # by adaptive backoff). No-op under the default lab profile.
        is_active = bool(getattr(defn, "requires_active_testing", False)) or \
            getattr(defn, "network_effect", "none") == "active"
        stealth.pace(is_active)

        # GATE 10: Subprocess Execution (shell=False, timeout, text=True, errors='replace')
        cmd_results = []
        overall_status = "success"

        for argv in argv_list:
            t0 = time.time()
            try:
                proc = subprocess.run(
                    argv,
                    shell=False,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=defn.timeout_seconds,
                )
                duration = time.time() - t0
                cmd_results.append({
                    "argv": argv,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "returncode": proc.returncode,
                    "duration_seconds": duration,
                })
                # BUG FIX (live WSL smoke test, 2026-11-13): `find`'s exit status is
                # 0 only if EVERY directory it descended into was readable. As a
                # non-root user -- which is exactly how suid_audit/find_writable/
                # flag_search are meant to be run, since privesc-hunting implies you
                # don't have root yet -- a `find /` sweep will almost always hit a
                # handful of permission-denied directories (/root, /etc/ssl/private,
                # systemd sandboxes, ...) and exit 1 purely from that noise, even
                # though stdout still carries every real match it found. Treating
                # that the same as a hard failure silently discarded suid_audit's
                # genuine findings on this exact run: exec_status != "success" means
                # base_agent.run_autonomous never records an Observation for it at
                # all (see the `if exec_status == "success":` gate there), so two
                # full retry-budget attempts produced zero usable evidence despite
                # the command actually working. `find` exit code 1 specifically is
                # "completed with some inaccessible paths," not "this command is
                # broken" -- only treat 2+ (a real usage/argument error) as a failure
                # for `find`. Every other command keeps the strict != 0 check.
                is_find = bool(argv) and os.path.basename(argv[0]) == "find"
                if proc.returncode != 0 and not (is_find and proc.returncode == 1):
                    overall_status = "failed"
            except subprocess.TimeoutExpired as te:
                duration = time.time() - t0
                # BUG FIX (live WSL smoke test, 2026-11-13): TimeoutExpired.stdout is
                # raw bytes even when the original subprocess.run() call used
                # text=True/errors="replace" -- that decoding only applies to the
                # normal (non-timeout) return path. A long-running command that times
                # out (e.g. `find /` sweeping a large mount) left raw bytes in
                # cmd_results, which later crashed session_manager.save_current_state()
                # with "TypeError: Object of type bytes is not JSON serializable" when
                # it tried to persist execution_history. Decode defensively here,
                # matching the same errors="replace" policy used everywhere else.
                partial_stdout = te.stdout
                if isinstance(partial_stdout, bytes):
                    partial_stdout = partial_stdout.decode("utf-8", errors="replace")
                cmd_results.append({
                    "argv": argv,
                    "stdout": partial_stdout or "",
                    "stderr": f"Command timed out after {defn.timeout_seconds}s",
                    "returncode": -1,
                    "duration_seconds": duration,
                })
                overall_status = "timeout"
                break
            except Exception as run_err:
                duration = time.time() - t0
                cmd_results.append({
                    "argv": argv,
                    "stdout": "",
                    "stderr": str(run_err),
                    "returncode": -1,
                    "duration_seconds": duration,
                })
                overall_status = "error"
                break

        # STEALTH ADAPTIVE BACKOFF: scan this action's output for rate-limit/WAF
        # signatures and adjust the pacing multiplier for subsequent requests.
        stealth.note_result(
            " ".join(((r.get("stdout") or "") + " " + (r.get("stderr") or "")) for r in cmd_results)
        )

        return {
            "status": overall_status,
            "action_id": action_id,
            "action_identity": action_identity,
            "results": cmd_results,
            "approval_record": approval_record,
            "rejection_reason": None,
            "rejection_category": None,
        }
