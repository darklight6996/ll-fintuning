"""
agents/bug_bounty/single_host_mode.py

Single-host variant of the BugBountyAgent's action registry and completion
criteria. Used whenever Scope.single_host_only=True -- i.e. the engagement is
authorized against exactly one pinned host/URL instance, not a whole domain
with its own subdomains. This is NOT specific to PortSwigger: any single-URL
target (a PortSwigger Web Security Academy lab, a single staging endpoint, one
client-provided test URL/IP) is the same situation.

Why this exists (not just always using build_bug_bounty_registry() /
BUG_BOUNTY_COMPLETION):

- SUBDOMAIN_ENUM (subfinder) and PORT_SCAN (nmap) both target the domain/host
  as a whole, not the one URL instance the operator actually authorized.
  Running them would mean either scanning infrastructure outside what was
  authorized (e.g. the shared parent domain behind a single lab instance), or
  wasting steps on recon that cannot possibly produce anything for a target
  that has no further hosts under it.
- The decision to leave these two actions out is made HERE, deterministically,
  from an explicit Scope flag the operator sets -- never inferred by the AI
  planner at runtime, and never guessed from the shape of the target string.
  Same posture as the framework's other scope/safety gates: the AI proposes
  actions from whatever registry it's given; the framework decides which
  registry that is.
- Every other action (endpoint_discovery, dir_brute, param_fuzz,
  header_analysis, idor_probe, sqli_probe, xss_probe, vuln_validation) targets
  a URL/path on the given target and stays correctly scoped either way.

This module only subtracts actions/phases -- it does not add or redefine any
command_builder, so every action it keeps behaves identically to the main
BugBountyAgent. Nothing here changes agents/bug_bounty/actions.py,
completion.py, or BugBountyAgent's default (multi-host) behavior; a Scope
without single_host_only=True gets the exact same registry/criteria as before
this module existed.
"""

from typing import Dict, List

from core.action_contract import ActionRegistry
from agents.bug_bounty.actions import (
    ENDPOINT_DISCOVERY,
    DIR_BRUTE,
    PARAM_FUZZ,
    HEADER_ANALYSIS,
    IDOR_PROBE,
    SQLI_PROBE,
    XSS_PROBE,
    VULN_VALIDATION,
    SQLI_DEEP_SCAN,
    NUCLEI_SCAN,
    DALFOX_SCAN,
)
from agents.bug_bounty.completion import BugBountyCompletionCriteria

# Same action_id -> phase mapping as completion.py, minus subdomain_enum/
# port_scan (neither action is registered in single-host mode, so neither
# phase can ever be completed -- they are dropped from required_phases below
# too, or verify_completion() would reject every single-host run as
# unsatisfiable no matter how much real work the agent did).
SINGLE_HOST_ACTION_TO_PHASE: Dict[str, str] = {
    "endpoint_discovery": "endpoint_discovery",
    "browser_probe": "endpoint_discovery",
    "dir_brute": "endpoint_discovery",
    "param_fuzz": "param_fuzzing",
    "header_analysis": "recon",
    "recon_scan": "recon",
    "idor_probe": "vuln_validation",
    "sqli_probe": "vuln_validation",
    "sqli_deep_scan": "vuln_validation",
    "nuclei_scan": "vuln_validation",
    "dalfox_scan": "vuln_validation",
    "http_probe": "vuln_validation",
    "self_script": "vuln_validation",
    "nuclei_template": "vuln_validation",
    "raw_http": "vuln_validation",
    "browser_exec": "vuln_validation",
    "race_probe": "vuln_validation",
    "xxe_probe": "vuln_validation",
    "ssrf_probe": "vuln_validation",
    "xss_probe": "vuln_validation",
    "vuln_validation": "vuln_validation",
}

_SINGLE_HOST_PHASE_TO_ACTIONS: Dict[str, List[str]] = {}
for _action_id, _phase in SINGLE_HOST_ACTION_TO_PHASE.items():
    _SINGLE_HOST_PHASE_TO_ACTIONS.setdefault(_phase, []).append(_action_id)


def build_single_host_bug_bounty_registry(allow_scanners: bool = True) -> ActionRegistry:
    """
    Same as build_bug_bounty_registry() but WITHOUT subdomain_enum/port_scan --
    the two whole-domain actions that are scope-inappropriate for a single
    pinned host/URL target.
    """
    # Each register() MUST pass the builder callable as its second argument --
    # a bare register(DEFN) leaves the registry's builder map empty and every
    # action is rejected at Gate 7 ("Command builder ... is not registered").
    # This file was written before that framework-wide bug was fixed in
    # agents/*/actions.py (2026-11-14) and the programmatic sweep that fixed the
    # main registries did not touch this module, so it is wired here explicitly.
    # Builders are imported from bug_bounty's own command_builders.py (the same
    # ones build_bug_bounty_registry uses); guarded by
    # tests/test_all_registries_have_builders.py.
    from agents.bug_bounty.command_builders import (
        build_endpoint_discovery,
        build_dir_brute,
        build_param_fuzz,
        build_header_analysis,
        build_idor_probe,
        build_sqli_probe,
        build_sqli_deep_scan,
        build_nuclei_scan,
        build_dalfox_scan,
        build_xss_probe,
        build_vuln_validation,
    )

    registry = ActionRegistry()
    registry.register(ENDPOINT_DISCOVERY, build_endpoint_discovery)
    registry.register(PARAM_FUZZ, build_param_fuzz)
    registry.register(HEADER_ANALYSIS, build_header_analysis)
    registry.register(IDOR_PROBE, build_idor_probe)
    registry.register(SQLI_PROBE, build_sqli_probe)
    registry.register(XSS_PROBE, build_xss_probe)
    registry.register(VULN_VALIDATION, build_vuln_validation)
    # LOUD AUTOMATED SCANNERS (wordlist dir-brute + sqlmap/nuclei/dalfox). Many
    # bug bounty programs prohibit automated scanning, so on a real target these
    # are registered ONLY when the operator asserts the program permits them
    # (Scope.allow_scanners -> run_target.py --allow-scanners). allow_scanners
    # defaults True, so labs and existing callers get every action as before.
    if allow_scanners:
        registry.register(DIR_BRUTE, build_dir_brute)
        registry.register(SQLI_DEEP_SCAN, build_sqli_deep_scan)
        registry.register(NUCLEI_SCAN, build_nuclei_scan)
        registry.register(DALFOX_SCAN, build_dalfox_scan)
    # Framework-level shared primitives (cross-agent).
    from agents.common.http_primitive import HTTP_PROBE, build_http_probe
    registry.register(HTTP_PROBE, build_http_probe)
    from agents.common.browser_primitive import BROWSER_PROBE, build_browser_probe
    registry.register(BROWSER_PROBE, build_browser_probe)
    # Breadth-first recon sweep, CLAMPED to the single-host mode set (no
    # ports/subdomains -- those are whole-domain probes outside the one
    # pinned URL instance the operator authorized; same reasoning as the
    # SUBDOMAIN_ENUM/PORT_SCAN exclusion above).
    from agents.common.recon_primitive import RECON_SCAN, build_recon_scan_single_host
    registry.register(RECON_SCAN, build_recon_scan_single_host)
    # Self-scripting is OPT-IN (runs model-written code): registered only when
    # the operator sets PENTEST_ENABLE_SELF_SCRIPTING=1.
    from agents.common.self_scripting import SELF_SCRIPT, build_self_script, self_scripting_enabled
    if self_scripting_enabled():
        registry.register(SELF_SCRIPT, build_self_script)
        # Shape C: raw-socket HTTP primitive (request smuggling / cache poisoning /
        # Host-header). Model-authored like self_script, gated by the same switch
        # (NOT allow_scanners: a single-connection timing/marker probe, quieter than
        # the loud scanners).
        from agents.common.raw_http_authoring import RAW_HTTP, build_raw_http
        registry.register(RAW_HTTP, build_raw_http)
        # Shape D: browser execution probe (DOM/reflected/stored XSS via real JS run).
        from agents.common.browser_exec_authoring import BROWSER_EXEC, build_browser_exec
        registry.register(BROWSER_EXEC, build_browser_exec)
        # Shape B reliable engine: race / limit-overrun (sequential control vs concurrent, robust CSRF).
        from agents.common.race_authoring import RACE_PROBE, build_race_probe
        registry.register(RACE_PROBE, build_race_probe)
        # Class-sweep deterministic engines (2026-11-25): control-vs-attack, OBJECTIVE
        # verdict oracles for the two classes previously had no typed engine at all --
        # XXE and SSRF were only reachable via self_script's own hand-rolled logic,
        # which is exactly what produced the live false-positive this task's own
        # instructions flagged (self_script claimed "response changed" on two
        # different XML-parser errors, neither ever reaching real app logic).
        from agents.common.xxe_authoring import XXE_PROBE, build_xxe_probe
        registry.register(XXE_PROBE, build_xxe_probe)
        from agents.common.ssrf_authoring import SSRF_PROBE, build_ssrf_probe
        registry.register(SSRF_PROBE, build_ssrf_probe)
        # Authoring + running a nuclei template is scanner-class; gate it with the
        # same allow_scanners switch so a no-automation program never gets it.
        if allow_scanners:
            from agents.common.nuclei_authoring import NUCLEI_TEMPLATE, build_nuclei_template
            registry.register(NUCLEI_TEMPLATE, build_nuclei_template)
    # bounty_platform_recon is OPT-IN (calls a third-party platform with the
    # operator's own credentials), independent of allow_scanners/self-scripting
    # -- it's engagement setup (program scope/policy), not a target probe, so
    # single-host vs. domain-wide makes no difference to it. Registered only
    # when the operator sets PENTEST_ENABLE_BOUNTY_PLATFORM=1.
    from agents.common.bounty_platform_primitive import (
        BOUNTY_PLATFORM_RECON, build_bounty_platform_recon, bounty_platform_enabled,
    )
    if bounty_platform_enabled():
        registry.register(BOUNTY_PLATFORM_RECON, build_bounty_platform_recon)
    return registry


# "recon" is kept as a required phase -- header_analysis still satisfies it --
# but "subdomain_enum" is dropped entirely (unregistered action, unsatisfiable
# phase).
SINGLE_HOST_BUG_BOUNTY_COMPLETION = BugBountyCompletionCriteria(
    required_phases=["recon", "endpoint_discovery"],
    evidence_required_for_phases=["recon", "endpoint_discovery"],
    minimum_findings=1,
    allow_unresolved_errors=False,
    phase_actions=_SINGLE_HOST_PHASE_TO_ACTIONS,
)
