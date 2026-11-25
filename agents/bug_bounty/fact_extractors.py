"""
agents/bug_bounty/fact_extractors.py

C5: best-effort structured fact extraction from bug_bounty action output, so the
AI planner sees "subdomains: [...]" / "open_ports: [...]" instead of 120 raw
characters of tool stdout it has to re-derive from scratch every step.

extract_facts() never raises -- a parsing failure or unrecognized output format
yields {} for that observation, never an exception that would break the loop.
"""

import json
import re
from typing import Any, Dict

from core.fact_parsing import parse_http_headers, parse_line_list, parse_nmap_open_ports

_SEVERITY_ORDER = {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1, "unknown": 0}


def _extract_nuclei_facts(stdout: str, stderr: str) -> Dict[str, Any]:
    """
    Parse nuclei's JSONL output (one match per line) into a vuln_signal.

    Same convention as the sqlmap extractor: a verdict-based engine ALWAYS emits
    a signal so it never falls through to the generic exit-0 milestone floor.
      - one or more matches -> confirmed=True, severity = the HIGHEST matched
        template's severity (nuclei's severity vocabulary is identical to ours),
        summary names the match count + top finding.
      - no matches          -> confirmed=False (clean, recorded non-finding).
    The scan is already filtered to low+ severity at the command level, so a
    match here is a real issue, not tech-fingerprint noise.
    """
    matches = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        info = obj.get("info", {}) or {}
        sev = str(info.get("severity", "unknown")).lower()
        name = info.get("name") or obj.get("template-id") or obj.get("templateID") or "nuclei match"
        at = obj.get("matched-at") or obj.get("matched_at") or obj.get("host") or ""
        matches.append({"severity": sev, "name": name, "at": at})

    if not matches:
        return {
            "vuln_signal": {
                "confirmed": False,
                "vuln_class": "nuclei",
                "summary": "nuclei: no low-or-higher severity templates matched.",
            }
        }

    top = max(matches, key=lambda m: _SEVERITY_ORDER.get(m["severity"], 0))
    top_sev = top["severity"] if top["severity"] in ("info", "low", "medium", "high", "critical") else "medium"
    return {
        "vuln_signal": {
            "confirmed": True,
            "severity": top_sev,
            "vuln_class": "nuclei",
            "summary": (
                f"nuclei matched {len(matches)} issue(s); highest severity "
                f"{top['severity']}: {top['name']}" + (f" @ {top['at']}" if top["at"] else "") + "."
            ),
            "matches": matches[:10],
        }
    }


def _extract_sqlmap_facts(stdout: str, stderr: str) -> Dict[str, Any]:
    """
    Turn sqlmap's textual output into a structured verdict the evidence chain
    can reason pass/fail on. Emits a 'vuln_signal' dict -- the SAME convention
    every specialist detection engine (nuclei, dalfox, ...) will use -- that
    base_agent._promote_evidence_if_warranted reads as its oracle:

        {"confirmed": True,  "severity": "high", "vuln_class": "sqli",
         "summary": "...", "parameter": "id", "dbms": "MySQL",
         "techniques": ["boolean-based blind", "time-based blind"]}
      or
        {"confirmed": False, "vuln_class": "sqli", "summary": "..."}

    A confirmed verdict becomes a real high-severity Finding; an explicit
    not-confirmed verdict becomes a recorded observation but NO finding (a clean
    scan is a valid negative, not a false positive). If sqlmap's output is
    unrecognizable, no vuln_signal is emitted and the generic exit-0 milestone
    behavior applies -- never a fabricated verdict either way.
    """
    out = f"{stdout}\n{stderr}"
    low = out.lower()

    confirmed_markers = (
        "sqlmap identified the following injection point",
        "is vulnerable",
        "the following injection point",
    )
    negative_markers = (
        "all tested parameters do not appear to be injectable",
        "all tested parameters appear to be not injectable",
        "does not seem to be injectable",
        # No parameter was testable at all (bare URL, nothing crawlable). sqlmap
        # tested nothing -- a non-finding, NOT a milestone. Folding it in here
        # (rather than returning {}) is deliberate: sqli_deep_scan is a
        # verdict-based detection engine, so it must ALWAYS emit a vuln_signal.
        # Returning {} would let base_agent's generic exit-0 promotion manufacture
        # a bogus "info" finding from a scan that never tested a parameter
        # (observed live 2026-11-14, before --crawl=2 was added).
        "no parameter(s) found for testing",
    )

    is_confirmed = any(m in low for m in confirmed_markers)
    is_negative = any(m in low for m in negative_markers)

    if is_confirmed:
        param_m = re.search(r"Parameter:\s*([^\(\n]+)", out)
        dbms_m = re.search(r"back-end DBMS:\s*([^\n]+)", out)
        techniques = re.findall(r"^\s*Type:\s*(.+)$", out, re.MULTILINE)
        parameter = param_m.group(1).strip() if param_m else None
        dbms = dbms_m.group(1).strip() if dbms_m else None
        techs = [t.strip() for t in techniques] or None
        bits = []
        if parameter:
            bits.append(f"parameter '{parameter}'")
        if dbms:
            bits.append(f"DBMS {dbms}")
        if techs:
            bits.append("techniques: " + ", ".join(techs))
        detail = ("; ".join(bits)) if bits else "see sqlmap output"
        return {
            "vuln_signal": {
                "confirmed": True,
                "severity": "high",
                "vuln_class": "sqli",
                "parameter": parameter,
                "dbms": dbms,
                "techniques": techs,
                "summary": f"sqlmap CONFIRMED SQL injection ({detail}).",
            }
        }

    if is_negative:
        return {
            "vuln_signal": {
                "confirmed": False,
                "vuln_class": "sqli",
                "summary": "sqlmap tested the target and found no injectable parameter.",
            }
        }

    # Unrecognized output (connection error, WAF block, truncated, etc.):
    # inconclusive. Still emit a confirmed=False signal -- a verdict-based engine
    # never falls through to the generic exit-0 milestone promotion, so an
    # inconclusive scan records an observation (progress) but NEVER a finding.
    return {
        "vuln_signal": {
            "confirmed": False,
            "vuln_class": "sqli",
            "inconclusive": True,
            "summary": "sqlmap run was inconclusive (no clear injectable/not-injectable verdict in output).",
        }
    }


def _extract_dalfox_facts(stdout: str, stderr: str) -> Dict[str, Any]:
    """
    Parse dalfox JSON output into a vuln_signal. dalfox emits a JSON array (or
    JSON lines) of findings, each with a 'type' (V=verified, R=reflected,
    G=grep) and a 'severity'. Verdict-based engine -> always emits a signal:
      - any VERIFIED ('V') finding   -> confirmed=True XSS at that severity.
      - only reflected/grep hits     -> a LEAD (confirmed=False, lead=True): a
                                        reflection/sink surface, not proven XSS.
      - no findings                  -> clean non-finding.
    """
    findings = []
    text = (stdout or "").strip()
    # dalfox may print a JSON array or one JSON object per line.
    candidates = []
    if text.startswith("["):
        try:
            candidates = json.loads(text)
        except Exception:
            candidates = []
    if not candidates:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    candidates.append(json.loads(line))
                except Exception:
                    pass
    for obj in candidates or []:
        if not isinstance(obj, dict):
            continue
        ftype = str(obj.get("type", "")).upper()
        sev = str(obj.get("severity", "")).lower()
        param = obj.get("param") or ""
        findings.append({"type": ftype, "severity": sev, "param": param,
                         "data": (obj.get("data") or "")[:200]})

    if not findings:
        return {"vuln_signal": {"confirmed": False, "vuln_class": "xss",
                                "summary": "dalfox tested the target and found no XSS."}}

    verified = [f for f in findings if f["type"] == "V"]
    if verified:
        top = verified[0]
        sev = top["severity"] if top["severity"] in ("low", "medium", "high", "critical") else "high"
        return {"vuln_signal": {
            "confirmed": True, "severity": sev, "vuln_class": "xss",
            "parameter": top["param"],
            "summary": f"dalfox VERIFIED XSS ({len(verified)} verified finding(s); "
                       f"parameter {top['param']!r}).",
        }}
    # Reflected/grep only -> a lead, not a confirmed bug.
    return {"vuln_signal": {
        "confirmed": False, "lead": True, "severity": "low", "vuln_class": "xss",
        "summary": f"Lead: dalfox found {len(findings)} reflected/grep hit(s) but no VERIFIED XSS; "
                   f"a reflection/sink surface worth manual confirmation.",
    }}


def extract_facts(action_id: str, stdout: str, stderr: str) -> Dict[str, Any]:
    try:
        if action_id == "sqli_deep_scan":
            return _extract_sqlmap_facts(stdout, stderr)

        if action_id == "nuclei_scan":
            return _extract_nuclei_facts(stdout, stderr)

        if action_id == "dalfox_scan":
            return _extract_dalfox_facts(stdout, stderr)

        if action_id == "http_probe":
            from agents.common.http_primitive import extract_http_probe_facts
            return extract_http_probe_facts(stdout, stderr)

        if action_id == "browser_probe":
            from agents.common.browser_primitive import extract_browser_facts
            return extract_browser_facts(stdout, stderr)

        if action_id == "recon_scan":
            from agents.common.recon_primitive import extract_recon_scan_facts
            return extract_recon_scan_facts(stdout, stderr)

        if action_id == "bounty_platform_recon":
            from agents.common.bounty_platform_primitive import extract_bounty_platform_facts
            return extract_bounty_platform_facts(stdout, stderr)

        if action_id == "self_script":
            from agents.common.self_scripting import extract_self_script_facts
            return extract_self_script_facts(stdout, stderr)

        if action_id == "nuclei_template":
            from agents.common.nuclei_authoring import extract_nuclei_template_facts
            return extract_nuclei_template_facts(stdout, stderr)

        if action_id == "raw_http":
            from agents.common.raw_http_authoring import extract_raw_http_facts
            return extract_raw_http_facts(stdout, stderr)

        if action_id == "browser_exec":
            from agents.common.browser_exec_authoring import extract_browser_exec_facts
            return extract_browser_exec_facts(stdout, stderr)

        if action_id == "race_probe":
            from agents.common.race_authoring import extract_race_probe_facts
            return extract_race_probe_facts(stdout, stderr)

        if action_id == "subdomain_enum":
            subs = parse_line_list(stdout)
            return {"subdomains": subs} if subs else {}

        if action_id == "port_scan":
            ports = parse_nmap_open_ports(stdout)
            return {"open_ports": ports} if ports else {}

        if action_id == "dir_brute":
            # ffuf -silent prints one matching URL/line per hit; exact format
            # varies by ffuf version, so this is a best-effort line capture.
            hits = parse_line_list(stdout)
            return {"discovered_paths": hits} if hits else {}

        if action_id == "endpoint_discovery":
            # endpoint_discovery runs `curl -s -g -L -k --max-time 15 <url>` which
            # returns the HTML body (no headers). parse_http_headers() only understands
            # HTTP status/header lines and was returning {} for every endpoint_discovery
            # call, so links/forms/params were never extracted. Use parse_html_endpoints
            # so the planner actually sees what endpoints and forms the page exposes.
            from core.fact_parsing import parse_html_endpoints
            return parse_html_endpoints(stdout)

        if action_id in (
            "param_fuzz", "header_analysis",
            "idor_probe", "sqli_probe", "xss_probe", "vuln_validation",
        ):
            return parse_http_headers(stdout)
    except Exception:
        pass
    return {}
