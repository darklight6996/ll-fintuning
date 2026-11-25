"""
agents/common/surface_characterize.py

--sweep mode's CHARACTERIZE step (cheap, non-exploit baselines): turns the
SAME Observations the autonomous loop already records (browser_probe forms,
recon_scan routes/forms/tech) into SurfaceEntry tag dicts the class gate
(core/tools/class_gate.py) can act on.

Scope, stated honestly: this pass characterizes from NAMING/shape heuristics
only (param name looks url/path/numeric-shaped, content-type of the page
that hosts the form, a same-page resource literally named like XML/JSON).
It does NOT send a differential probe to check "does this actually reflect"
or "does it error on a quote" -- that is a real probe (cost + risk), not a
read of an existing Observation, and is left to the per-class ENGINE the
gate routes to (http_probe/xxe_probe/ssrf_probe/etc.) to confirm. This keeps
CHARACTERIZE itself free (no new network calls), exactly as the task asks
("cheap, non-exploit baselines"), while being explicit that it is a
NAME-BASED proxy for those signals, not the signals themselves.
"""

import re
from typing import Any, Dict, List

_URL_NAME_RE = re.compile(r"(url|uri|link|callback|webhook|fetch|redirect|src|host|endpoint|api)", re.I)
_PATH_NAME_RE = re.compile(r"(file|path|doc|template|page|include|dir|folder)", re.I)
_ID_NAME_RE = re.compile(r"(^id$|id$|Id$|guid|uuid)", re.I)
_XML_RESOURCE_RE = re.compile(r"xml", re.I)
_TEMPLATE_NAME_RE = re.compile(r"(template|view|layout)", re.I)


def _endpoint_key(method: str, action: str, param: str) -> str:
    return f"{(method or 'GET').upper()} {action}::{param}"


def characterize_observation(action_id: str, parsed_facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a list of surface-entry dicts (endpoint_key + tag fields), one
    per (endpoint, param) discovered in this single Observation's
    parsed_facts. Never raises -- malformed/missing facts just yield [].
    """
    entries: List[Dict[str, Any]] = []
    if not isinstance(parsed_facts, dict):
        return entries

    # browser_probe: {"forms": [{"action":..., "method":..., "inputs": [...]}], ...}
    forms = parsed_facts.get("forms")
    if isinstance(forms, list):
        # A resource literally named e.g. "xmlStockCheckPayload.js" is a strong
        # clue that a form on this page expects an XML body -- the exact clue
        # that was MISSED in the live false positive this task's own context
        # cites (work.md, 2026-11-25). Surface it as accepts_xml on every form
        # on this observation rather than silently dropping it.
        resources = parsed_facts.get("resources") or parsed_facts.get("scripts") or []
        page_hints_xml = any(_XML_RESOURCE_RE.search(str(r)) for r in resources) if isinstance(resources, list) else False

        for form in forms:
            if not isinstance(form, dict):
                continue
            action = str(form.get("action") or "")
            method = str(form.get("method") or "GET")
            inputs = form.get("inputs") or []
            for inp in inputs:
                name = inp if isinstance(inp, str) else (inp or {}).get("name", "")
                name = str(name or "")
                if not name or not action:
                    continue
                entries.append({
                    "endpoint_key": _endpoint_key(method, action, name),
                    "endpoint": action, "method": method.upper(), "param": name,
                    "value_url_shaped": bool(_URL_NAME_RE.search(name)),
                    "fetch_named_param": bool(_URL_NAME_RE.search(name)),
                    "value_path_shaped": bool(_PATH_NAME_RE.search(name)),
                    "file_named_param": bool(_PATH_NAME_RE.search(name)),
                    "numeric_or_guid_id": bool(_ID_NAME_RE.search(name)),
                    "template_reflection": bool(_TEMPLATE_NAME_RE.search(name)),
                    "accepts_xml": page_hints_xml,
                    "query_param": method.upper() == "GET",
                    "state_changing_limited_resource": method.upper() in ("POST", "PUT", "PATCH"),
                })

    # recon_scan: {"recon": {"forms": [{"action":..., "param":...}], ...}}
    recon = parsed_facts.get("recon")
    if isinstance(recon, dict):
        for form in (recon.get("forms") or []):
            if not isinstance(form, dict):
                continue
            action = str(form.get("action") or "")
            name = str(form.get("param") or "")
            if not name or not action:
                continue
            entries.append({
                "endpoint_key": _endpoint_key("GET", action, name),
                "endpoint": action, "method": "GET", "param": name,
                "value_url_shaped": bool(_URL_NAME_RE.search(name)),
                "fetch_named_param": bool(_URL_NAME_RE.search(name)),
                "value_path_shaped": bool(_PATH_NAME_RE.search(name)),
                "file_named_param": bool(_PATH_NAME_RE.search(name)),
                "numeric_or_guid_id": bool(_ID_NAME_RE.search(name)),
                "query_param": True,
            })

    return entries


# Which (gated) engine action, if any, actually confirms this class today --
# used to populate the coverage ledger's "engine" column and to log
# "no typed engine for <class> -- low confidence" per the task's own fallback
# instruction when a class has none.
CLASS_TO_ENGINE: Dict[str, str] = {
    "xss": "browser_exec",
    "xxe": "xxe_probe",
    "ssrf": "ssrf_probe",
    "lfi": "http_probe",
    "sqli": "sqli_deep_scan",
    "race": "race_probe",
    "smuggling": "raw_http",
    "idor": "idor_probe",
    # ssti has no typed engine yet -- falls back to self_script/manual-payload.
}
