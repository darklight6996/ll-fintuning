"""
agents/firewall_bypass/fact_extractors.py

C5: best-effort structured fact extraction from firewall_bypass action output.
Every action in this agent is an HTTP probe (curl against the target with some
mutation), so a single shared header/WAF-signature parser covers the whole
action set. See core/fact_parsing.py for the parsing logic itself.
"""

from typing import Any, Dict

from core.fact_parsing import parse_http_headers

_HTTP_PROBE_ACTIONS = (
    "waf_fingerprint",
    "rule_enum",
    "chunked_encoding",
    "unicode_normalization",
    "header_mutation",
    "content_type_confusion",
    "path_normalization",
    "parameter_pollution",
    "payload_mutation",
)


def extract_facts(action_id: str, stdout: str, stderr: str) -> Dict[str, Any]:
    try:
        if action_id in _HTTP_PROBE_ACTIONS:
            facts = parse_http_headers(stdout)
            # Status code changing between probes (e.g. 200 -> 403 -> 200) is the
            # single most decision-relevant signal for this agent: it tells the
            # planner whether a given mutation technique got through the WAF.
            if "status_code" in facts:
                facts["bypass_signal"] = (
                    "likely_blocked" if facts["status_code"] in (401, 403, 406, 429, 503)
                    else "likely_passed"
                )
            return facts
    except Exception:
        pass
    return {}
