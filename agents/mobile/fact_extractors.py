"""
agents/mobile/fact_extractors.py

C5: best-effort structured fact extraction from mobile action output. Several
mobile actions (perm_audit, hardcoded_secrets, insecure_storage, dynamic_instrumentation,
webview_audit, mobile_report) are currently placeholder print() stubs in
command_builders.py with no real tool output to parse yet -- those intentionally
return {} here rather than fabricating facts from the stub's own echoed text.
"""

from typing import Any, Dict

from core.fact_parsing import parse_http_headers, parse_line_list


def extract_facts(action_id: str, stdout: str, stderr: str) -> Dict[str, Any]:
    try:
        if action_id == "http_probe":
            from agents.common.http_primitive import extract_http_probe_facts
            return extract_http_probe_facts(stdout, stderr)
        if action_id == "browser_probe":
            from agents.common.browser_primitive import extract_browser_facts
            return extract_browser_facts(stdout, stderr)
        if action_id == "self_script":
            from agents.common.self_scripting import extract_self_script_facts
            return extract_self_script_facts(stdout, stderr)
        if action_id in ("static_analysis", "decompile_app"):
            lines = parse_line_list(stdout, max_items=30)
            return {"raw_lines": lines} if lines else {}

        if action_id in ("tls_pinning_check", "auth_token_test"):
            return parse_http_headers(stdout)
    except Exception:
        pass
    return {}
