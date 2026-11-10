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
        if action_id in ("static_analysis", "decompile_app"):
            lines = parse_line_list(stdout, max_items=30)
            return {"raw_lines": lines} if lines else {}

        if action_id in ("tls_pinning_check", "auth_token_test"):
            return parse_http_headers(stdout)
    except Exception:
        pass
    return {}
