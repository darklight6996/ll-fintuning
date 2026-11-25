"""
agents/ctf/fact_extractors.py

C5: best-effort structured fact extraction from CTF action output, so the AI
planner sees "suid_binaries: [...]" / "user: root" instead of 120 raw
characters of find/id/ps stdout it has to re-derive from scratch every step --
in particular, it needs to carry a discovered SUID binary PATH forward to
parameterize strings_analysis(binary_path=...) or path_hijack_validation(...).
"""

from typing import Any, Dict

from core.fact_parsing import parse_find_paths, parse_line_list, parse_unix_identity


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
        if action_id == "environment_recon":
            return parse_unix_identity(stdout)

        if action_id == "suid_audit":
            paths = parse_find_paths(stdout)
            return {"suid_binaries": paths} if paths else {}

        if action_id == "find_writable":
            paths = parse_find_paths(stdout)
            return {"writable_paths": paths} if paths else {}

        if action_id == "flag_search":
            paths = parse_find_paths(stdout)
            return {"candidate_flag_files": paths} if paths else {}

        if action_id == "read_file":
            # Often IS the flag -- keep the full (small) content rather than truncating.
            content = (stdout or "").strip()
            return {"file_contents": content[:500]} if content else {}

        if action_id in ("process_list", "cron_job_audit", "kernel_exploit_check"):
            lines = parse_line_list(stdout)
            return {"raw_lines": lines} if lines else {}

        if action_id == "strings_analysis":
            lines = parse_line_list(stdout, max_items=30)
            return {"extracted_strings": lines} if lines else {}
    except Exception:
        pass
    return {}
