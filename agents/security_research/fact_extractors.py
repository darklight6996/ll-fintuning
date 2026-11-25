"""
agents/security_research/fact_extractors.py

C5: best-effort structured fact extraction from security_research action output.
root_cause_analysis/harness_generation/benign_poc_dev/exploitability_audit/
variation_fuzzing/remediation_guide/research_writeup/cvss_calculator are currently
placeholder print() stubs in command_builders.py -- those intentionally return
{} rather than fabricating facts from the stub's own echoed text.
"""

import json
from typing import Any, Dict

from core.fact_parsing import parse_line_list


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
        if action_id == "cve_lookup":
            # cveawg.mitre.org returns JSON; best-effort extraction of the fields
            # most relevant to triage. Falls back to {} on any non-JSON response
            # (rate limiting, HTML error page, network failure already captured
            # in stderr) rather than raising.
            try:
                data = json.loads(stdout)
            except (json.JSONDecodeError, TypeError):
                return {}
            facts: Dict[str, Any] = {}
            cve_id = data.get("cveMetadata", {}).get("cveId")
            if cve_id:
                facts["cve_id"] = cve_id
            descriptions = (
                data.get("containers", {}).get("cna", {}).get("descriptions", [])
            )
            if descriptions:
                facts["cve_description"] = descriptions[0].get("value", "")[:300]
            return facts

        if action_id == "source_diff":
            lines = parse_line_list(stdout, max_items=30)
            return {"diff_lines": lines} if lines else {}
    except Exception:
        pass
    return {}
