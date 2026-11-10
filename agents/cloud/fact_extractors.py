"""
agents/cloud/fact_extractors.py

C5: best-effort structured fact extraction from cloud action output. Most cloud
actions (iam_audit, resource_enum, misconfig_validation, assume_role_audit,
security_group_audit, cloudtrail_audit, privesc_path_audit, cloud_report) are
currently placeholder print() stubs in command_builders.py (no cloud-provider
SDK/CLI call yet) -- those intentionally return {} rather than fabricating
facts from the stub's own echoed text. Only the two real HTTP probes parse.
"""

from typing import Any, Dict

from core.fact_parsing import parse_http_headers


def extract_facts(action_id: str, stdout: str, stderr: str) -> Dict[str, Any]:
    try:
        if action_id in ("s3_bucket_audit", "metadata_service_audit"):
            return parse_http_headers(stdout)
    except Exception:
        pass
    return {}
