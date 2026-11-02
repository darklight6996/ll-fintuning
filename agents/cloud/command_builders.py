"""
agents/cloud/command_builders.py

Pure Python command translators for CloudAgent actions.
Translates typed action parameter models into executable argument vectors.

SAFETY INVARIANTS:
  - Returns strictly List[List[str]]
  - Never shell strings
  - No shell=True
  - Cross-platform compatible
"""

import sys
from typing import List

from agents.cloud.actions import (
    IamAuditParams,
    ResourceEnumParams,
    MisconfigValidationParams,
    S3BucketAuditParams,
    AssumeRoleAuditParams,
    SecurityGroupAuditParams,
    CloudTrailAuditParams,
    MetadataServiceAuditParams,
    PrivescPathAuditParams,
    CloudReportParams,
)


def build_iam_audit(params: IamAuditParams) -> List[List[str]]:
    """Build command vector for IAM permission and policy inspection."""
    return [
        [sys.executable, "-c", f"print('Auditing {params.provider.upper()} IAM policies for account {params.target}')"]
    ]


def build_resource_enum(params: ResourceEnumParams) -> List[List[str]]:
    """Build command vector for cloud asset inventory enumeration."""
    return [
        [sys.executable, "-c", f"print('Enumerating {params.provider.upper()} resources ({params.service_filter}) for {params.target}')"]
    ]


def build_misconfig_validation(params: MisconfigValidationParams) -> List[List[str]]:
    """Build command vector for verifying specific cloud resource exposure."""
    return [
        [sys.executable, "-c", f"print('Validating misconfiguration ({params.check_type}) on resource {params.resource_id}')"]
    ]


def build_s3_bucket_audit(params: S3BucketAuditParams) -> List[List[str]]:
    """Build command vector for testing public access on Amazon S3 buckets."""
    url = f"https://{params.bucket_name}.s3.amazonaws.com"
    return [
        ["curl", "-s", "-I", "-k", "--max-time", "10", url]
    ]


def build_assume_role_audit(params: AssumeRoleAuditParams) -> List[List[str]]:
    """Build command vector for evaluating trust policies and assume-role privileges."""
    return [
        [sys.executable, "-c", f"print('Auditing assume-role trust policy for {params.role_arn}')"]
    ]


def build_security_group_audit(params: SecurityGroupAuditParams) -> List[List[str]]:
    """Build command vector for auditing wide-open security group ingress rules."""
    return [
        [sys.executable, "-c", f"print('Auditing security group rules ({params.group_id}) for {params.target}')"]
    ]


def build_cloudtrail_audit(params: CloudTrailAuditParams) -> List[List[str]]:
    """Build command vector for verifying CloudTrail audit logging."""
    return [
        [sys.executable, "-c", f"print('Auditing CloudTrail logging state for {params.target}')"]
    ]


def build_metadata_service_audit(params: MetadataServiceAuditParams) -> List[List[str]]:
    """Build command vector for querying metadata service (IMDSv1)."""
    url = f"{params.endpoint.rstrip('/')}/latest/meta-data/"
    return [
        ["curl", "-s", "-k", "--max-time", "5", url]
    ]


def build_privesc_path_audit(params: PrivescPathAuditParams) -> List[List[str]]:
    """Build command vector for tracing IAM administrative privilege escalation."""
    return [
        [sys.executable, "-c", f"print('Synthesizing IAM privilege escalation path for {params.principal}')"]
    ]


def build_cloud_report(params: CloudReportParams) -> List[List[str]]:
    """Build command vector for cloud security report generation."""
    return [
        [sys.executable, "-c", f"print('Generating cloud security posture report for {params.target}')"]
    ]

