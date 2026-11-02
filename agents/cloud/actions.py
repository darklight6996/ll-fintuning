"""
agents/cloud/actions.py

Action definitions and registry factory for CloudAgent.
Covers AWS, Azure, and GCP cloud environments: IAM permissions, asset inventory,
S3 bucket exposures, security groups, metadata service (IMDS) auditing, and posture verification.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition, ActionRegistry


# ---------------------------------------------------------------------------
# Typed Action Parameter Models (Pydantic v2, extra="forbid")
# ---------------------------------------------------------------------------

class IamAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target cloud account ID, project ID, or tenant")
    provider: str = Field("aws", description="Cloud service provider (aws, azure, gcp)")


class ResourceEnumParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target cloud account ID or project ID")
    provider: str = Field("aws", description="Cloud service provider (aws, azure, gcp)")
    service_filter: str = Field("s3,iam,ec2,rds,lambda", description="Comma-separated service list")


class MisconfigValidationParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target cloud environment identifier")
    resource_id: str = Field(..., description="ARN, resource ID, or URI to audit")
    check_type: str = Field("public_access", description="Security check type (public_access, unencrypted, wildcards)")


class S3BucketAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target AWS account ID or region")
    bucket_name: str = Field(..., description="S3 bucket name to test for public access and ACLs")


class AssumeRoleAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target cloud environment identifier")
    role_arn: str = Field(..., description="Role ARN to evaluate for trust policy and escalation")


class SecurityGroupAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target cloud environment identifier")
    group_id: str = Field("all", description="Security group ID or 'all' for wide ingress audit")


class CloudTrailAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target cloud environment identifier")


class MetadataServiceAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target instance IP or base URL")
    endpoint: str = Field("http://169.254.169.254", description="Metadata service endpoint to test for IMDSv1")


class PrivescPathAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target cloud environment identifier")
    principal: str = Field("current_identity", description="IAM principal evaluated for privilege escalation")


class CloudReportParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target cloud scope evaluated")
    output_format: str = Field("markdown", description="Report output format (markdown, json)")


# ---------------------------------------------------------------------------
# Action Definitions
# ---------------------------------------------------------------------------

IAM_AUDIT = ActionDefinition(
    action_id="iam_audit",
    phase="iam_audit",
    description="Audit IAM roles, policies, inline attachments, and wildcard permissions (*:*)",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=IamAuditParams,
    allowed_target_kinds=["domain", "ip", "local"],
    command_builder="build_iam_audit",
    timeout_seconds=45,
    max_retries=2,
    satisfies_phase=True,
)

RESOURCE_ENUM = ActionDefinition(
    action_id="resource_enum",
    phase="resource_enum",
    description="Enumerate cloud assets, storage buckets, compute instances, and database endpoints",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=ResourceEnumParams,
    allowed_target_kinds=["domain", "ip", "local"],
    command_builder="build_resource_enum",
    timeout_seconds=60,
    max_retries=2,
    satisfies_phase=True,
)

MISCONFIG_VALIDATION = ActionDefinition(
    action_id="misconfig_validation",
    phase="misconfig_validation",
    description="Execute safe, targeted probe to confirm public exposure or policy misconfiguration",
    requires_active_testing=True,
    network_effect="read_only",
    state_change="none",
    risk_level="medium",
    params_model=MisconfigValidationParams,
    allowed_target_kinds=["domain", "ip", "url", "local"],
    command_builder="build_misconfig_validation",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

S3_BUCKET_AUDIT = ActionDefinition(
    action_id="s3_bucket_audit",
    phase="misconfig_validation",
    description="Inspect Amazon S3 bucket policies, ACLs, and test unauthenticated public list/read",
    requires_active_testing=True,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=S3BucketAuditParams,
    allowed_target_kinds=["domain", "url", "local"],
    command_builder="build_s3_bucket_audit",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

ASSUME_ROLE_AUDIT = ActionDefinition(
    action_id="assume_role_audit",
    phase="iam_audit",
    description="Test IAM role assume-role trust relationships and potential cross-account escalation",
    requires_active_testing=True,
    network_effect="read_only",
    state_change="none",
    risk_level="medium",
    params_model=AssumeRoleAuditParams,
    allowed_target_kinds=["domain", "local"],
    command_builder="build_assume_role_audit",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

SECURITY_GROUP_AUDIT = ActionDefinition(
    action_id="security_group_audit",
    phase="resource_enum",
    description="Audit VPC security group ingress rules for unrestricted CIDR blocks (0.0.0.0/0)",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=SecurityGroupAuditParams,
    allowed_target_kinds=["domain", "ip", "local"],
    command_builder="build_security_group_audit",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=True,
)

CLOUDTRAIL_AUDIT = ActionDefinition(
    action_id="cloudtrail_audit",
    phase="resource_enum",
    description="Verify cloud audit logging configuration, multi-region coverage, and log validation",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=CloudTrailAuditParams,
    allowed_target_kinds=["domain", "local"],
    command_builder="build_cloudtrail_audit",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=True,
)

METADATA_SERVICE_AUDIT = ActionDefinition(
    action_id="metadata_service_audit",
    phase="misconfig_validation",
    description="Audit EC2/GCE metadata service endpoints for unauthenticated IMDSv1 token access",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=MetadataServiceAuditParams,
    allowed_target_kinds=["ip", "url", "domain"],
    command_builder="build_metadata_service_audit",
    timeout_seconds=20,
    max_retries=1,
    satisfies_phase=True,
)

PRIVESC_PATH_AUDIT = ActionDefinition(
    action_id="privesc_path_audit",
    phase="iam_audit",
    description="Correlate IAM policy permissions to determine full administrative escalation paths",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=PrivescPathAuditParams,
    allowed_target_kinds=["domain", "local"],
    command_builder="build_privesc_path_audit",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

CLOUD_REPORT = ActionDefinition(
    action_id="cloud_report",
    phase="misconfig_validation",
    description="Synthesize verified cloud findings into a formal security posture audit report",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=CloudReportParams,
    allowed_target_kinds=["domain", "ip", "local"],
    command_builder="build_cloud_report",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)


def build_cloud_registry() -> ActionRegistry:
    """Build and return the ActionRegistry for CloudAgent."""
    registry = ActionRegistry()
    registry.register(IAM_AUDIT)
    registry.register(RESOURCE_ENUM)
    registry.register(MISCONFIG_VALIDATION)
    registry.register(S3_BUCKET_AUDIT)
    registry.register(ASSUME_ROLE_AUDIT)
    registry.register(SECURITY_GROUP_AUDIT)
    registry.register(CLOUDTRAIL_AUDIT)
    registry.register(METADATA_SERVICE_AUDIT)
    registry.register(PRIVESC_PATH_AUDIT)
    registry.register(CLOUD_REPORT)
    return registry

