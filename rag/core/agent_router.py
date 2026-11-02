"""
core/agent_router.py

Maps assessment types and interactive operator menu choices to specialized Agent classes.
"""

from typing import Dict, Any, Type, Optional, List
from dataclasses import dataclass


@dataclass
class AgentMetadata:
    key: str
    name: str
    description: str
    module_path: str
    class_name: str
    default_tools: List[str]


AGENT_REGISTRY: Dict[str, AgentMetadata] = {
    "1": AgentMetadata(
        key="bug_bounty",
        name="Bug Bounty Agent",
        description="Reconnaissance, subdomain enumeration, endpoint discovery, parameter fuzzing, vuln validation",
        module_path="agents.bug_bounty.agent",
        class_name="BugBountyAgent",
        default_tools=["curl", "nmap", "ffuf", "subfinder"]
    ),
    "2": AgentMetadata(
        key="red_team",
        name="Red Team Agent",
        description="Reconnaissance, initial access, privilege escalation, lateral movement, objective impact",
        module_path="agents.red_team.agent",
        class_name="RedTeamAgent",
        default_tools=["nmap", "curl", "whois"]
    ),
    "3": AgentMetadata(
        key="pentest",
        name="Penetration Testing Agent",
        description="Formal PTES: Recon, port enumeration, exploitation, post-exploitation, report synthesis",
        module_path="agents.pentest.agent",
        class_name="PentestAgent",
        default_tools=["nmap", "nikto", "sqlmap", "curl"]
    ),
    "4": AgentMetadata(
        key="ctf",
        name="CTF / Wargame Agent",
        description="Environment reconnaissance, local enumeration, privesc vectors, flag capture verification",
        module_path="agents.ctf.agent",
        class_name="CTFAgent",
        default_tools=["python", "curl"]
    ),
    "5": AgentMetadata(
        key="mobile",
        name="Mobile App Security Agent",
        description="APK/IPA static analysis, dynamic runtime testing, auth validation, data storage & network audit",
        module_path="agents.mobile.agent",
        class_name="MobileAppAgent",
        default_tools=["apktool", "semgrep"]
    ),
    "6": AgentMetadata(
        key="firewall_bypass",
        name="Firewall / WAF Bypass Agent",
        description="WAF fingerprinting, rule enumeration, payload mutation and bypass technique testing",
        module_path="agents.firewall_bypass.agent",
        class_name="FirewallBypassAgent",
        default_tools=["curl", "nmap"]
    ),
    "7": AgentMetadata(
        key="security_research",
        name="Security Research Agent",
        description="CVE analysis, advisory analysis, proof-of-concept synthesis, patch analysis & technical write-up",
        module_path="agents.security_research.agent",
        class_name="SecurityResearchAgent",
        default_tools=["git", "python", "curl"]
    ),
    "8": AgentMetadata(
        key="active_directory",
        name="Active Directory Agent",
        description="AD domain enumeration, Kerberoasting, AS-REP roasting, DCSync analysis, domain trust audit",
        module_path="agents.active_directory.agent",
        class_name="ActiveDirectoryAgent",
        default_tools=["nmap", "dig"]
    ),
    "9": AgentMetadata(
        key="cloud",
        name="Cloud Security Agent",
        description="IAM policy audit, cloud resource enumeration, misconfiguration validation (AWS/Azure/GCP)",
        module_path="agents.cloud.agent",
        class_name="CloudAgent",
        default_tools=["curl", "python", "git"]
    )
}


class AgentRouter:
    """
    Routes operator menu selections to the appropriate agent instance.
    """

    @classmethod
    def list_agents(cls) -> List[AgentMetadata]:
        return list(AGENT_REGISTRY.values())

    @classmethod
    def get_agent_meta(cls, identifier: str) -> Optional[AgentMetadata]:
        clean_id = identifier.strip().lower()
        if clean_id in AGENT_REGISTRY:
            return AGENT_REGISTRY[clean_id]
        for meta in AGENT_REGISTRY.values():
            if clean_id == meta.key or clean_id == meta.name.lower():
                return meta
        return None

    @classmethod
    def instantiate_agent(
        cls,
        identifier: str,
        scope: Any,
        mode: str = "manual",
        session_manager: Optional[Any] = None,
        chat_assistant: Optional[Any] = None,
        command_executor: Optional[Any] = None,
    ) -> Any:
        meta = cls.get_agent_meta(identifier)
        if not meta:
            raise ValueError(f"Unknown agent identifier: '{identifier}'. Must be 1-9 or valid agent key.")

        import importlib
        mod = importlib.import_module(meta.module_path)
        agent_cls = getattr(mod, meta.class_name)

        return agent_cls(
            scope=scope,
            mode=mode,
            session_manager=session_manager,
            chat_assistant=chat_assistant,
            command_executor=command_executor,
        )

