"""
agents package initialization.
Exports BaseAgent, PhaseResult, and all 9 specialized assessment agents.
"""

from agents.base_agent import BaseAgent, PhaseResult
from agents.bug_bounty.agent import BugBountyAgent
from agents.red_team.agent import RedTeamAgent
from agents.pentest.agent import PentestAgent
from agents.ctf.agent import CTFAgent
from agents.mobile.agent import MobileAppAgent
from agents.firewall_bypass.agent import FirewallBypassAgent
from agents.security_research.agent import SecurityResearchAgent
from agents.active_directory.agent import ActiveDirectoryAgent
from agents.cloud.agent import CloudAgent

__all__ = [
    "BaseAgent",
    "PhaseResult",
    "BugBountyAgent",
    "RedTeamAgent",
    "PentestAgent",
    "CTFAgent",
    "MobileAppAgent",
    "FirewallBypassAgent",
    "SecurityResearchAgent",
    "ActiveDirectoryAgent",
    "CloudAgent",
]
