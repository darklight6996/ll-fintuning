"""
core/intelligence/base.py

Base interface contract for intelligence extraction plugins.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseParser(ABC):
    """
    Abstract interface for specialized cybersecurity intelligence parsers.
    Each parser focuses on extracting a single semantic domain from terminal
    streams and session notes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this parser plugin."""
        pass

    @abstractmethod
    def parse(self, text: str, session_state: Dict[str, Any]) -> Any:
        """
        Extract domain-specific intelligence from combined session output.

        Parameters
        ----------
        text : str
            Aggregated terminal stdout, evidence, and notes text.
        session_state : Dict[str, Any]
            Full session state dictionary containing targets, notes, and evidence objects.

        Returns
        -------
        Domain-specific dataclass (e.g. UserIntel, PortIntel, etc.)
        """
        pass
