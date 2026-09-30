"""
core/intelligence/extractor.py

IntelligenceExtractor orchestrates decoupled parser plugins to transform raw session notes,
terminal evidence, and completed steps into structured cybersecurity intelligence.
"""

from typing import Dict, Any, List, Optional, Tuple
import threading

from core.intelligence.base import BaseParser
from core.intelligence.models import (
    SessionIntelligence,
    UserIntel,
    KernelIntel,
    PortIntel,
    BannerIntel,
    BinaryIntel,
    SUIDIntel,
    StringsIntel,
    FlagIntel,
    FilesystemIntel
)
from core.intelligence.parsers import (
    UserParser,
    KernelParser,
    PortParser,
    BannerParser,
    BinaryParser,
    SUIDParser,
    StringsParser,
    FlagParser,
    FilesystemParser
)

_GLOBAL_EXTRACTOR: Optional["IntelligenceExtractor"] = None
_LOCK = threading.RLock()


class IntelligenceExtractor:
    """
    Plugin-driven intelligence extraction engine.
    Eliminates monolithic regex routines by dispatching text across dedicated,
    single-responsibility parser plugins.
    """

    def __init__(self, register_defaults: bool = True):
        self._parsers: Dict[str, BaseParser] = {}
        if register_defaults:
            self._register_default_parsers()

    def _register_default_parsers(self) -> None:
        """Register the 9 standard cybersecurity forensic parsers."""
        default_parsers = [
            UserParser(),
            KernelParser(),
            PortParser(),
            BannerParser(),
            BinaryParser(),
            SUIDParser(),
            StringsParser(),
            FlagParser(),
            FilesystemParser(),
        ]
        for parser in default_parsers:
            self.register_parser(parser)

    def register_parser(self, parser: BaseParser) -> None:
        """Register a new intelligence parser plugin."""
        self._parsers[parser.name] = parser

    def unregister_parser(self, parser_name: str) -> Optional[BaseParser]:
        """Unregister a parser plugin by name."""
        return self._parsers.pop(parser_name, None)

    def get_parser(self, parser_name: str) -> Optional[BaseParser]:
        """Retrieve a registered parser plugin."""
        return self._parsers.get(parser_name)

    @property
    def registered_parsers(self) -> List[str]:
        """List names of all registered parser plugins."""
        return list(self._parsers.keys())

    def aggregate_text(self, session_state: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Aggregate text across notes, evidence, and completed step outputs.
        """
        if not session_state:
            return "", []

        notes = session_state.get("notes", [])
        evidence = session_state.get("evidence", [])
        completed = session_state.get("completed_steps", [])

        all_blocks = []
        raw_notes = []

        for n in notes:
            content = n.get("content", "") if isinstance(n, dict) else str(n)
            if content:
                all_blocks.append(content)
                raw_notes.append(content)

        for e in evidence:
            if isinstance(e, dict):
                out = e.get("output", "") or e.get("stdout", "")
                if out:
                    all_blocks.append(out)
            elif isinstance(e, str) and e:
                all_blocks.append(e)

        for s in completed:
            if isinstance(s, dict):
                out = s.get("stdout", "") or s.get("evidence", "")
                if out:
                    all_blocks.append(out)

        combined_text = "\n".join(all_blocks)
        return combined_text, raw_notes

    def extract(self, session_state: Dict[str, Any]) -> SessionIntelligence:
        """
        Run all registered parsers against the session state and return
        a structured SessionIntelligence object.
        """
        combined_text, raw_notes = self.aggregate_text(session_state)

        user_intel = UserIntel()
        kernel_intel = KernelIntel()
        port_intel = PortIntel()
        banner_intel = BannerIntel()
        binary_intel = BinaryIntel()
        suid_intel = SUIDIntel()
        strings_intel = StringsIntel()
        flag_intel = FlagIntel()
        filesystem_intel = FilesystemIntel()
        custom_facts: Dict[str, Any] = {}

        # Dispatch across all registered parsers
        for name, parser in self._parsers.items():
            try:
                result = parser.parse(combined_text, session_state)
                if isinstance(result, UserIntel):
                    user_intel = result
                elif isinstance(result, KernelIntel):
                    kernel_intel = result
                elif isinstance(result, PortIntel):
                    port_intel = result
                elif isinstance(result, BannerIntel):
                    banner_intel = result
                elif isinstance(result, BinaryIntel):
                    binary_intel = result
                elif isinstance(result, SUIDIntel):
                    suid_intel = result
                elif isinstance(result, StringsIntel):
                    strings_intel = result
                elif isinstance(result, FlagIntel):
                    flag_intel = result
                elif isinstance(result, FilesystemIntel):
                    filesystem_intel = result
                else:
                    custom_facts[name] = result
            except Exception as e:
                # Isolated failure in one parser must never crash the co-pilot
                custom_facts[f"{name}_error"] = str(e)

        return SessionIntelligence(
            user_intel=user_intel,
            kernel_intel=kernel_intel,
            port_intel=port_intel,
            banner_intel=banner_intel,
            binary_intel=binary_intel,
            suid_intel=suid_intel,
            strings_intel=strings_intel,
            flag_intel=flag_intel,
            filesystem_intel=filesystem_intel,
            raw_notes_text=raw_notes,
            custom_facts=custom_facts
        )

    def extract_dict(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract session intelligence and convert to the legacy dictionary contract.
        """
        return self.extract(session_state).to_dict()


def get_intelligence_extractor() -> IntelligenceExtractor:
    """
    Process-wide singleton getter for IntelligenceExtractor.
    """
    global _GLOBAL_EXTRACTOR
    if _GLOBAL_EXTRACTOR is None:
        with _LOCK:
            if _GLOBAL_EXTRACTOR is None:
                _GLOBAL_EXTRACTOR = IntelligenceExtractor()
    return _GLOBAL_EXTRACTOR
