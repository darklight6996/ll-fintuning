"""
core/intelligence package

Cybersecurity forensic artifact extraction and intelligence decomposition:
- BaseParser: Abstract plugin contract
- IntelligenceExtractor: Plugin registry and aggregator
- Structured Intel Dataclasses: UserIntel, KernelIntel, PortIntel, BannerIntel,
  BinaryIntel, SUIDIntel, StringsIntel, FlagIntel, FilesystemIntel, SessionIntelligence
- Parsers: 9 single-responsibility domain parsers
"""

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
from core.intelligence.extractor import (
    IntelligenceExtractor,
    get_intelligence_extractor
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

__all__ = [
    "BaseParser",
    "SessionIntelligence",
    "UserIntel",
    "KernelIntel",
    "PortIntel",
    "BannerIntel",
    "BinaryIntel",
    "SUIDIntel",
    "StringsIntel",
    "FlagIntel",
    "FilesystemIntel",
    "IntelligenceExtractor",
    "get_intelligence_extractor",
    "UserParser",
    "KernelParser",
    "PortParser",
    "BannerParser",
    "BinaryParser",
    "SUIDParser",
    "StringsParser",
    "FlagParser",
    "FilesystemParser",
]
