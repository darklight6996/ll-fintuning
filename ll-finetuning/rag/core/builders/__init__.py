"""
core/builders package.

NOTE: PromptBuilder is intentionally NOT re-exported here.
It lives in core.prompts and core.builders.prompt_builder (alias).
Importing it from this __init__ creates a circular dependency:
  core.builders -> core.builders.prompt_builder
               -> core.prompts.prompt_builder
               -> core.builders.context_builder
               -> core.builders.__init__  (circular!)
"""
from .context_builder import ContextBuilder, ContextPackage

__all__ = ["ContextBuilder", "ContextPackage"]
