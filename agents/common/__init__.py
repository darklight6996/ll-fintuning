"""
agents.common -- framework-level actions shared across ALL agents.

Capabilities that belong to the whole framework (the rich HTTP primitive, and
later the headless-browser and self-scripting primitives) live here rather than
inside one agent, so every agent registers them with a single line and they stay
consistent everywhere.
"""
