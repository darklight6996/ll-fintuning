"""
core/technique_advisor.py

FRAMEWORK-LEVEL technique retrieval -- the "playbook" layer.

Gives the AI planner the attack methodology/technique knowledge it needs FROM
THE KNOWLEDGE BASE, instead of relying on the base model's own weights. This is
the core of the design thesis: a small (7B) local model does not need to
*invent* techniques -- request-smuggling methodology, SSTI payloads, IDOR
patterns, the SSRF->metadata->creds chain all already exist and are exactly
what the knowledge base holds. The model's job narrows to orchestration,
chaining, and oracle judgment; the techniques are retrieved.

This is consumed by core/ai_planner.py, which is shared by EVERY agent
(bug_bounty, pentest, ctf, red_team, active_directory, cloud, firewall_bypass,
mobile, security_research). So wiring it here lights it up for all of them at
once -- it is cross-agent by construction, not a bug_bounty bolt-on.

Two knowledge sources, BOTH optional and BOTH failing silently to empty:
  1. RAGEngine (core.rag_engine): embedded doc corpus (HTB / PortSwigger / OWASP
     / cybersec fine-tune data) -> free-text technique/methodology chunks.
  2. YAMLSearch (core.yaml_search): embedded YAML assessment-scenario templates
     -> structured methodology templates.

INVARIANTS (non-negotiable -- the planner must never regress because of this):
  * NEVER raises into the planner. Any import/model/index/download failure
    degrades to an empty advice string, and the planner behaves EXACTLY as it
    did before this module existed. The knowledge base is an enhancement, never
    a dependency. A box with no FAISS index built simply gets no advice yet.
  * Each source is initialized at most once (try-once-then-cache), and the
    underlying engines are process singletons / cache their own index, so
    per-planner-step retrieval is cheap after the first call.
  * Output is token-bounded (max_chars) so it informs the decision without
    blowing the prompt budget.
  * Fully disable-able via env var PENTEST_DISABLE_TECHNIQUE_ADVISOR=1 (a clean
    kill switch for debugging a run without the knowledge base in the loop).
"""

import os
from typing import Any, List, Optional, Tuple

_ADVISOR: Optional["TechniqueAdvisor"] = None

_DISABLE_ENV = "PENTEST_DISABLE_TECHNIQUE_ADVISOR"


class TechniqueAdvisor:
    def __init__(self) -> None:
        # None  = not yet attempted; False = attempted and unavailable;
        # object = the live engine. Keeps us from re-paying a failed init.
        self._rag: Any = None
        self._yaml: Any = None
        self._rag_tried = False
        self._yaml_tried = False

    # -- lazy, fail-silent source init ------------------------------------

    def _get_rag(self):
        if self._rag_tried:
            return self._rag or None
        self._rag_tried = True
        try:
            from core.rag_engine import get_global_rag_engine
            self._rag = get_global_rag_engine() or False
        except Exception:
            self._rag = False
        return self._rag or None

    def _get_yaml(self):
        if self._yaml_tried:
            return self._yaml or None
        self._yaml_tried = True
        try:
            from core.yaml_search import YAMLSearch
            ys = YAMLSearch(auto_sync=True, verbose=False)
            # Only keep it if it actually has templates indexed.
            self._yaml = ys if len(ys) > 0 else False
        except Exception:
            self._yaml = False
        return self._yaml or None

    # -- retrieval --------------------------------------------------------

    def _rag_snippets(self, query: str, k: int) -> List[str]:
        eng = self._get_rag()
        if eng is None:
            return []
        try:
            ranked = eng.retrieve(query)
            if not ranked:
                return []
            out = []
            for meta, _score in ranked[:k]:
                text = ""
                if isinstance(meta, dict):
                    text = (meta.get("text") or meta.get("content")
                            or meta.get("chunk") or meta.get("body") or "")
                text = str(text).strip().replace("\n", " ")
                if text:
                    out.append(text)
            return out
        except Exception:
            return []

    def _yaml_snippets(self, query: str, k: int) -> List[str]:
        ys = self._get_yaml()
        if ys is None:
            return []
        try:
            templates = ys.search(query, top_k=k)
            out = []
            for t in templates or []:
                if not isinstance(t, dict):
                    continue
                name = (t.get("name") or t.get("title") or t.get("id") or "template")
                desc = (t.get("description") or t.get("objective")
                        or t.get("summary") or "")
                tags = t.get("tags") or []
                tag_str = f" [{', '.join(str(x) for x in tags[:5])}]" if tags else ""
                line = f"{name}{tag_str}: {str(desc).strip()}".strip()
                if line:
                    out.append(line.replace("\n", " "))
            return out
        except Exception:
            return []

    def advise(
        self,
        query: str,
        rag_k: int = 3,
        yaml_k: int = 2,
        max_chars: int = 1600,
    ) -> str:
        """
        Return a compact, token-bounded technique-knowledge block for `query`,
        or "" when nothing is available (no index, disabled, or no hits).
        """
        if os.environ.get(_DISABLE_ENV) == "1":
            return ""
        if not query or not query.strip():
            return ""

        rag = self._rag_snippets(query, rag_k)
        yml = self._yaml_snippets(query, yaml_k)
        if not rag and not yml:
            return ""

        lines: List[str] = []
        if yml:
            lines.append("Methodology templates:")
            for s in yml:
                lines.append(f"  - {s}")
        if rag:
            lines.append("Reference technique notes:")
            for s in rag:
                lines.append(f"  - {s}")

        block = "\n".join(lines).strip()
        if len(block) > max_chars:
            block = block[: max_chars - 3].rstrip() + "..."
        return block


def get_technique_advisor() -> TechniqueAdvisor:
    """Process-wide singleton so engine/model init is paid at most once."""
    global _ADVISOR
    if _ADVISOR is None:
        _ADVISOR = TechniqueAdvisor()
    return _ADVISOR
