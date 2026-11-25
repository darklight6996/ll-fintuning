"""
core/tools/class_gate.py

Pure, deterministic CONTEXT GATE for --sweep mode: maps the cheap
characterization tags collected for one surface entry (endpoint/param) to the
vuln classes actually worth testing there. This is the "context-gated" half of
the class-sweep spec -- it decides WHICH classes are even plausible for a given
param, so the model's job stays classification/prioritization, never payload
authorship, and the framework never wastes a budget testing XXE against a JSON
API or SQLi against a static asset path.

No network calls, no AI calls -- a table lookup over booleans. Tested directly
with synthetic tag sets.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

# Each gate rule: (tag_name, vuln_class). A surface entry's tags are an
# arbitrary Set[str]; a class fires if ANY of its tags is present (an OR
# across the rule rows that name it), matching the task's rule table where
# e.g. SSRF fires on "value url/host-shaped OR fetch-named".
_GATE_RULES: List[tuple] = [
    ("reflects_unencoded", "xss"),
    ("accepts_xml", "xxe"),
    ("value_url_shaped", "ssrf"),
    ("fetch_named_param", "ssrf"),
    ("value_path_shaped", "lfi"),
    ("file_named_param", "lfi"),
    ("errors_on_quote", "sqli"),
    ("query_param", "sqli"),
    ("state_changing_limited_resource", "race"),
    ("raw_request_available", "smuggling"),
    ("template_reflection", "ssti"),
]

# idor additionally requires auth context -- not a simple OR, so handled as a
# combination rule instead of a single-tag row above.
_COMBINATION_RULES: List[tuple] = [
    (("numeric_or_guid_id", "behind_auth"), "idor"),
]

ALL_SWEEP_CLASSES = sorted({cls for _, cls in _GATE_RULES} | {cls for _, cls in _COMBINATION_RULES})


def characterize_tags(entry: Dict[str, Any]) -> Set[str]:
    """
    Derive the boolean CHARACTERIZATION tags for one surface entry from its
    cheap, non-exploit observed facts. `entry` is expected to carry whichever
    of these keys are known (missing/None keys are simply False):
      reflects_unencoded, errors_on_quote, content_type, value_url_shaped,
      value_path_shaped, numeric_or_guid_id, behind_auth,
      state_changing_limited_resource, accepts_xml, template_reflection,
      raw_request_available, fetch_named_param, file_named_param, query_param.
    Returns the SET of tags that are truthy -- never raises, missing/odd
    input just yields fewer tags.
    """
    tag_names = (
        "reflects_unencoded", "errors_on_quote", "value_url_shaped",
        "value_path_shaped", "numeric_or_guid_id", "behind_auth",
        "state_changing_limited_resource", "accepts_xml", "template_reflection",
        "raw_request_available", "fetch_named_param", "file_named_param",
        "query_param",
    )
    tags = {name for name in tag_names if entry.get(name)}
    content_type = str(entry.get("content_type") or "").lower()
    if "xml" in content_type:
        tags.add("accepts_xml")
    return tags


def gate_classes_for_tags(tags: Set[str]) -> List[str]:
    """
    The context gate itself: which vuln classes are plausible given this
    entry's tags. Deterministic, order-stable (ALL_SWEEP_CLASSES order), no
    AI call -- this is what keeps the sweep from testing every class against
    every endpoint.
    """
    fired: Set[str] = set()
    for tag, cls in _GATE_RULES:
        if tag in tags:
            fired.add(cls)
    for required_tags, cls in _COMBINATION_RULES:
        if all(t in tags for t in required_tags):
            fired.add(cls)
    return [c for c in ALL_SWEEP_CLASSES if c in fired]


def gate_classes_for_entry(entry: Dict[str, Any]) -> List[str]:
    """Convenience: characterize + gate in one call."""
    return gate_classes_for_tags(characterize_tags(entry))
