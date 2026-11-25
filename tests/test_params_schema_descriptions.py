"""
tests/test_params_schema_descriptions.py

Regression test for a bug found live (2026-11-25, two consecutive real
PortSwigger runs): AIPlanner._format_params_schema() rendered each action's
parameter catalog line as bare "name:type (required/optional)" and silently
DROPPED every field's Pydantic Field(description=...) text. This meant
carefully-written prompt guidance -- e.g. self_script's "code" field, which
teaches the model to recognize a backend XML-parser-error signature
(SAXParseException / "Entities are not allowed") and rewrite its probe as
raw XML instead of mislabeling the result as SQLi/SSRF -- was present in the
schema but had ZERO effect on the model, because it never reached the
prompt at all. Confirmed on a live run: the model hit exactly that error
signature twice in a row and never once attempted an XML body.

This test asserts the rendered parameter line actually contains each
field's description text, not just its name/type/required-ness.
"""

import pytest

from core.ai_planner import AIPlanner
from agents.common.self_scripting import SelfScriptParams
from agents.common.recon_primitive import ReconScanParams


def _planner():
    # _format_params_schema needs no instance state; stub constructor args.
    return AIPlanner(registry=None, chat_assistant=None, scope=None)


def test_field_description_is_included_in_rendered_schema():
    schema = SelfScriptParams.model_json_schema()["properties"]
    required = SelfScriptParams.model_json_schema().get("required", [])
    rendered = _planner()._format_params_schema(schema, required)
    assert "SAXParseException" in rendered
    assert "Entities are not allowed" in rendered
    assert "CONFIRMATION RIGOR" in rendered


def test_required_and_type_still_rendered_correctly():
    schema = SelfScriptParams.model_json_schema()["properties"]
    required = SelfScriptParams.model_json_schema().get("required", [])
    rendered = _planner()._format_params_schema(schema, required)
    assert "target:string (required)" in rendered
    assert "code:string (required)" in rendered


def test_field_without_description_still_renders_bare():
    rendered = _planner()._format_params_schema(
        {"foo": {"type": "string"}}, required=["foo"])
    assert rendered == "foo:string (required)"


def test_very_long_description_is_capped_not_unbounded():
    long_desc = "x" * 5000
    rendered = _planner()._format_params_schema(
        {"foo": {"type": "string", "description": long_desc}}, required=["foo"])
    assert len(rendered) < 2500
    assert rendered.endswith("...")


def test_no_schema_returns_no_parameters_placeholder():
    assert _planner()._format_params_schema({}, []) == "(no parameters)"


def test_optional_field_with_default_still_rendered():
    schema = ReconScanParams.model_json_schema()["properties"]
    required = ReconScanParams.model_json_schema().get("required", [])
    rendered = _planner()._format_params_schema(schema, required)
    assert "modes:" in rendered
    assert "(optional)" in rendered
    # The modes field's own guidance should also now be visible.
    assert "single-host engagement" in rendered


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
