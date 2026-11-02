"""
tests/conftest.py

Pytest configuration for the Autonomous AI Security Assessment System test suite.

Markers:
    slow    : Tests that load the TinyLlama LLM model (GPU-heavy, ~2-3 min each).
              Skip with: pytest -m "not slow"
              Run with:  pytest --runslow   OR   pytest -m slow

Usage examples:
    # Fast CI (skip LLM tests):
    pytest tests/test_autonomous_framework.py tests/test_multi_agent_framework.py -m "not slow" -v

    # Full regression (include GPU tests):
    pytest tests/ --runslow -v

    # Only autonomous framework (always fast, no GPU):
    pytest tests/test_autonomous_framework.py -v

    # Only LLM/AI tests:
    pytest tests/test_multi_agent_framework.py -m slow -v
"""

import pytest


def pytest_addoption(parser):
    """Add --runslow flag to pytest CLI."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Include tests marked @pytest.mark.slow (loads TinyLlama on GPU, ~2-3 min).",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (loads LLM model on GPU; deselect with '-m not slow')",
    )
    config.addinivalue_line(
        "markers",
        "agent_registry: marks tests that verify the ActionRegistry for a specific agent",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks end-to-end integration tests that require full environment",
    )


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.slow tests unless --runslow is passed."""
    if config.getoption("--runslow"):
        # --runslow given: do not skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="Skipped: GPU-heavy LLM test. Pass --runslow to include.")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

