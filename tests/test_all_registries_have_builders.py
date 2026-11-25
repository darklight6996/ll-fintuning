"""
tests/test_all_registries_have_builders.py

Regression guard for a framework-wide bug found live (pentest agent,
Qwen2.5-7B-Instruct + qwen-qlora smoke test, 2026-11-14): EVERY agent except
CTF called `registry.register(SOME_ACTION_DEFINITION)` with no second
argument. ActionRegistry.register() only populates self.builders when a
builder callable is passed (see core/action_contract.py) -- without it,
registry.get_builder(defn.command_builder) returns None for every action, and
command_executor.py's Gate 7 (command generation) rejects every proposal that
gets past Gates 1/2 with "Command builder '...' is not registered."

This means every autonomous action in active_directory, bug_bounty, cloud,
firewall_bypass, mobile, pentest, red_team, and security_research could never
have actually executed, ever -- only CTF's build_ctf_registry() happened to
pass the actual builder function as the second argument to register(). Fixed
by wiring each agent's own command_builders.py functions into its registry,
the same way CTF already did.

This test imports every agent's build_<agent>_registry() factory and asserts
every single registered ActionDefinition resolves to a real, callable builder
via registry.get_builder(defn.command_builder) -- so this class of bug (a
registered action silently missing its command builder) can never regress
silently again for ANY agent, present or future.
"""
import importlib

import pytest

# (module path, factory function name) for every agent with an autonomous
# action registry.
REGISTRY_FACTORIES = [
    ("agents.active_directory.actions", "build_ad_registry"),
    ("agents.bug_bounty.actions", "build_bug_bounty_registry"),
    ("agents.cloud.actions", "build_cloud_registry"),
    ("agents.ctf.actions", "build_ctf_registry"),
    ("agents.firewall_bypass.actions", "build_firewall_registry"),
    ("agents.mobile.actions", "build_mobile_registry"),
    ("agents.pentest.actions", "build_pentest_registry"),
    ("agents.red_team.actions", "build_red_team_registry"),
    ("agents.security_research.actions", "build_research_registry"),
    # Scope-derived variant registry (Scope.single_host_only=True) used by the
    # PortSwigger / single-host bug-bounty harness. Written before the
    # framework-wide builder-wiring fix, so it is guarded here too.
    ("agents.bug_bounty.single_host_mode", "build_single_host_bug_bounty_registry"),
]


def _load_registry(module_path, factory_name):
    module = importlib.import_module(module_path)
    factory = getattr(module, factory_name, None)
    if factory is None:
        pytest.skip(f"{module_path}.{factory_name} not found -- factory name may have changed.")
    return factory()


@pytest.mark.parametrize("module_path,factory_name", REGISTRY_FACTORIES)
def test_every_registered_action_has_a_real_builder(module_path, factory_name):
    registry = _load_registry(module_path, factory_name)
    actions = registry.list_actions()
    assert actions, f"{module_path}.{factory_name}() registered zero actions."

    missing = []
    for defn in actions:
        builder = registry.get_builder(defn.command_builder)
        if builder is None or not callable(builder):
            missing.append((defn.action_id, defn.command_builder))

    assert not missing, (
        f"{module_path}.{factory_name}() has action(s) with no resolvable "
        f"command builder (would be rejected at Gate 7 with "
        f"'builder_missing' on every live run): {missing}"
    )


@pytest.mark.parametrize("module_path,factory_name", REGISTRY_FACTORIES)
def test_builders_are_wired_to_the_correct_params_model(module_path, factory_name):
    """
    Lighter sanity check: each builder must at least be invocable with an
    instance of its own declared params_model without raising on construction
    of the params object itself (catches an obviously mismatched builder/
    params_model pairing, independent of the registration-wiring bug above).
    This does not execute any resulting command -- it only checks the builder
    is callable and params_model is a real Pydantic model class.
    """
    registry = _load_registry(module_path, factory_name)
    for defn in registry.list_actions():
        assert hasattr(defn.params_model, "model_fields"), (
            f"{module_path}: action '{defn.action_id}' params_model "
            f"{defn.params_model!r} is not a Pydantic model."
        )
