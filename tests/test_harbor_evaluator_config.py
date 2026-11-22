import json
import sys
from pathlib import Path

import pytest
from conftest import fixture_recipe_config, init_fixture_workspace, init_workspace_from_config

from evolve import workspace as workspace_module
from evolve.workspace import InitOptions, _eval_env


def test_eval_env_uses_configured_harbor_agent() -> None:
    env = _eval_env(
        "exp",
        "swe-bench-lite",
        n_concurrent=2,
        tasks_per_round=3,
        trials=1,
        partial_floor=0.8,
        agent="evolve.integrations.harbor.miniswe_candidate:MiniSweSourceAgent",
        model="openai/gpt-5.4-2026-03-05",
    )

    assert "EVOLVE_HARBOR_AGENT=evolve.integrations.harbor.miniswe_candidate:MiniSweSourceAgent\n" in env
    assert "EVOLVE_HARBOR_MODEL=openai/gpt-5.4-2026-03-05\n" in env
    assert "CheckoutTargetAgent" not in env


def test_eval_env_freezes_configured_model() -> None:
    env = _eval_env(
        "exp",
        "terminal-bench-2",
        n_concurrent=2,
        tasks_per_round=30,
        trials=1,
        partial_floor=0.8,
        agent="evolve.integrations.harbor.miniswe_candidate:MiniSweSourceAgent",
        model="openai/gpt-5.4-2026-03-05",
    )

    assert "EVOLVE_HARBOR_MODEL=openai/gpt-5.4-2026-03-05\n" in env


def test_harbor_engine_exposes_rollout_model_to_verifier() -> None:
    engine = (Path(__file__).resolve().parents[1] / "scaffolds" / "evaluators" / "harbor" / "engine.sh").read_text()

    assert '--ve "EVOLVE_HARBOR_MODEL=$EVOLVE_HARBOR_MODEL"' in engine
    assert '--ve "EVOLVE_HARBOR_MODEL=openai/$OPENAI_MODEL"' in engine


def test_harbor_engine_supports_optional_frozen_evaluator_runtime_without_changing_defaults() -> None:
    engine = (Path(__file__).resolve().parents[1] / "scaffolds" / "evaluators" / "harbor" / "engine.sh").read_text()

    assert "if [ -f evaluator/prepare-runtime.sh ]; then" in engine
    assert 'EVOLVE_HARBOR_ENVIRONMENT="${EVOLVE_HARBOR_ENVIRONMENT:-}"' in engine
    assert 'EVOLVE_WORKSPACE="$EVOLVE_WORKSPACE"' in engine
    assert 'sh evaluator/prepare-runtime.sh "$EVOLVE_RUN_DIR" "$evaluator_runtime_env"' in engine
    assert '--ae "$evaluator_runtime_entry" --ve "$evaluator_runtime_entry"' in engine
    assert "local execution runtime requires Harbor LocalEnvironment; refusing Docker fallback" in engine


def test_harbor_engine_runs_with_frozen_weighted_resource_budget() -> None:
    engine = (Path(__file__).resolve().parents[1] / "scaffolds" / "evaluators" / "harbor" / "engine.sh").read_text()

    assert 'set -- python -I -m evolve.integrations.harbor._resource_scheduler "$@"' in engine
    assert "export HARBOR_RESOURCE_CPU_SLOTS HARBOR_RESOURCE_MEMORY_MB HARBOR_RESOURCE_POLICY" in engine
    assert "resource-scheduler.log" in engine


def test_eval_env_freezes_agent_timeout_multiplier() -> None:
    env = _eval_env(
        "exp",
        "terminal-bench-2",
        n_concurrent=8,
        tasks_per_round=30,
        trials=1,
        partial_floor=0.8,
        agent="evolve.integrations.harbor.miniswe_candidate:MiniSweSourceAgent",
        agent_timeout_multiplier=4,
    )

    assert "EVOLVE_HARBOR_AGENT_TIMEOUT_MULTIPLIER=4\n" in env


def test_eval_env_freezes_verifier_timeout_multiplier() -> None:
    env = _eval_env(
        "exp",
        "terminal-bench-2",
        n_concurrent=8,
        tasks_per_round=30,
        trials=1,
        partial_floor=0.8,
        agent="evolve.integrations.harbor.miniswe_candidate:MiniSweSourceAgent",
        verifier_timeout_multiplier=2,
    )

    assert "EVOLVE_HARBOR_VERIFIER_TIMEOUT_MULTIPLIER=2\n" in env


def test_eval_env_freezes_weighted_resource_scheduler() -> None:
    env = _eval_env(
        "exp",
        "terminal-bench-4.1",
        n_concurrent=30,
        tasks_per_round=30,
        trials=1,
        partial_floor=1.0,
        agent="custom:Agent",
        resource_cpu_slots=22,
        resource_memory_mb=30720,
        resource_policy="calibrated-largest-fitting",
        resource_max_active=8,
        resource_cpu_fraction=0.5,
        resource_memory_fraction=0.25,
        resource_memory_floor_mb=1024,
        resource_memory_cap_mb=6144,
        resource_min_available_memory_mb=4096,
        resource_profile="evaluator/resource-profile.json",
    )

    assert "HARBOR_RESOURCE_CPU_SLOTS=22\n" in env
    assert "HARBOR_RESOURCE_MEMORY_MB=30720\n" in env
    assert "HARBOR_RESOURCE_POLICY=calibrated-largest-fitting\n" in env
    assert "HARBOR_RESOURCE_MAX_ACTIVE=8\n" in env
    assert "HARBOR_RESOURCE_CPU_FRACTION=0.5\n" in env
    assert "HARBOR_RESOURCE_MEMORY_FRACTION=0.25\n" in env
    assert "HARBOR_RESOURCE_MEMORY_FLOOR_MB=1024\n" in env
    assert "HARBOR_RESOURCE_MEMORY_CAP_MB=6144\n" in env
    assert "HARBOR_RESOURCE_MIN_AVAILABLE_MEMORY_MB=4096\n" in env
    assert "HARBOR_RESOURCE_PROFILE=evaluator/resource-profile.json\n" in env


def test_eval_env_omits_neutral_harbor_controls() -> None:
    env = _eval_env(
        "exp",
        "terminal-bench-2",
        n_concurrent=1,
        tasks_per_round=1,
        trials=1,
        partial_floor=0.9,
        agent="custom:Agent",
        setup_timeout_multiplier=1,
        agent_timeout_multiplier=1,
        verifier_timeout_multiplier=1,
        max_retries=0,
    )

    assert "EVOLVE_HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER" not in env
    assert "EVOLVE_HARBOR_AGENT_TIMEOUT_MULTIPLIER" not in env
    assert "EVOLVE_HARBOR_VERIFIER_TIMEOUT_MULTIPLIER" not in env
    assert "EVOLVE_HARBOR_MAX_RETRIES" not in env


def test_eval_env_and_environment_kwargs_render_local_backend() -> None:
    env = _eval_env(
        "exp",
        "tasks",
        n_concurrent=1,
        tasks_per_round=1,
        trials=1,
        partial_floor=0.8,
        agent="custom:Agent",
        environment="evolve.harbor_local:LocalEnvironment",
        execution_backend="local",
    )

    assert "EVOLVE_HARBOR_ENVIRONMENT=evolve.harbor_local:LocalEnvironment\n" in env
    assert "EVOLVE_EXECUTION_BACKEND=local\n" in env
    assert workspace_module._runtime_kwargs(
        {"workdir": "/workspace", "options": {"clean": True}}, "environment_kwargs"
    ) == ('options={"clean":true}\nworkdir="/workspace"\n')


def test_agent_env_renders_frozen_miniswe_limits_deterministically() -> None:
    assert workspace_module._agent_env(
        {
            "MINISWE_STEP_LIMIT": "100",
            "MINISWE_ENV_TIMEOUT": 30,
            "MINISWE_COST_LIMIT": 3.0,
        }
    ) == ("MINISWE_COST_LIMIT=3.0\nMINISWE_ENV_TIMEOUT=30\nMINISWE_STEP_LIMIT=100\n")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"BAD-NAME": "1"}, "invalid evaluator.agent_env name"),
        ({"GOOD": "first\nsecond"}, "single-line"),
        ({"GOOD": "first\0second"}, "NUL"),
        ({"GOOD": ["nested"]}, "scalar"),
        (["not", "a", "mapping"], "must be a mapping"),
    ],
)
def test_agent_env_rejects_unsafe_values(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        workspace_module._agent_env(value)


def test_environment_kwargs_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        workspace_module._runtime_kwargs(["bad"], "environment_kwargs")
    with pytest.raises(ValueError, match="invalid evaluator.environment_kwargs name"):
        workspace_module._runtime_kwargs({"bad-name": True}, "environment_kwargs")


def test_init_real_harbor_recipe_requires_evaluator_agent(tmp_path: Path) -> None:
    config = fixture_recipe_config("hill_climb-smoke", "broken")
    config["evaluator"].pop("agent")

    with pytest.raises(ValueError, match="evaluator.agent is required"):
        init_workspace_from_config(InitOptions(workspace=tmp_path / "w"), config)


def test_init_writes_recipe_harbor_agent_to_eval_env(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"

    init_fixture_workspace(workspace)

    env = (workspace / "evaluator" / "eval.env").read_text()
    assert "EVOLVE_HARBOR_AGENT=target.agent:HarborAgent\n" in env
    assert "CheckoutTargetAgent" not in env
    assert (workspace / "evaluator" / "agent.env").read_text() == ""
    assert not (workspace / "evaluator" / "checkout_agent.py").exists()


def test_environment_build_timeout_is_frozen_with_other_phase_limits() -> None:
    env = _eval_env(
        "exp",
        "tasks",
        1,
        1,
        1,
        1.0,
        "custom:Agent",
        build_timeout_multiplier=12,
        setup_timeout_multiplier=20,
        agent_timeout_multiplier=12,
        verifier_timeout_multiplier=12,
    )
    values = dict(line.split("=", 1) for line in env.splitlines())
    assert values["EVOLVE_HARBOR_ENVIRONMENT_BUILD_TIMEOUT_MULTIPLIER"] == "12"
    assert values["EVOLVE_HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER"] == "20"
    assert values["EVOLVE_HARBOR_AGENT_TIMEOUT_MULTIPLIER"] == "12"
    assert values["EVOLVE_HARBOR_VERIFIER_TIMEOUT_MULTIPLIER"] == "12"


@pytest.mark.parametrize(
    ("route", "expected"),
    [("inherit", "localhost"), ("bypass", "localhost,model.example"), (None, "localhost,model.example")],
)
def test_harbor_engine_proxy_projection_respects_frozen_runtime(tmp_path, monkeypatch, capsys, route, expected) -> None:
    engine = (Path(__file__).resolve().parents[1] / "scaffolds/evaluators/harbor/engine.sh").read_text()
    projection = engine.split("proxy_bypass=$(\n", 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n)", 1)[0]
    monkeypatch.chdir(tmp_path)
    if route is not None:
        (tmp_path / "evaluator").mkdir()
        (tmp_path / "evaluator/runtime.json").write_text(json.dumps({"proxy": {"model_endpoint": route}}))
    monkeypatch.setattr(sys, "argv", ["proxy", "https://model.example/v1", "", "localhost", ""])

    exec(compile(projection, "harbor-engine-proxy", "exec"), {})

    assert capsys.readouterr().out.strip() == expected
