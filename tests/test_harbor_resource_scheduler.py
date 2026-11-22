"""Exercise Harbor's batch submission API without starting containers or models."""

import asyncio
from pathlib import Path

import pytest
from harbor.models.trial.config import TrialConfig
from harbor.trial.queue import TrialQueue

from evolve.integrations.harbor._resource_scheduler import Claim, WeightedTrialQueue


def trial(tmp_path: Path, name: str, cpus: float, mib: int, extra: str = "") -> TrialConfig:
    path = tmp_path / name
    path.mkdir()
    (path / "task.toml").write_text(f"[environment]\ncpus = {cpus}\nmemory_mb = {mib}\n{extra}", encoding="utf-8")
    return TrialConfig(task={"path": path}, trial_name=name)


@pytest.fixture
def queue(monkeypatch):
    monkeypatch.setenv("HARBOR_RESOURCE_CPU_SLOTS", "22")
    monkeypatch.setenv("HARBOR_RESOURCE_MEMORY_MB", "30720")
    monkeypatch.setenv("HARBOR_RESOURCE_POLICY", "greedy-largest-fitting")
    return WeightedTrialQueue(30)


def test_batch_backfills_while_largest_trial_keeps_running(queue, tmp_path, monkeypatch):
    inputs = [("a", 2, 4096), ("b", 2, 4096), ("c", 2, 4096), ("mid", 4, 8192), ("big", 8, 16384), ("later", 8, 16384)]
    configs = [trial(tmp_path, *item) for item in inputs]
    started = {}
    finish = {}
    order = []

    async def execute(self, config):
        name = config.trial_name
        order.append(name)
        started[name].set()
        await finish[name].wait()
        return name

    async def scenario():
        for config in configs:
            started[config.trial_name] = asyncio.Event()
            finish[config.trial_name] = asyncio.Event()
        jobs = [asyncio.create_task(work) for work in queue.submit_batch(configs)]
        await started["a"].wait()
        assert order == ["big", "mid", "a"]
        assert (queue.cpu, queue.memory_mb) == (14, 28672)
        finish["mid"].set()
        await started["c"].wait()
        assert order == ["big", "mid", "a", "b", "c"]
        assert queue.memory_mb == 28672
        for event in finish.values():
            event.set()
        assert await asyncio.gather(*jobs) == [row[0] for row in inputs]
        assert (queue.active, queue.cpu, queue.memory_mb) == (0, 0, 0)

    monkeypatch.setattr(TrialQueue, "_execute_trial_with_retries", execute)
    asyncio.run(asyncio.wait_for(scenario(), timeout=3))


def test_cpu_limit_and_cancelled_or_failed_trials_release_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("HARBOR_RESOURCE_CPU_SLOTS", "4")
    monkeypatch.setenv("HARBOR_RESOURCE_MEMORY_MB", "32768")
    queue = WeightedTrialQueue(30)
    specs = [("large", 3, 8192), ("cancel-pending", 3, 4096), ("next", 3, 4096), ("fits", 1, 4096)]
    configs = [trial(tmp_path, *spec) for spec in specs]

    async def scenario():
        started = {name: asyncio.Event() for name, _, _ in specs}
        keep_running = asyncio.Event()

        async def execute(self, config):
            started[config.trial_name].set()
            if config.trial_name == "next":
                raise RuntimeError("fake trial failure")
            await keep_running.wait()

        monkeypatch.setattr(TrialQueue, "_execute_trial_with_retries", execute)
        jobs = [asyncio.create_task(work) for work in queue.submit_batch(configs)]
        await started["fits"].wait()
        assert queue.cpu == 4
        assert not started["cancel-pending"].is_set()
        jobs[1].cancel()
        jobs[0].cancel()
        await started["next"].wait()
        keep_running.set()
        results = await asyncio.gather(*jobs, return_exceptions=True)
        assert [type(r) for r in results] == [asyncio.CancelledError, asyncio.CancelledError, RuntimeError, type(None)]
        assert not queue.waiting
        assert (queue.active, queue.cpu, queue.memory_mb) == (0, 0, 0)

    asyncio.run(asyncio.wait_for(scenario(), timeout=3))


def test_uses_effective_verifier_budget_and_resource_overrides(queue, tmp_path):
    config = trial(tmp_path, "phases", 1.25, 4096, "[verifier.environment]\ncpus = 1\nmemory_mb = 16384\n")
    assert queue.requirements(config) == Claim(2, 16384)
    config.environment.override_cpus = 4
    config.environment.override_memory_mb = 8192
    assert queue.requirements(config) == Claim(4, 8192)


def test_intermediate_verifier_overlaps_main(queue, tmp_path):
    config = trial(
        tmp_path,
        "steps",
        4,
        8192,
        """
[[steps]]
name = "first"
[steps.verifier.environment]
cpus = 2
memory_mb = 8192
[[steps]]
name = "last"
[steps.verifier.environment]
cpus = 2
memory_mb = 8192
""",
    )
    assert queue.requirements(config) == Claim(6, 16384)


def test_infeasible_or_undeclared_task_rejected_before_submission(queue, tmp_path):
    small = trial(tmp_path, "fits", 2, 4096)
    huge = trial(tmp_path, "too-large", 2, 32768)
    with pytest.raises(ValueError, match="too-large exceeds"):
        queue.submit_batch([small, huge])
    (small.task.path / "task.toml").write_text("[environment]\ncpus = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing positive memory"):
        queue.submit(small)
    assert queue.active == 0 and not queue.waiting
