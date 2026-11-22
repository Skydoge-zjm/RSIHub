"""Run one Harbor job with weighted trial admission in this CLI process."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from collections.abc import Coroutine
from importlib.metadata import version
from pathlib import Path
from typing import Any, NamedTuple

from harbor.models.task.config import EnvironmentConfig, TaskConfig
from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.queue import TrialQueue

_LOG = logging.getLogger("evolve.harbor.admission")


class Claim(NamedTuple):
    cpu: int
    memory_mb: int


def _limit(name: str) -> int | None:
    value = os.environ.get(name)
    if not value:
        return None
    if not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class WeightedTrialQueue(TrialQueue):
    def __init__(self, n_concurrent: int, *args: Any, **kwargs: Any):
        super().__init__(n_concurrent, *args, **kwargs)
        self.cpu_limit = _limit("HARBOR_RESOURCE_CPU_SLOTS")
        self.memory_limit = _limit("HARBOR_RESOURCE_MEMORY_MB")
        if self.cpu_limit is None and self.memory_limit is None:
            raise ValueError("weighted admission requires a CPU or memory budget")
        if os.getenv("HARBOR_RESOURCE_POLICY", "greedy-largest-fitting") != "greedy-largest-fitting":
            raise ValueError("unknown Harbor resource policy")
        self.count_limit = n_concurrent
        self.active = 0
        self.cpu = 0
        self.memory_mb = 0
        self.waiting: dict[asyncio.Future[None], tuple[str, Claim]] = {}
        self.dispatch_handle: asyncio.Handle | None = None

    def requirements(self, trial: TrialConfig) -> Claim:
        # Harbor's trial task config is an identifier. Read the frozen task snapshot.
        path = trial.task.path
        if path is None or trial.task.git_url:
            raise ValueError(f"{trial.trial_name}: weighted admission requires a local task snapshot")
        task = TaskConfig.model_validate_toml((path / "task.toml").read_text(encoding="utf-8"))

        def request(env: EnvironmentConfig) -> Claim:
            cpu = trial.environment.override_cpus if trial.environment.override_cpus is not None else env.cpus
            memory = trial.environment.override_memory_mb
            if memory is None:
                memory = env.memory_mb
            if self.cpu_limit is not None and (cpu is None or cpu <= 0):
                raise ValueError(f"{trial.trial_name}: missing positive CPU declaration")
            if self.memory_limit is not None and (memory is None or memory <= 0):
                raise ValueError(f"{trial.trial_name}: missing positive memory declaration")
            return Claim(math.ceil(cpu or 0), memory or 0)

        main = request(task.environment)
        phases = [main]
        if not trial.install_only and not trial.verifier.disable:
            steps = task.steps or [None]
            for index, step in enumerate(steps):
                verifier = resolve_effective_verifier_env_config(task, step)
                if verifier is not None:
                    need = request(verifier)
                    # During an intermediate separate verifier the agent is still live.
                    if index < len(steps) - 1:
                        need = Claim(main.cpu + need.cpu, main.memory_mb + need.memory_mb)
                    phases.append(need)
        peak = Claim(max(p.cpu for p in phases), max(p.memory_mb for p in phases))
        if not self._fits(peak.cpu, peak.memory_mb):
            raise ValueError(f"{trial.trial_name} exceeds the job budget: {peak}")
        return peak

    def _fits(self, cpu: int, memory: int) -> bool:
        return (self.cpu_limit is None or cpu <= self.cpu_limit) and (
            self.memory_limit is None or memory <= self.memory_limit
        )

    def _event(self, event: str, name: str, claim: Claim) -> None:
        _LOG.info(
            "%s %s cpu=%d memory_mb=%d running=%d reserved_cpu=%d reserved_memory_mb=%d",
            event,
            name,
            claim.cpu,
            claim.memory_mb,
            self.active,
            self.cpu,
            self.memory_mb,
        )

    def _dispatch(self) -> None:
        self.dispatch_handle = None
        # Stable ties preserve submission order. A non-fitting trial cannot block smaller ones.
        for ticket, (name, need) in sorted(
            self.waiting.items(), key=lambda item: (item[1][1].memory_mb, item[1][1].cpu), reverse=True
        ):
            if self.active >= self.count_limit:
                break
            if ticket.cancelled() or not self._fits(self.cpu + need.cpu, self.memory_mb + need.memory_mb):
                continue
            del self.waiting[ticket]
            self.active += 1
            self.cpu += need.cpu
            self.memory_mb += need.memory_mb
            ticket.set_result(None)
            self._event("admit", name, need)

    def _schedule(self) -> None:
        # Harbor starts the whole batch in one TaskGroup. Coalesce arrivals and releases
        # for this loop turn so the first coroutine does not get an automatic head start.
        if self.waiting and self.dispatch_handle is None:
            self.dispatch_handle = asyncio.get_running_loop().call_soon(self._dispatch)

    async def _bounded(self, trial: TrialConfig, need: Claim) -> TrialResult:
        ticket = asyncio.get_running_loop().create_future()
        self.waiting[ticket] = trial.trial_name, need
        self._schedule()
        try:
            await ticket
            return await super().submit(trial)
        finally:
            self.waiting.pop(ticket, None)
            if ticket.done() and not ticket.cancelled():
                self.active -= 1
                self.cpu -= need.cpu
                self.memory_mb -= need.memory_mb
                self._event("release", trial.trial_name, need)
            self._schedule()

    def submit(self, trial_config: TrialConfig) -> Coroutine[Any, Any, TrialResult]:
        return self._bounded(trial_config, self.requirements(trial_config))

    def submit_batch(self, configs: list[TrialConfig]) -> list[Coroutine[Any, Any, TrialResult]]:
        self._validate_agent_concurrency(configs)
        requests = [self.requirements(config) for config in configs]
        return [self._bounded(config, need) for config, need in zip(configs, requests, strict=True)]


def main() -> None:
    from unittest.mock import patch

    from harbor import job
    from harbor.cli.main import app

    if version("harbor") != "0.27.0":
        raise RuntimeError("weighted admission requires pinned Harbor 0.27.0")
    handler = logging.FileHandler(Path(os.environ["RSIHUB_RESOURCE_LOG"]), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False
    _LOG.addHandler(handler)
    # The pinned CLI's Job constructs its queue through this single module binding.
    try:
        with patch.object(job, "TrialQueue", WeightedTrialQueue):
            app(prog_name="harbor")
    finally:
        _LOG.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    main()
