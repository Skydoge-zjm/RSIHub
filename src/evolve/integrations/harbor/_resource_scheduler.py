"""Run one Harbor job with calibrated weighted trial admission."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Coroutine
from importlib.metadata import version
from pathlib import Path
from typing import Any

from harbor.models.task.config import EnvironmentConfig, TaskConfig
from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.queue import TrialQueue

from ._resource_policy import Admission, Claim, ResourcePolicy, host_memory_available_mb

_LOG = logging.getLogger("evolve.harbor.admission")


class WeightedTrialQueue(TrialQueue):
    def __init__(self, n_concurrent: int, *args: Any, **kwargs: Any):
        super().__init__(n_concurrent, *args, **kwargs)
        self.resources = ResourcePolicy.from_environment(n_concurrent)
        self.cpu_limit = self.resources.cpu_limit
        self.memory_limit = self.resources.memory_limit
        self.max_active = self.resources.max_active
        self.active = 0
        self.cpu = 0
        self.memory_mb = 0
        self.waiting: dict[asyncio.Future[None], tuple[str, Admission]] = {}
        self.dispatch_handle: asyncio.Handle | None = None
        self.last_host_defer_log = 0.0

    def _task(self, trial: TrialConfig) -> tuple[str, TaskConfig]:
        path = trial.task.path
        if path is None or trial.task.git_url:
            raise ValueError(f"{trial.trial_name}: weighted admission requires a local task snapshot")
        task = TaskConfig.model_validate_toml((path / "task.toml").read_text(encoding="utf-8"))
        return Path(path).name, task

    def official_requirements(self, trial: TrialConfig) -> Claim:
        _, task = self._task(trial)

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
        return Claim(max(p.cpu for p in phases), max(p.memory_mb for p in phases))

    def admission(self, trial: TrialConfig) -> Admission:
        task_name, _ = self._task(trial)
        return self.resources.admission(
            trial.trial_name,
            task_name,
            self.official_requirements(trial),
        )

    def requirements(self, trial: TrialConfig) -> Claim:
        return self.admission(trial).reserved

    def _fits(self, cpu: int, memory: int) -> bool:
        return self.resources.fits(cpu, memory)

    def _event(self, event: str, name: str, need: Admission) -> None:
        _LOG.info(
            "%s %s reserved_cpu=%d reserved_memory_mb=%d official_cpu=%d "
            "official_memory_mb=%d source=%s running=%d reserved_total_cpu=%d "
            "reserved_total_memory_mb=%d",
            event,
            name,
            need.reserved.cpu,
            need.reserved.memory_mb,
            need.official.cpu,
            need.official.memory_mb,
            need.source,
            self.active,
            self.cpu,
            self.memory_mb,
        )

    def _host_memory_ok(self) -> bool:
        return self.resources.host_memory_ok()

    def _defer_for_host_memory(self) -> None:
        now = time.monotonic()
        if now - self.last_host_defer_log >= 30:
            _LOG.info(
                "defer host_memory_available_mb=%s minimum_memory_mb=%d running=%d",
                host_memory_available_mb(),
                self.resources.min_available_memory_mb,
                self.active,
            )
            self.last_host_defer_log = now
        self.dispatch_handle = asyncio.get_running_loop().call_later(2, self._dispatch)

    def _dispatch(self) -> None:
        self.dispatch_handle = None
        if not self.waiting:
            return
        if not self._host_memory_ok():
            self._defer_for_host_memory()
            return
        # Stable ties preserve submission order. A non-fitting trial cannot block smaller ones.
        for ticket, (name, need) in sorted(
            self.waiting.items(),
            key=lambda item: (item[1][1].reserved.memory_mb, item[1][1].reserved.cpu),
            reverse=True,
        ):
            if self.active >= self.max_active:
                break
            reservation = need.reserved
            if ticket.cancelled() or not self._fits(
                self.cpu + reservation.cpu,
                self.memory_mb + reservation.memory_mb,
            ):
                continue
            del self.waiting[ticket]
            self.active += 1
            self.cpu += reservation.cpu
            self.memory_mb += reservation.memory_mb
            ticket.set_result(None)
            self._event("admit", name, need)

    def _schedule(self) -> None:
        # Harbor starts the whole batch in one TaskGroup. Coalesce arrivals and releases
        # for this loop turn so the first coroutine does not get an automatic head start.
        if self.waiting and self.dispatch_handle is None:
            self.dispatch_handle = asyncio.get_running_loop().call_soon(self._dispatch)

    async def _bounded(self, trial: TrialConfig, admission: Admission) -> TrialResult:
        ticket = asyncio.get_running_loop().create_future()
        self.waiting[ticket] = trial.trial_name, admission
        self._schedule()
        try:
            await ticket
            return await super().submit(trial)
        finally:
            self.waiting.pop(ticket, None)
            if ticket.done() and not ticket.cancelled():
                self.active -= 1
                self.cpu -= admission.reserved.cpu
                self.memory_mb -= admission.reserved.memory_mb
                self._event("release", trial.trial_name, admission)
            self._schedule()

    def submit(self, trial_config: TrialConfig) -> Coroutine[Any, Any, TrialResult]:
        return self._bounded(trial_config, self.admission(trial_config))

    def submit_batch(self, configs: list[TrialConfig]) -> list[Coroutine[Any, Any, TrialResult]]:
        self._validate_agent_concurrency(configs)
        admissions = [self.admission(config) for config in configs]
        return [
            self._bounded(config, admission)
            for config, admission in zip(configs, admissions, strict=True)
        ]


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
