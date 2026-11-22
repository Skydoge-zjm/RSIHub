"""Run one Harbor job with calibrated weighted trial admission."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
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
_POLICIES = {"greedy-largest-fitting", "calibrated-largest-fitting"}


class Claim(NamedTuple):
    cpu: int
    memory_mb: int


class Admission(NamedTuple):
    official: Claim
    reserved: Claim
    source: str


def _positive_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if not value:
        return default
    if not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    if not value.isascii() or not value.isdecimal() or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _fraction(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number in (0, 1]") from error
    if not 0 < parsed <= 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return parsed


def _load_profiles() -> dict[str, Claim]:
    raw_path = os.environ.get("HARBOR_RESOURCE_PROFILE", "").strip()
    if not raw_path:
        return {}
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load resource profile {path}: {error}") from error
    if payload.get("schema_version") != 1 or not isinstance(payload.get("tasks"), dict):
        raise ValueError("resource profile must contain schema_version=1 and a tasks mapping")
    profiles: dict[str, Claim] = {}
    for task_name, values in payload["tasks"].items():
        if not isinstance(task_name, str) or not isinstance(values, dict):
            raise ValueError("resource profile task entries must be named mappings")
        cpu = values.get("cpu")
        memory_mb = values.get("memory_mb")
        if (
            isinstance(cpu, bool)
            or not isinstance(cpu, int)
            or cpu < 1
            or isinstance(memory_mb, bool)
            or not isinstance(memory_mb, int)
            or memory_mb < 1
        ):
            raise ValueError(f"resource profile for {task_name} requires positive integer cpu and memory_mb")
        profiles[task_name] = Claim(cpu, memory_mb)
    return profiles


def _host_memory_available_mb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


class WeightedTrialQueue(TrialQueue):
    def __init__(self, n_concurrent: int, *args: Any, **kwargs: Any):
        super().__init__(n_concurrent, *args, **kwargs)
        self.cpu_limit = _positive_int("HARBOR_RESOURCE_CPU_SLOTS")
        self.memory_limit = _positive_int("HARBOR_RESOURCE_MEMORY_MB")
        if self.cpu_limit is None and self.memory_limit is None:
            raise ValueError("weighted admission requires a CPU or memory budget")
        self.policy = os.getenv("HARBOR_RESOURCE_POLICY", "greedy-largest-fitting")
        if self.policy not in _POLICIES:
            raise ValueError(f"unknown Harbor resource policy: {self.policy}")
        self.cpu_fraction = _fraction("HARBOR_RESOURCE_CPU_FRACTION", 0.5)
        self.memory_fraction = _fraction("HARBOR_RESOURCE_MEMORY_FRACTION", 0.25)
        self.memory_floor_mb = _positive_int("HARBOR_RESOURCE_MEMORY_FLOOR_MB", 1024)
        self.memory_cap_mb = _positive_int("HARBOR_RESOURCE_MEMORY_CAP_MB", 6144)
        if self.memory_floor_mb is None or self.memory_cap_mb is None or self.memory_floor_mb > self.memory_cap_mb:
            raise ValueError("resource memory floor must not exceed the memory cap")
        self.max_active = min(
            n_concurrent,
            _positive_int("HARBOR_RESOURCE_MAX_ACTIVE", n_concurrent) or n_concurrent,
        )
        self.min_available_memory_mb = _nonnegative_int("HARBOR_RESOURCE_MIN_AVAILABLE_MEMORY_MB")
        self.profiles = _load_profiles()
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
        official = self.official_requirements(trial)
        if not self._fits(official.cpu, official.memory_mb):
            raise ValueError(f"{trial.trial_name} exceeds the job budget: {official}")
        if self.policy == "greedy-largest-fitting":
            return Admission(official, official, "declared")

        task_name, _ = self._task(trial)
        calibrated = self.profiles.get(task_name)
        if calibrated is not None:
            if calibrated.cpu > official.cpu or calibrated.memory_mb > official.memory_mb:
                raise ValueError(
                    f"{trial.trial_name}: profile reservation {calibrated} exceeds declared limit {official}"
                )
            return Admission(official, calibrated, "profile")

        cpu = max(1, math.ceil(official.cpu * self.cpu_fraction))
        memory_mb = max(
            self.memory_floor_mb,
            math.ceil(official.memory_mb * self.memory_fraction),
        )
        memory_mb = min(self.memory_cap_mb, memory_mb)
        reserved = Claim(min(official.cpu, cpu), min(official.memory_mb, memory_mb))
        return Admission(official, reserved, "fraction")

    def requirements(self, trial: TrialConfig) -> Claim:
        return self.admission(trial).reserved

    def _fits(self, cpu: int, memory: int) -> bool:
        return (self.cpu_limit is None or cpu <= self.cpu_limit) and (
            self.memory_limit is None or memory <= self.memory_limit
        )

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
        if self.min_available_memory_mb <= 0:
            return True
        available = _host_memory_available_mb()
        return available is None or available >= self.min_available_memory_mb

    def _defer_for_host_memory(self) -> None:
        now = time.monotonic()
        if now - self.last_host_defer_log >= 30:
            _LOG.info(
                "defer host_memory_available_mb=%s minimum_memory_mb=%d running=%d",
                _host_memory_available_mb(),
                self.min_available_memory_mb,
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
