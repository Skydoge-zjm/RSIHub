"""Resolve calibrated CPU and memory admission claims for Harbor trials."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import NamedTuple

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
            raise ValueError(
                f"resource profile for {task_name} requires positive integer cpu and memory_mb"
            )
        profiles[task_name] = Claim(cpu, memory_mb)
    return profiles


def host_memory_available_mb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


class ResourcePolicy:
    @classmethod
    def from_environment(cls, n_concurrent: int) -> ResourcePolicy:
        return cls(
            _positive_int("HARBOR_RESOURCE_CPU_SLOTS"),
            _positive_int("HARBOR_RESOURCE_MEMORY_MB"),
            n_concurrent,
        )

    def __init__(self, cpu_limit: int | None, memory_limit: int | None, n_concurrent: int):
        if cpu_limit is None and memory_limit is None:
            raise ValueError("weighted admission requires a CPU or memory budget")
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.name = os.getenv("HARBOR_RESOURCE_POLICY", "greedy-largest-fitting")
        if self.name not in _POLICIES:
            raise ValueError(f"unknown Harbor resource policy: {self.name}")
        self.cpu_fraction = _fraction("HARBOR_RESOURCE_CPU_FRACTION", 0.5)
        self.memory_fraction = _fraction("HARBOR_RESOURCE_MEMORY_FRACTION", 0.25)
        self.memory_floor_mb = _positive_int("HARBOR_RESOURCE_MEMORY_FLOOR_MB", 1024)
        self.memory_cap_mb = _positive_int("HARBOR_RESOURCE_MEMORY_CAP_MB", 6144)
        if (
            self.memory_floor_mb is None
            or self.memory_cap_mb is None
            or self.memory_floor_mb > self.memory_cap_mb
        ):
            raise ValueError("resource memory floor must not exceed the memory cap")
        self.max_active = min(
            n_concurrent,
            _positive_int("HARBOR_RESOURCE_MAX_ACTIVE", n_concurrent) or n_concurrent,
        )
        self.min_available_memory_mb = _nonnegative_int("HARBOR_RESOURCE_MIN_AVAILABLE_MEMORY_MB")
        self.profiles = _load_profiles()

    def fits(self, cpu: int, memory_mb: int) -> bool:
        return (self.cpu_limit is None or cpu <= self.cpu_limit) and (
            self.memory_limit is None or memory_mb <= self.memory_limit
        )

    def admission(self, trial_name: str, task_name: str, official: Claim) -> Admission:
        if not self.fits(official.cpu, official.memory_mb):
            raise ValueError(f"{trial_name} exceeds the job budget: {official}")
        if self.name == "greedy-largest-fitting":
            return Admission(official, official, "declared")

        calibrated = self.profiles.get(task_name)
        if calibrated is not None:
            if calibrated.cpu > official.cpu or calibrated.memory_mb > official.memory_mb:
                raise ValueError(
                    f"{trial_name}: profile reservation {calibrated} "
                    f"exceeds declared limit {official}"
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

    def host_memory_ok(self) -> bool:
        if self.min_available_memory_mb <= 0:
            return True
        available = host_memory_available_mb()
        return available is None or available >= self.min_available_memory_mb
