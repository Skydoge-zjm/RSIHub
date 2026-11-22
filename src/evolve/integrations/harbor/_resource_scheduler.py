"""Install the RSIHub weighted resource scheduler into Harbor's trial queue."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from pathlib import Path

MARKER = "_RSIHUB_RESOURCE_SCHEDULER_V1"


PATCHED_QUEUE = r'''
@dataclass(frozen=True)
class _ResourcePermit:
    request_id: int
    cpu: int
    memory_mb: int


class _ResourcePool:
    """Greedy weighted resource pool for task-level Docker reservations."""

    def __init__(self, *, cpu_limit: int, memory_limit_mb: int):
        self.cpu_limit = cpu_limit
        self.memory_limit_mb = memory_limit_mb
        self.used_cpu = 0
        self.used_memory_mb = 0
        self.next_request_id = 0
        self.waiting: dict[int, tuple[int, int]] = {}
        self.condition = asyncio.Condition()

    @classmethod
    def from_environment(cls) -> "_ResourcePool | None":
        raw_cpu = os.environ.get("HARBOR_RESOURCE_CPU_SLOTS", "").strip()
        raw_memory = os.environ.get("HARBOR_RESOURCE_MEMORY_MB", "").strip()
        if not raw_cpu and not raw_memory:
            return None
        try:
            cpu_limit = int(raw_cpu or "1000000")
            memory_limit_mb = int(raw_memory or "1000000000")
        except ValueError as exc:
            raise ValueError("HARBOR_RESOURCE_* values must be integers") from exc
        if cpu_limit < 1 or memory_limit_mb < 1:
            raise ValueError("HARBOR_RESOURCE_* values must be positive")
        policy = os.environ.get("HARBOR_RESOURCE_POLICY", "greedy-largest-fitting")
        if policy != "greedy-largest-fitting":
            raise ValueError(f"Unsupported HARBOR_RESOURCE_POLICY={policy!r}")
        return cls(cpu_limit=cpu_limit, memory_limit_mb=memory_limit_mb)

    @staticmethod
    def _environment_resources(environment: object | None) -> tuple[int, int]:
        if environment is None:
            return 0, 0
        if isinstance(environment, dict):
            cpu = int(environment.get("cpus", 0) or 0)
            memory_mb = int(environment.get("memory_mb", 0) or 0)
            return cpu, memory_mb
        cpu = int(getattr(environment, "cpus", 0) or 0)
        memory_mb = int(getattr(environment, "memory_mb", 0) or 0)
        return cpu, memory_mb

    @classmethod
    def _task_resource_tables(cls, config: TrialConfig) -> list[object]:
        task_path = getattr(config.task, "path", None)
        if task_path is None:
            raise RuntimeError(
                f"Cannot determine resources for {config.trial_name}: "
                "the trial task is not a local task directory."
            )
        task_path = Path(task_path)
        task_file = task_path if task_path.is_file() else task_path / "task.toml"
        if not task_file.is_file():
            raise RuntimeError(
                f"Cannot determine resources for {config.trial_name}: "
                f"missing {task_file}."
            )
        try:
            task_data = tomllib.loads(task_file.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise RuntimeError(
                f"Cannot determine resources for {config.trial_name}: "
                f"cannot read {task_file}."
            ) from exc

        tables: list[object] = [task_data.get("environment")]
        verifier = task_data.get("verifier")
        if isinstance(verifier, dict):
            tables.append(verifier.get("environment"))
        for step in task_data.get("steps") or []:
            if not isinstance(step, dict):
                continue
            tables.append(step.get("environment"))
            step_verifier = step.get("verifier")
            if isinstance(step_verifier, dict):
                tables.append(step_verifier.get("environment"))
        return tables

    @classmethod
    def requirements(cls, config: TrialConfig) -> tuple[int, int]:
        resources = [
            cls._environment_resources(environment)
            for environment in cls._task_resource_tables(config)
        ]
        cpu = max((item[0] for item in resources), default=1)
        memory_mb = max((item[1] for item in resources), default=1)
        return max(cpu, 1), max(memory_mb, 1)

    async def acquire(self, config: TrialConfig) -> _ResourcePermit:
        cpu, memory_mb = self.requirements(config)
        if cpu > self.cpu_limit or memory_mb > self.memory_limit_mb:
            raise RuntimeError(
                f"Trial {config.trial_name} requires {cpu} CPU slots and "
                f"{memory_mb} MiB, exceeding the configured resource pool "
                f"({self.cpu_limit} CPU slots, {self.memory_limit_mb} MiB)."
            )
        request_id = self.next_request_id
        self.next_request_id += 1
        async with self.condition:
            self.waiting[request_id] = (cpu, memory_mb)
            try:
                while True:
                    available_cpu = self.cpu_limit - self.used_cpu
                    available_memory = self.memory_limit_mb - self.used_memory_mb
                    fitting = [
                        (candidate_id, need_cpu, need_memory)
                        for candidate_id, (need_cpu, need_memory) in self.waiting.items()
                        if need_cpu <= available_cpu and need_memory <= available_memory
                    ]
                    if fitting:
                        selected_id, selected_cpu, selected_memory = max(
                            fitting,
                            key=lambda item: (item[2], item[1], -item[0]),
                        )
                        if selected_id == request_id:
                            del self.waiting[request_id]
                            self.used_cpu += selected_cpu
                            self.used_memory_mb += selected_memory
                            return _ResourcePermit(request_id, selected_cpu, selected_memory)
                    await self.condition.wait()
            except BaseException:
                self.waiting.pop(request_id, None)
                self.condition.notify_all()
                raise

    async def release(self, permit: _ResourcePermit) -> None:
        async with self.condition:
            self.used_cpu -= permit.cpu
            self.used_memory_mb -= permit.memory_mb
            self.condition.notify_all()
'''


def _patch_queue(queue_path: Path) -> None:
    source = queue_path.read_text(encoding="utf-8")
    if MARKER in source:
        return
    if importlib.metadata.version("harbor") != "0.27.0":
        raise RuntimeError("RSIHub resource scheduler requires Harbor 0.27.0")

    imports = "import asyncio\nimport shutil\n"
    patched_imports = "import asyncio\nfrom pathlib import Path\nimport shutil\nimport tomllib\n"
    if imports not in source:
        raise RuntimeError("unexpected Harbor queue import block")
    source = source.replace(imports, patched_imports, 1)
    source = source.replace("class TrialQueue:\n", PATCHED_QUEUE + "\nclass TrialQueue:\n", 1)

    init_old = """        self._agent_pool_limits: dict[str, int] = {}
        self._agent_pool_semaphores: dict[str, asyncio.Semaphore] = {}
        self._held_agent_permits: dict[str, list[_AgentSemaphorePermit]] = {}
"""
    init_new = init_old + "        self._resource_pool = _ResourcePool.from_environment()\n"
    if init_old not in source:
        raise RuntimeError("unexpected Harbor queue initializer")
    source = source.replace(init_old, init_new, 1)

    run_old = """    async def _run_trial(self, trial_config: TrialConfig) -> TrialResult:
        \"\"\"Execute a single trial, acquiring the semaphore for concurrency control.\"\"\"
        async with self._semaphore:
            return await self._execute_trial_with_retries(trial_config)
"""
    run_new = """    async def _run_trial(self, trial_config: TrialConfig) -> TrialResult:
        \"\"\"Execute one trial under count and weighted resource limits.\"\"\"
        async with self._semaphore:
            permit = (
                await self._resource_pool.acquire(trial_config)
                if self._resource_pool is not None
                else None
            )
            try:
                return await self._execute_trial_with_retries(trial_config)
            finally:
                if permit is not None:
                    await self._resource_pool.release(permit)
"""
    if run_old not in source:
        raise RuntimeError("unexpected Harbor trial runner")
    source = source.replace(run_old, run_new, 1)
    source = source.replace("class _ResourcePermit:\n", f"# {MARKER}\nclass _ResourcePermit:\n", 1)
    queue_path.write_text(source, encoding="utf-8")


def main() -> None:
    if not os.environ.get("HARBOR_RESOURCE_CPU_SLOTS") and not os.environ.get("HARBOR_RESOURCE_MEMORY_MB"):
        print("resource scheduler disabled")
        return
    queue_module = importlib.import_module("harbor.trial.queue")
    queue_path = Path(queue_module.__file__).resolve()
    _patch_queue(queue_path)
    print(f"resource scheduler installed at {queue_path}")


if __name__ == "__main__":
    main()

