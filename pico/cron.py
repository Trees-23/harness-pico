"""Small asyncio cron service aligned with Nanobot's scheduler shape."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class CronSchedule:
    kind: Literal["at", "every", "cron"]
    at_ms: int | None = None
    every_ms: int | None = None
    expr: str | None = None
    tz: str | None = None


@dataclass
class CronPayload:
    kind: Literal["system_event", "agent_turn"] = "agent_turn"
    message: str = ""
    deliver: bool = False
    channel: str | None = None
    to: str | None = None


@dataclass
class CronRunRecord:
    run_at_ms: int
    status: Literal["ok", "error", "skipped"]
    duration_ms: int = 0
    error: str | None = None


@dataclass
class CronJobState:
    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: Literal["ok", "error", "skipped"] | None = None
    last_error: str | None = None
    run_history: list[CronRunRecord] = field(default_factory=list)


@dataclass
class CronJob:
    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CronJob":
        state_payload = dict(payload.get("state", {}) or {})
        state_payload["run_history"] = [
            item if isinstance(item, CronRunRecord) else CronRunRecord(**item)
            for item in state_payload.get("run_history", [])
        ]
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            enabled=bool(payload.get("enabled", True)),
            schedule=CronSchedule(**dict(payload.get("schedule", {"kind": "every"}))),
            payload=CronPayload(**dict(payload.get("payload", {}) or {})),
            state=CronJobState(**state_payload),
            created_at_ms=int(payload.get("created_at_ms", payload.get("createdAtMs", 0)) or 0),
            updated_at_ms=int(payload.get("updated_at_ms", payload.get("updatedAtMs", 0)) or 0),
            delete_after_run=bool(payload.get("delete_after_run", payload.get("deleteAfterRun", False))),
        )


@dataclass
class CronStore:
    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None
    if schedule.kind == "every":
        return now_ms + schedule.every_ms if schedule.every_ms and schedule.every_ms > 0 else None
    if schedule.kind == "cron":
        return None
    return None


class CronService:
    _MAX_RUN_HISTORY = 20

    def __init__(
        self,
        store_path: str | Path,
        on_job: Callable[[CronJob], Awaitable[str | None]] | None = None,
        max_sleep_ms: int = 300_000,
    ):
        self.store_path = Path(store_path)
        self.on_job = on_job
        self.max_sleep_ms = int(max_sleep_ms)
        self._store: CronStore | None = None
        self._timer_task: asyncio.Task | None = None
        self._running = False
        self._timer_active = False

    def _load_store(self) -> CronStore:
        if self._store is not None:
            return self._store
        if not self.store_path.exists():
            self._store = CronStore()
            return self._store
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
            self._store = CronStore(
                version=int(payload.get("version", 1)),
                jobs=[CronJob.from_dict(item) for item in payload.get("jobs", [])],
            )
        except Exception:
            self._store = CronStore()
        return self._store

    def _save_store(self) -> None:
        store = self._store or CronStore()
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": store.version,
            "jobs": [asdict(job) for job in store.jobs],
        }
        self.store_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    async def start(self) -> None:
        self._running = True
        self._load_store()
        self._recompute_next_runs()
        self._save_store()
        self._arm_timer()

    def stop(self) -> None:
        self._running = False
        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None

    def _recompute_next_runs(self) -> None:
        store = self._store or self._load_store()
        now_ms = _now_ms()
        for job in store.jobs:
            if job.enabled:
                job.state.next_run_at_ms = _compute_next_run(job.schedule, now_ms)

    def _get_next_wake_ms(self) -> int | None:
        store = self._store or self._load_store()
        times = [job.state.next_run_at_ms for job in store.jobs if job.enabled and job.state.next_run_at_ms]
        return min(times) if times else None

    def _arm_timer(self) -> None:
        if self._timer_task is not None:
            self._timer_task.cancel()
        if not self._running:
            return
        next_wake = self._get_next_wake_ms()
        delay_ms = self.max_sleep_ms if next_wake is None else min(self.max_sleep_ms, max(0, next_wake - _now_ms()))

        async def tick() -> None:
            await asyncio.sleep(delay_ms / 1000)
            if self._running:
                await self._on_timer()

        try:
            self._timer_task = asyncio.create_task(tick())
        except RuntimeError:
            self._timer_task = None

    async def _on_timer(self) -> None:
        store = self._load_store()
        self._timer_active = True
        try:
            now_ms = _now_ms()
            for job in list(store.jobs):
                if job.enabled and job.state.next_run_at_ms and now_ms >= job.state.next_run_at_ms:
                    await self._execute_job(job)
            self._save_store()
        finally:
            self._timer_active = False
        self._arm_timer()

    async def _execute_job(self, job: CronJob) -> None:
        started = _now_ms()
        try:
            if self.on_job is not None:
                await self.on_job(job)
            job.state.last_status = "ok"
            job.state.last_error = None
        except Exception as exc:
            job.state.last_status = "error"
            job.state.last_error = str(exc)
        finished = _now_ms()
        job.state.last_run_at_ms = started
        job.updated_at_ms = finished
        job.state.run_history.append(
            CronRunRecord(
                run_at_ms=started,
                status=job.state.last_status or "error",
                duration_ms=finished - started,
                error=job.state.last_error,
            )
        )
        job.state.run_history = job.state.run_history[-self._MAX_RUN_HISTORY :]
        if job.schedule.kind == "at":
            if job.delete_after_run:
                store = self._store or self._load_store()
                store.jobs = [item for item in store.jobs if item.id != job.id]
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
        else:
            job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        store = self._load_store()
        jobs = store.jobs if include_disabled else [job for job in store.jobs if job.enabled]
        return sorted(jobs, key=lambda job: job.state.next_run_at_ms or float("inf"))

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
    ) -> CronJob:
        store = self._load_store()
        now_ms = _now_ms()
        job = CronJob(
            id=str(uuid.uuid4())[:8],
            name=name,
            enabled=True,
            schedule=schedule,
            payload=CronPayload(message=message, deliver=deliver, channel=channel, to=to),
            state=CronJobState(next_run_at_ms=_compute_next_run(schedule, now_ms)),
            created_at_ms=now_ms,
            updated_at_ms=now_ms,
            delete_after_run=delete_after_run,
        )
        store.jobs.append(job)
        self._save_store()
        self._arm_timer()
        return job

    def register_system_job(self, job: CronJob) -> CronJob:
        store = self._load_store()
        now_ms = _now_ms()
        job.state.next_run_at_ms = _compute_next_run(job.schedule, now_ms)
        job.created_at_ms = job.created_at_ms or now_ms
        job.updated_at_ms = now_ms
        store.jobs = [item for item in store.jobs if item.id != job.id]
        store.jobs.append(job)
        self._save_store()
        self._arm_timer()
        return job

    def remove_job(self, job_id: str) -> Literal["removed", "protected", "not_found"]:
        store = self._load_store()
        job = next((item for item in store.jobs if item.id == job_id), None)
        if job is None:
            return "not_found"
        if job.payload.kind == "system_event":
            return "protected"
        store.jobs = [item for item in store.jobs if item.id != job_id]
        self._save_store()
        self._arm_timer()
        return "removed"

    def status(self) -> dict[str, Any]:
        jobs = self.list_jobs(include_disabled=True)
        return {"running": self._running, "jobs": len(jobs)}


def now_ms_from_datetime(value: datetime) -> int:
    return int(value.timestamp() * 1000)
