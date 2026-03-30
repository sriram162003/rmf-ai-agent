"""
Task failure monitor and replanner.

Subscribes to RMF task state updates via the RMFClient Socket.IO stream.
When a task enters a failure state it notifies the agent core to replan.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Callable, Coroutine

logger = logging.getLogger(__name__)

FAILURE_STATES = {"failed", "error", "killed"}
TERMINAL_STATES = {"completed", "canceled"} | FAILURE_STATES

# Max replanning attempts before escalating to HITL
MAX_REPLAN_ATTEMPTS = 3


class TaskMonitor:
    """Track in-flight tasks and detect failures."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}
        self._replan_counts: dict[str, int] = defaultdict(int)
        self._failure_callbacks: list[Callable[[str, str, dict], Coroutine]] = []

    def track(self, task_id: str, metadata: dict | None = None) -> None:
        self._tasks[task_id] = metadata or {}
        logger.debug("Tracking task: %s", task_id)

    def on_failure(self, callback: Callable[[str, str, dict], Coroutine]) -> None:
        """Register callback(task_id, state, original_metadata) for failed tasks."""
        self._failure_callbacks.append(callback)

    async def handle_task_update(self, data: dict) -> None:
        task_id = data.get("booking", {}).get("id") or data.get("id", "")
        status = data.get("status", "")

        if not task_id or task_id not in self._tasks:
            return

        logger.debug("Task %s → %s", task_id, status)

        if status in FAILURE_STATES:
            metadata = self._tasks[task_id]
            count = self._replan_counts[task_id]
            self._replan_counts[task_id] += 1

            if count < MAX_REPLAN_ATTEMPTS:
                logger.warning(
                    "Task %s failed (%s) — triggering replan attempt %d/%d",
                    task_id,
                    status,
                    count + 1,
                    MAX_REPLAN_ATTEMPTS,
                )
                for cb in self._failure_callbacks:
                    await cb(task_id, status, metadata)
            else:
                logger.error(
                    "Task %s exceeded max replan attempts — escalating to HITL",
                    task_id,
                )
                for cb in self._failure_callbacks:
                    metadata["_escalate_to_hitl"] = True
                    await cb(task_id, status, metadata)

        if status in TERMINAL_STATES:
            self._tasks.pop(task_id, None)

    def active_tasks(self) -> list[str]:
        return list(self._tasks.keys())

    def replan_count(self, task_id: str) -> int:
        return self._replan_counts[task_id]
