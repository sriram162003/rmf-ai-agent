"""Tests for the TaskMonitor / replanner."""

import asyncio
import pytest
from agent.replanner import TaskMonitor


@pytest.mark.asyncio
async def test_failure_callback():
    monitor = TaskMonitor()
    monitor.track("task-1", {"command": "patrol"})
    received = []

    async def on_fail(task_id, state, meta):
        received.append((task_id, state))

    monitor.on_failure(on_fail)
    await monitor.handle_task_update({"booking": {"id": "task-1"}, "status": "failed"})
    assert received == [("task-1", "failed")]


@pytest.mark.asyncio
async def test_no_callback_for_completed():
    monitor = TaskMonitor()
    monitor.track("task-2", {})
    received = []

    async def on_fail(task_id, state, meta):
        received.append(task_id)

    monitor.on_failure(on_fail)
    await monitor.handle_task_update({"booking": {"id": "task-2"}, "status": "completed"})
    assert received == []


@pytest.mark.asyncio
async def test_task_removed_on_terminal():
    monitor = TaskMonitor()
    monitor.track("task-3", {})
    await monitor.handle_task_update({"booking": {"id": "task-3"}, "status": "completed"})
    assert "task-3" not in monitor.active_tasks()
