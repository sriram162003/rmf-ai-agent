"""Tests for the HITL approval gate."""

import asyncio
import pytest
from agent.hitl import HITLGate, ApprovalStatus


@pytest.mark.asyncio
async def test_approve():
    gate = HITLGate(auto_approve_timeout=5)
    task = asyncio.create_task(gate.request_approval("test_action", "desc"))
    await asyncio.sleep(0.01)
    pending = gate.pending_requests()
    assert len(pending) == 1
    gate.approve(pending[0]["id"])
    result = await task
    assert result is True


@pytest.mark.asyncio
async def test_reject():
    gate = HITLGate(auto_approve_timeout=5)
    task = asyncio.create_task(gate.request_approval("test_action", "desc"))
    await asyncio.sleep(0.01)
    pending = gate.pending_requests()
    gate.reject(pending[0]["id"], reason="not allowed")
    result = await task
    assert result is False


@pytest.mark.asyncio
async def test_timeout():
    gate = HITLGate(auto_approve_timeout=1)
    result = await gate.request_approval("test_action", "desc")
    assert result is False
    reqs = gate.all_requests()
    assert reqs[0]["status"] == ApprovalStatus.TIMEOUT
