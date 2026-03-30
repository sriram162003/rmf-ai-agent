"""
Human-in-the-Loop (HITL) approval gate.

The agent calls request_approval() and awaits a future that resolves
when a human approves or rejects via the dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class ApprovalRequest:
    id: str
    action: str
    description: str
    payload: dict
    status: ApprovalStatus = ApprovalStatus.PENDING
    reason: str = ""
    _future: asyncio.Future = field(default_factory=asyncio.Future, repr=False)


class HITLGate:
    """
    Manages a queue of approval requests.

    Usage:
        gate = HITLGate(auto_approve_timeout=60)
        approved = await gate.request_approval("dispatch_task", "Send robot to floor 3", payload)
    """

    def __init__(self, auto_approve_timeout: int = 60) -> None:
        self._timeout = auto_approve_timeout
        self._requests: dict[str, ApprovalRequest] = {}

    async def request_approval(
        self, action: str, description: str, payload: dict | None = None
    ) -> bool:
        """
        Block until a human approves or rejects.
        Returns True if approved, False if rejected/timeout.
        """
        req = ApprovalRequest(
            id=str(uuid.uuid4()),
            action=action,
            description=description,
            payload=payload or {},
        )
        self._requests[req.id] = req
        logger.info("HITL approval requested: [%s] %s", req.id[:8], description)

        try:
            if self._timeout > 0:
                approved = await asyncio.wait_for(req._future, timeout=self._timeout)
            else:
                approved = await req._future
        except asyncio.TimeoutError:
            req.status = ApprovalStatus.TIMEOUT
            logger.warning("HITL timeout for [%s] — action blocked", req.id[:8])
            return False

        return approved

    def approve(self, request_id: str, reason: str = "") -> bool:
        req = self._requests.get(request_id)
        if req and req.status == ApprovalStatus.PENDING:
            req.status = ApprovalStatus.APPROVED
            req.reason = reason
            if not req._future.done():
                req._future.set_result(True)
            logger.info("HITL approved: [%s]", request_id[:8])
            return True
        return False

    def reject(self, request_id: str, reason: str = "") -> bool:
        req = self._requests.get(request_id)
        if req and req.status == ApprovalStatus.PENDING:
            req.status = ApprovalStatus.REJECTED
            req.reason = reason
            if not req._future.done():
                req._future.set_result(False)
            logger.info("HITL rejected: [%s] reason=%s", request_id[:8], reason)
            return True
        return False

    def pending_requests(self) -> list[dict]:
        return [
            {
                "id": r.id,
                "action": r.action,
                "description": r.description,
                "payload": r.payload,
                "status": r.status.value,
            }
            for r in self._requests.values()
            if r.status == ApprovalStatus.PENDING
        ]

    def all_requests(self) -> list[dict]:
        return [
            {
                "id": r.id,
                "action": r.action,
                "description": r.description,
                "payload": r.payload,
                "status": r.status.value,
                "reason": r.reason,
            }
            for r in self._requests.values()
        ]
